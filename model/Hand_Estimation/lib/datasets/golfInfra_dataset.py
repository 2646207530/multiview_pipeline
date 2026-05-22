import os
import cv2
import yaml
import warnings
import random
import numpy as np
import torch
from typing import List
from collections import defaultdict
from torch.utils.data import Dataset
from termcolor import colored
from manotorch.manolayer import ManoLayer

from lib.utils.builder import DATASET
from lib.utils.config import CN
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.transform import get_annot_center, get_annot_scale
from lib.datasets.hdata import HDataset


@DATASET.register_module()
class GolfInfraDataset(HDataset):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.camnum = cfg.N_VIEWS
        self.use_left_hand = cfg.USE_LEFT_HAND
        self.filter_invisible_hand = cfg.FILTER_INVISIBLE_HAND

        self.test_mano_right = ManoLayer(flat_hand_mean=False, side="right", mano_assets_root="assets/mano_v1_2",
                                         use_pca=True, ncomps=45)
        self.test_mano_left = ManoLayer(flat_hand_mean=False, side="left", mano_assets_root="assets/mano_v1_2",
                                        use_pca=True, ncomps=45) if self.use_left_hand else None

        self.root = cfg.DATA_ROOT
        self._mapping = []

        self.build_mapping()
        self.load_dataset()
        self.intrinsics_cache = self._preload_intrinsics()

    def build_mapping(self):
        pseudo_dir = os.path.join(self.root, 'pseudo_label_wilor')
        pesudo_files = sorted(os.listdir(pseudo_dir))
        for pesudo_file in pesudo_files:
            if not pesudo_file.endswith('.npz'):
                continue
            base_name = pesudo_file.replace('.npz', '')
            parts = base_name.rsplit('_', 3)
            if len(parts) != 4:
                continue
            seq_name, camera_id, frame_id, hand_id = parts[0], parts[1], int(parts[2]), int(parts[3])
            self._mapping.append([seq_name, camera_id, frame_id, hand_id])
        print('finish mapping')

    def _preload(self):
        self.name = "Golf_Hand"
        os.environ["UST_Hand_DIR"] = self.root

    def load_dataset(self):
        self._preload()
        self.raw_size = (1440, 1080)
        self.pseudo_dir = os.path.join(self.root, 'pseudo_label_wilor')
        self.data_dir = self.root

        count = 0
        self.sample_idxs = []
        self.rgb_file = []
        self.pseudo_file = []

        for i, (seq_name, cam_id, frame_id, hand_id) in enumerate(etqdm(self._mapping)):
            rgb_file = os.path.join(self.data_dir, seq_name, cam_id, 'images_undistorted', f"{frame_id:06d}.jpg")
            self.rgb_file.append(rgb_file)
            pseudo_file = os.path.join(self.pseudo_dir, f'{seq_name}_{cam_id}_{frame_id}_{hand_id}.npz')
            self.pseudo_file.append(pseudo_file)
            self.sample_idxs.append(count)
            count += 1

        logger.info(
            f"{self.name} Got {colored(len(self.sample_idxs), 'yellow', attrs=['bold'])} samples for data_split {self.data_split}")

    def _preload_intrinsics(self):
        intrinsics_cache = {}
        if not hasattr(self, '_mapping') or not self._mapping:
            return {}

        unique_seqs = set(item[0] for item in self._mapping)
        for seq_name in unique_seqs:
            intrinsics_cache[seq_name] = {}
            for cam_id in ['0', '1']:
                yaml_path = os.path.join(self.data_dir, seq_name, 'calib_undistorted', f'{cam_id}.yaml')
                if os.path.exists(yaml_path):
                    with open(yaml_path, 'r') as f:
                        calib_data = yaml.safe_load(f)
                        intrinsics_cache[seq_name][cam_id] = np.array(calib_data['K'], dtype=np.float32)
        return intrinsics_cache

    def __len__(self):
        return len(self.sample_idxs)

    def get_sample_idx(self) -> List[int]:
        return self.sample_idxs

    def get_label(self, label_file: str):
        return np.load(label_file)

    def get_image(self, idx, order='RGB'):
        path = self.rgb_file[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if order == 'RGB':
            img = img[:, :, ::-1].copy()
        return img

    def get_image_path(self, idx):
        return self.rgb_file[idx]

    def get_pseudo_path(self, idx):
        return self.pseudo_file[idx]

    def get_rawimage_size(self, idx):
        return self.raw_size

    def get_cam_center(self, idx):
        intr = self.get_cam_intr(idx)
        return np.array([intr[0, 2], intr[1, 2]])

    def get_cam_intr(self, idx):
        sample_idx = self.sample_idxs[idx]
        seq_name, cam_id, _, _ = self._mapping[sample_idx]
        intrinsics = self.intrinsics_cache.get(seq_name, {}).get(cam_id)
        if intrinsics is None:
            return np.eye(3, dtype=np.float32)
        # ⚠️ 必须返回 copy: getitem_test 里 `cam_intr[0,2] = W - cam_intr[0,2]`
        # 是 in-place 改, 不 copy 就会污染 cache, 导致下次访问拿到已翻转的值,
        # 出现 "隔一帧 cam_intr 反一下" 的诡异 bug (overlay 在左右手之间跳).
        return intrinsics.copy()

    def get_bbox_center_scale(self, idx):
        joints2d = self.get_joints_2d(idx)
        #joints2d = self.get_update_2d(idx)
        center = get_annot_center(joints2d)
        scale = get_annot_scale(joints2d)
        return center, scale

    def get_update_2d(self, idx):
        pseudo_path = self.get_pseudo_path(idx)
        file_name = os.path.basename(pseudo_path)

        new_pseudo_dir = os.path.join('/home/cyc/UST-Hand', 'update_pesudo')
        pseudo_path = os.path.join(new_pseudo_dir, file_name)

        label = self.get_label(pseudo_path)
        return label["joints_2d"]

    def get_joints_2d(self, idx):
        label = self.get_label(self.get_pseudo_path(idx))
        return label["joints_2d"]

    def get_joints_2d_vis(self, joints_2d=None, raw_size=None, **kwargs):
        joints_vis = ((joints_2d[:, 0] >= 0) & (joints_2d[:, 0] < raw_size[0])) & \
                     ((joints_2d[:, 1] >= 0) & (joints_2d[:, 1] < raw_size[1]))
        return joints_vis.astype(np.float32)

    def get_sides(self, idx):
        w_results = np.load(self.get_pseudo_path(idx))
        return 'right' if w_results['is_right'][0] > 0.5 else 'left'

    def getitem_test(self, idx):
        hand_side = self.get_sides(idx)
        bbox_center, bbox_scale = self.get_bbox_center_scale(idx)
        # 与 GolfDataset (师兄版本) 同步:
        #   - 不 copy cam_intr (跟训练时同行为)
        #   - flip_hand 时 in-place 翻 cam_intr[0,2] 和 cam_center[0]
        cam_intr = self.get_cam_intr(idx)
        cam_center = self.get_cam_center(idx)
        bbox_scale = bbox_scale * self.bbox_expand_ratio
        image_path = self.get_image_path(idx)
        image = self.get_image(idx)
        raw_size = [image.shape[1], image.shape[0]]

        if hand_side != self.sides:
            bbox_center[0] = raw_size[0] - bbox_center[0]
            image = image[:, ::-1, :]

            cam_intr[0, 2] = raw_size[0] - cam_intr[0, 2]
            cam_center[0] = raw_size[0] - cam_center[0]

        label = {
            "idx": idx,
            "bbox_center": bbox_center,
            "bbox_scale": bbox_scale,
            "cam_center": cam_center.copy(),
            "image_path": image_path,
            "raw_size": raw_size,
            "cam_intr": cam_intr.copy(),
        }
        return self.transform(image, label)

    # 接口占位符以兼容 HDataset
    def get_image_mask(self, idx):
        pass

    def get_mano_shape(self, idx):
        pass

    def get_mano_pose(self, idx):
        pass

    def get_verts_3d(self, idx):
        return np.zeros((778, 3), dtype=np.float32)

    def get_verts_uvd(self, idx):
        return np.zeros((778, 3), dtype=np.float32)

    def get_bone_scale(self, idx):
        pass

    def get_sample_identifier(self, idx):
        pass

    def get_joints_3d(self, idx):
        pass

    def get_joints_uvd(self, idx):
        pass




@DATASET.register_module()
class GolfInfraMultiViewTempo(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.name = type(self).__name__
        self.cfg = cfg
        self.root = self.cfg.DATA_ROOT
        self.n_views = cfg.N_VIEWS
        self.data_split = cfg.DATA_SPLIT
        self.skip_frames = cfg.get("SKIP_FRAMES", 0)
        self.window_size = cfg.get("WINDOW_SIZE", 3)
        self.step_size = cfg.get("STEP_SIZE", self.window_size)
        self.master_system = cfg.get("MASTER_SYSTEM", "as_first_camera")
        self.data_mode = cfg.DATA_MODE
        self.CONST_CAM_SERIAL = '0'

        _valset = self._single_view_test()
        self.set_mappings = {f"test": _valset}
        source_set = self.set_mappings[f"test"]

        multivew_mapping = {}
        for i, idx in enumerate(source_set.sample_idxs):
            seq_name, cam_id, frame_id, hand_id = source_set._mapping[idx]
            if (seq_name, frame_id, hand_id) not in multivew_mapping:
                multivew_mapping[(seq_name, frame_id, hand_id)] = [(cam_id, i)]
            else:
                multivew_mapping[(seq_name, frame_id, hand_id)].append((cam_id, i))

        new_multiview_mapping = {}
        for (seq_name, frame_id, hand_id), cam_idx_list in multivew_mapping.items():
            if len(cam_idx_list) < self.n_views:
                continue
            cam_idx_list.sort(key=lambda x: x[0])
            if len(cam_idx_list) > self.n_views:
                cam_idx_list = cam_idx_list[:self.n_views]

            view_idx_list = [x[1] for x in cam_idx_list]
            view_info_list = [{"seq_name": seq_name, "cam_id": cid, "frame_id": frame_id, "hand_id": hand_id} for cid, _
                              in cam_idx_list]
            new_multiview_mapping[(seq_name, frame_id, hand_id)] = (view_idx_list, view_info_list)

        seq_frame_mapping = defaultdict(list)
        for (seq_name, frame_id, hand_id), (idx_list, info_list) in new_multiview_mapping.items():
            seq_frame_mapping[(seq_name, hand_id)].append((frame_id, idx_list, info_list))

        self.multiview_sample_idxs = []
        self.multiview_sample_infos = []
        for (seq_name, hand_id), frame_items in seq_frame_mapping.items():
            frame_items.sort(key=lambda x: x[0])
            n_frames = len(frame_items)
            if n_frames == 0: continue

            start_idxs = []
            current_start = 0
            while current_start + self.window_size <= n_frames:
                start_idxs.append(current_start)
                current_start += self.step_size

            if (n_frames - current_start) > 0:
                start_idxs.append(current_start)

            for start in start_idxs:
                end = start + self.window_size
                window_frames = []
                for i in range(start, end):
                    if i < n_frames:
                        window_frames.append(frame_items[i])
                    else:
                        window_frames.append(frame_items[-1])

                self.multiview_sample_idxs.append([f[1] for f in window_frames])
                self.multiview_sample_infos.append([f[2] for f in window_frames])

        total_windows = len(self.multiview_sample_idxs)
        self.valid_sample_idx_list = [i for i in range(total_windows) if
                                      i % (self.skip_frames + 1) == 0] if self.skip_frames != 0 else list(
            range(total_windows))
        self.len = len(self.valid_sample_idx_list)

        self.extrinsics_cache = {}
        self._preload_all_extrinsics()

    def _preload_all_extrinsics(self):
        unique_seqs = set(self.multiview_sample_infos[idx][0][0]['seq_name'] for idx in self.valid_sample_idx_list)
        for seq_name in unique_seqs:
            parsed_data = {}
            for cam_id in ['0', '1']:
                yaml_path = os.path.join(self.root, seq_name, "calib_undistorted", f"{cam_id}.yaml")
                if os.path.exists(yaml_path):
                    with open(yaml_path, 'r') as f:
                        calib = yaml.safe_load(f)
                    T = np.eye(4, dtype=np.float32)
                    T[:3, :3] = np.array(calib['R'], dtype=np.float32)
                    T[:3, 3] = np.array(calib['t'], dtype=np.float32).flatten()
                    parsed_data[cam_id] = T
            self.extrinsics_cache[seq_name] = parsed_data

    def _single_view_test(self):
        cfg_test = dict(
            DATA_SPLIT="test", DATA_MODE=self.data_mode, NAME=self.cfg.NAME, TYPE=self.cfg.TYPE,
            DATA_ROOT=self.cfg.DATA_ROOT, USE_LEFT_HAND=self.cfg.USE_LEFT_HAND,
            FILTER_INVISIBLE_HAND=self.cfg.FILTER_INVISIBLE_HAND, TRANSFORM=self.cfg.TRANSFORM,
            DATA_PRESET=self.cfg.DATA_PRESET, N_VIEWS=self.cfg.N_VIEWS
        )
        return GolfInfraDataset(CN(cfg_test))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        window_idx = self.valid_sample_idx_list[idx]
        window_idxs = self.multiview_sample_idxs[window_idx]
        window_infos = self.multiview_sample_infos[window_idx]
        window_data = dict()
        frame_ids = []

        for t in range(self.window_size):
            time_idxs = window_idxs[t]
            time_infos = window_infos[t]
            time_data = dict()
            frame_ids.append(time_infos[0]["frame_id"])

            current_seq = time_infos[0]["seq_name"]
            all_extrinsics = self.extrinsics_cache.get(current_seq, {})

            # 跟 GolfMultiViewTempo (师兄版本) 同步:
            # 在 yaml 原始 world→cam_i 阶段就对左手做 X 轴共轭镜像 (R → M·R·M, t → M·t),
            # 等价于把"相机 + 世界"一起沿世界 X=0 平面整体镜像. 配合 getitem_test 的
            # 翻图 + 翻 K cx, 训练后模型输出 3D 落在 X-mirrored 世界系, parse 端补
            # 一次 X → -X 拉回真实世界.
            target_cam_extr_list = []
            for info in time_infos:
                T = all_extrinsics.get(info["cam_id"], np.eye(4, dtype=np.float32)).copy()
                if info["hand_id"] == 1:
                    M = np.array([[-1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]], dtype=np.float32)
                    T[:3, :3] = M @ T[:3, :3] @ M
                    T[:3, 3] = M @ T[:3, 3]
                target_cam_extr_list.append(T)
            time_data["target_cam_extr"] = target_cam_extr_list

            sample_need = [
                'image', 'image_ori', 'cam_intr', "target_cam_intr", 'affine', 'affine_inv',
                'frame_num', 'cam_num', 'file_num', 'rotation', 'cfd', "extr_prerot", "pse_joints_vis", "image_path"
            ]

            time_data["sample_idx"] = time_idxs
            time_data["cam_serial"] = [info["cam_id"] for info in time_infos]

            for vi, (s_idx, info) in enumerate(zip(time_idxs, time_infos)):
                src_sample = self.set_mappings["test"][s_idx]
                for query, value in src_sample.items():
                    if query not in sample_need: continue
                    if query in time_data:
                        time_data[query].append(value)
                    else:
                        time_data[query] = [value]

            new_master_id = 0 if self.master_system == "as_first_camera" else time_data["cam_serial"].index(
                self.CONST_CAM_SERIAL)
            T_master = time_data["target_cam_extr"][new_master_id].copy()

            for i, T_m2c in enumerate(time_data["target_cam_extr"]):
                extr_prerot = time_data.get("extr_prerot", [np.eye(3, dtype=np.float32) for _ in
                                                            range(len(time_data["target_cam_extr"]))])[i]
                extr_prerot_tf_inv = np.eye(4).astype(np.float32)
                extr_prerot_tf_inv[:3, :3] = extr_prerot.T
                T_new_master_2_cam = T_m2c @ np.linalg.inv(T_master)
                time_data["target_cam_extr"][i] = T_new_master_2_cam @ extr_prerot_tf_inv

            for k in time_data.keys():
                if isinstance(time_data[k][0], (np.ndarray, torch.Tensor, int, float)):
                    time_data[k] = np.stack(time_data[k])
            time_data["master_id"] = new_master_id

            for k, v in time_data.items():
                if k in window_data:
                    window_data[k].append(v)
                else:
                    window_data[k] = [v]

        for k in window_data.keys():
            if isinstance(window_data[k][0], np.ndarray):
                window_data[k] = np.stack(window_data[k])

        window_data["window_size"] = self.window_size
        window_data["seq_name"] = window_infos[0][0]["seq_name"]
        window_data['frame_id'] = frame_ids
        window_data['hand_id'] = window_infos[0][0]["hand_id"]
        return window_data