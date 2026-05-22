import hashlib
import json
import os
import pickle
import random
import cv2
import warnings
from typing import List

import copy
import imageio
import numpy as np
import torch
import torch.nn as nn
import yaml  # 新增：用于解析相机标定参数
from manotorch.manolayer import ManoLayer, MANOOutput
from scipy.spatial.distance import cdist
from termcolor import colored
from collections import defaultdict
from copy import deepcopy

from lib.utils.builder import DATASET
from lib.utils.config import CN
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.transform import (batch_ref_bone_len, bbox_xywh_to_xyxy, cal_transform_mean, get_annot_center,
                                 get_annot_scale, persp_project, SE3_transform)
from lib.datasets.hdata import HDataset, kpId2vertices
from torch.utils.data import Dataset


@DATASET.register_module()
class GolfDataset(HDataset):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.camnum = cfg.N_VIEWS
        self.use_left_hand = cfg.USE_LEFT_HAND
        self.filter_invisible_hand = cfg.FILTER_INVISIBLE_HAND

        self.test_mano_right = ManoLayer(
            flat_hand_mean=False,
            side="right",
            mano_assets_root="assets/mano_v1_2",
            use_pca=True,
            ncomps=45,
        )
        self.test_mano_left = ManoLayer(
            flat_hand_mean=False,
            side="left",
            mano_assets_root="assets/mano_v1_2",
            use_pca=True,
            ncomps=45,
        ) if self.use_left_hand else None

        self.root = cfg.DATA_ROOT
        self._mapping = []

        self.build_mapping()
        self.load_dataset()
        self.intrinsics_cache = self._preload_intrinsics()

    def build_mapping(self):
        # 读取 WiLoR 生成的伪标签 (格式: dataset_2026xxxx_xxxxxx_0_125_0.npz)
        pseudo_dir = os.path.join(self.root,'pseudo_label_wilor')
        pesudo_files = sorted(os.listdir(pseudo_dir))
        for pesudo_file in pesudo_files:
            if not pesudo_file.endswith('.npz'):
                continue

            base_name = pesudo_file.replace('.npz', '')
            # 使用 rsplit 从后往前切分 3 次，保证 dataset_xxxx_xxxx 名字完整保留
            parts = base_name.rsplit('_', 3)
            if len(parts) != 4:
                continue

            seq_name = parts[0]  # e.g., dataset_20260107_112118
            camera_id = parts[1]  # e.g., '0' or '1'
            frame_id = int(parts[2])  # e.g., 125
            hand_id = int(parts[3])  # e.g., 0 (right) or 1 (left)

            # 现在的 mapping 结构: [序列名, 相机ID, 帧ID, 手部ID]
            cur_hand = [seq_name, camera_id, frame_id, hand_id]
            self._mapping.append(cur_hand)

        print('finish mapping')

    def _preload(self):
        self.name = "Golf_Hand"
        os.environ["UST_Hand_DIR"] = self.root

    def load_dataset(self):
        self._preload()

        ust_name = f"{self.data_split}"
        logger.info(f"Golf_Hand use split: {ust_name}")
        self.raw_size = (1440, 1080)  # 根据你提供的yaml文件修改了默认分辨率
        self.pseudo_dir = os.path.join(self.root,'pseudo_label_wilor')
        self.data_dir = os.path.join(self.root)

        count = 0
        self.sample_idxs = []
        self.rgb_file = []
        self.pseudo_file = []

        for i, (seq_name, cam_id, frame_id, hand_id) in enumerate(etqdm(self._mapping)):
            # 适配高尔夫数据集的图片路径: 去畸变后的 jpg 文件
            rgb_file = os.path.join(self.data_dir, seq_name, cam_id, 'images_undistorted', f"{frame_id:06d}.jpg")
            self.rgb_file.append(rgb_file)

            pseudo_file = f'{seq_name}_{cam_id}_{frame_id}_{hand_id}.npz'
            pseudo_file = os.path.join(self.pseudo_dir, pseudo_file)
            self.pseudo_file.append(pseudo_file)

            self.sample_idxs.append(count)
            count = count + 1

        logger.info(f"{self.name} Got {colored(len(self.sample_idxs), 'yellow', attrs=['bold'])}"
                    f"/ samples for data_split {self.data_split}")

    def _preload_intrinsics(self):
        """
        加载高尔夫数据集的去畸变内参矩阵
        Cache 结构: { seq_name: { '0': K_matrix, '1': K_matrix } }
        """
        intrinsics_cache = {}

        if not hasattr(self, '_mapping') or not self._mapping:
            return {}

        unique_seqs = set(item[0] for item in self._mapping)
        logger.info(f"Preloading intrinsics for sequences: {unique_seqs}")

        for seq_name in unique_seqs:
            intrinsics_cache[seq_name] = {}
            for cam_id in ['0', '1']:
                # 指向我们刚生成的去畸变标定文件夹
                yaml_path = os.path.join(self.data_dir, seq_name, 'calib_undistorted', f'{cam_id}.yaml')

                if os.path.exists(yaml_path):
                    try:
                        with open(yaml_path, 'r') as f:
                            calib_data = yaml.safe_load(f)
                            # 从 yaml 提取 K 并转为 float32 矩阵
                            intrinsics_cache[seq_name][cam_id] = np.array(calib_data['K'], dtype=np.float32)
                    except Exception as e:
                        logger.error(f"Failed to load intrinsics for {seq_name} cam {cam_id}: {e}")
                else:
                    logger.warning(f"Intrinsics file not found: {yaml_path}")

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
        if not isinstance(img, np.ndarray):
            raise IOError("Fail to read %s" % path)

        if order == 'RGB':
            img = img[:, :, ::-1].copy()
        return img

    def get_image_path(self, idx):
        return self.rgb_file[idx]

    def get_pseudo_path(self, idx):
        return self.pseudo_file[idx]



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

    def get_rawimage_size(self, idx):
        return (1440, 1080)  # 适配你的yaml分辨率



    def get_bbox_center_scale(self, idx):
        joints2d = self.get_joints_2d(idx)
        #joints2d = self.get_update_2d(idx)
        center = get_annot_center(joints2d)
        scale = get_annot_scale(joints2d)
        return center, scale

    def get_joints_2d(self, idx):
        pseudo_path = self.get_pseudo_path(idx)
        label = self.get_label(pseudo_path)
        return label["joints_2d"]

    def get_update_2d(self, idx):
        pseudo_path = self.get_pseudo_path(idx)
        file_name = os.path.basename(pseudo_path)

        new_pseudo_dir = os.path.join('/home/cyc/UST-Hand', 'update_pesudo')
        pseudo_path = os.path.join(new_pseudo_dir, file_name)

        label = self.get_label(pseudo_path)
        return label["joints_2d"]

    def get_joints_2d_vis(self, joints_2d=None, raw_size=None, **kwargs):
        joints_vis = ((joints_2d[:, 0] >= 0) & (joints_2d[:, 0] < raw_size[0])) & \
                     ((joints_2d[:, 1] >= 0) & (joints_2d[:, 1] < raw_size[1]))
        return joints_vis.astype(np.float32)

    def get_sides(self, idx):
        pseudo_file = self.get_pseudo_path(idx)
        w_results = np.load(pseudo_file)
        is_right = w_results['is_right'][0]
        hand_side = 'right' if is_right > 0.5 else 'left'
        return hand_side

    def getitem_test(self, idx):
        hand_side = self.get_sides(idx)
        bbox_center, bbox_scale = self.get_bbox_center_scale(idx)
        cam_intr = self.get_cam_intr(idx)
        cam_center = self.get_cam_center(idx)
        bbox_scale = bbox_scale * self.bbox_expand_ratio
        joints_2d = self.get_joints_2d(idx)
        image_path = self.get_image_path(idx)
        image = self.get_image(idx)

        raw_size = [image.shape[1], image.shape[0]]
        joints_vis = self.get_joints_2d_vis(joints_2d=joints_2d, raw_size=raw_size)

        flip_hand = True if hand_side != self.sides else False

        if flip_hand:
            bbox_center[0] = raw_size[0] - bbox_center[0]
            joints_2d = self.flip_2d(joints_2d, raw_size[0])
            image = image[:, ::-1, :]

            cam_intr[0, 2] = raw_size[0] - cam_intr[0, 2]
            cam_center[0] = raw_size[0] - cam_center[0]

        label = {
            "idx": idx,
            "bbox_center": bbox_center,
            "bbox_scale": bbox_scale,
            "cam_center": cam_center.copy(),
            "joints_2d": joints_2d.copy(),
            'pseudo_2d': joints_2d.copy(),
            "joints_vis": joints_vis.copy(),
            'pseudo_vis': joints_vis.copy(),
            "image_path": image_path,
            "raw_size": raw_size,
            "cam_intr": cam_intr.copy(),
        }

        results = self.transform(image, label)
        return results


@DATASET.register_module()
class GolfDatasetMultiView(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.name = type(self).__name__

        self.cfg = cfg
        self.n_views = cfg.N_VIEWS
        self.data_split = cfg.DATA_SPLIT
        self.skip_frames = cfg.get("SKIP_FRAMES", 0)
        assert self.data_split in ["train", "val", "test"], f"{self.name} unsupport data split {self.data_split}"

        self.master_system = cfg.get("MASTER_SYSTEM", "default")
        if self.master_system == "default":
            self.master_system = "as_first_camera"
            warnings.warn(f"MultiView dataset require you to specify the filed: MASTER_SYSTEM in config file."
                          f"Currently the default value: as_first_camera is used.")

        self.setup = cfg.SETUP
        self.data_mode = cfg.DATA_MODE
        self.const_cam_view_id = '0'  # 高尔夫中是字符 '0'

        self.center_idx = cfg.DATA_PRESET.CENTER_IDX

        _trainset, _valset = self._single_view_test()

        self.set_mappings = {
            f"train": _trainset,
            f"val": _valset,
        }

        self.root = self.set_mappings["train"].root

        source_set_name = f"{self.data_split}"
        source_set = self.set_mappings[source_set_name]

        self.multiview_sample_idxs = []
        self.multiview_sample_infos = []

        multivew_mapping = {}
        for i, idx in enumerate(source_set.sample_idxs):
            seq_name, cam_id, frame_id, hand_id = source_set._mapping[idx]
            # 去掉了 subject_id 和 seq_id 的双重绑定，仅使用 seq_name
            key = (seq_name, frame_id, hand_id)
            if key not in multivew_mapping:
                multivew_mapping[key] = [(cam_id, i)]
            else:
                multivew_mapping[key].append((cam_id, i))

        for key, value in multivew_mapping.items():
            seq_name, frame_id, hand_id = key
            self.multiview_sample_idxs.append([i for (_, i) in value])
            self.multiview_sample_infos.append([{
                "seq_name": seq_name,
                "cam_id": cam_id,
                "frame_id": frame_id,
                "hand_id": hand_id
            } for (cam_id, _) in value])

        total_len = len(self.multiview_sample_idxs)

        if self.skip_frames != 0:
            self.valid_sample_idx_list = [i for i in range(total_len) if i % (self.skip_frames + 1) == 0]
        else:
            self.valid_sample_idx_list = [i for i in range(total_len)]

        self.len = len(self.valid_sample_idx_list)
        logger.warning(f"{self.name} Init Done. Skip frames: {self.skip_frames}, total {self.len} samples")

    def _single_view_test(self):
        cfg_train = dict(
            DATA_SPLIT="train", DATA_MODE=self.data_mode, NAME=self.cfg.NAME, TYPE=self.cfg.TYPE,
            DATA_ROOT=self.cfg.DATA_ROOT, USE_LEFT_HAND=self.cfg.USE_LEFT_HAND,
            FILTER_INVISIBLE_HAND=self.cfg.FILTER_INVISIBLE_HAND, TRANSFORM=self.cfg.TRANSFORM,
            DATA_PRESET=self.cfg.DATA_PRESET, N_VIEWS=self.cfg.N_VIEWS
        )
        cfg_val = cfg_train.copy()
        cfg_val["DATA_SPLIT"] = "val"

        return GolfDataset(CN(cfg_train)), GolfDataset(CN(cfg_val))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        idx = self.valid_sample_idx_list[idx]
        multiview_id_list = self.multiview_sample_idxs[idx]
        multiview_info_list = self.multiview_sample_infos[idx]

        if self.master_system == "as_first_camera":
            if self.data_split == "train":
                lists_to_shuffle = list(zip(multiview_id_list, multiview_info_list))
                random.shuffle(lists_to_shuffle)
                multiview_id_list, multiview_info_list = zip(*lists_to_shuffle)

        elif self.master_system == "as_constant_camera":
            const_v_id = -101
            for vi, info in enumerate(multiview_info_list):
                if info["cam_id"] == self.const_cam_view_id:
                    const_v_id = vi
                    break
            assert const_v_id != -101, f"Cannot find constant camera"
            curr_idx = multiview_id_list.pop(const_v_id)
            curr_info = multiview_info_list.pop(const_v_id)
            multiview_id_list.insert(0, curr_idx)
            multiview_info_list.insert(0, curr_info)

        # 读取 YAML 获取外参 T = [R|t]
        seq_name = multiview_info_list[0]["seq_name"]
        camera_extr_mapping = {}
        for cam_id in ['0', '1']:
            yaml_path = os.path.join(self.root, seq_name, 'calib_undistorted', f'{cam_id}.yaml')
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r') as f:
                    calib = yaml.safe_load(f)
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = np.array(calib['R'], dtype=np.float32)
                T[:3, 3] = np.array(calib['t'], dtype=np.float32).flatten()
                # 注意：高尔夫数据集中 t 已是米级别，无需原代码中的 /= 1000.0
                camera_extr_mapping[cam_id] = T

        sample = dict()
        sample["sample_idx"] = multiview_id_list
        sample["target_cam_extr"] = list()

        for info in multiview_info_list:
            cam_id = info["cam_id"]
            T_master_2_cam = camera_extr_mapping.get(cam_id, np.eye(4, dtype=np.float32))
            sample["target_cam_extr"].append(T_master_2_cam)

        sample_need = [
            'image', 'image_ori', 'cam_intr', "target_cam_intr", "target_cam_extr",
            "pse_joints_2d", "pse_pose", "pse_shape", 'target_pseudo_uv', 'target_joints_2d',
            'target_joints_uvd', 'target_joints_3d', 'target_joints_heatmap',
            "pse_joints_heatmap", "pse_joints_vis", "target_joints_vis", "joints_vis",
            'affine_inv', 'frame_num', 'cam_num', 'file_num', 'rotation', 'extr_prerot'
        ]

        for i, info in zip(multiview_id_list, multiview_info_list):
            source_set = self.set_mappings[f"{self.data_split}"]
            src_sample = source_set[i]
            for query, value in src_sample.items():
                if query not in sample_need:
                    continue
                if query in sample:
                    sample[query].append(value)
                else:
                    sample[query] = [value]

        if self.master_system == "as_constant_camera":
            master_id = 0
            T_master_2_new_master = sample["target_cam_extr"][master_id].copy()

        for i, T_m2c in enumerate(sample["target_cam_extr"]):
            # 如果源样本没有 extr_prerot，补上单位阵避免报错
            extr_prerot = \
            sample.get("extr_prerot", [np.eye(3, dtype=np.float32) for _ in range(len(sample["target_cam_extr"]))])[i]
            extr_prerot_tf_inv = np.eye(4).astype(np.float32)
            extr_prerot_tf_inv[:3, :3] = extr_prerot.T
            if self.master_system == "as_constant_camera":
                T_new_master_2_cam = T_master_2_new_master @ np.linalg.inv(T_m2c)
            else:
                T_new_master_2_cam = sample["target_cam_extr"][0].copy() @ np.linalg.inv(T_m2c)
            sample["target_cam_extr"][i] = T_new_master_2_cam @ extr_prerot_tf_inv

        for query in sample.keys():
            if isinstance(sample[query][0], (int, float, np.ndarray, torch.Tensor)):
                sample[query] = np.stack(sample[query])

        return sample


@DATASET.register_module()
class GolfMultiViewTempo(Dataset):

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
        self.CONST_CAM_SERIAL = '0'  # 高尔夫中为 '0'

        _trainset, _valset = self._single_view_test()
        self.set_mappings = {f"train": _trainset, f"test": _valset}
        source_set = self.set_mappings[f"{self.data_split}"]

        multivew_mapping = {}
        for i, idx in enumerate(source_set.sample_idxs):
            seq_name, cam_id, frame_id, hand_id = source_set._mapping[idx]
            key = (seq_name, frame_id, hand_id)
            if key not in multivew_mapping:
                multivew_mapping[key] = [(cam_id, i)]
            else:
                multivew_mapping[key].append((cam_id, i))

        new_multiview_mapping = {}
        for (seq_name, frame_id, hand_id), cam_idx_list in multivew_mapping.items():
            if len(cam_idx_list) < self.n_views:
                continue
            #if hand_id == 1:  # 跳过左手
            #    continue
            cam_idx_list.sort(key=lambda x: x[0])
            if len(cam_idx_list) > self.n_views:
                cam_idx_list = cam_idx_list[:self.n_views]

            view_idx_list = [x[1] for x in cam_idx_list]
            view_info_list = [{
                "seq_name": seq_name, "cam_id": cid, "frame_id": frame_id, "hand_id": hand_id
            } for cid, _ in cam_idx_list]
            new_multiview_mapping[(seq_name, frame_id, hand_id)] = (view_idx_list, view_info_list)

        seq_frame_mapping = defaultdict(list)
        for (seq_name, frame_id, hand_id), (idx_list, info_list) in new_multiview_mapping.items():
            seq_frame_mapping[(seq_name, hand_id)].append((frame_id, idx_list, info_list))

        self.multiview_sample_idxs = []
        self.multiview_sample_infos = []
        for (seq_name, hand_id), frame_items in seq_frame_mapping.items():
            frame_items.sort(key=lambda x: x[0])
            n_frames = len(frame_items)
            if n_frames == 0:
                continue

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
        """扫描所有数据集，从 yaml 加载外参矩阵"""
        unique_seqs = set()
        for idx in self.valid_sample_idx_list:
            unique_seqs.add(self.multiview_sample_infos[idx][0][0]['seq_name'])

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
        cfg_train = dict(
            DATA_SPLIT="train", DATA_MODE=self.data_mode, NAME=self.cfg.NAME, TYPE=self.cfg.TYPE,
            DATA_ROOT=self.cfg.DATA_ROOT, USE_LEFT_HAND=self.cfg.USE_LEFT_HAND,
            FILTER_INVISIBLE_HAND=self.cfg.FILTER_INVISIBLE_HAND, TRANSFORM=self.cfg.TRANSFORM,
            DATA_PRESET=self.cfg.DATA_PRESET, N_VIEWS=self.cfg.N_VIEWS
        )
        cfg_val = cfg_train.copy()
        cfg_val["DATA_SPLIT"] = "val"
        return GolfDataset(CN(cfg_train)), GolfDataset(CN(cfg_val))

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
            #time_data["target_cam_extr"] = [all_extrinsics.get(info["cam_id"], np.eye(4, dtype=np.float32)) for info in
            #                                time_infos]

            target_cam_extr_list = []
            for info in time_infos:
                # 加上 .copy() 防止直接修改全局缓存
                T = all_extrinsics.get(info["cam_id"], np.eye(4, dtype=np.float32)).copy()


                # 如果是左手 (hand_id == 1)，对相机外参进行 X 轴镜像
                if info["hand_id"] == 1:
                    M = np.array([[-1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]], dtype=np.float32)
                    T[:3, :3] = M @ T[:3, :3] @ M
                    T[:3, 3] = M @ T[:3, 3]



                target_cam_extr_list.append(T)

            time_data["target_cam_extr"] = target_cam_extr_list

            sample_need = [
                'image', 'image_ori', 'cam_intr', "target_cam_intr",'affine',
                "pse_joints_2d", 'target_pseudo_uv', "pse_joints_heatmap", "pse_joints_vis", 'affine_inv', 'frame_num',
                'cam_num', 'file_num', 'rotation', 'cfd', "extr_prerot","image_path"
            ]

            time_data["sample_idx"] = time_idxs
            time_data["cam_serial"] = [info["cam_id"] for info in time_infos]

            for vi, (s_idx, info) in enumerate(zip(time_idxs, time_infos)):
                src_sample = self.set_mappings[self.data_split][s_idx]
                for query, value in src_sample.items():
                    if query not in sample_need: continue
                    if query in time_data:
                        time_data[query].append(value)
                    else:
                        time_data[query] = [value]

            if self.master_system == "as_first_camera":
                new_master_id = 0
            else:
                new_master_id = time_data["cam_serial"].index(self.CONST_CAM_SERIAL)
            T_master = time_data["target_cam_extr"][new_master_id].copy()

            for i, T_m2c in enumerate(time_data["target_cam_extr"]):
                extr_prerot = time_data.get("extr_prerot", [np.eye(3, dtype=np.float32) for _ in
                                                            range(len(time_data["target_cam_extr"]))])[i]
                extr_prerot_tf_inv = np.eye(4).astype(np.float32)
                extr_prerot_tf_inv[:3, :3] = extr_prerot.T
                T_new_master_2_cam =T_m2c @ np.linalg.inv(T_master)#np.linalg.inv(T_m2c)@ T_master
                T_rel = T_new_master_2_cam @ extr_prerot_tf_inv

                time_data["target_cam_extr"][i] = T_rel

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