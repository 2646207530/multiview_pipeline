import os
import sys
import random
import json  # <--- 新增：用于保存JSON
from argparse import Namespace
from time import time

import matplotlib.pyplot as plt
import lib.models
import lib.utils.transform
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from lib.utils.mano import MANO
from lib.datasets import create_dataset
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import get_config
from lib.utils.logger import logger
from lib.utils.misc import CONST, format_args_cfg
from lib.utils.net_utils import setup_seed
from lib.utils.config import CN
from lib.utils.io_utils import load_model
from lib.utils.recorder import Recorder
from lib.utils.summary_writer import DDPSummaryWriter
from lib.utils.triangulation import batch_triangulate_dlt_torch
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.utils.transform import batch_cam_extr_transf, batch_cam_intr_projection


def get_proj_2d(pred_joints, T_c2m, K, n_views):
    pred_joints = pred_joints.unsqueeze(1).repeat(1, n_views, 1, 1)  # (B, N, 21, 3)
    pred_joints_in_cam = batch_cam_extr_transf(T_c2m, pred_joints)
    pred_joints_2d = batch_cam_intr_projection(K, pred_joints_in_cam)  # (B*N, 21, 2)
    return pred_joints_2d


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    else:
        return data


def flatten_strings(obj):
    """递归摊平多层嵌套的列表或元组字符串"""
    if isinstance(obj, str):
        return [obj]
    elif isinstance(obj, (list, tuple)):
        res = []
        for item in obj:
            res.extend(flatten_strings(item))
        return res
    return []


def draw_hand_skeleton(img, pts):
    """在 OpenCV 图像上绘制 21 关键点手部骨架"""
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # 大拇指
        (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
        (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
        (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
        (0, 17), (17, 18), (18, 19), (19, 20)  # 小拇指
    ]

    for edge in edges:
        pt1 = (int(pts[edge[0]][0]), int(pts[edge[0]][1]))
        pt2 = (int(pts[edge[1]][0]), int(pts[edge[1]][1]))
        cv2.line(img, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)

    for pt in pts:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)


def get_rendered_frame_at_t(batch, preds, t_idx):
    """
    通过逆仿射变换将指定时间步长 (t_idx) 的预测投影回原图，
    并在水平方向上拼接多个视角，返回单帧画面。
    """
    img = batch["image"]
    B = img.size(0)
    T = img.size(1)
    N = img.size(2)

    hand_id = batch.get('hand_id', torch.tensor([0])).item()

    K = batch['target_cam_intr'].flatten(0, 1)  # (BT, N, 3, 3)
    T_c2m = batch['target_cam_extr'].flatten(0, 1)

    pred_proj_2d = get_proj_2d(preds['master_joints_mvf'], T_c2m, K, N)
    pred_proj_2d = pred_proj_2d.reshape(B, T, N, 21, 2)

    image_paths = flatten_strings(batch['image_path'])
    affines = batch['affine'].reshape(B, T, N, -1, 3).cpu().numpy()

    view_images = []
    for view_id in range(N):
        idx = t_idx * N + view_id
        path = image_paths[idx]

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            logger.warning(f"Could not read image: {path}")
            continue

        pts_crop = pred_proj_2d[0, t_idx, view_id].detach().cpu().numpy()

        affine = affines[0, t_idx, view_id]
        A = np.eye(3)
        A[:2, :] = affine[:2, :]
        A_inv = np.linalg.inv(A)

        pts_crop_homo = np.concatenate([pts_crop, np.ones((21, 1))], axis=1)
        pts_orig = np.dot(A_inv, pts_crop_homo.T).T[:, :2]

        if hand_id == 1:
            img_width = img_bgr.shape[1]
            pts_orig[:, 0] = img_width - pts_orig[:, 0]

        draw_hand_skeleton(img_bgr, pts_orig)
        view_images.append(img_bgr)

    if len(view_images) > 0:
        stitched = np.concatenate(view_images, axis=1)
        h, w = stitched.shape[:2]
        return cv2.resize(stitched, (w // 2, h // 2))

    return None


def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def _init_fn(worker_id):
    seed = worker_id * int(torch.initial_seed()) % CONST.INT_MAX
    np.random.seed(seed)
    random.seed(seed)


def verify_mano(mano_pose,mano_shape,mano_trans,device):
    mano_layer = MANO('right').layer
    mano_layer.to(device)
    batch = mano_pose.shape[0]
    mano_mesh_cam, _ = mano_layer(mano_pose, mano_shape)

    mano_mesh = mano_mesh_cam / 1000
    mano_joint = torch.bmm(
        torch.from_numpy(MANO().joint_regressor).to(mano_pose.device)[None, :, :].repeat(batch, 1, 1), mano_mesh)
    mano_trans=mano_trans.unsqueeze(1)
    mano_joint=mano_joint-mano_joint[:, 9, None]
    mano_joint+=mano_trans

    return mano_joint



def main_worker(cfg: CN, arg: Namespace):
    if hasattr(arg, 'output_dir') and arg.output_dir:
        save_dir = os.path.abspath(arg.output_dir)
    else:
        if hasattr(sys, 'frozen'):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(base_dir, 'mano_vis')
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"!!! IMPORTANT: Results will be saved to: {save_dir}")

    device = torch.device(arg.device)
    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    model.to(device)

    print(f"[{'!' * 10}] 查看模型是否加载在显卡上了: {next(model.parameters()).device}")

    ckpt_rel_path = 'checkpoints'
    ckpt_path = get_resource_path(ckpt_rel_path)

    logger.info(f"Loading checkpoint from: {ckpt_path}")

    if not os.path.exists(ckpt_path):
        fallback_path = 'exp/new/checkpoints/checkpoint_30'
        if os.path.exists(fallback_path):
            ckpt_path = fallback_path
        else:
            raise FileNotFoundError(f"Cannot find checkpoint at {ckpt_path} or {fallback_path}")

    load_model(model, resume_path=ckpt_path, map_location=arg.device)

    val_data = create_dataset(cfg.DATASET.TEST, data_preset=cfg.DATA_PRESET)

    val_loader = DataLoader(val_data,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=int(arg.workers),
                            drop_last=False,
                            worker_init_fn=_init_fn)

    sequence_frames_dict = {}

    # === 新增：用于存放 MANO 参数的字典 ===
    # 结构: sequence_params_dict[seq_name][hand_id][frame_id] = { 'rot': ..., 'pose': ... }
    sequence_params_dict = {}

    with torch.no_grad():
        model.eval()
        for bidx, batch in enumerate(val_loader):
            batch = to_device(batch, device)
            preds = model(batch, 0, 0, "test")

            mano_pose = preds['pose_euler_mv']
            mano_shape = preds['shape_mv']
            mano_cam = preds['cam_mv']
            global_scale = preds['global_scale_mv']
            # 关键: 用 transformer 后的 master_joints_mvf 第 9 个 joint (OpenPose Middle MCP,
            # 跟 verify_mano 的 mano_joint - mano_joint[:, 9] zeroing 约定一致) 作为 trans,
            # 而不是 preds['global_trans_mv'] (= anchor_center, 三角化得到的中间产物).
            #
            # 为什么:
            #   * master_joints_mvf 经过 transformer + MANO decoder 输出, 受 2D 投影 loss
            #     直接监督, 在 (翻图 + cam_intr 翻 cx + target_cam_extr 首行取反) 三件套
            #     新约定下能很快收敛到真实世界系.
            #   * global_trans_mv = anchor_center 来自 CVINet 的 2D anchor 三角化, 没有
            #     直接的 2D 投影 loss 反传, finetune 几个 epoch 收敛慢, 会停在介于旧
            #     X-mirrored frame 和真实世界之间的中间位置 (实测 trans_l X ≈ -0.15,
            #     既不是旧的 -0.32 也不是真的 +0.32).
            # HE 自己的 overlay (master_joints_mvf 投影) 已经对了, 这里让 npy trans 跟它一致.
            global_trans = preds['master_joints_mvf'][:, 9, :]
            mano_cam = global_trans
            # ==== 核心修改：将参数 flatten 以避免时间维度引起的索引错误 ====
            #mano_pose_flat = mano_pose.view(-1, 48)  # [B*T, 48]
            #mano_shape_flat = mano_shape.view(-1, 10)  # [B*T, 10]
            #mano_cam_flat = mano_cam.view(-1, 3)  # [B*T, 3]

            mano_joint=verify_mano(mano_pose,mano_shape,mano_cam,device)



            rot_flat = mano_pose[:, :3]  # 全局旋转 [B*T, 3]
            pose_flat = mano_pose[:, 3:]  # 手部关节姿态 [B*T, 45]

            frames = [f.item() for f in batch['frame_id']]
            seq_name = batch['seq_name'][0] if 'seq_name' in batch else f"seq_{bidx}"
            hand_id = batch.get('hand_id', torch.tensor([0])).item()

            seq_folder = os.path.join(save_dir, f"{seq_name}_hand{hand_id}")
            os.makedirs(seq_folder, exist_ok=True)

            if seq_folder not in sequence_frames_dict:
                sequence_frames_dict[seq_folder] = {}

            # 初始化该 seq 的参数字典
            if seq_name not in sequence_params_dict:
                sequence_params_dict[seq_name] = {0: {}, 1: {}}

            for i, frame_id in enumerate(frames):
                # 记录这一帧的 MANO 参数（由于有滑动窗口，通过判断是否在字典内来去重）
                if frame_id not in sequence_params_dict[seq_name][hand_id]:
                    sequence_params_dict[seq_name][hand_id][frame_id] = {
                        'rot': rot_flat[i].cpu().numpy().tolist(),
                        'pose': pose_flat[i].cpu().numpy().tolist(),
                        'trans': mano_cam[i].cpu().numpy().tolist(),
                        'shape': mano_shape[i].cpu().numpy().tolist()
                    }

                # 记录这帧图片
                if frame_id not in sequence_frames_dict[seq_folder]:
                    frame_img = get_rendered_frame_at_t(batch, preds, i)
                    if frame_img is not None:
                        img_path = os.path.join(seq_folder, f"{frame_id:06d}.jpg")
                        cv2.imwrite(img_path, frame_img)
                        sequence_frames_dict[seq_folder][frame_id] = img_path

    logger.info("Dataloader sequence finished. Now compiling continuous videos and exporting JSON...")

    # ==== 新增：统一导出 JSON 文件 ====
    # 假设 0:右手 (right_hand), 1:左手 (left_hand)
    for seq_name, hand_data in sequence_params_dict.items():
        out_dict = {
            'right_hand': {'rot_r': [], 'pose_r': [], 'trans_r': [], 'shape_r': []},
            'left_hand': {'rot_l': [], 'pose_l': [], 'trans_l': [], 'shape_l': []}
        }

        # 整理右手 (hand_id=0) 数据，按照 frame_id 顺序写入
        if len(hand_data[0]) > 0:
            sorted_frames_r = sorted(hand_data[0].keys())
            out_dict['right_hand']['rot_r'] = [hand_data[0][f]['rot'] for f in sorted_frames_r]
            out_dict['right_hand']['pose_r'] = [hand_data[0][f]['pose'] for f in sorted_frames_r]
            out_dict['right_hand']['trans_r'] = [hand_data[0][f]['trans'] for f in sorted_frames_r]
            out_dict['right_hand']['shape_r'] = [hand_data[0][f]['shape'] for f in sorted_frames_r]

        # 整理左手 (hand_id=1) 数据，按照 frame_id 顺序写入
        if len(hand_data[1]) > 0:
            sorted_frames_l = sorted(hand_data[1].keys())
            out_dict['left_hand']['rot_l'] = [hand_data[1][f]['rot'] for f in sorted_frames_l]
            out_dict['left_hand']['pose_l'] = [hand_data[1][f]['pose'] for f in sorted_frames_l]
            out_dict['left_hand']['trans_l'] = [hand_data[1][f]['trans'] for f in sorted_frames_l]
            out_dict['left_hand']['shape_l'] = [hand_data[1][f]['shape'] for f in sorted_frames_l]

        json_path = os.path.join(save_dir, f"{seq_name}_mano.json")
        with open(json_path, 'w') as f:
            json.dump(out_dict, f)
        logger.info(f"Generated MANO JSON parameters: {json_path}")

    # ==== 新增：用 raw 图给缺伪标签的帧填占位, 让可视化覆盖到全部帧 ====
    # 背景: GolfDataset.build_mapping 只把 pseudo_label_wilor/ 下存在 .npz 的
    #       (seq, cam, frame, hand) 加进 _mapping. WiLoR 漏检的帧对应的 .npz 不存在,
    #       推理跳过 → seq_folder 里那帧的 jpg 也不写, mp4 出现帧序跳变 (例如 780 → 982).
    # 修法: 推理 loop 结束后, 扫描 <root>/<seq>/<cam>/images_undistorted/ 下所有原图,
    #       对每个还没写 jpg 的 frame_id, 直接把多视角原图水平拼接 + 缩半 (跟
    #       get_rendered_frame_at_t 输出格式一致) 写成 placeholder, 不画 overlay.
    #       mp4 合成那一步会把这些占位帧也吃进去, 视频里就没空洞了.
    logger.info("Filling missing frames with raw placeholders for full-frame visualization...")
    try:
        try:
            _dataset_root = val_data.root
        except AttributeError:
            _dataset_root = val_loader.dataset.root
        _view_cams = ['0', '1']  # golf 数据集固定 2 视角 (与 const_cam_view_id 一致)
        for _seq_folder, _frame_map in sequence_frames_dict.items():
            _folder_base = os.path.basename(_seq_folder)
            if "_hand" not in _folder_base:
                continue
            _seq_name = _folder_base.rsplit("_hand", 1)[0]
            _cam0_dir = os.path.join(_dataset_root, _seq_name, _view_cams[0], "images_undistorted")
            if not os.path.isdir(_cam0_dir):
                logger.warning(f"raw image dir not found, skip filling: {_cam0_dir}")
                continue

            _all_fids = []
            for _jpg in sorted(os.listdir(_cam0_dir)):
                if not _jpg.endswith(".jpg"):
                    continue
                try:
                    _all_fids.append(int(os.path.splitext(_jpg)[0]))
                except ValueError:
                    continue

            _n_filled = 0
            for _fid in _all_fids:
                if _fid in _frame_map:
                    continue
                _view_imgs = []
                for _cam in _view_cams:
                    _p = os.path.join(_dataset_root, _seq_name, _cam, "images_undistorted",
                                      f"{_fid:06d}.jpg")
                    _im = cv2.imread(_p)
                    if _im is None:
                        _view_imgs = []
                        break
                    _view_imgs.append(_im)
                if not _view_imgs:
                    continue
                _stitched = np.concatenate(_view_imgs, axis=1)
                _h, _w = _stitched.shape[:2]
                _stitched = cv2.resize(_stitched, (_w // 2, _h // 2))
                _out_path = os.path.join(_seq_folder, f"{_fid:06d}.jpg")
                cv2.imwrite(_out_path, _stitched)
                _frame_map[_fid] = _out_path
                _n_filled += 1
            logger.info(f"[{_seq_folder}] filled {_n_filled} missing frames "
                        f"(total now {len(_frame_map)} / {len(_all_fids)})")
    except Exception as e:
        logger.warning(f"Fill-missing-frames step failed ({e}), 视频会有帧序跳变.")

    # 最后统一排序并合成视频
    for seq_folder, frame_map in sequence_frames_dict.items():
        sorted_frame_ids = sorted(frame_map.keys())
        if len(sorted_frame_ids) == 0:
            continue

        video_path = f"{seq_folder}.mp4"
        logger.info(f"Generating video: {video_path}")

        video_writer = None
        for fid in sorted_frame_ids:
            img_path = frame_map[fid]
            img = cv2.imread(img_path)
            if img is None:
                continue

            if video_writer is None:
                h, w, _ = img.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (w, h))

            video_writer.write(img)

        if video_writer is not None:
            video_writer.release()

    logger.info(f"All sequence videos generated and saved to {save_dir}!")


if __name__ == "__main__":
    arg, _ = parse_exp_args()

    cuda_available = torch.cuda.is_available()

    if cuda_available:
        actual_gpu_count = torch.cuda.device_count()

        if arg.gpu_id is not None:
            try:
                requested_ids = [int(x) for x in arg.gpu_id.split(',')]
                for rid in requested_ids:
                    if rid >= actual_gpu_count:
                        logger.warning(
                            f"Requested GPU ID {rid} exceeds available GPU count {actual_gpu_count}. Resetting to 0.")
                        arg.gpu_id = "0"
                        break
            except ValueError:
                logger.error(f"Invalid gpu_id format: {arg.gpu_id}. Use comma-separated integers like '0' or '0,1'.")
                arg.gpu_id = "0"

            os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu_id
            logger.info(f"Using GPU(s): {arg.gpu_id}")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            logger.info("No GPU ID specified, defaulting to GPU 0")

        arg.device = "cuda"
        arg.n_gpus = torch.cuda.device_count()
    else:
        logger.warning("No NVIDIA GPU detected. Switching to CPU mode.")
        arg.device = "cpu"
        arg.n_gpus = 0
        arg.gpu_id = ""

    bundled_config_rel_path = os.path.join('config', 'WORK1_Inference.yaml')

    potential_config_path = get_resource_path(bundled_config_rel_path)

    if hasattr(sys, '_MEIPASS'):
        logger.info(f"Running in frozen mode. Overriding config path to: {potential_config_path}")
        arg.cfg = potential_config_path
    else:
        if not os.path.exists(arg.cfg) and os.path.exists(potential_config_path):
            arg.cfg = potential_config_path

    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)
    cfg.defrost()

    real_mano_dir = get_resource_path('mano_data')

    if 'MANO' in cfg.MODEL.SV_HEAD:
        cfg.MODEL.SV_HEAD.MANO.MODEL_PATH = real_mano_dir
        base_mean_file = os.path.basename(cfg.MODEL.SV_HEAD.MANO.MEAN_PARAMS)
        cfg.MODEL.SV_HEAD.MANO.MEAN_PARAMS = os.path.join(real_mano_dir, base_mean_file)

    if 'TRANSFORMER' in cfg.MODEL.MV_HEAD and 'MANO' in cfg.MODEL.MV_HEAD.TRANSFORMER:
        cfg.MODEL.MV_HEAD.TRANSFORMER.MANO.MODEL_PATH = real_mano_dir
        base_mean_file = os.path.basename(cfg.MODEL.MV_HEAD.TRANSFORMER.MANO.MEAN_PARAMS)
        cfg.MODEL.MV_HEAD.TRANSFORMER.MANO.MEAN_PARAMS = os.path.join(real_mano_dir, base_mean_file)

    if arg.input_dir:
        if not os.path.exists(arg.input_dir):
            sys.exit(1)
        cfg.DATASET.TEST.DATA_ROOT = arg.input_dir

    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)

    logger.warning(f"final args and cfg: \n{format_args_cfg(arg, cfg)}")
    logger.info("====> Evaluation on single GPU (Data Parallel) <====")

    main_worker(cfg, arg)