"""
一键手部 + 物体参数提取脚本

功能:
  1. HaMER 手部推理 (vGesture 环境)
  2. 物体位姿注入 (从逐帧 4x4 txt 文件夹读取)
  3. 聚合所有参数写入单个 npy
  4. (可选) 手物 Concat: 将左右手平移对齐到物体上的指定局部坐标点
  5. (可选) Contact 优化: 调用 train_contact.py 优化手物接触
  6. (可选) 力闭合优化: 调用 finetune_force_closure.py 优化力闭合
  7. (可选) 手物重放: 将参考帧的手物相对位姿刚性重放到整个序列

用法 (仅手部):
  python run_hamer_to_npy.py \
    --input /path/to/images \
    --output /path/to/data.npy \
    --seq_name seq_136 \
    --cam_k /path/to/cam_K.txt

用法 (手部 + 物体位姿提取 + 物体位姿平滑):
  python run_hamer_to_npy.py \
    --input /home/pt/fbs/FoundationPoseROS2/FoundationPose/demo_data/ours/rgb \
    --output /home/pt/fbs/data_smooth.npy \
    --seq_name seq_136 \
    --cam_k /home/pt/fbs/FoundationPoseROS2/FoundationPose/demo_data/ours/cam_K.txt \
    --obj_folder /home/pt/fbs/FoundationPoseROS2/FoundationPose/debug/ob_in_cam \
    --obj_pose_sigma 2.0

用法 (手部 + 物体 + 手物Concat):
  python run_hamer_to_npy.py \
    --input /home/pt/fbs/FoundationPoseROS2/FoundationPose/demo_data/ours/rgb \
    --output /home/pt/fbs/data_fixed.npy \
    --seq_name seq_136 \
    --cam_k /home/pt/fbs/FoundationPoseROS2/FoundationPose/demo_data/ours/cam_K.txt \
    --obj_folder /home/pt/fbs/FoundationPoseROS2/FoundationPose/debug/ob_in_cam \
    --point_r 0.36 0.0 0.0 \
    --point_l 0.56 0.0 0.0

用法 (手部 + 物体 + 手物Concat + Contact优化 + 力闭合优化 + 手物重放):
  python run_hamer_to_npy.py \
    --input /home/pt/fbs/FoundationPoseROS2/FoundationPose/demo_data/ours/rgb \
    --output /home/pt/fbs/data_fixed.npy \
    --seq_name seq_136 \
    --cam_k /home/pt/fbs/FoundationPoseROS2/FoundationPose/demo_data/ours/cam_K.txt \
    --obj_folder /home/pt/fbs/FoundationPoseROS2/FoundationPose/debug/ob_in_cam \
    --point_r 0.36 0.0 0.0 \
    --point_l 0.56 0.0 0.0 \
    --opt_range 120 200 \
    --force_closure_range 120 200 \
    --ref_frame 182
"""

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import sys
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'model'))
sys.path.insert(0, os.path.join(_ROOT, 'model', 'hamer'))

import argparse
import glob
import subprocess

import cv2
import numpy as np
import torch
import smplx
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from model.hamer.infer import (
    hamer_inference,
    matrix_to_axis_angle,
    load_intrinsics,
)
from model.rootnet.Model_RGB import get_model
from config.hamer_config import hamer_opt
from config.yolo_config import yolo_opt
from yolo.detector import Detector


def parse_detections(dets):
    """将 YOLO 检测结果统一为 [['right', [x1,y1,x2,y2]], ...] 格式"""
    if not isinstance(dets, list) or len(dets) == 0:
        return []
    if isinstance(dets[0], list) and len(dets[0]) > 0 and isinstance(dets[0][0], list):
        return dets[0]
    return dets


def extract_hand_params(output, mano_params, hand_label):
    """
    从 HaMER 推理结果中提取单只手的 MANO 参数，
    返回 dict 或在异常时返回 None。
    """
    betas_np = mano_params['betas'].detach().cpu().numpy().squeeze()

    hand_pose_mats = mano_params['hand_pose'].detach().cpu().numpy().squeeze()
    hand_pose_aa = matrix_to_axis_angle(hand_pose_mats)

    global_orient_mat = mano_params['global_orient'].detach().cpu().numpy().squeeze()
    if global_orient_mat.ndim == 3:
        global_orient_mat = global_orient_mat[0]
    global_orient_aa, _ = cv2.Rodrigues(global_orient_mat)
    global_orient_aa = global_orient_aa.flatten()

    cam_t_np = output['pred_cam_t_full'].detach().cpu().numpy().squeeze()

    return {
        'betas': betas_np,            # (10,)
        'pose_hand': hand_pose_aa,    # (45,)
        'pose_global': global_orient_aa,  # (3,)
        'cam_t': cam_t_np,            # (3,)
        'is_right': (hand_label == 'right'),
    }


NAN_RIGHT = {
    'rot':   np.full((3,),  np.nan, dtype=np.float32),
    'pose':  np.full((45,), np.nan, dtype=np.float32),
    'shape': np.full((10,), np.nan, dtype=np.float32),
    'trans': np.full((3,),  np.nan, dtype=np.float32),
}
NAN_LEFT = NAN_RIGHT  # 结构相同

MANO_MODEL_PATH = os.environ.get(
    "MANO_MODEL_PATH",
    os.path.join(_ROOT, "MANO"),
)
_MIRROR_LEFT = np.array([1.0, -1.0, -1.0], dtype=np.float32)


def _right_to_left_mano(rot, pose):
    """将右手 MANO 轴角参数镜像为左手：(x, y, z) → (x, -y, -z)。"""
    return rot * _MIRROR_LEFT, (pose.reshape(-1, 3) * _MIRROR_LEFT).reshape(pose.shape)


def _clean(arr):
    """将 NaN / Inf 置零并统一为 float32。"""
    if hasattr(arr, "__len__") and len(arr) and np.issubdtype(np.asarray(arr).dtype, np.floating):
        return np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(arr, dtype=np.float32)


def _interpolate_nan(arr):
    """
    对 (N, D) 数组做逐列线性插值，就地填补 NaN 间隙。
    序列首尾的 NaN 使用最近有效值做常量外推。
    返回本次填补的缺失帧数。
    """
    if not np.isnan(arr).any():
        return 0

    N, D = arr.shape
    valid_mask = ~np.isnan(arr).any(axis=1)

    if not valid_mask.any():
        return 0

    all_idx = np.arange(N)
    valid_idx = all_idx[valid_mask]
    valid_data = arr[valid_mask]

    for d in range(D):
        arr[:, d] = np.interp(all_idx, valid_idx, valid_data[:, d])

    return int(N - valid_mask.sum())


def _slerp_interpolate_nan(rot_arr):
    """
    对 (N, 3) 的轴角数组用 SLERP 做旋转插值，就地修改。
    避免逐分量线性插值导致旋转"绕远路"的问题。
    序列首尾超出有效帧范围的部分用最近有效值常量外推。
    返回填补的帧数。
    """
    if not np.isnan(rot_arr).any():
        return 0

    N = rot_arr.shape[0]
    valid_mask = ~np.isnan(rot_arr).any(axis=1)

    if not valid_mask.any() or valid_mask.all():
        return 0

    valid_idx = np.where(valid_mask)[0]
    valid_rotations = R.from_rotvec(rot_arr[valid_idx])

    slerp = Slerp(valid_idx.astype(float), valid_rotations)

    interp_min, interp_max = valid_idx[0], valid_idx[-1]
    all_idx = np.arange(N)

    inner_mask = (all_idx >= interp_min) & (all_idx <= interp_max)
    inner_idx = all_idx[inner_mask].astype(float)
    interpolated = slerp(inner_idx)
    rot_arr[inner_mask] = interpolated.as_rotvec().astype(np.float32)

    if interp_min > 0:
        rot_arr[:interp_min] = rot_arr[interp_min]
    if interp_max < N - 1:
        rot_arr[interp_max + 1:] = rot_arr[interp_max]

    return int(N - valid_mask.sum())


def concat_hand_to_object(root, seq_name, point_r, point_l):
    """
    就地修改 root dict：将左右手 trans 重新计算，
    使 MANO 关节几何中心对齐到物体局部坐标系中的指定点。
    """
    params = root["data_dict"][seq_name]["params"]
    hand_r = params["right hand"]
    hand_l = params["left hand"]
    obj = params["object"]

    if obj["obj_rot"] is None or obj["obj_trans"] is None:
        print("\n[手物Concat] 物体位姿不可用，跳过手物对齐")
        return

    obj_trans = _clean(obj["obj_trans"])
    obj_rot   = _clean(obj["obj_rot"])

    for k in ("pose_r", "shape_r", "rot_r", "trans_r"):
        if k in hand_r:
            hand_r[k] = _clean(hand_r[k])
    for k in ("pose_l", "shape_l", "rot_l", "trans_l"):
        if k in hand_l:
            hand_l[k] = _clean(hand_l[k])

    pose_r, shape_r, rot_r = hand_r["pose_r"], hand_r["shape_r"], hand_r["rot_r"]
    pose_l, shape_l, rot_l = hand_l["pose_l"], hand_l["shape_l"], hand_l["rot_l"]

    n_frames = min(
        obj_trans.shape[0],
        pose_r.shape[0], shape_r.shape[0], rot_r.shape[0],
        pose_l.shape[0], shape_l.shape[0], rot_l.shape[0],
    )

    obj_trans = obj_trans[:n_frames]
    obj_rot   = obj_rot[:n_frames]
    pose_r  = pose_r[:n_frames];  shape_r = shape_r[:n_frames]; rot_r = rot_r[:n_frames]
    pose_l  = pose_l[:n_frames];  shape_l = shape_l[:n_frames]; rot_l = rot_l[:n_frames]

    print(f"\n[手物Concat] 总帧数: {n_frames}")
    print(f"[手物Concat] 右手目标点 (物体局部): {point_r}")
    print(f"[手物Concat] 左手目标点 (物体局部): {point_l}")

    r_obj = R.from_rotvec(obj_rot)
    target_cam_r = obj_trans + r_obj.apply(point_r)
    target_cam_l = obj_trans + r_obj.apply(point_l)

    # ── 右手 MANO ──
    print("[手物Concat] 创建右手 MANO 模型 ...")
    mano_r = smplx.create(
        model_path=os.path.dirname(MANO_MODEL_PATH),
        model_type="MANO", is_rhand=True,
        use_pca=False, flat_hand_mean=True,
    )
    with torch.no_grad():
        out_r = mano_r(
            global_orient=torch.tensor(rot_r, dtype=torch.float32),
            hand_pose=torch.tensor(pose_r, dtype=torch.float32),
            betas=torch.tensor(shape_r, dtype=torch.float32),
            transl=torch.zeros((n_frames, 3), dtype=torch.float32),
        )
    joints_offset_r = np.mean(out_r.joints.cpu().numpy(), axis=1)
    hand_r["trans_r"] = (target_cam_r - joints_offset_r).astype(np.float32)
    print("[手物Concat] 已更新 trans_r，右手关节中心 → 物体 point_r")

    # ── 左手 MANO（npy 左手槽位存右手参数，先镜像 Y/Z）──
    print("[手物Concat] 创建左手 MANO 模型 ...")
    rot_l_m, pose_l_m = _right_to_left_mano(rot_l, pose_l)
    mano_l = smplx.create(
        model_path=os.path.dirname(MANO_MODEL_PATH),
        model_type="MANO", is_rhand=False,
        use_pca=False, flat_hand_mean=True,
    )
    with torch.no_grad():
        out_l = mano_l(
            global_orient=torch.tensor(rot_l_m, dtype=torch.float32),
            hand_pose=torch.tensor(pose_l_m, dtype=torch.float32),
            betas=torch.tensor(shape_l, dtype=torch.float32),
            transl=torch.zeros((n_frames, 3), dtype=torch.float32),
        )
    joints_offset_l = np.mean(out_l.joints.cpu().numpy(), axis=1)
    hand_l["trans_l"] = (target_cam_l - joints_offset_l).astype(np.float32)
    print("[手物Concat] 已更新 trans_l，左手关节中心 → 物体 point_l")

    # ── 帧数裁剪（保持所有数组长度一致）──
    n_orig = obj["obj_trans"].shape[0] if isinstance(obj["obj_trans"], np.ndarray) else -1
    if n_frames < n_orig:
        for store in (obj, hand_r, hand_l):
            for key, arr in list(store.items()):
                if isinstance(arr, np.ndarray) and arr.shape[0] == n_orig:
                    store[key] = arr[:n_frames].astype(np.float32)

    print("[手物Concat] 手物对齐完成")


# ── 手物重放 (rehand) 辅助函数 ──

def _make_T(rot_mat, trans):
    """构建 4x4 齐次变换矩阵。"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot_mat
    T[:3, 3] = trans
    return T


def _build_obj_T(rot_vec, trans):
    """从轴角 + 平移构建物体的 4x4 变换。"""
    return _make_T(R.from_rotvec(rot_vec).as_matrix(), trans)


def _build_hand_T(rot_vec, trans, j0):
    """构建 MANO 手在世界坐标系中的真实 4x4 刚体变换。

    MANO 的 global_orient 绕根关节 j0 旋转:
      v_world = R @ (v - j0) + j0 + transl = R @ v + (I-R)@j0 + transl
    所以有效平移 t_eff = (I-R)@j0 + transl
    """
    R_mat = R.from_rotvec(rot_vec).as_matrix()
    t_eff = (np.eye(3) - R_mat) @ j0 + trans
    return _make_T(R_mat, t_eff)


def _decompose_hand_T(T, j0):
    """从 4x4 有效变换中反解 MANO 的 rot (轴角) 和 transl。"""
    R_mat = T[:3, :3]
    t_eff = T[:3, 3]
    rot_vec = R.from_matrix(R_mat).as_rotvec().astype(np.float32)
    transl = (t_eff - (np.eye(3) - R_mat) @ j0).astype(np.float32)
    return rot_vec, transl


def _compute_j0(mano_model, hand_pose, betas):
    """用 MANO 前向传播计算根关节(手腕)在零姿态零平移时的位置。"""
    with torch.no_grad():
        out = mano_model(
            global_orient=torch.zeros(1, 3, dtype=torch.float32),
            hand_pose=torch.tensor(hand_pose[np.newaxis], dtype=torch.float32),
            betas=torch.tensor(betas[np.newaxis], dtype=torch.float32),
            transl=torch.zeros(1, 3, dtype=torch.float32),
        )
    j0 = out.joints[0, 0].numpy().astype(np.float64)
    return j0


def rehand_by_object(root, seq_name, ref_frame):
    """
    将参考帧的手"粘"到物体上，整个序列的手刚性跟随物体运动。
    就地修改 root dict。

    步骤:
      1. 用 MANO 计算手腕根关节位置 j0（global_orient 的旋转中心）
      2. 构建手的真实刚体变换: T_hand_eff = [R, (I-R)@j0 + transl; 0, 1]
      3. 计算手在物体局部坐标系下的相对变换: T_rel = inv(T_obj[ref]) @ T_hand_eff[ref]
      4. 整个序列: T_hand_eff_new = T_obj[i] @ T_rel → 反解 MANO 的 rot 和 transl
      5. 同时锁定参考帧的 pose / shape
    """
    params = root["data_dict"][seq_name]["params"]
    obj = params["object"]

    if obj["obj_rot"] is None or obj["obj_trans"] is None:
        print("\n[手物重放] 物体位姿不可用，跳过")
        return

    obj_rot = np.asarray(obj["obj_rot"], dtype=np.float64)
    obj_trans = np.asarray(obj["obj_trans"], dtype=np.float64)
    n_frames = obj_rot.shape[0]

    if ref_frame >= n_frames:
        raise ValueError(f"[手物重放] 参考帧 {ref_frame} 超出总帧数 {n_frames}")

    print(f"\n[手物重放] 序列: {seq_name}, 总帧数: {n_frames}, 参考帧: {ref_frame}")

    T_obj_ref = _build_obj_T(obj_rot[ref_frame], obj_trans[ref_frame])
    T_obj_ref_inv = np.linalg.inv(T_obj_ref)

    # ── 右手 ──
    print("[手物重放] 处理右手 ...")
    hand_r = params["right hand"]
    rot_r = np.asarray(hand_r["rot_r"], dtype=np.float64)
    trans_r = np.asarray(hand_r["trans_r"], dtype=np.float64)
    pose_r = np.asarray(hand_r["pose_r"], dtype=np.float64)
    shape_r = np.asarray(hand_r["shape_r"], dtype=np.float64)

    mano_r = smplx.create(
        model_path=os.path.dirname(MANO_MODEL_PATH),
        model_type="MANO", is_rhand=True,
        use_pca=False, flat_hand_mean=True,
    )
    j0_r = _compute_j0(mano_r, pose_r[ref_frame].astype(np.float32),
                        shape_r[ref_frame].astype(np.float32))
    print(f"  右手 j0 (根关节偏移) = {j0_r}")

    T_hand_ref_r = _build_hand_T(rot_r[ref_frame], trans_r[ref_frame], j0_r)
    T_rel_r = T_obj_ref_inv @ T_hand_ref_r

    ref_pose_r = pose_r[ref_frame].copy()
    ref_shape_r = shape_r[ref_frame].copy()

    for i in range(n_frames):
        T_obj_i = _build_obj_T(obj_rot[i], obj_trans[i])
        T_new = T_obj_i @ T_rel_r
        rot_r[i], trans_r[i] = _decompose_hand_T(T_new, j0_r)
        pose_r[i] = ref_pose_r
        shape_r[i] = ref_shape_r

    hand_r["rot_r"] = rot_r.astype(np.float32)
    hand_r["trans_r"] = trans_r.astype(np.float32)
    hand_r["pose_r"] = pose_r.astype(np.float32)
    hand_r["shape_r"] = shape_r.astype(np.float32)
    print(f"  右手: 已将参考帧 {ref_frame} 的手物相对位姿重放到全部 {n_frames} 帧")

    # ── 左手 ──
    if "left hand" in params:
        print("[手物重放] 处理左手 ...")
        hand_l = params["left hand"]
        rot_l = np.asarray(hand_l["rot_l"], dtype=np.float64)
        trans_l = np.asarray(hand_l["trans_l"], dtype=np.float64)
        pose_l = np.asarray(hand_l["pose_l"], dtype=np.float64)
        shape_l = np.asarray(hand_l["shape_l"], dtype=np.float64)

        # 左手参数在 npy 中以右手形式存储，镜像为真实左手参数
        rot_l_actual_ref = (rot_l[ref_frame].astype(np.float32) * _MIRROR_LEFT).astype(np.float64)
        pose_l_actual_ref = (pose_l[ref_frame].reshape(-1, 3).astype(np.float32)
                             * _MIRROR_LEFT).reshape(-1)

        mano_l = smplx.create(
            model_path=os.path.dirname(MANO_MODEL_PATH),
            model_type="MANO", is_rhand=False,
            use_pca=False, flat_hand_mean=True,
        )
        j0_l = _compute_j0(mano_l, pose_l_actual_ref,
                            shape_l[ref_frame].astype(np.float32))
        print(f"  左手 j0 (根关节偏移) = {j0_l}")

        T_hand_ref_l = _build_hand_T(rot_l_actual_ref, trans_l[ref_frame], j0_l)
        T_rel_l = T_obj_ref_inv @ T_hand_ref_l

        ref_pose_l = pose_l[ref_frame].copy()
        ref_shape_l = shape_l[ref_frame].copy()

        for i in range(n_frames):
            T_obj_i = _build_obj_T(obj_rot[i], obj_trans[i])
            T_new = T_obj_i @ T_rel_l
            rot_l_actual_new, trans_l[i] = _decompose_hand_T(T_new, j0_l)
            rot_l[i] = (rot_l_actual_new * _MIRROR_LEFT).astype(np.float64)
            pose_l[i] = ref_pose_l
            shape_l[i] = ref_shape_l

        hand_l["rot_l"] = rot_l.astype(np.float32)
        hand_l["trans_l"] = trans_l.astype(np.float32)
        hand_l["pose_l"] = pose_l.astype(np.float32)
        hand_l["shape_l"] = shape_l.astype(np.float32)
        print(f"  左手: 已将参考帧 {ref_frame} 的手物相对位姿重放到全部 {n_frames} 帧")

    print("[手物重放] 完成")


def inject_object_pose_from_txt(txt_folder, root_dict, seq_name):
    """
    读取 FoundationPose 输出的 4x4 位姿 txt 文件，
    提取旋转向量和平移，写入 root_dict 的 object 字段。
    """
    txt_files = sorted(glob.glob(os.path.join(txt_folder, "*.txt")))

    if not txt_files:
        print("[物体位姿] 未找到 txt 文件，跳过。")
        return

    obj_rot_list = []
    obj_trans_list = []

    for f_path in txt_files:
        matrix = np.loadtxt(f_path)
        translation = matrix[:3, 3]
        rotation_mat = matrix[:3, :3]
        rot_vec = R.from_matrix(rotation_mat).as_rotvec()
        obj_trans_list.append(translation)
        obj_rot_list.append(rot_vec)

    target = root_dict["data_dict"][seq_name]["params"]["object"]
    target["obj_rot"] = np.array(obj_rot_list, dtype=np.float32)
    target["obj_trans"] = np.array(obj_trans_list, dtype=np.float32)

    print(f"[物体位姿] 已注入 {len(txt_files)} 帧: "
          f"obj_rot {target['obj_rot'].shape}, obj_trans {target['obj_trans'].shape}")


def smooth_object_pose_temporal(root_dict, seq_name, sigma=2.0):
    """
    对物体位姿做时序上的高斯滤波，减轻从 txt 逐帧读取带来的抖动。
    平移：沿时间轴对 obj_trans 做 1D 高斯平滑。
    旋转：转为四元数并统一符号后，对四元数分量做 1D 高斯平滑并再归一化，再转回轴角。
    在 hand-object contact 等步骤前调用；sigma<=0 时跳过。
    """
    if sigma <= 0:
        return
    obj = root_dict["data_dict"][seq_name]["params"]["object"]
    if obj["obj_rot"] is None or obj["obj_trans"] is None:
        return
    obj_rot = np.array(obj["obj_rot"], dtype=np.float64)
    obj_trans = np.array(obj["obj_trans"], dtype=np.float64)
    N = obj_rot.shape[0]
    if N < 2:
        return

    # 平移：沿时间轴高斯平滑
    obj_trans_smooth = gaussian_filter1d(obj_trans, sigma=sigma, axis=0, mode="nearest")
    obj["obj_trans"] = obj_trans_smooth.astype(np.float32)

    # 旋转：轴角 -> 四元数，统一符号后平滑四元数分量，再归一化并转回轴角
    quats = R.from_rotvec(obj_rot).as_quat()  # (N, 4) xyzw
    for i in range(1, N):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    quats_smooth = gaussian_filter1d(quats, sigma=sigma, axis=0, mode="nearest")
    quats_smooth /= np.linalg.norm(quats_smooth, axis=1, keepdims=True)
    obj_rot_smooth = R.from_quat(quats_smooth).as_rotvec()
    obj["obj_rot"] = obj_rot_smooth.astype(np.float32)

    print(f"[物体位姿滤波] 已用时序高斯滤波平滑 (sigma={sigma})")


def run(input_folder, output_npy, seq_name, cam_k_path=None,
        obj_folder=None, obj_pose_sigma=0.0, point_r=None, point_l=None,
        opt_range=None, force_closure_range=None, ref_frame=None):
    # ── 1. 加载相机内参 ──
    k_use = None
    if cam_k_path:
        k_use = load_intrinsics(cam_k_path)
        if k_use is not None:
            print(f"已加载相机内参: {cam_k_path}")

    # ── 2. 初始化模型 (只加载一次) ──
    print("正在加载模型...")
    hamer = hamer_inference(hamer_opt)
    detector = Detector(yolo_opt)
    sar = get_model()
    print("模型加载完成。")

    # ── 3. 扫描图片并按帧号排序 ──
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
        image_paths.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"错误: 在 {input_folder} 中未找到图片文件")
        return

    # 建立 帧号 -> 图片路径 的映射
    frame_map = {}
    max_frame_idx = -1
    for img_path in image_paths:
        try:
            idx = int(os.path.splitext(os.path.basename(img_path))[0])
            frame_map[idx] = img_path
            max_frame_idx = max(max_frame_idx, idx)
        except ValueError:
            continue

    if max_frame_idx < 0:
        print("错误: 未能从文件名中解析出有效帧号 (期望数字命名，如 000000.jpg)")
        return

    total_frames = max_frame_idx + 1
    print(f"共 {len(frame_map)} 张有效图片，序列对齐长度: {total_frames} 帧")

    # ── 4. 逐帧推理 + 在内存中收集参数 ──
    r_rot, r_pose, r_shape, r_trans = [], [], [], []
    l_rot, l_pose, l_shape, l_trans = [], [], [], []
    stats = {'missing_file': 0, 'missing_right': 0, 'missing_left': 0}

    for i in tqdm(range(total_frames), desc="推理进度"):
        if i not in frame_map:
            # 该帧图片缺失，全部填 NaN
            stats['missing_file'] += 1
            for lst in (r_rot, l_rot):   lst.append(NAN_RIGHT['rot'].copy())
            for lst in (r_pose, l_pose): lst.append(NAN_RIGHT['pose'].copy())
            for lst in (r_shape, l_shape): lst.append(NAN_RIGHT['shape'].copy())
            for lst in (r_trans, l_trans): lst.append(NAN_RIGHT['trans'].copy())
            continue

        img_path = frame_map[i]
        image = cv2.imread(img_path)
        if image is None:
            stats['missing_file'] += 1
            for lst in (r_rot, l_rot):   lst.append(NAN_RIGHT['rot'].copy())
            for lst in (r_pose, l_pose): lst.append(NAN_RIGHT['pose'].copy())
            for lst in (r_shape, l_shape): lst.append(NAN_RIGHT['shape'].copy())
            for lst in (r_trans, l_trans): lst.append(NAN_RIGHT['trans'].copy())
            continue

        # YOLO 检测
        _, dets = detector.detect(image)
        detection_list = parse_detections(dets)

        # 本帧结果缓存
        frame_result = {'right': None, 'left': None}

        for bbox in detection_list:
            hand_label = bbox[0]
            try:
                output, params = hamer.estimate_from_rgb(image, [bbox], k_use)
                mano_params = output['pred_mano_params']
                hand_data = extract_hand_params(output, mano_params, hand_label)
                frame_result[hand_label] = hand_data
            except Exception as e:
                print(f"帧 {i} 处理 {hand_label} 手出错: {e}")
                continue

        # 收集右手
        rh = frame_result['right']
        if rh is not None:
            r_rot.append(rh['pose_global'])
            r_pose.append(rh['pose_hand'])
            r_shape.append(rh['betas'])
            r_trans.append(rh['cam_t'])
        else:
            stats['missing_right'] += 1
            r_rot.append(NAN_RIGHT['rot'].copy())
            r_pose.append(NAN_RIGHT['pose'].copy())
            r_shape.append(NAN_RIGHT['shape'].copy())
            r_trans.append(NAN_RIGHT['trans'].copy())

        # 收集左手
        lh = frame_result['left']
        if lh is not None:
            l_rot.append(lh['pose_global'])
            l_pose.append(lh['pose_hand'])
            l_shape.append(lh['betas'])
            l_trans.append(lh['cam_t'])
        else:
            stats['missing_left'] += 1
            l_rot.append(NAN_LEFT['rot'].copy())
            l_pose.append(NAN_LEFT['pose'].copy())
            l_shape.append(NAN_LEFT['shape'].copy())
            l_trans.append(NAN_LEFT['trans'].copy())

    # ── 5. 聚合为 NumPy 数组 ──
    r_rot_arr   = np.array(r_rot,   dtype=np.float32)  # (N, 3)
    r_pose_arr  = np.array(r_pose,  dtype=np.float32)  # (N, 45)
    r_shape_arr = np.array(r_shape, dtype=np.float32)  # (N, 10)
    r_trans_arr = np.array(r_trans, dtype=np.float32)   # (N, 3)

    l_rot_arr   = np.array(l_rot,   dtype=np.float32)
    l_pose_arr  = np.array(l_pose,  dtype=np.float32)
    l_shape_arr = np.array(l_shape, dtype=np.float32)
    l_trans_arr = np.array(l_trans, dtype=np.float32)

    # ── 5.5 插值填补 NaN 间隙 ──
    # rot (轴角) 用 SLERP 球面插值，避免旋转绕远路
    interp_count = 0
    interp_count += _slerp_interpolate_nan(r_rot_arr)
    interp_count += _slerp_interpolate_nan(l_rot_arr)

    # pose / shape / trans 用普通线性插值
    for arr in [r_pose_arr, r_shape_arr, r_trans_arr,
                l_pose_arr, l_shape_arr, l_trans_arr]:
        interp_count += _interpolate_nan(arr)

    if interp_count > 0:
        print(f"插值共填补了 {interp_count} 个缺失数据段 (rot 使用 SLERP)")
    else:
        print("未发现需要插值的 NaN 间隙")

    # ── 6. 构建模板并写入 ──
    params = {
        "right hand": {
            "rot_r":   r_rot_arr,
            "pose_r":  r_pose_arr,
            "trans_r": r_trans_arr,
            "shape_r": r_shape_arr,
        },
        "left hand": {
            "rot_l":   l_rot_arr,
            "pose_l":  l_pose_arr,
            "trans_l": l_trans_arr,
            "shape_l": l_shape_arr,
        },
        "object":  {"obj_rot": None, "obj_trans": None},
        "camera":  {"world2cam": None, "K": None, "k_use": k_use},
    }

    # ── 注入图片名与图片路径 ──
    imgnames = [os.path.basename(p) for p in image_paths]
    imgpath = os.path.abspath(input_folder)

    root = {
        "imgnames": imgnames,
        "imgpath":  imgpath,
        "data_dict": {
            seq_name: {"params": params}
        },
    }

    # ── 7. 物体位姿注入 (可选) ──
    if obj_folder:
        print(f"\n[物体位姿] 直接从文件夹读取: {obj_folder}")
        inject_object_pose_from_txt(obj_folder, root, seq_name)
        if obj_pose_sigma is not None and obj_pose_sigma > 0:
            smooth_object_pose_temporal(root, seq_name, sigma=obj_pose_sigma)
    else:
        print("\n[物体位姿] 未指定 --obj_folder，跳过物体位姿提取")

    # ── 7.5 手物 Concat (可选) ──
    if point_r is not None and point_l is not None:
        concat_hand_to_object(root, seq_name, point_r, point_l)
    elif point_r is not None or point_l is not None:
        print("\n[手物Concat] --point_r 和 --point_l 必须同时指定，跳过手物对齐")
    else:
        print("\n[手物Concat] 未指定 --point_r/--point_l，跳过手物对齐")

    # ── 7.6 Contact 优化 (可选) ──
    if opt_range is not None:
        opt_start, opt_end = opt_range
        print(f"\n[Contact优化] 帧区间: [{opt_start}, {opt_end})")

        # 先保存中间 npy，供 train_contact.py 读取
        np.save(output_npy, root, allow_pickle=True)
        print(f"[Contact优化] 中间结果已保存: {output_npy}")

        # 优化后的导出路径
        base, ext = os.path.splitext(output_npy)
        opt_contact_path = os.path.abspath(f"{base}_opt_contact{ext}")
        abs_output_npy = os.path.abspath(output_npy)

        sportgs_dir = os.path.join(_ROOT, 'model', 'SportGS')
        cmd = [
            sys.executable, 'train_contact.py',
            f'dataset.pose_path={abs_output_npy}',
            f'+dataset.opt_frame_start={opt_start}',
            f'+dataset.opt_frame_end={opt_end}',
            f'+dataset.export_data_path={abs_output_npy}',
            f'+dataset.export_output_path={opt_contact_path}',
        ]
        print(f"[Contact优化] 执行: cd {sportgs_dir} && {' '.join(cmd)}")
        subprocess.run(cmd, cwd=sportgs_dir, check=True)

        # 加载优化后的数据替换内存中的 root
        print(f"[Contact优化] 加载优化结果: {opt_contact_path}")
        root = np.load(opt_contact_path, allow_pickle=True).item()
        print("[Contact优化] 完成")
    else:
        print("\n[Contact优化] 未指定 --opt_range，跳过 Contact 优化")

    # ── 7.7 力闭合优化 (可选) ──
    if force_closure_range is not None:
        fc_start, fc_end = force_closure_range
        print(f"\n[力闭合优化] 帧区间: [{fc_start}, {fc_end})")

        # 先保存中间 npy，供 finetune_force_closure.py 读取
        np.save(output_npy, root, allow_pickle=True)
        print(f"[力闭合优化] 中间结果已保存: {output_npy}")

        # 优化后的导出路径
        base, ext = os.path.splitext(output_npy)
        opt_force_closure_path = os.path.abspath(f"{base}_opt_force_closure{ext}")
        abs_output_npy = os.path.abspath(output_npy)

        sportgs_dir = os.path.join(_ROOT, 'model', 'SportGS')
        cmd = [
            sys.executable, 'finetune_force_closure.py',
            f'dataset.pose_path={abs_output_npy}',
            f'+dataset.opt_frame_start={fc_start}',
            f'+dataset.opt_frame_end={fc_end}',
            f'+dataset.export_data_path={abs_output_npy}',
            f'+dataset.export_output_path={opt_force_closure_path}',
        ]
        print(f"[力闭合优化] 执行: cd {sportgs_dir} && {' '.join(cmd)}")
        subprocess.run(cmd, cwd=sportgs_dir, check=True)

        # 加载优化后的数据替换内存中的 root
        print(f"[力闭合优化] 加载优化结果: {opt_force_closure_path}")
        root = np.load(opt_force_closure_path, allow_pickle=True).item()
        print("[力闭合优化] 完成")
    else:
        print("\n[力闭合优化] 未指定 --force_closure_range，跳过力闭合优化")

    # ── 7.8 手物重放 (可选) ──
    if ref_frame is not None:
        rehand_by_object(root, seq_name, ref_frame)
    else:
        print("\n[手物重放] 未指定 --ref_frame，跳过手物重放")

    # ── 8. 保存 ──
    np.save(output_npy, root, allow_pickle=True)

    # ── 9. 打印摘要 ──
    obj = root["data_dict"][seq_name]["params"]["object"]
    print("\n========== 完成 ==========")
    print(f"输出文件: {output_npy}")
    print(f"序列名称: {seq_name}")
    print(f"总帧数:   {total_frames}")
    print(f"图片路径: {imgpath}")
    print(f"图片名数: {len(imgnames)} (首: {imgnames[0]}, 末: {imgnames[-1]})")
    print(f"文件缺失: {stats['missing_file']} 帧")
    print(f"右手缺失: {stats['missing_right']} 帧 (已插值)")
    print(f"左手缺失: {stats['missing_left']} 帧 (已插值)")
    print(f"插值填补: {interp_count} 段")
    print(f"右手 pose 形状: {r_pose_arr.shape}")
    print(f"左手 pose 形状: {l_pose_arr.shape}")
    if obj["obj_rot"] is not None:
        print(f"物体 rot 形状:  {obj['obj_rot'].shape}")
        print(f"物体 trans 形状: {obj['obj_trans'].shape}")
    else:
        print("物体位姿: 未注入")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='一键手部 + 物体参数提取')
    parser.add_argument('--input',    type=str, required=True, help='图片文件夹路径')
    parser.add_argument('--output',   type=str, required=True, help='输出 npy 文件路径')
    parser.add_argument('--seq_name', type=str, required=True, help='序列名称 (如 seq_136)')
    parser.add_argument('--cam_k',    type=str, default=None,  help='相机内参 txt 文件路径 (3x3 矩阵，可选)')
    parser.add_argument('--obj_folder',     type=str, default=None, help='物体位姿 txt 文件夹路径 (每帧一个 4x4 矩阵 txt)')
    parser.add_argument('--obj_pose_sigma', type=float, default=0.0,
                        help='物体位姿时序高斯滤波的 sigma（帧数），0 表示不滤波，默认 0（不做平滑）')
    parser.add_argument('--point_r', type=float, nargs=3, default=None,
                        help='物体局部坐标系中的右手目标点 [x y z]，如 0.36 0.0 0.0')
    parser.add_argument('--point_l', type=float, nargs=3, default=None,
                        help='物体局部坐标系中的左手目标点 [x y z]，如 0.56 0.0 0.0')
    parser.add_argument('--opt_range', type=int, nargs=2, default=None,
                        metavar=('START', 'END'),
                        help='Contact 优化帧区间 [START, END)，如 120 200')
    parser.add_argument('--force_closure_range', type=int, nargs=2, default=None,
                        metavar=('START', 'END'),
                        help='力闭合优化帧区间 [START, END)，如 120 200')
    parser.add_argument('--ref_frame', type=int, default=None,
                        help='参考帧索引 (0-based)，将该帧的手物相对位姿重放到整个序列')
    args = parser.parse_args()

    pr = np.array(args.point_r, dtype=np.float32) if args.point_r else None
    pl = np.array(args.point_l, dtype=np.float32) if args.point_l else None

    run(args.input, args.output, args.seq_name, args.cam_k,
        obj_folder=args.obj_folder, obj_pose_sigma=args.obj_pose_sigma,
        point_r=pr, point_l=pl,
        opt_range=args.opt_range, force_closure_range=args.force_closure_range,
        ref_frame=args.ref_frame)
