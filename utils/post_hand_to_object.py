#!/usr/bin/env python3
"""
将左右手同时移动到物体上的指定点（与 post_concat.py 相反）。

post_concat.py：物体局部点 → 移到手的关节中心（物体跟手走）
本脚本：        手的关节中心 → 移到物体局部点（手跟物体走）

所有帧同时处理：
  - 右手关节中心对齐到物体上的 point_r
  - 左手关节中心对齐到物体上的 point_l

手的"对齐点"为 MANO 所有关节的几何中心（与 post_concat.py 一致）。
"""
"""
python utils/post_hand_to_object.py \
    --data_npy /home/pt/fbs/data.npy \
    --point_r 0.36 0.0 0.0 \
    --point_l 0.36 0.0 0.0 \
    --output /home/pt/fbs/data_hand_to_obj.npy
"""

import numpy as np
import os
import argparse
from scipy.spatial.transform import Rotation as R
import torch
import smplx

TARGET_LOCAL_POINT_R = np.array([0.36, 0.0, 0.0], dtype=np.float32)
TARGET_LOCAL_POINT_L = np.array([0.56, 0.0, 0.0], dtype=np.float32)

MANO_MODEL_PATH = "/home/pt/fbs/MANO"

# npy 中左手槽位存的是右手 MANO 参数，送入左手 MANO 前需沿 Y、Z 取反
MIRROR_LEFT_PARAMS = np.array([1.0, -1.0, -1.0], dtype=np.float32)


def right_to_left_mano_params(rot: np.ndarray, pose: np.ndarray):
    """将右手 MANO 轴角参数镜像为左手：(x, y, z) → (x, -y, -z)。"""
    rot_out = rot * MIRROR_LEFT_PARAMS
    pose_out = (pose.reshape(-1, 3) * MIRROR_LEFT_PARAMS).reshape(pose.shape)
    return rot_out, pose_out


def clean(arr):
    if hasattr(arr, "__len__") and len(arr) and np.issubdtype(np.asarray(arr).dtype, np.floating):
        a = np.asarray(arr, dtype=np.float32)
        return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(arr, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="将左右手同时移动到物体上的两个指定点（R/L）"
    )
    parser.add_argument("--data_npy", type=str, default="/home/pt/fbs/data.npy",
                        help="输入的 data.npy 路径")
    parser.add_argument("--output", type=str, default="/home/pt/fbs/data_hand_to_obj.npy",
                        help="输出路径")
    parser.add_argument("--point_r", type=float, nargs=3, default=None,
                        help="物体局部坐标系中的右手目标点 [x y z]，默认 [0.36 0 0]")
    parser.add_argument("--point_l", type=float, nargs=3, default=None,
                        help="物体局部坐标系中的左手目标点 [x y z]，默认 [0.36 0 0]")
    args = parser.parse_args()

    point_r = np.array(args.point_r, dtype=np.float32) if args.point_r else TARGET_LOCAL_POINT_R.copy()
    point_l = np.array(args.point_l, dtype=np.float32) if args.point_l else TARGET_LOCAL_POINT_L.copy()

    if not os.path.exists(MANO_MODEL_PATH):
        raise FileNotFoundError(f"找不到 MANO 模型: {MANO_MODEL_PATH}")

    # ---------- 1. 加载 data.npy ----------
    print(f"加载: {args.data_npy}")
    data = np.load(args.data_npy, allow_pickle=True).item()
    if "data_dict" not in data:
        raise KeyError("data.npy 中缺少 data_dict")
    seq_name = next(iter(data["data_dict"].keys()))
    params = data["data_dict"][seq_name]["params"]
    hand_data_r = params["right hand"]
    obj_data = params["object"]

    if "left hand" not in params:
        raise KeyError("data.npy 中缺少 left hand 数据，无法同时移动左右手")
    hand_data_l = params["left hand"]

    obj_trans = clean(obj_data["obj_trans"])
    obj_rot = clean(obj_data["obj_rot"])
    n_frames_data = obj_trans.shape[0]

    for k in ["pose_r", "shape_r", "rot_r", "trans_r"]:
        if k in hand_data_r:
            hand_data_r[k] = clean(hand_data_r[k])
    for k in ["pose_l", "shape_l", "rot_l", "trans_l"]:
        if k in hand_data_l:
            hand_data_l[k] = clean(hand_data_l[k])

    pose_r = hand_data_r["pose_r"]
    shape_r = hand_data_r["shape_r"]
    rot_r = hand_data_r["rot_r"]
    pose_l = hand_data_l["pose_l"]
    shape_l = hand_data_l["shape_l"]
    rot_l = hand_data_l["rot_l"]

    n_frames = min(
        n_frames_data,
        pose_r.shape[0], shape_r.shape[0], rot_r.shape[0],
        pose_l.shape[0], shape_l.shape[0], rot_l.shape[0],
    )
    if n_frames < n_frames_data:
        print(f"警告: 帧数不一致，将统一裁剪到前 {n_frames} 帧")

    obj_trans = obj_trans[:n_frames]
    obj_rot = obj_rot[:n_frames]
    pose_r = pose_r[:n_frames]
    shape_r = shape_r[:n_frames]
    rot_r = rot_r[:n_frames]
    pose_l = pose_l[:n_frames]
    shape_l = shape_l[:n_frames]
    rot_l = rot_l[:n_frames]

    print(f"总帧数: {n_frames}")
    print(f"右手目标点 (物体局部): {point_r}")
    print(f"左手目标点 (物体局部): {point_l}")

    # ---------- 2. 物体目标点在相机坐标系中的位置 ----------
    r_obj = R.from_rotvec(obj_rot)
    target_cam_r = obj_trans + r_obj.apply(point_r)  # (N, 3)
    target_cam_l = obj_trans + r_obj.apply(point_l)  # (N, 3)

    # ---------- 3. 右手 MANO: 计算 transl=0 时关节几何中心偏移 ----------
    print("创建右手 MANO 模型 ...")
    mano_r = smplx.create(
        model_path=os.path.dirname(MANO_MODEL_PATH),
        model_type="MANO",
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=True,
    )
    with torch.no_grad():
        out_r = mano_r(
            global_orient=torch.tensor(rot_r, dtype=torch.float32),
            hand_pose=torch.tensor(pose_r, dtype=torch.float32),
            betas=torch.tensor(shape_r, dtype=torch.float32),
            transl=torch.zeros((n_frames, 3), dtype=torch.float32),
        )
    joints_mean_offset_r = np.mean(out_r.joints.cpu().numpy(), axis=1)  # (N, 3)

    hand_data_r["trans_r"] = (target_cam_r - joints_mean_offset_r).astype(np.float32)
    print("已计算新的 trans_r，使右手关节中心对齐到物体上的 point_r")

    # ---------- 4. 左手 MANO: 镜像参数后计算 transl=0 时关节几何中心偏移 ----------
    print("创建左手 MANO 模型（npy 左手槽位存的是右手参数，先镜像 Y/Z）...")
    rot_l_mirror, pose_l_mirror = right_to_left_mano_params(rot_l, pose_l)
    mano_l = smplx.create(
        model_path=os.path.dirname(MANO_MODEL_PATH),
        model_type="MANO",
        is_rhand=False,
        use_pca=False,
        flat_hand_mean=True,
    )
    with torch.no_grad():
        out_l = mano_l(
            global_orient=torch.tensor(rot_l_mirror, dtype=torch.float32),
            hand_pose=torch.tensor(pose_l_mirror, dtype=torch.float32),
            betas=torch.tensor(shape_l, dtype=torch.float32),
            transl=torch.zeros((n_frames, 3), dtype=torch.float32),
        )
    joints_mean_offset_l = np.mean(out_l.joints.cpu().numpy(), axis=1)  # (N, 3)

    hand_data_l["trans_l"] = (target_cam_l - joints_mean_offset_l).astype(np.float32)
    print("已计算新的 trans_l，使左手关节中心对齐到物体上的 point_l")

    # ---------- 5. 裁剪帧数（保持所有数组长度一致） ----------
    if n_frames < n_frames_data:
        for key in list(obj_data.keys()):
            arr = obj_data[key]
            if isinstance(arr, np.ndarray) and arr.shape[0] == n_frames_data:
                obj_data[key] = arr[:n_frames].astype(np.float32)
        for key in list(hand_data_r.keys()):
            arr = hand_data_r[key]
            if isinstance(arr, np.ndarray) and arr.shape[0] == n_frames_data:
                hand_data_r[key] = arr[:n_frames].astype(np.float32)
        for key in list(hand_data_l.keys()):
            arr = hand_data_l[key]
            if isinstance(arr, np.ndarray) and arr.shape[0] == n_frames_data:
                hand_data_l[key] = arr[:n_frames].astype(np.float32)

    # ---------- 6. 保存 ----------
    output_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    print(f"保存: {output_path}")
    np.save(output_path, data)
    print("完成。")


if __name__ == "__main__":
    main()
