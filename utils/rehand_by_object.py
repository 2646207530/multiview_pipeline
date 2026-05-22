#!/usr/bin/env python3
"""
将指定「锁定帧」的手粘到物体上，在重放区间内做刚性跟随；其他区间保持不变。
  不传 --ranges 时默认整条序列（本序列或 --replay_to 目标）均为重放区间。

步骤:
  1. 用 MANO 计算手腕根关节位置 j0（global_orient 的旋转中心）
  2. 以 --lock_frame 为参考帧，构建手相对物体的变换: T_rel = inv(T_obj[ref]) @ T_hand_eff[ref]
  3. 对 --ranges 中的每个区间 [start, end]（不传则 [0, N-1]）：区间内每一帧 T_hand_new[i] = T_obj[i] @ T_rel，并锁定 pose/shape
  4. 不在任何区间内的帧不修改
  5. 保存结果 npy

用法:
  # 不传 --ranges：重放整条序列（本序列或 --replay_to 目标序列）
  python utils/rehand_by_object.py --input data.npy --output out.npy --lock_frame 182

  # 锁定第 182 帧，只重放到本序列区间 [120, 200]
  python utils/rehand_by_object.py --input data.npy --output out.npy --lock_frame 182 --ranges 120 200

  # 锁定第 182 帧，重放到多区间 [10,20]、[50,60]、[100,150]
  python utils/rehand_by_object.py --input data.npy --output out.npy --lock_frame 182 --ranges 10 20 50 60 100 150

  # 重放到其他序列整条（不传 --ranges）
  python utils/rehand_by_object.py --input input.npy --replay_to target.npy --output out.npy --lock_frame 182

  # 重放到其他序列的指定区间 [0, 100]
  python utils/rehand_by_object.py --input input.npy --replay_to target.npy --output out.npy --lock_frame 182 --ranges 0 100
"""

import os
import numpy as np
import argparse
import torch
import smplx
from scipy.spatial.transform import Rotation as R

MANO_MODEL_DIR = "/home/pt/fbs"
MIRROR = np.array([1.0, -1.0, -1.0], dtype=np.float32)


def make_T(rot_mat, trans):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot_mat
    T[:3, 3] = trans
    return T


def build_hand_T(rot_vec, trans, j0):
    """构建 MANO 手在世界坐标系中的真实 4x4 刚体变换。

    MANO 的 global_orient 绕根关节 j0 旋转:
      v_world = R @ (v - j0) + j0 + transl = R @ v + (I-R)@j0 + transl
    所以有效平移 t_eff = (I-R)@j0 + transl
    """
    R_mat = R.from_rotvec(rot_vec).as_matrix()
    t_eff = (np.eye(3) - R_mat) @ j0 + trans
    return make_T(R_mat, t_eff)


def decompose_hand_T(T, j0):
    """从 4x4 有效变换中反解 MANO 的 rot (轴角) 和 transl。

    transl = t_eff - (I-R)@j0
    """
    R_mat = T[:3, :3]
    t_eff = T[:3, 3]
    rot_vec = R.from_matrix(R_mat).as_rotvec().astype(np.float32)
    transl = (t_eff - (np.eye(3) - R_mat) @ j0).astype(np.float32)
    return rot_vec, transl


def build_obj_T(obj_rot_vec, obj_trans):
    R_mat = R.from_rotvec(obj_rot_vec).as_matrix()
    return make_T(R_mat, obj_trans)


def compute_j0(mano_model, hand_pose, betas):
    """用 MANO 前向传播计算根关节(手腕)在零姿态零平移时的位置。"""
    with torch.no_grad():
        out = mano_model(
            global_orient=torch.zeros(1, 3, dtype=torch.float32),
            hand_pose=torch.tensor(hand_pose[np.newaxis], dtype=torch.float32),
            betas=torch.tensor(betas[np.newaxis], dtype=torch.float32),
            transl=torch.zeros(1, 3, dtype=torch.float32),
        )
    j0 = out.joints[0, 0].numpy().astype(np.float64)
    print(f"  j0 (根关节偏移) = {j0}")
    return j0


def mirror_axis_angle(vec):
    """轴角镜像: (x, y, z) → (x, -y, -z)，用于左手参数转换。"""
    return vec * MIRROR


def parse_ranges(args_list):
    """解析为一组 [start, end] 闭区间。

    - 2 个数: 单区间 -> [(a, b)]
    - 2k 个数: k 个区间 -> [(a1,b1), (a2,b2), ...]
    - 2k+1 个数: k 个区间 + 单帧 -> 最后一帧单独成区间 (n, n)
    """
    if not args_list:
        return []
    n = len(args_list)
    ranges = []
    i = 0
    while i < n:
        if i + 1 < n:
            ranges.append((int(args_list[i]), int(args_list[i + 1])))
            i += 2
        else:
            ranges.append((int(args_list[i]), int(args_list[i])))
            i += 1
    return ranges


def main():
    parser = argparse.ArgumentParser(
        description="对指定区间做锁定帧重放：手粘到物体上，区间内手刚性跟随物体运动"
    )
    parser.add_argument("--input", type=str, default="/home/pt/fbs/data_opt_hand_fc.npy",
                        help="输入 npy 路径（锁定帧的手物相对位姿从此文件取）")
    parser.add_argument("--replay_to", type=str, default=None,
                        help="重放目标：不传则重放到本序列 (--input)；传入则重放到该 npy 序列上，此时 --ranges 为该目标序列的帧区间")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 npy 路径（必须显式指定，不覆盖源文件）")
    parser.add_argument("--lock_frame", type=int, default=182,
                        help="锁定帧（参考帧索引，在 --input 中取该帧的手部姿态用于重放）")
    parser.add_argument("--ranges", type=int, nargs="*", default=None,
                        help="重放区间：不传则整条序列；单区间写 start end；多区间写 a1 b1 a2 b2 ... (闭区间)")
    args = parser.parse_args()

    output_path = args.output
    ref = args.lock_frame

    print(f"加载源序列 (锁定帧来源): {args.input}")
    data = np.load(args.input, allow_pickle=True).item()
    seq_name_src = next(iter(data["data_dict"].keys()))
    params_src = data["data_dict"][seq_name_src]["params"]

    # 锁定帧必须在源序列有效范围内
    obj_rot_src = np.asarray(params_src["object"]["obj_rot"], dtype=np.float64)
    obj_trans_src = np.asarray(params_src["object"]["obj_trans"], dtype=np.float64)
    n_frames_src = obj_rot_src.shape[0]
    if ref < 0 or ref >= n_frames_src:
        raise ValueError(f"锁定帧 {ref} 超出源序列有效范围 0~{n_frames_src - 1}")

    # 决定重放目标：本序列 或 其他 npy
    if args.replay_to is None:
        target_data = data
        seq_name = seq_name_src
        params = params_src
        obj_rot = obj_rot_src
        obj_trans = obj_trans_src
        n_frames = n_frames_src
        print(f"重放目标: 本序列 ({args.input})")
    else:
        print(f"加载重放目标序列: {args.replay_to}")
        target_data = np.load(args.replay_to, allow_pickle=True).item()
        seq_name = next(iter(target_data["data_dict"].keys()))
        params = target_data["data_dict"][seq_name]["params"]
        obj_rot = np.asarray(params["object"]["obj_rot"], dtype=np.float64)
        obj_trans = np.asarray(params["object"]["obj_trans"], dtype=np.float64)
        n_frames = obj_rot.shape[0]
        print(f"重放目标: 其他序列 ({args.replay_to}), 序列名: {seq_name}, 总帧数: {n_frames}")

    # 未指定 --ranges 或为空则重放整条序列
    replay_ranges = parse_ranges(args.ranges) if args.ranges else [(0, n_frames - 1)]
    if not replay_ranges:
        replay_ranges = [(0, n_frames - 1)]
    print(f"序列: {seq_name}, 总帧数: {n_frames}, 锁定帧(来自源): {ref}, 重放区间: {replay_ranges}")
    for start, end in replay_ranges:
        if start < 0 or end >= n_frames or start > end:
            raise ValueError(f"区间 [{start}, {end}] 非法（目标有效帧 0~{n_frames - 1}）")

    # ============ 右手 ============
    print("\n===== 处理右手 =====")
    # 锁定帧的手、物位姿一律从源序列取
    rot_r_src = np.asarray(params_src["right hand"]["rot_r"], dtype=np.float64)
    trans_r_src = np.asarray(params_src["right hand"]["trans_r"], dtype=np.float64)
    pose_r_src = np.asarray(params_src["right hand"]["pose_r"], dtype=np.float64)
    shape_r_src = np.asarray(params_src["right hand"]["shape_r"], dtype=np.float64)
    rot_r = np.asarray(params["right hand"]["rot_r"], dtype=np.float64)
    trans_r = np.asarray(params["right hand"]["trans_r"], dtype=np.float64)
    pose_r = np.asarray(params["right hand"]["pose_r"], dtype=np.float64)
    shape_r = np.asarray(params["right hand"]["shape_r"], dtype=np.float64)

    mano_r = smplx.create(MANO_MODEL_DIR, "MANO", use_pca=False,
                           is_rhand=True, flat_hand_mean=True)
    T_obj_ref = build_obj_T(obj_rot_src[ref], obj_trans_src[ref])
    T_obj_ref_inv = np.linalg.inv(T_obj_ref)
    j0_r = compute_j0(mano_r, pose_r_src[ref].astype(np.float32),
                       shape_r_src[ref].astype(np.float32))
    T_hand_ref_r = build_hand_T(rot_r_src[ref], trans_r_src[ref], j0_r)
    T_rel_r = T_obj_ref_inv @ T_hand_ref_r
    ref_pose_r = pose_r_src[ref].copy()
    ref_shape_r = shape_r_src[ref].copy()

    total_locked_r = 0
    for start, end in replay_ranges:
        for i in range(start, end + 1):
            T_obj_i = build_obj_T(obj_rot[i], obj_trans[i])
            T_new = T_obj_i @ T_rel_r
            rot_r[i], trans_r[i] = decompose_hand_T(T_new, j0_r)
            pose_r[i] = ref_pose_r
            shape_r[i] = ref_shape_r
        num = end - start + 1
        total_locked_r += num
        print(f"  右手 区间 [{start}, {end}]: 已用锁定帧 {ref} 重放 {num} 帧")

    params["right hand"]["rot_r"] = rot_r.astype(np.float32)
    params["right hand"]["trans_r"] = trans_r.astype(np.float32)
    params["right hand"]["pose_r"] = pose_r.astype(np.float32)
    params["right hand"]["shape_r"] = shape_r.astype(np.float32)
    print(f"右手: 共锁定重放 {total_locked_r} 帧")

    # ============ 左手 ============
    if "left hand" in params and "left hand" in params_src:
        print("\n===== 处理左手 =====")
        rot_l = np.asarray(params["left hand"]["rot_l"], dtype=np.float64)
        trans_l = np.asarray(params["left hand"]["trans_l"], dtype=np.float64)
        pose_l = np.asarray(params["left hand"]["pose_l"], dtype=np.float64)
        shape_l = np.asarray(params["left hand"]["shape_l"], dtype=np.float64)
        rot_l_src = np.asarray(params_src["left hand"]["rot_l"], dtype=np.float64)
        trans_l_src = np.asarray(params_src["left hand"]["trans_l"], dtype=np.float64)
        pose_l_src = np.asarray(params_src["left hand"]["pose_l"], dtype=np.float64)
        shape_l_src = np.asarray(params_src["left hand"]["shape_l"], dtype=np.float64)

        mano_l = smplx.create(MANO_MODEL_DIR, "MANO", use_pca=False,
                               is_rhand=False, flat_hand_mean=True)

        rot_l_actual_ref = mirror_axis_angle(rot_l_src[ref].astype(np.float32)).astype(np.float64)
        pose_l_actual_ref = (pose_l_src[ref].reshape(-1, 3).astype(np.float32) * MIRROR).reshape(-1)
        j0_l = compute_j0(mano_l, pose_l_actual_ref, shape_l_src[ref].astype(np.float32))
        T_obj_ref = build_obj_T(obj_rot_src[ref], obj_trans_src[ref])
        T_obj_ref_inv = np.linalg.inv(T_obj_ref)
        T_hand_ref_l = build_hand_T(rot_l_actual_ref, trans_l_src[ref], j0_l)
        T_rel_l = T_obj_ref_inv @ T_hand_ref_l
        ref_pose_l = pose_l_src[ref].copy()
        ref_shape_l = shape_l_src[ref].copy()

        total_locked_l = 0
        for start, end in replay_ranges:
            for i in range(start, end + 1):
                T_obj_i = build_obj_T(obj_rot[i], obj_trans[i])
                T_new = T_obj_i @ T_rel_l
                rot_l_actual_new, trans_l[i] = decompose_hand_T(T_new, j0_l)
                rot_l[i] = mirror_axis_angle(rot_l_actual_new).astype(np.float64)
                pose_l[i] = ref_pose_l
                shape_l[i] = ref_shape_l
            num = end - start + 1
            total_locked_l += num
            print(f"  左手 区间 [{start}, {end}]: 已用锁定帧 {ref} 重放 {num} 帧")

        params["left hand"]["rot_l"] = rot_l.astype(np.float32)
        params["left hand"]["trans_l"] = trans_l.astype(np.float32)
        params["left hand"]["pose_l"] = pose_l.astype(np.float32)
        params["left hand"]["shape_l"] = shape_l.astype(np.float32)
        print(f"左手: 共锁定重放 {total_locked_l} 帧")

    print(f"\n保存: {output_path}")
    np.save(output_path, target_data)
    print("完成。")


if __name__ == "__main__":
    main()
