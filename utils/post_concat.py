#!/usr/bin/env python3
"""
用 3d_keypoints.json（COCO17）中的右手腕 3D 位置，作为“希望的手中心”，
通过 MANO 重新计算手的平移向量，并可选地把物体局部点 [-0.43, 0, 0] 移动到这个手中心。

关键点：
- COCO17: 9 = 左手腕, 10 = 右手腕
- 原来 test.py 里真正对齐的是“手的关节几何中心 (Mean Joint)”，不是单纯手腕坐标
- 之前版本直接把手腕写进 trans_r，物体是对齐到手腕，所以和“手的中心”之间会有一个固定偏移

现在脚本的做法：
1. 用 MANO（和 test.py 一样）先在 transl=0 时算出每帧的 joints_mean_offset
2. 把 JSON 右手腕当成“目标手中心” desired_center_world
3. 令新的手平移 trans_r = desired_center_world - joints_mean_offset
   这样再重建时，手的几何中心就会落在 JSON 给的点上
4. 若不加 --no_object_align，则物体平移也更新，使物体局部点 [-0.43,0,0] 落在同一个中心
"""

import numpy as np
import json
import os
import argparse
from scipy.spatial.transform import Rotation as R
import torch
import smplx

# npy 中左手槽位存的是右手 MANO 参数，用左手 MANO 加载时轴角沿 Y、Z 取反
MIRROR_LEFT_PARAMS = np.array([1.0, -1.0, -1.0], dtype=np.float32)


def right_to_left_mano_params(rot: np.ndarray, pose: np.ndarray):
    """将右手 MANO 参数转为左手 MANO 输入：轴角 (x,y,z) 取 (x, -y, -z)。"""
    rot_l = rot * MIRROR_LEFT_PARAMS
    pose_l = (pose.reshape(-1, 3) * MIRROR_LEFT_PARAMS).reshape(pose.shape)
    return rot_l, pose_l

# ================= 配置 =================
COCO17_RIGHT_WRIST_INDEX = 10   # COCO17 右手腕
COCO17_LEFT_WRIST_INDEX = 9    # COCO17 左手腕

# 物体局部坐标系中「指定位置」，右手和左手分别对应不同的点
TARGET_CENTER_LOCAL_R = np.array([0.36, 0.0, 0.0], dtype=np.float32)
TARGET_CENTER_LOCAL_L = np.array([0.56, 0.0, 0.0], dtype=np.float32)

# MANO 模型路径
MANO_MODEL_PATH = "/home/pt/fbs/MANO"


def uvz_to_cam_xyz_with_K(keypoints_uvz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    仅当 JSON 存的是 (u, v, z) 时使用：用真实相机内参 K 反投影为相机坐标 (X, Y, Z)。
    keypoints_uvz: (..., 3)，最后一维为 [u, v, z]（像素 u,v + 深度 z）。
    K: (3, 3) 相机内参矩阵（来自 data.npy 中的 camera.k_use）。
    返回: (..., 3) 相机坐标 (X, Y, Z)。
    """
    K = np.asarray(K, dtype=np.float32)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = keypoints_uvz[..., 0]
    v = keypoints_uvz[..., 1]
    z = keypoints_uvz[..., 2]
    X = (u - cx) / fx * z
    Y = (v - cy) / fy * z
    Z = z
    return np.stack([X, Y, Z], axis=-1).astype(np.float32)


def load_json_keypoints(json_path: str):
    """加载 3d_keypoints.json，返回 (num_frames, 17, 3) 的数组。"""
    with open(json_path, "r") as f:
        raw = json.load(f)
    kp = np.array(raw["keypoints_3d_coco17"], dtype=np.float32)
    return kp, raw.get("num_frames", len(kp))


def main():
    parser = argparse.ArgumentParser(
        description="用 JSON 手腕位置作为目标手中心，重新计算 MANO 平移，并可选地让物体指定点对齐到该中心"
    )
    parser.add_argument("--data_npy", type=str, default="/home/pt/fbs/data.npy", help="输入的 data.npy 路径")
    parser.add_argument(
        "--keypoints_json",
        type=str,
        default="/home/pt/fbs/dataset/DA5298464_Video_20251017143917258_w1440_h1080_pBayerRG8_f97/3d_keypoints.json",
        help="3d_keypoints.json 路径（COCO17）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/pt/fbs/data_fixed.npy",
        help="输出 data.npy 路径（不会覆盖原始 data.npy）",
    )
    parser.add_argument(
        "--no_object_align",
        action="store_true",
        help="若设置，只根据 JSON 重新计算 hand trans_r，不根据中心位置更新物体平移",
    )
    parser.add_argument(
        "--ref_frame",
        type=int,
        default=182,
        help="用于确定手物对齐固定点的参考帧索引（0-based），取该帧绑定后的根节点位置作为物体对齐目标",
    )
    parser.add_argument(
        "--json_format",
        type=str,
        choices=("xyz", "xyz_virtual", "uvz"),
        default="xyz",
        help=(
            "JSON 中关键点格式："
            "xyz=已经在真实相机坐标系下的 (X,Y,Z)，直接使用；"
            "xyz_virtual=虚拟内参 (f=sqrt(w^2+h^2), cx=w/2, cy=h/2) 相机坐标系下的 (X,Y,Z)，"
            "需要配合 --json_img_width/height 和 data.npy 中的真实 K 进行转换；"
            "uvz=(u,v,z) 像素+深度，使用 data.npy 中的真实 K 反投影为 (X,Y,Z)"
        ),
    )
    parser.add_argument(
        "--json_img_width",
        type=float,
        default=1440,
        help="生成虚拟内参时使用的图像宽度 w（默认 1440，对应示例路径中的 w1440）",
    )
    parser.add_argument(
        "--json_img_height",
        type=float,
        default=1080,
        help="生成虚拟内参时使用的图像高度 h（默认 1080，对应示例路径中的 h1080）",
    )
    args = parser.parse_args()

    output_path = args.output or args.data_npy

    if not os.path.exists(MANO_MODEL_PATH):
        raise FileNotFoundError(f"找不到 MANO 模型: {MANO_MODEL_PATH}")

    # 1. 加载 data.npy
    print(f"加载: {args.data_npy}")
    data = np.load(args.data_npy, allow_pickle=True).item()
    if "data_dict" not in data:
        raise KeyError("data.npy 中缺少 data_dict")
    seq_name = next(iter(data["data_dict"].keys()))
    params = data["data_dict"][seq_name]["params"]
    hand_data = params["right hand"]
    obj_data = params["object"]
    has_left_hand = "left hand" in params

    cam_data = params.get("camera", None)
    K_real = None
    if cam_data is not None and "k_use" in cam_data:
        K_real = np.asarray(cam_data["k_use"], dtype=np.float32)

    # 清洗并统一为 float32
    def clean(arr):
        if hasattr(arr, "__len__") and len(arr) and np.issubdtype(np.asarray(arr).dtype, np.floating):
            a = np.asarray(arr, dtype=np.float32)
            return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        return np.asarray(arr, dtype=np.float32)

    obj_trans = clean(obj_data["obj_trans"])
    obj_rot = clean(obj_data["obj_rot"])
    n_frames_data = obj_trans.shape[0]

    for k in ["pose_r", "shape_r", "rot_r", "trans_r"]:
        if k in hand_data:
            hand_data[k] = clean(hand_data[k])

    pose_r = hand_data["pose_r"]
    shape_r = hand_data["shape_r"]
    rot_r = hand_data["rot_r"]

    if has_left_hand:
        hand_data_l = params["left hand"]
        for k in ["pose_l", "shape_l", "rot_l", "trans_l"]:
            if k in hand_data_l:
                hand_data_l[k] = clean(hand_data_l[k])

    # 2. 加载 JSON 关键点
    print(f"加载: {args.keypoints_json}")
    keypoints_3d, num_frames_json = load_json_keypoints(args.keypoints_json)
    n_frames_json = keypoints_3d.shape[0]
    right_wrist = keypoints_3d[:, COCO17_RIGHT_WRIST_INDEX, :]  # (num_frames_json, 3)
    left_wrist = keypoints_3d[:, COCO17_LEFT_WRIST_INDEX, :]  # (num_frames_json, 3)

    # 2.1
    #   - json_format=xyz：JSON 已经存的是“真实相机坐标系下”的 (X,Y,Z)，可直接使用；
    #   - json_format=xyz_virtual：JSON 存的是“虚拟内参相机坐标系下”的 (X,Y,Z)，
    #       需先用虚拟内参恢复 (u,v,z)，再用真实 K 反投影到真实相机坐标系；
    #   - json_format=uvz：JSON 存的是 (u,v,z)，直接用真实 K 反投影到真实相机坐标系。
    if args.json_format == "xyz":
        # 不做任何转换，避免把 (X,Y,Z) 误当 (u,v,z) 反投影导致错位
        print("JSON 格式 xyz：认为存的是“真实相机坐标系” (X,Y,Z)，直接作为真实相机坐标使用")
    elif args.json_format == "xyz_virtual":
        if args.json_img_width is None or args.json_img_height is None:
            raise ValueError("json_format=xyz_virtual 时须同时指定 --json_img_width 与 --json_img_height")
        if K_real is None:
            raise KeyError("data.npy 中缺少 camera.k_use，无法在 json_format=xyz_virtual 时将虚拟相机坐标转换到真实相机坐标系")

        # 虚拟相机内参（与生成 JSON 时一致）：f = sqrt(w^2+h^2), cx=w/2, cy=h/2
        img_w = float(args.json_img_width)
        img_h = float(args.json_img_height)
        f_v = (img_w ** 2 + img_h ** 2) ** 0.5
        cx_v, cy_v = img_w / 2.0, img_h / 2.0

        # 先用虚拟内参把虚拟相机系下的 (X_v, Y_v, Z_v) 转成 (u, v, z)
        def virtual_xyz_to_uvz(xyz: np.ndarray) -> np.ndarray:
            Xv = xyz[..., 0]
            Yv = xyz[..., 1]
            Zv = xyz[..., 2]
            u = f_v * Xv / np.maximum(Zv, 1e-6) + cx_v
            v = f_v * Yv / np.maximum(Zv, 1e-6) + cy_v
            z = Zv
            return np.stack([u, v, z], axis=-1).astype(np.float32)

        right_wrist_uvz = virtual_xyz_to_uvz(right_wrist)
        left_wrist_uvz = virtual_xyz_to_uvz(left_wrist)

        # 再用真实 K 把 (u,v,z) 反投影到真实相机坐标系 (X,Y,Z)
        right_wrist = uvz_to_cam_xyz_with_K(right_wrist_uvz, K_real)
        left_wrist = uvz_to_cam_xyz_with_K(left_wrist_uvz, K_real)
        print(
            f"JSON 格式 xyz_virtual：已按虚拟内参 (w={img_w}, h={img_h}, f=sqrt(w^2+h^2), cx=w/2, cy=h/2) "
            f"将 (X_v,Y_v,Z_v) 映射为 (u,v,z)，再用真实 K (camera.k_use) 反投影为真实相机坐标 (X,Y,Z)"
        )
    elif args.json_format == "uvz":
        if K_real is None:
            raise KeyError("data.npy 中缺少 camera.k_use，无法在 json_format=uvz 时用真实 K 反投影 (u,v,z) 到 (X,Y,Z)")
        right_wrist = uvz_to_cam_xyz_with_K(
            right_wrist,
            K_real,
        )
        left_wrist = uvz_to_cam_xyz_with_K(
            left_wrist,
            K_real,
        )
        print(
            "已将 JSON 手腕 (u,v,z) 用 data.npy 中的真实相机内参 K (camera.k_use) "
            "反投影为相机坐标 (X,Y,Z)"
        )

    # 3. 帧数对齐：取最小长度（若有左手则一并纳入）
    n_frames = min(n_frames_data, n_frames_json, pose_r.shape[0], shape_r.shape[0], rot_r.shape[0])
    if has_left_hand:
        hand_data_l = params["left hand"]
        pose_l = hand_data_l["pose_l"]
        n_frames = min(n_frames, pose_l.shape[0], hand_data_l["shape_l"].shape[0], hand_data_l["rot_l"].shape[0])
    if not (n_frames_data == n_frames_json == pose_r.shape[0] == shape_r.shape[0] == rot_r.shape[0]):
        print(
            f"警告: 帧数不一致，将统一裁剪到前 {n_frames} 帧: "
            f"data={n_frames_data}, json={n_frames_json}, pose_r={pose_r.shape[0]}, "
            f"shape_r={shape_r.shape[0]}, rot_r={rot_r.shape[0]}"
            + (f", pose_l={pose_l.shape[0]}" if has_left_hand else "")
        )

    obj_trans = obj_trans[:n_frames]
    obj_rot = obj_rot[:n_frames]
    pose_r = pose_r[:n_frames]
    shape_r = shape_r[:n_frames]
    rot_r = rot_r[:n_frames]
    right_wrist = right_wrist[:n_frames]
    left_wrist = left_wrist[:n_frames]
    if has_left_hand:
        hand_data_l = params["left hand"]
        for k in ["pose_l", "shape_l", "rot_l", "trans_l"]:
            if k in hand_data_l and hand_data_l[k].shape[0] > n_frames:
                hand_data_l[k] = hand_data_l[k][:n_frames].astype(np.float32)
        pose_l = hand_data_l["pose_l"][:n_frames]
        shape_l = hand_data_l["shape_l"][:n_frames]
        rot_l = hand_data_l["rot_l"][:n_frames]

    # 4. MANO: 计算在 transl=0 时，每帧关节几何中心相对手腕的偏移
    print("创建 MANO 模型并计算关节几何中心偏移 ...")
    mano_layer = smplx.create(
        model_path=os.path.dirname(MANO_MODEL_PATH),
        model_type="MANO",
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=True,
    )

    pose_tensor = torch.tensor(pose_r, dtype=torch.float32)
    shape_tensor = torch.tensor(shape_r, dtype=torch.float32)
    rot_tensor = torch.tensor(rot_r, dtype=torch.float32)
    zero_trans = torch.zeros((n_frames, 3), dtype=torch.float32)

    with torch.no_grad():
        out_zero = mano_layer(
            global_orient=rot_tensor,
            hand_pose=pose_tensor,
            betas=shape_tensor,
            transl=zero_trans,
        )

    joints = out_zero.joints.cpu().numpy()  # (N, 16, 3)，关节 0 为手腕
    # 手腕在 transl=0 时在相机系下的位置（由 global_orient + hand_pose 决定）
    wrist_cam_when_transl_zero = joints[:, 0, :]  # (N, 3)
    # 手的关节平均点（transl=0 时）在相机系下的偏移
    joints_mean_offset = np.mean(joints, axis=1)  # (N, 3)

    # 5. 实际手腕 = trans_r + wrist_cam_when_transl_zero。令其等于 JSON 手腕（真实相机系）
    #    => trans_r_new = json_wrist_cam - wrist_cam_when_transl_zero（每帧独立）
    desired_wrist_cam = right_wrist.astype(np.float32)  # (N, 3)，每帧使用 JSON 手腕在真实相机系下的位置
    hand_trans_new = desired_wrist_cam - wrist_cam_when_transl_zero
    hand_data["trans_r"] = hand_trans_new.astype(np.float32)
    # 调试：若手腕几乎不动，检查 desired_wrist_cam 是否随帧变化
    w_std = np.std(desired_wrist_cam, axis=0)
    print(
        f"已根据 JSON 右手腕（索引 {COCO17_RIGHT_WRIST_INDEX}）按「trans + rot + MANO 手腕」反算 trans_r，共 {n_frames} 帧；"
        f"desired_wrist_cam 各轴标准差 [x y z] = [{w_std[0]:.4f} {w_std[1]:.4f} {w_std[2]:.4f}]（若接近 0 说明每帧手腕几乎不变，请核对 --json_format 与数据）"
    )

    # 5b. 左手腕与 JSON 左手腕（真实相机系）对齐
    #     npy 中左手存的是右手 MANO 参数，需要镜像后才能送入左手 MANO
    if has_left_hand:
        mano_layer_l = smplx.create(
            model_path=os.path.dirname(MANO_MODEL_PATH),
            model_type="MANO",
            is_rhand=False,
            use_pca=False,
            flat_hand_mean=True,
        )
        rot_l_mirrored, pose_l_mirrored = right_to_left_mano_params(rot_l, pose_l)
        pose_tensor_l = torch.tensor(pose_l_mirrored, dtype=torch.float32)
        shape_tensor_l = torch.tensor(shape_l, dtype=torch.float32)
        rot_tensor_l = torch.tensor(rot_l_mirrored, dtype=torch.float32)
        zero_trans_l = torch.zeros((n_frames, 3), dtype=torch.float32)
        with torch.no_grad():
            out_zero_l = mano_layer_l(
                global_orient=rot_tensor_l,
                hand_pose=pose_tensor_l,
                betas=shape_tensor_l,
                transl=zero_trans_l,
            )
        joints_l = out_zero_l.joints.cpu().numpy()  # (N, 16, 3)
        wrist_cam_when_transl_zero_l = joints_l[:, 0, :]  # (N, 3)
        desired_left_wrist_cam = left_wrist.astype(np.float32)
        hand_trans_new_l = desired_left_wrist_cam - wrist_cam_when_transl_zero_l
        hand_data_l["trans_l"] = hand_trans_new_l.astype(np.float32)
        print(f"已根据 JSON 左手腕（索引 {COCO17_LEFT_WRIST_INDEX}）按「镜像参数 + 左手 MANO 手腕」反算 trans_l，共 {n_frames} 帧")

    # 6. 物体不动，手跟着物体走：
    #    在参考帧，先把手部平均点对齐到物体指定点，得到此时的根节点位置，
    #    以该根节点与物体指定点的偏移为基准，所有帧都保持这个偏移 → 重算 trans_r (和 trans_l)
    if not args.no_object_align:
        ref = min(args.ref_frame, n_frames - 1)

        r_obj = R.from_rotvec(obj_rot)

        # 右手 → 物体上的 TARGET_CENTER_LOCAL_R
        obj_target_r = obj_trans + r_obj.apply(TARGET_CENTER_LOCAL_R)  # (N, 3)

        ref_trans_r_aligned = obj_target_r[ref] - joints_mean_offset[ref]
        ref_wrist_r = ref_trans_r_aligned + wrist_cam_when_transl_zero[ref]
        offset_r = ref_wrist_r - obj_target_r[ref]

        hand_trans_new = obj_target_r + offset_r - wrist_cam_when_transl_zero
        hand_data["trans_r"] = hand_trans_new.astype(np.float32)

        print(
            f"参考帧 {ref}: 右手均值点对齐物体点 {TARGET_CENTER_LOCAL_R} 后 → 根节点 = {ref_wrist_r}, "
            f"物体指定点 = {obj_target_r[ref]}, 相对偏移 = {offset_r}"
        )

        # 左手 → 物体上的 TARGET_CENTER_LOCAL_L
        if has_left_hand:
            obj_target_l = obj_trans + r_obj.apply(TARGET_CENTER_LOCAL_L)  # (N, 3)

            joints_mean_offset_l = np.mean(joints_l, axis=1)  # (N, 3)
            ref_trans_l_aligned = obj_target_l[ref] - joints_mean_offset_l[ref]
            ref_wrist_l = ref_trans_l_aligned + wrist_cam_when_transl_zero_l[ref]
            offset_l = ref_wrist_l - obj_target_l[ref]

            hand_trans_new_l = obj_target_l + offset_l - wrist_cam_when_transl_zero_l
            hand_data_l["trans_l"] = hand_trans_new_l.astype(np.float32)
            print(
                f"参考帧 {ref}: 左手均值点对齐物体点 {TARGET_CENTER_LOCAL_L} 后 → 根节点 = {ref_wrist_l}, "
                f"相对偏移 = {offset_l}"
            )

        print(f"已固定第 {ref} 帧的手-物相对偏移，物体不变")

    # 7. 若裁剪了帧数，保持手/物体参数内部长度一致
    if n_frames < n_frames_data:
        for key in list(obj_data.keys()):
            arr = obj_data[key]
            if isinstance(arr, np.ndarray) and arr.shape[0] == n_frames_data:
                obj_data[key] = arr[:n_frames].astype(np.float32)
        for key in list(hand_data.keys()):
            arr = hand_data[key]
            if isinstance(arr, np.ndarray) and arr.shape[0] == n_frames_data:
                hand_data[key] = arr[:n_frames].astype(np.float32)
        if has_left_hand:
            for key in list(hand_data_l.keys()):
                arr = hand_data_l[key]
                if isinstance(arr, np.ndarray) and arr.shape[0] == n_frames_data:
                    hand_data_l[key] = arr[:n_frames].astype(np.float32)

    # 7b. 将 camera.k_use 覆盖为虚拟相机内参（f = sqrt(w^2+h^2), cx=w/2, cy=h/2）
    if cam_data is not None:
        img_w = float(args.json_img_width) if args.json_img_width is not None else None
        img_h = float(args.json_img_height) if args.json_img_height is not None else None
        if img_w is not None and img_h is not None:
            f_v = (img_w ** 2 + img_h ** 2) ** 0.5
            cx_v, cy_v = img_w / 2.0, img_h / 2.0
            K_virtual = np.array(
                [[f_v, 0.0, cx_v],
                 [0.0, f_v, cy_v],
                 [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            cam_data["k_use"] = K_virtual

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    print(f"保存: {output_path}")
    np.save(output_path, data)
    print("完成。")


if __name__ == "__main__":
    main()
