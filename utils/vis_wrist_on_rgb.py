#!/usr/bin/env python3
"""
在虚拟相机内参下，把 3d_keypoints.json 中的手腕（及可选其他关键点）重投影回 RGB 图并保存。

假设 JSON 中关键点为虚拟相机坐标系下的 (X, Y, Z)。
投影公式：u = fx*X/Z + cx, v = fy*Y/Z + cy，其中虚拟 K 为 f=sqrt(w^2+h^2), cx=w/2, cy=h/2。
"""

import os
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm

# COCO17: 9=左手腕, 10=右手腕
COCO17_RIGHT_WRIST = 10
COCO17_LEFT_WRIST = 9


def virtual_K_from_image_size(img_w: float, img_h: float):
    """虚拟相机内参：f = sqrt(w^2+h^2)，主点 (w/2, h/2)。"""
    f = float((img_w ** 2 + img_h ** 2) ** 0.5)
    cx = img_w / 2.0
    cy = img_h / 2.0
    return f, cx, cy


def project_xyz_to_uv(xyz: np.ndarray, f: float, cx: float, cy: float) -> np.ndarray:
    """
    将虚拟相机下的 (X, Y, Z) 投影到像素 (u, v)。
    xyz: (..., 3)，最后一维为 X, Y, Z
    返回: (..., 2) 为 u, v；Z<=0 的位置会得到 nan，调用方需过滤。
    """
    X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    z_ok = Z > 1e-6
    u = np.where(z_ok, f * X / Z + cx, np.nan)
    v = np.where(z_ok, f * Y / Z + cy, np.nan)
    return np.stack([u, v], axis=-1)


def load_json_keypoints(json_path: str):
    """加载 3d_keypoints.json，返回 (num_frames, 17, 3)。"""
    with open(json_path, "r") as f:
        raw = json.load(f)
    kp = np.array(raw["keypoints_3d_coco17"], dtype=np.float32)
    return kp, raw.get("num_frames", len(kp))


def main():
    parser = argparse.ArgumentParser(
        description="用虚拟相机内参将 JSON 手腕 3D 重投影到 RGB 图并保存"
    )
    parser.add_argument(
        "--keypoints_json",
        type=str,
        default="/home/pt/fbs/dataset/DA5298464_Video_20251017143917258_w1440_h1080_pBayerRG8_f97/3d_keypoints.json",
        help="3d_keypoints.json 路径",
    )
    parser.add_argument(
        "--rgb_dir",
        type=str,
        default="/home/pt/fbs/FoundationPoseROS2/FoundationPose/demo_data/ours/rgb",
        help="RGB 图像文件夹（含 000000.jpg, 000001.jpg, ...）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/pt/fbs/test/wrist_reproj_frames",
        help="标注后的图像输出目录",
    )
    parser.add_argument(
        "--img_width",
        type=float,
        default=1440.0,
        help="虚拟相机图像宽度（用于虚拟内参）",
    )
    parser.add_argument(
        "--img_height",
        type=float,
        default=1080.0,
        help="虚拟相机图像高度（用于虚拟内参）",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="每 N 帧保存一张（1=每帧都保存）",
    )
    parser.add_argument(
        "--right_wrist_only",
        action="store_true",
        help="只画右手腕；否则画右手腕+左手腕",
    )
    args = parser.parse_args()

    print(f"加载: {args.keypoints_json}")
    keypoints_3d, num_frames = load_json_keypoints(args.keypoints_json)
    # (N, 17, 3) 虚拟相机下的 (X, Y, Z)
    n_frames = keypoints_3d.shape[0]

    f, cx, cy = virtual_K_from_image_size(args.img_width, args.img_height)
    print(f"虚拟内参: f={f:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # 右手腕索引 10，左手腕 9
    indices = [COCO17_RIGHT_WRIST]
    if not args.right_wrist_only:
        indices.append(COCO17_LEFT_WRIST)

    # 投影所有帧的选定关键点到 (u,v)
    uv_list = []
    for idx in indices:
        xyz = keypoints_3d[:, idx, :]  # (N, 3)
        uv = project_xyz_to_uv(xyz, f, cx, cy)  # (N, 2)
        uv_list.append(uv)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"输出目录: {args.output_dir}，每 {args.interval} 帧保存一张")

    # 图像命名：与常见序列一致 000000.jpg, 000001.jpg, ...
    for i in tqdm(range(0, n_frames, args.interval), desc="重投影并保存"):
        img_name = f"{i:06d}.jpg"
        img_path = os.path.join(args.rgb_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # 右手腕红点，左手腕蓝点
        colors = [(0, 0, 255), (255, 0, 0)]  # BGR: 红、蓝
        for k, uv in enumerate(uv_list):
            u, v = uv[i, 0], uv[i, 1]
            if not np.isfinite(u) or not np.isfinite(v):
                continue
            px, py = int(round(u)), int(round(v))
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(img, (px, py), 10, colors[k], -1)
                cv2.circle(img, (px, py), 10, (255, 255, 255), 2)

        out_path = os.path.join(args.output_dir, img_name)
        cv2.imwrite(out_path, img)

    n_saved = len(list(range(0, n_frames, args.interval)))
    print(f"完成，共保存 {n_saved} 张到 {args.output_dir}")


if __name__ == "__main__":
    main()
