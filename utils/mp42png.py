#!/usr/bin/env python3
"""
将指定 MP4 视频转换为帧序列 PNG 图片。
输出到名为 rgb 的文件夹，帧命名为 000000.png, 000001.png, ...
"""
import argparse
import os

import cv2


def main():
    parser = argparse.ArgumentParser(description="MP4 转 PNG 帧序列（输出到 rgb 文件夹）")
    parser.add_argument("mp4_path", type=str, help="输入的 MP4 文件路径")
    parser.add_argument(
        "-o", "--output-dir",
        type=str, default=None,
        help="输出根目录，默认为 MP4 所在目录",
    )
    args = parser.parse_args()

    mp4_path = os.path.abspath(args.mp4_path)
    if not os.path.isfile(mp4_path):
        raise FileNotFoundError(f"找不到视频文件: {mp4_path}")

    if args.output_dir is None:
        base_dir = os.path.dirname(mp4_path)
    else:
        base_dir = os.path.abspath(args.output_dir)

    rgb_dir = os.path.join(base_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {mp4_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(rgb_dir, f"{frame_idx:06d}.png")
        cv2.imwrite(out_path, frame)
        frame_idx += 1

    cap.release()
    print(f"已写入 {frame_idx} 帧到: {rgb_dir}")


if __name__ == "__main__":
    main()
