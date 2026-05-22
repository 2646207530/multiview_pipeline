#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将两个视频左右排列合并为一个，并在各自画面左上角标注文件名。

用法示例：
    python utils/merge_videos_side_by_side.py left.mp4 right.mp4 output.mp4

依赖：
    - 系统需安装 ffmpeg，并在 PATH 中可直接调用 `ffmpeg`
"""

import argparse
import os
import shlex
import subprocess


def build_drawtext(label: str, alias: str, fontsize: int) -> str:
    """
    为某一路视频构造 drawtext 滤镜。
    alias: 该路视频在 filter_complex 中的标签名（如 v0、v1）。
    """
    # 简单转义，避免常见字符导致 ffmpeg 解析出错
    safe_label = label.replace(":", r"\:").replace("'", r"\'")
    return (
        f"[{alias}]drawtext=text='{safe_label}':"
        f"x=10:y=10:fontsize={fontsize}:fontcolor=white:borderw=2:{''}"
    )


def main():
    parser = argparse.ArgumentParser(description="左右拼接两个视频并添加文件名标注")
    parser.add_argument("left", help="左侧视频路径")
    parser.add_argument("right", help="右侧视频路径")
    parser.add_argument("output", help="输出视频路径")
    parser.add_argument(
        "--fontsize", type=int, default=28, help="文字字号（默认：28）"
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="ffmpeg 可执行文件名或绝对路径（默认：ffmpeg）",
    )
    args = parser.parse_args()

    left_path = os.path.abspath(args.left)
    right_path = os.path.abspath(args.right)
    output_path = os.path.abspath(args.output)

    if not os.path.exists(left_path):
        raise FileNotFoundError(f"左侧视频不存在：{left_path}")
    if not os.path.exists(right_path):
        raise FileNotFoundError(f"右侧视频不存在：{right_path}")

    left_label = os.path.basename(left_path)
    right_label = os.path.basename(right_path)

    # 说明：为了简单起见，这里假设两路视频的分辨率/高度一致；
    # 如果不一致，ffmpeg 的 hstack 会报错，可在此基础上自行增加 scale 逻辑。

    # 过滤链：
    #   [0:v] 命名为 v0，加文字 -> lv
    #   [1:v] 命名为 v1，加文字 -> rv
    #   [lv][rv] hstack=inputs=2 -> [v]
    draw_left = build_drawtext(left_label, "v0", args.fontsize) + "[lv]"
    draw_right = build_drawtext(right_label, "v1", args.fontsize) + "[rv]"
    filter_complex = (
        "[0:v]setpts=PTS-STARTPTS[v0];"
        "[1:v]setpts=PTS-STARTPTS[v1];"
        f"{draw_left};"
        f"{draw_right};"
        "[lv][rv]hstack=inputs=2[v]"
    )

    cmd = [
        args.ffmpeg,
        "-y",  # 覆盖输出
        "-i",
        left_path,
        "-i",
        right_path,
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",      # 视频映射为合成结果
        "-map",
        "0:a?",     # 音频优先用左侧，如无则忽略
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-shortest",
        output_path,
    ]

    print("运行命令：")
    print(" ".join(shlex.quote(c) for c in cmd))

    subprocess.run(cmd, check=True)
    print(f"合成完成：{output_path}")


if __name__ == "__main__":
    main()

