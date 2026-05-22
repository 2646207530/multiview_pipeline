#!/usr/bin/env python3
"""
从 data_fixed.npy 中裁剪手和物体的时间片段，保留第 n 帧到第 m 帧（含），
并保存为 /home/pt/fbs/data_fixed_crop.npy。

用法示例：
    python -m utils.crop_data_fixed --start 100 --end 200

注意：
- 帧索引为 0 基（0 表示第一帧）；
- 本脚本会同时裁剪 hand / object 以及与帧数对齐的一些其它字段（若存在）。
"""

import argparse
import os
from typing import Any, Dict

import numpy as np


def crop_sequence(
    data: Dict[str, Any],
    start: int,
    end: int,
) -> Dict[str, Any]:
    if "data_dict" not in data:
        raise KeyError("data.npy 中缺少 data_dict")

    # 假定所有序列的帧数一致，沿用 post_concat.py 的约定
    for seq_name, seq in data["data_dict"].items():
        if "params" not in seq:
            continue
        params = seq["params"]

        if "object" not in params:
            raise KeyError(f"序列 {seq_name} 的 params 中缺少 object")

        obj_data = params["object"]
        obj_trans = obj_data.get("obj_trans", None)
        if not isinstance(obj_trans, np.ndarray) or obj_trans.ndim < 1:
            raise ValueError(f"序列 {seq_name} 的 obj_trans 不是有效的时间序列数组")

        n_frames_total = obj_trans.shape[0]

        if not (0 <= start <= end < n_frames_total):
            raise ValueError(
                f"非法帧区间 [{start}, {end}]，总帧数为 {n_frames_total}（索引为 0~{n_frames_total - 1})"
            )

        sl = slice(start, end + 1)

        # 裁剪 object 相关的所有帧对齐数组
        for key in list(obj_data.keys()):
            arr = obj_data[key]
            if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] == n_frames_total:
                obj_data[key] = arr[sl]

        # 裁剪右手
        if "right hand" in params:
            hand_r = params["right hand"]
            for key in list(hand_r.keys()):
                arr = hand_r[key]
                if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] == n_frames_total:
                    hand_r[key] = arr[sl]

        # 裁剪左手（若存在）
        if "left hand" in params:
            hand_l = params["left hand"]
            for key in list(hand_l.keys()):
                arr = hand_l[key]
                if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] == n_frames_total:
                    hand_l[key] = arr[sl]

        # 裁剪与帧数对齐的顶层字段（例如 frame_ids / image_names 等）
        for key in list(seq.keys()):
            if key == "params":
                continue
            value = seq[key]
            if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == n_frames_total:
                seq[key] = value[sl]
            elif isinstance(value, list) and len(value) == n_frames_total:
                seq[key] = value[start : end + 1]

    # 裁剪顶层帧对齐字段（如 imgnames）
    sl_top = slice(start, end + 1)
    for key in list(data.keys()):
        if key == "data_dict":
            continue
        value = data[key]
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == n_frames_total:
            data[key] = value[sl_top]
        elif isinstance(value, list) and len(value) == n_frames_total:
            data[key] = value[start : end + 1]

    return data


def main():
    parser = argparse.ArgumentParser(
        description="从 data_fixed.npy 中裁剪手和物体的第 n~m 帧（0 基索引，含两端）"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/home/pt/fbs/data_fixed.npy",
        help="输入的 data_fixed.npy 路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/pt/fbs/ManipTrans/data/ours/data_fixed_crop.npy",
        help="输出裁剪后的 data.npy 路径",
    )
    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="起始帧索引（0 基，含此帧）",
    )
    parser.add_argument(
        "--end",
        type=int,
        required=True,
        help="结束帧索引（0 基，含此帧）",
    )

    args = parser.parse_args()

    if args.end < args.start:
        raise ValueError(f"--end ({args.end}) 不能小于 --start ({args.start})")

    print(f"加载: {args.input}")
    data = np.load(args.input, allow_pickle=True).item()

    data_cropped = crop_sequence(data, args.start, args.end)

    output_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    print(f"保存裁剪结果到: {output_path}")
    np.save(output_path, data_cropped)
    print("完成。")


if __name__ == "__main__":
    main()

