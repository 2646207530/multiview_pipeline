#!/usr/bin/env python3
"""
将包含时间序列的 data.npy 从 n FPS 重采样为 m FPS。

假设：
- data.npy 的结构与本项目其它脚本一致，为一个 dict，包含 "data_dict"；
- 每个序列的时间维度沿 axis=0（即 shape[0] 是帧数）；
- 对于数值类型的 numpy 数组，使用线性插值进行重采样；
- 对于长度随帧数变化的 list 或字符串数组，使用就近帧索引进行采样；
- 会尽量保持原有字典结构不变，只修改时间相关字段的长度与内容。

用法示例：
    python -m utils.resample_fps \\
        --input /home/pt/fbs/ManipTrans/data/ours/data.npy \\
        --output /home/pt/fbs/ManipTrans/data/ours/data_60fps.npy \\
        --src_fps 30 --dst_fps 60
"""

import argparse
import os
from typing import Any, Dict

import numpy as np


def compute_new_length(n_frames: int, src_fps: float, dst_fps: float) -> int:
    """
    根据原始帧数与 FPS 计算新的帧数，保持总时长大致不变。

    时长约为 (n_frames - 1) / src_fps，新的帧数:
        new_n = round(duration * dst_fps) + 1
    """
    if n_frames <= 1 or src_fps <= 0 or dst_fps <= 0:
        return n_frames
    duration = (n_frames - 1) / float(src_fps)
    new_n = int(round(duration * float(dst_fps))) + 1
    return max(1, new_n)


def build_time_axes(n_frames: int, src_fps: float, dst_fps: float):
    """构建原始与目标时间轴（单位：秒）。"""
    if n_frames <= 1 or src_fps <= 0 or dst_fps <= 0:
        t_src = np.arange(n_frames, dtype=np.float32)
        return t_src, t_src

    duration = (n_frames - 1) / float(src_fps)
    n_new = compute_new_length(n_frames, src_fps, dst_fps)
    t_src = np.linspace(0.0, duration, n_frames, dtype=np.float32)
    t_dst = np.linspace(0.0, duration, n_new, dtype=np.float32)
    return t_src, t_dst


def resample_numeric_array(arr: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """
    对数值型 numpy 数组沿 axis=0 进行线性插值重采样。
    """
    if arr.ndim == 0:
        return arr
    if arr.shape[0] != t_src.shape[0]:
        # 不是以帧数为第 0 维的数组，直接返回
        return arr

    orig_dtype = arr.dtype
    flat = arr.reshape(arr.shape[0], -1).astype(np.float32)
    n_src, feat_dim = flat.shape

    out = np.empty((t_dst.shape[0], feat_dim), dtype=np.float32)
    for i in range(feat_dim):
        out[:, i] = np.interp(t_dst, t_src, flat[:, i])

    out = out.reshape((t_dst.shape[0],) + arr.shape[1:])
    # 若原来是整数，四舍五入后转回原类型；否则保持 float32
    if np.issubdtype(orig_dtype, np.integer):
        out = np.rint(out).astype(orig_dtype)
    return out


def resample_list_by_index(seq_list: Any, new_indices: np.ndarray):
    """
    对 list 或一维数组，按照就近帧索引进行重采样。
    """
    if isinstance(seq_list, list):
        return [seq_list[i] for i in new_indices]
    arr = np.asarray(seq_list)
    if arr.ndim >= 1 and arr.shape[0] >= 1:
        return arr[new_indices]
    return seq_list


def resample_sequence_dict(
    seq: Dict[str, Any],
    src_fps: float,
    dst_fps: float,
) -> Dict[str, Any]:
    """
    对单个 sequence（通常是 data['data_dict'][seq_name]）内部的时间序列字段做 FPS 重采样。
    """
    if "params" not in seq or "object" not in seq["params"]:
        return seq

    params = seq["params"]
    obj_data = params["object"]
    obj_trans = obj_data.get("obj_trans", None)
    if not isinstance(obj_trans, np.ndarray) or obj_trans.ndim < 1:
        return seq

    n_frames_total = obj_trans.shape[0]
    if n_frames_total <= 1 or src_fps <= 0 or dst_fps <= 0 or src_fps == dst_fps:
        return seq

    # 时间轴与索引映射
    t_src, t_dst = build_time_axes(n_frames_total, src_fps, dst_fps)
    # new_indices 用于 list / 非数值数组 的最近邻采样
    new_indices = np.clip(
        np.round(
            (t_dst / (t_src[-1] if t_src[-1] > 0 else 1.0)) * (n_frames_total - 1)
        ).astype(int),
        0,
        n_frames_total - 1,
    )

    def maybe_resample_any(value: Any) -> Any:
        # 数值 numpy 数组，且第 0 维是时间维
        if isinstance(value, np.ndarray):
            if value.ndim >= 1 and value.shape[0] == n_frames_total and np.issubdtype(
                value.dtype, np.number
            ):
                return resample_numeric_array(value, t_src, t_dst)
            # 非数值数组但长度随帧数变化，做索引重采样
            if value.ndim >= 1 and value.shape[0] == n_frames_total:
                return value[new_indices]
            return value

        # list，且长度等于帧数
        if isinstance(value, list) and len(value) == n_frames_total:
            return resample_list_by_index(value, new_indices)

        return value

    # 1) 先重采样 object 下的所有字段
    for key in list(obj_data.keys()):
        obj_data[key] = maybe_resample_any(obj_data[key])

    # 2) 重采样右手
    if "right hand" in params:
        hand_r = params["right hand"]
        for key in list(hand_r.keys()):
            hand_r[key] = maybe_resample_any(hand_r[key])

    # 3) 重采样左手（若存在）
    if "left hand" in params:
        hand_l = params["left hand"]
        for key in list(hand_l.keys()):
            hand_l[key] = maybe_resample_any(hand_l[key])

    # 4) 重采样与帧数对齐的顶层字段（例如 frame_ids / image_names 等）
    for key in list(seq.keys()):
        if key == "params":
            continue
        value = seq[key]
        seq[key] = maybe_resample_any(value)

    return seq


def resample_data_dict(
    data: Dict[str, Any],
    src_fps: float,
    dst_fps: float,
) -> Dict[str, Any]:
    """
    对整个 data.npy（顶层 dict）进行 FPS 重采样。
    """
    if "data_dict" not in data:
        raise KeyError("data.npy 中缺少 data_dict")

    for seq_name, seq in data["data_dict"].items():
        data["data_dict"][seq_name] = resample_sequence_dict(seq, src_fps, dst_fps)

    # 可选：若顶层有 fps 相关字段，可在这里同步更新（根据项目实际字段名自行修改）
    # 例如：
    # if "fps" in data:
    #     data["fps"] = float(dst_fps)

    return data


def main():
    parser = argparse.ArgumentParser(
        description="将 data.npy 中按时间对齐的序列从 n FPS 重采样为 m FPS"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/home/pt/fbs/ManipTrans/data/ours/data.npy",
        help="输入 data.npy 路径（包含 data_dict）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="输出 data.npy 路径（默认与输入同目录，文件名追加 '_{dst_fps}fps' 后缀）",
    )
    parser.add_argument(
        "--src_fps",
        type=float,
        required=True,
        help="原始帧率（n FPS）",
    )
    parser.add_argument(
        "--dst_fps",
        type=float,
        required=True,
        help="目标帧率（m FPS）",
    )

    args = parser.parse_args()
    if args.src_fps <= 0 or args.dst_fps <= 0:
        raise ValueError("--src_fps 与 --dst_fps 必须为正数")

    input_path = os.path.abspath(args.input)
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        in_dir, in_name = os.path.split(input_path)
        name, ext = os.path.splitext(in_name)
        suffix = f"_{int(args.dst_fps)}fps" if float(args.dst_fps).is_integer() else f"_{args.dst_fps}fps"
        output_path = os.path.join(in_dir, name + suffix + ext)

    print(f"加载: {input_path}")
    data = np.load(input_path, allow_pickle=True).item()

    print(f"从 {args.src_fps} FPS 重采样到 {args.dst_fps} FPS ...")
    data_resampled = resample_data_dict(data, args.src_fps, args.dst_fps)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"保存到: {output_path}")
    np.save(output_path, data_resampled)
    print("完成。")


if __name__ == "__main__":
    main()

