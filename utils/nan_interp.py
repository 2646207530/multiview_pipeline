"""NaN 插值辅助函数 (从旧 CLI ``run_hamer_to_npy.py`` 抽出).

被 pipeline 的 step5 (inference) 复用: 多视角推理缺帧会留 NaN, 这里做
逐列线性插值 (平移/形状) + SLERP 旋转插值 (轴角) 填补. 纯 numpy/scipy.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


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
