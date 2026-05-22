#!/usr/bin/env python3
"""
对指定 OBJ 网格做奇异值分解 (SVD)，用于分析主方向 / 尺度。

用法示例：
    python utils/svd_obj.py --obj /home/pt/fbs/球杆_补洞.obj

可选将点云按主轴对齐并导出新的 OBJ：
    python utils/svd_obj.py --obj input.obj --save_aligned aligned.obj
"""

import argparse
import os

import numpy as np


def compute_svd(vertices: np.ndarray):
    """
    对顶点 (N,3) 做中心化后 SVD：
        Vc = V - mean
        Vc = U S V^T
    返回:
        mean: (3,)
        U: (N,N) 左奇异向量（通常不关心）
        S: (3,) 奇异值（越大说明该主方向上的尺度越大）
        VT: (3,3) 右奇异向量转置，每一行是一个主方向（类似 PCA 主轴）
    """
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices 形状必须是 (N,3)，当前为 {vertices.shape}")

    mean = vertices.mean(axis=0, keepdims=True)
    centered = vertices - mean  # (N,3)

    # 只对 3 维特征做 SVD，速度更快（相当于对协方差矩阵做特征分解）
    # 也可以直接对 (N,3) 做 np.linalg.svd(centered, full_matrices=False)
    cov = centered.T @ centered / max(len(vertices) - 1, 1)  # (3,3)
    # 协方差矩阵的特征分解等价于 SVD 中的右奇异向量和奇异值平方 / (N-1)
    eigvals, eigvecs = np.linalg.eigh(cov)  # 返回升序
    # 按特征值从大到小排序
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # 奇异值 S 与特征值关系：S_i = sqrt( (N-1) * λ_i )
    S = np.sqrt(np.maximum(eigvals, 0.0) * max(len(vertices) - 1, 1))
    VT = eigvecs.T  # (3,3) 每一行是一个主轴方向

    # 特征向量只定义到符号：eigh 可能给出一组“左手系”主轴，导致 VT.T 为反射（镜像）
    # 强制为右手系：若 det(VT.T) < 0，翻转第三主轴（最后一行）
    R = VT.T
    if np.linalg.det(R) < 0:
        VT = VT.copy()
        VT[-1, :] *= -1

    return mean[0], S, VT


def main():
    parser = argparse.ArgumentParser(
        description="对指定 OBJ 做 SVD，打印主方向与奇异值，并可选导出按主轴对齐后的 OBJ。"
    )
    parser.add_argument(
        "--obj",
        type=str,
        required=True,
        help="输入 OBJ 文件路径",
    )
    parser.add_argument(
        "--save_aligned",
        type=str,
        default=None,
        help="若指定，则保存按主轴对齐且以质心为原点的 OBJ 到该路径",
    )
    args = parser.parse_args()

    obj_path = os.path.abspath(args.obj)
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"OBJ 文件不存在: {obj_path}")

    print(f"加载 OBJ: {obj_path}")
    with open(obj_path, "r") as f:
        lines = f.readlines()

    # 解析所有顶点行 v x y z，记录其索引和坐标
    vertex_indices = []
    vertex_positions = []
    for idx, line in enumerate(lines):
        if not line.lstrip().startswith("v "):
            continue
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        try:
            x, y, z = map(float, parts[1:4])
        except ValueError:
            continue
        vertex_indices.append(idx)
        vertex_positions.append([x, y, z])

    if not vertex_positions:
        raise ValueError("OBJ 中未找到任何顶点行 (v x y z)")

    vertices = np.asarray(vertex_positions, dtype=np.float64)
    print(f"顶点数量: {vertices.shape[0]}")

    mean, S, VT = compute_svd(vertices)

    print("\n==== 顶点质心 (mean，在原始坐标系下) ====")
    print(mean)

    print("\n==== 奇异值 S（对应三个主方向的尺度，从大到小） ====")
    print(S)

    print("\n==== 主方向（右奇异向量的转置 VT，每一行是一个单位方向向量） ====")
    print("行 0: 最大主轴方向")
    print("行 1: 次大主轴方向")
    print("行 2: 最小主轴方向")
    print(VT)

    if args.save_aligned is not None:
        # 将点云移到以质心为原点的坐标系，并用 VT 旋转到主轴对齐坐标系，
        # 然后整体再缩放到原始尺度的 0.4 倍
        centered = vertices - mean  # (N,3)
        aligned = centered @ VT.T  # (N,3)
        aligned *= 0.4

        # 在原始 OBJ 文本上，仅替换 v 行的坐标，其余 (vn/vt/f/...) 完全保持不变
        new_lines = list(lines)
        for (line_idx, (x, y, z)) in zip(vertex_indices, aligned):
            prefix = lines[line_idx].split("#", 1)[0]  # 保留行内注释前的部分结构
            parts = prefix.strip().split()
            # 保持 "v" 开头，其余坐标用新的值（格式简洁输出）
            parts[0] = "v"
            parts[1:4] = [f"{x:.8f}", f"{y:.8f}", f"{z:.8f}"]
            new_line = " ".join(parts)
            # 若原行有 # 注释，拼回去
            if "#" in lines[line_idx]:
                comment = lines[line_idx].split("#", 1)[1].rstrip("\n")
                new_line = f"{new_line}  # {comment}"
            new_lines[line_idx] = new_line + "\n"

        out_path = os.path.abspath(args.save_aligned)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            f.writelines(new_lines)
        print(f"\n已保存按主轴对齐后的 OBJ 到: {out_path}（仅修改 v 行坐标，f/vt/vn 等完全保持不变）")


if __name__ == "__main__":
    main()

