"""
翻转 OBJ 文件中所有面的法向量方向。
操作:
  1. 将所有顶点法向量 (vn) 取反
  2. 反转每个面 (f) 的顶点绕序 (winding order)

用法:
  python flip_obj_normals.py <input.obj> [output.obj]
  如果不指定 output，则默认输出为 <input>_flipped.obj
"""

import argparse
import os
import re


def flip_normals(input_path: str, output_path: str | None = None):
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_flipped{ext}"

    with open(input_path, "r") as f:
        lines = f.readlines()

    out_lines = []
    vn_count = 0
    f_count = 0

    for line in lines:
        stripped = line.strip()

        # ---------- 翻转顶点法向量 ----------
        if stripped.startswith("vn "):
            parts = stripped.split()
            # parts: ['vn', x, y, z]
            nx, ny, nz = float(parts[1]), float(parts[2]), float(parts[3])
            out_lines.append(f"vn {-nx:.4f} {-ny:.4f} {-nz:.4f}\n")
            vn_count += 1

        # ---------- 反转面的顶点绕序 ----------
        elif stripped.startswith("f "):
            parts = stripped.split()
            # parts: ['f', v1//vn1, v2//vn2, v3//vn3, ...]
            verts = parts[1:]  # 保留每个顶点分量 (可能是 v, v/vt, v/vt/vn, v//vn)
            reversed_verts = list(reversed(verts))
            out_lines.append("f " + " ".join(reversed_verts) + "\n")
            f_count += 1

        else:
            out_lines.append(line)

    with open(output_path, "w") as f:
        f.writelines(out_lines)

    print(f"✅ 完成! 翻转了 {vn_count} 条法向量, 反转了 {f_count} 个面的绕序")
    print(f"   输入: {input_path}")
    print(f"   输出: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="翻转 OBJ 文件的面法向量方向")
    parser.add_argument("input", type=str, help="输入 OBJ 文件路径")
    parser.add_argument("output", type=str, nargs="?", default=None,
                        help="输出 OBJ 文件路径 (默认: <input>_flipped.obj)")
    args = parser.parse_args()
    flip_normals(args.input, args.output)
