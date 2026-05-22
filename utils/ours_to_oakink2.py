#python utils/ours_to_oakink2.py --input /path/to/data.npy --obj /path/to/final.obj --output /path/to/output.pkl
import os
import argparse
import json
import pickle

import numpy as np
import torch
import smplx
from scipy.spatial.transform import Rotation as R


def load_ours_data(npy_path: str):
    """加载我们的数据格式"""
    data = np.load(npy_path, allow_pickle=True).item()
    imgnames = data["imgnames"]

    # 动态获取 sequence 的 key (例如 'seq_136')
    seq_key = list(data["data_dict"].keys())[0]
    params = data["data_dict"][seq_key]["params"]

    rot_r = params["right hand"]["rot_r"]  # (T, 3)
    pose_r = params["right hand"]["pose_r"]  # (T, 45)
    trans_r = params["right hand"]["trans_r"]  # (T, 3)
    shape_r = params["right hand"]["shape_r"]  # (T, 10)

    obj_rot = params["object"]["obj_rot"]  # (T, 3)
    obj_trans = params["object"]["obj_trans"]  # (T, 3)

    return {
        "imgnames": imgnames,
        "rot_r": rot_r,
        "pose_r": pose_r,
        "trans_r": trans_r,
        "shape_r": shape_r,
        "obj_rot": obj_rot,
        "obj_trans": obj_trans,
    }


def convert_mano_to_smplx_params(
    rot_r: np.ndarray,
    pose_r: np.ndarray,
    trans_r: np.ndarray,
    shape_r: np.ndarray,
):
    """
    将 MANO 参数转换为 SMPLX 格式的参数。
    
    SMPLX 需要：
    - betas: (T, 10) 形状参数
    - global_orient: (T, 3) 全局旋转（轴角）
    - body_pose: (T, 63) 身体姿态（21个关节，每个3维轴角）
    - left_hand_pose: (T, 45) 左手姿态
    - right_hand_pose: (T, 45) 右手姿态
    - transl: (T, 3) 平移
    
    对于右手数据，我们只需要：
    - betas: 使用 shape_r
    - global_orient: 使用 rot_r
    - body_pose: 全零（因为我们只有手部）
    - left_hand_pose: 全零
    - right_hand_pose: 使用 pose_r
    - transl: 使用 trans_r
    """
    T = rot_r.shape[0]
    
    # 创建 SMPLX 参数字典
    smplx_params = {
        "betas": shape_r.astype(np.float32),  # (T, 10)
        "global_orient": rot_r.astype(np.float32),  # (T, 3)
        "body_pose": np.zeros((T, 63), dtype=np.float32),  # (T, 63) 全零
        "left_hand_pose": np.zeros((T, 45), dtype=np.float32),  # (T, 45) 全零
        "right_hand_pose": pose_r.astype(np.float32),  # (T, 45)
        "transl": trans_r.astype(np.float32),  # (T, 3)
    }
    
    return smplx_params


def convert_obj_rot_trans_to_transf(obj_rot: np.ndarray, obj_trans: np.ndarray):
    """
    将物体旋转（轴角）和平移转换为 4x4 变换矩阵。
    
    Args:
        obj_rot: (T, 3) 轴角表示的旋转
        obj_trans: (T, 3) 平移向量
    
    Returns:
        obj_transf: (T, 4, 4) 变换矩阵
    """
    T = obj_rot.shape[0]
    obj_transf = np.zeros((T, 4, 4), dtype=np.float32)
    
    for t in range(T):
        # 将轴角转换为旋转矩阵
        R_obj = R.from_rotvec(obj_rot[t]).as_matrix()
        
        # 构建 4x4 变换矩阵
        obj_transf[t, :3, :3] = R_obj
        obj_transf[t, :3, 3] = obj_trans[t]
        obj_transf[t, 3, 3] = 1.0
    
    return obj_transf


def build_oakink2_anno(
    ours: dict,
    obj_id: str = "object_0",
    fps: int = 120,
):
    """
    构建 oakink2 格式的 anno 字典。
    
    Args:
        ours: 我们的数据字典
        obj_id: 物体ID，默认为 "object_0"
        fps: 帧率，默认为 120Hz
    
    Returns:
        anno: oakink2 格式的字典
    """
    T = ours["rot_r"].shape[0]
    
    # 1. 生成帧ID列表（从0开始，120Hz）
    mocap_frame_id_list = list(range(T))
    
    # 2. 转换手部参数为 SMPLX 格式
    smplx_params = convert_mano_to_smplx_params(
        ours["rot_r"],
        ours["pose_r"],
        ours["trans_r"],
        ours["shape_r"],
    )
    
    # 3. 构建 raw_smplx 字典（按帧ID索引）
    raw_smplx = {}
    for frame_id in mocap_frame_id_list:
        raw_smplx[frame_id] = {
            "betas": smplx_params["betas"][frame_id],
            "global_orient": smplx_params["global_orient"][frame_id],
            "body_pose": smplx_params["body_pose"][frame_id],
            "left_hand_pose": smplx_params["left_hand_pose"][frame_id],
            "right_hand_pose": smplx_params["right_hand_pose"][frame_id],
            "transl": smplx_params["transl"][frame_id],
        }
    
    # 4. 转换物体变换
    obj_transf_4x4 = convert_obj_rot_trans_to_transf(
        ours["obj_rot"],
        ours["obj_trans"],
    )
    
    # 5. 构建 obj_transf 字典（按物体ID和帧ID索引）
    obj_transf = {
        obj_id: {}
    }
    for frame_id in mocap_frame_id_list:
        obj_transf[obj_id][frame_id] = obj_transf_4x4[frame_id]
    
    # 6. 物体列表
    obj_list = [obj_id]
    
    # 7. 构建完整的 anno 字典
    anno = {
        "mocap_frame_id_list": mocap_frame_id_list,
        "raw_smplx": raw_smplx,
        "obj_transf": obj_transf,
        "obj_list": obj_list,
    }
    
    return anno


def create_program_info(
    total_frames: int,
    output_path: str,
    obj_id: str = "object_0",
):
    """
    创建 program_info JSON 文件。
    
    Args:
        total_frames: 总帧数
        output_path: 输出JSON文件路径
        obj_id: 物体ID
    """
    # 创建简单的 program_info，包含一个阶段，覆盖所有帧
    # 格式: {(start_frame, end_frame): {"obj_list_rh": [obj_id], "obj_list_lh": None}}
    program_info = {
        str((0, total_frames - 1)): {
            "obj_list_rh": [obj_id],
            "obj_list_lh": None,
        }
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(program_info, f, indent=2)
    
    print(f"Program info 已保存到: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ours data.npy to oakink2 format (pickle + program_info)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/home/pt/fbs/ManipTrans/data/ours/data.npy",
        help="输入 ours 的 data.npy 路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/pt/fbs/ManipTrans/data/OakInk-v2/anno_preview/ours_seq.pkl",
        help="输出 pickle 文件路径",
    )
    parser.add_argument(
        "--obj_id",
        type=str,
        default="object_0",
        help="物体ID（用于 oakink2 格式）",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=120,
        help="帧率（默认 120Hz，oakink2 标准）",
    )
    parser.add_argument(
        "--create_program_info",
        action="store_true",
        help="是否创建 program_info JSON 文件",
    )
    parser.add_argument(
        "--program_info_output",
        type=str,
        default=None,
        help="program_info JSON 文件输出路径（如果未指定，则根据 output 路径自动生成）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    assert os.path.isfile(args.input), f"input npy 不存在: {args.input}"

    print(f"正在加载数据: {args.input}")
    ours = load_ours_data(args.input)
    
    print(f"数据帧数: {ours['rot_r'].shape[0]}")
    print(f"正在转换为 oakink2 格式...")
    
    # 构建 oakink2 anno
    anno = build_oakink2_anno(ours, obj_id=args.obj_id, fps=args.fps)
    
    # 保存 pickle 文件
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(anno, f)
    print(f"Pickle 文件已保存: {args.output}")
    
    # 创建 program_info（如果需要）
    if args.create_program_info:
        if args.program_info_output is None:
            # 根据 output 路径自动生成 program_info 路径
            base_name = os.path.splitext(os.path.basename(args.output))[0]
            program_info_dir = os.path.join(
                os.path.dirname(os.path.dirname(args.output)),
                "program",
                "program_info"
            )
            program_info_output = os.path.join(program_info_dir, f"{base_name}.json")
        else:
            program_info_output = args.program_info_output
        
        create_program_info(
            total_frames=ours["rot_r"].shape[0],
            output_path=program_info_output,
            obj_id=args.obj_id,
        )
    
    print("\n转换完成！")
    print(f"\n使用方法：")
    print(f"1. 将生成的 pickle 文件放在: {os.path.dirname(args.output)}")
    if args.create_program_info:
        print(f"2. Program info 文件在: {program_info_output}")
    print(f"3. 使用 data_idx 格式: <hash>@0")
    print(f"   其中 <hash> 是 pickle 文件名的前5位字符")
    print(f"   例如，如果文件是 'ours_seq_abcde12345.pkl'，则使用 'abcde@0'")


if __name__ == "__main__":
    main()
