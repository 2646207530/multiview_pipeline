#python utils/ours_to_sv_dict.py --vis_frames 5
import os
import argparse

import numpy as np
import torch
import smplx
import trimesh
from scipy.spatial.transform import Rotation as R


def load_ours_data(npy_path: str):
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


def build_mano_verts(
    rot_r: np.ndarray,
    pose_r: np.ndarray,
    trans_r: np.ndarray,
    shape_r: np.ndarray,
    mano_model_dir: str,
):
    """
    使用 MANO 右手模型，将我们的数据转换为每帧的手部顶点 (T, 778, 3)。
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mano_layer = smplx.create(
        mano_model_dir,
        "MANO",
        use_pca=False,
        is_rhand=True,
        flat_hand_mean=True,
    ).to(device)

    rot_r_t = torch.tensor(rot_r, dtype=torch.float32, device=device)
    pose_r_t = torch.tensor(pose_r, dtype=torch.float32, device=device)
    trans_r_t = torch.tensor(trans_r, dtype=torch.float32, device=device)
    shape_r_t = torch.tensor(shape_r, dtype=torch.float32, device=device)

    with torch.no_grad():
        out = mano_layer(
            global_orient=rot_r_t,
            hand_pose=pose_r_t,
            betas=shape_r_t,
            transl=trans_r_t,
        )
        verts = out.vertices  # (T, 778, 3)

    return verts.cpu().numpy().astype(np.float32)


def load_object_mesh(obj_path: str):
    """
    读取物体 obj，并返回：
      - obj_verts_centered: (V, 3) 以网格质心为中心的顶点
      - obj_faces: (F, 3) faces
      - obj_vertex_normals: (V, 3) 顶点法线
      - pc: (N, 3) 物体表面点云（同样以质心为中心）
      - pc_normals: (N, 3) 点云法线
    """
    mesh = trimesh.load(obj_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        # 取第一个 geometry 作为主体
        mesh = mesh.dump()[0]

    verts = mesh.vertices.astype(np.float64)
    faces = mesh.faces.astype(np.int64)

    # 顶点法线（trimesh 会自动计算）
    v_normals = mesh.vertex_normals.astype(np.float64)

    # 以质心为中心，便于数值范围与 grab_demo 接近
    centroid = verts.mean(axis=0, keepdims=True)
    verts_centered = verts - centroid

    # 采样表面点云
    num_points = 8000
    pc, face_idx = trimesh.sample.sample_surface(mesh, num_points)
    pc_centered = pc - centroid
    f_normals = mesh.face_normals[face_idx]

    return verts_centered, faces, v_normals, pc_centered.astype(
        np.float32
    ), f_normals.astype(np.float32)


def build_sv_dict(
    ours: dict,
    mano_model_dir: str,
    obj_path: str,
):
    rot_r = ours["rot_r"]
    pose_r = ours["pose_r"]
    trans_r = ours["trans_r"]
    shape_r = ours["shape_r"]  # (T, 10)
    obj_rot = ours["obj_rot"]
    obj_trans = ours["obj_trans"]

    T = rot_r.shape[0]

    # 统一使用数据平均的 beta（MANO 10 维 shape）
    beta_unified = np.mean(shape_r, axis=0).astype(np.float32)  # (10,)
    shape_r_unified = np.broadcast_to(beta_unified, (T, 10))

    # 手部顶点（每帧使用同一套 beta）
    rhand_verts = build_mano_verts(
        rot_r, pose_r, trans_r, shape_r_unified, mano_model_dir
    )

    # 物体网格与点云
    obj_verts, obj_faces, obj_vnormals, obj_pc, obj_pc_normals = load_object_mesh(
        obj_path
    )

    # 点云 & 法线在时间上复制（每一帧使用同一个物体点云）
    object_pc = np.repeat(obj_pc[None, ...], T, axis=0)
    object_normal = np.repeat(obj_pc_normals[None, ...], T, axis=0)

    sv_dict = {
        "rhand_global_orient_gt": rot_r.astype(np.float32),
        "rhand_transl": trans_r.astype(np.float32),
        "rhand_verts": rhand_verts.astype(np.float32),
        "object_pc": object_pc.astype(np.float32),
        "object_normal": object_normal.astype(np.float32),
        "object_global_orient": obj_rot.astype(np.float32),
        "object_transl": obj_trans.astype(np.float32),
        "obj_faces": obj_faces.astype(np.int64),
        "obj_verts": obj_verts.astype(np.float64),
        "obj_vertex_normals": obj_vnormals.astype(np.float64),
    }

    return sv_dict, beta_unified


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ours data.npy to sv_dict format similar to grab_demo."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/home/pt/fbs/ManipTrans/data/ours/data_fixed_crop.npy",
        help="输入 ours 的 data.npy 路径",
    )
    parser.add_argument(
        "--obj",
        type=str,
        default="/home/pt/fbs/ManipTrans/data/ours/111.obj",
        help="物体 obj 模型路径（如 final.obj）",
    )
    parser.add_argument(
        "--mano_dir",
        type=str,
        default="/home/pt/fbs",
        help="MANO 模型根目录（包含 MANO_RIGHT.pkl 等）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/pt/fbs/ManipTrans/data/ours/ours_sv_dict.npy",
        help="输出 sv_dict npy 路径",
    )
    parser.add_argument(
        "--beta_output",
        type=str,
        default=None,
        help="输出统一 beta (MANO 10 维 shape) 的 npy 路径；默认与 output 同目录下的 ours_beta.npy",
    )
    parser.add_argument(
        "--vis_frame",
        type=int,
        default=-1,
        help="如果 >=0，则从输出 npy 中可视化该帧并导出为 .ply 场景文件",
    )
    parser.add_argument(
        "--vis_output_dir",
        type=str,
        default="/home/pt/fbs/test/debug_meshes_sv",
        help="可视化导出的 .ply 保存目录",
    )
    parser.add_argument(
        "--vis_frames",
        type=int,
        default=0,
        help="转换后直接保存几帧的点云文件（PLY格式），0 表示不保存",
    )
    parser.add_argument(
        "--vis_frame_indices",
        type=str,
        default=None,
        help="指定要保存的帧索引，用逗号分隔，例如 '0,10,20'。如果提供则忽略 --vis_frames",
    )
    parser.add_argument(
        "--vis_frames_output_dir",
        type=str,
        default="/home/pt/fbs/test/debug_frames_vis",
        help="转换后可视化帧的点云文件保存目录（用于 --vis_frames）",
    )
    return parser.parse_args()


def visualize_frames_interactive(
    sv_dict: dict,
    mano_model_dir: str,
    num_frames: int = 5,
    frame_indices: list = None,
    out_dir: str = None,
):
    """
    保存转换后的几帧数据为点云文件（PLY格式），方便后续查看。
    
    Args:
        sv_dict: 转换后的 sv_dict 字典
        mano_model_dir: MANO 模型目录
        num_frames: 要可视化的帧数（如果 frame_indices 为 None，则取前 num_frames 帧）
        frame_indices: 指定要可视化的帧索引列表，如果提供则忽略 num_frames
        out_dir: 输出目录，如果为 None 则使用默认目录
    """
    rhand_verts = sv_dict["rhand_verts"]  # (T, 778, 3)
    object_pc = sv_dict["object_pc"]  # (T, N, 3)
    object_global_orient = sv_dict["object_global_orient"]  # (T, 3)
    object_transl = sv_dict["object_transl"]  # (T, 3)
    obj_faces = sv_dict["obj_faces"]  # (F, 3)
    obj_verts = sv_dict["obj_verts"]  # (V, 3)
    
    T = rhand_verts.shape[0]
    
    # 确定要可视化的帧索引
    if frame_indices is None:
        frame_indices = list(range(min(num_frames, T)))
    else:
        frame_indices = [idx for idx in frame_indices if 0 <= idx < T]
    
    if not frame_indices:
        print("没有有效的帧索引可可视化")
        return
    
    # 获取 MANO faces
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mano_layer = smplx.create(
        mano_model_dir,
        "MANO",
        use_pca=False,
        is_rhand=True,
        flat_hand_mean=True,
    ).to(device)
    # 修复：mano_layer.faces 可能是 numpy 数组或 tensor
    mano_faces = mano_layer.faces
    if isinstance(mano_faces, torch.Tensor):
        mano_faces = mano_faces.cpu().numpy()
    else:
        mano_faces = np.array(mano_faces)
    
    if out_dir is None:
        out_dir = "/home/pt/fbs/test/debug_frames_vis"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"保存可视化帧到: {out_dir}")
    print(f"帧索引: {frame_indices}")
    
    for frame_idx in frame_indices:
        scene_meshes = []
        
        # 1. 手部网格
        hand_v = rhand_verts[frame_idx].astype(np.float64)
        hand_mesh = trimesh.Trimesh(hand_v, mano_faces, process=False)
        hand_mesh.visual.face_colors = [200, 160, 140, 255]  # 珊瑚色
        scene_meshes.append(hand_mesh)
        
        # 2. 物体网格（应用该帧的姿态）
        rot_vec = object_global_orient[frame_idx]
        transl = object_transl[frame_idx]
        R_obj = R.from_rotvec(rot_vec).as_matrix()
        obj_v_transformed = (R_obj @ obj_verts.T).T + transl
        
        obj_mesh = trimesh.Trimesh(
            obj_v_transformed.astype(np.float64), 
            obj_faces.astype(np.int64), 
            process=False
        )
        obj_mesh.visual.face_colors = [50, 200, 50, 255]  # 绿色
        scene_meshes.append(obj_mesh)
        
        # 3. 物体点云（用小球表示，采样一部分以控制大小）
        obj_pc = object_pc[frame_idx].astype(np.float64)
        obj_pc_transformed = (R_obj @ obj_pc.T).T + transl
        step = max(1, obj_pc_transformed.shape[0] // 1000)  # 最多取约 1000 个点
        pc_sampled = obj_pc_transformed[::step]
        for p in pc_sampled:
            s = trimesh.creation.icosphere(subdivisions=1, radius=0.004)
            s.apply_translation(p)
            s.visual.face_colors = [0, 150, 255, 200]  # 蓝色
            scene_meshes.append(s)
        
        # 合并场景并保存
        scene = trimesh.util.concatenate(scene_meshes)
        out_path = os.path.join(out_dir, f"frame_{frame_idx:04d}_scene.ply")
        scene.export(out_path)
        print(f"  帧 {frame_idx} 已保存: {out_path}")
    
    print(f"所有可视化文件已保存到: {out_dir}")


def visualize_sv_frame(npy_path: str, mano_model_dir: str, frame_idx: int, out_dir: str):
    """
    从 sv_dict npy 中取出某一帧，导出一个包含：
      - MANO 右手网格
      - 物体网格（应用该帧的 object_global_orient + object_transl）
      - 物体点云（object_pc）的 3D 场景到 .ply 文件，方便用 Meshlab/CloudCompare 打开查看。
    """
    assert os.path.isfile(npy_path), f"sv_dict npy 不存在: {npy_path}"
    arr = np.load(npy_path, allow_pickle=True)
    if arr.shape == () and arr.dtype == object:
        d = arr.item()
    else:
        d = arr

    required_keys = [
        "rhand_verts",
        "object_pc",
        "object_global_orient",
        "object_transl",
        "obj_faces",
        "obj_verts",
    ]
    for k in required_keys:
        if k not in d:
            raise KeyError(f"sv_dict 缺少 key: {k}")

    rhand_verts = d["rhand_verts"]  # (T, 778, 3)
    object_pc = d["object_pc"]  # (T, 8000, 3)
    object_global_orient = d["object_global_orient"]  # (T, 3)
    object_transl = d["object_transl"]  # (T, 3)
    obj_faces = d["obj_faces"]  # (F, 3)
    obj_verts = d["obj_verts"]  # (V, 3)

    T = rhand_verts.shape[0]
    if frame_idx < 0 or frame_idx >= T:
        raise ValueError(f"vis_frame 超出范围: 0 <= idx < {T}, 当前为 {frame_idx}")

    os.makedirs(out_dir, exist_ok=True)

    # MANO faces 用于构建手部网格
    mano_layer = smplx.create(
        mano_model_dir,
        "MANO",
        use_pca=False,
        is_rhand=True,
        flat_hand_mean=True,
    )
    mano_faces = mano_layer.faces

    scene_meshes = []

    # 1) 手部网格
    hand_v = rhand_verts[frame_idx].astype(np.float64)
    hand_mesh = trimesh.Trimesh(hand_v, mano_faces, process=False)
    hand_mesh.visual.face_colors = [200, 160, 140, 255]
    scene_meshes.append(hand_mesh)

    # 2) 物体网格（应用该帧的姿态）
    obj_mesh = trimesh.Trimesh(obj_verts.astype(np.float64), obj_faces.astype(np.int64), process=False)
    rot_vec = object_global_orient[frame_idx]
    transl = object_transl[frame_idx]
    R_obj = R.from_rotvec(rot_vec).as_matrix()
    T_mat = np.eye(4, dtype=np.float64)
    T_mat[:3, :3] = R_obj
    T_mat[:3, 3] = transl
    obj_mesh.apply_transform(T_mat)
    obj_mesh.visual.face_colors = [50, 200, 50, 255]
    scene_meshes.append(obj_mesh)

    # 3) 物体点云（用小球表示，抽样一部分以控制大小）
    pc = object_pc[frame_idx].astype(np.float64)
    step = max(1, pc.shape[0] // 1000)  # 最多取约 1000 个点
    pc_sampled = pc[::step]
    for p in pc_sampled:
        s = trimesh.creation.icosphere(subdivisions=1, radius=0.004)
        s.apply_translation(p)
        s.visual.face_colors = [0, 150, 255, 200]
        scene_meshes.append(s)

    scene = trimesh.util.concatenate(scene_meshes)
    out_path = os.path.join(out_dir, f"frame_{frame_idx:04d}_scene_sv.ply")
    scene.export(out_path)
    print(f"可视化帧 {frame_idx} 已导出到: {out_path}")


def main():
    args = parse_args()

    assert os.path.isfile(args.input), f"input npy 不存在: {args.input}"
    assert os.path.isfile(args.obj), f"obj 模型不存在: {args.obj}"

    ours = load_ours_data(args.input)
    sv_dict, beta_unified = build_sv_dict(
        ours, mano_model_dir=args.mano_dir, obj_path=args.obj
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, sv_dict)
    print(f"保存完成: {args.output}")

    # 统一 beta：MANO 10 维 shape，形状 (10,)
    beta_out_path = args.beta_output
    if beta_out_path is None:
        out_dir = os.path.dirname(args.output)
        beta_out_path = os.path.join(out_dir, "ours_beta.npy")
    os.makedirs(os.path.dirname(beta_out_path) or ".", exist_ok=True)
    np.save(beta_out_path, beta_unified)
    print(f"统一 beta (数据平均): shape={beta_unified.shape}")
    print(f"  beta = {beta_unified}")
    print(f"已保存: {beta_out_path}")

    # 转换后直接可视化几帧
    if args.vis_frames > 0 or args.vis_frame_indices is not None:
        frame_indices = None
        if args.vis_frame_indices is not None:
            try:
                frame_indices = [int(x.strip()) for x in args.vis_frame_indices.split(',')]
            except ValueError:
                print(f"警告: --vis_frame_indices 格式错误，使用 --vis_frames 参数")
                frame_indices = None
        
        num_frames = args.vis_frames if frame_indices is None else 0
        visualize_frames_interactive(
            sv_dict, 
            args.mano_dir, 
            num_frames=num_frames,
            frame_indices=frame_indices,
            out_dir=args.vis_frames_output_dir
        )

    if args.vis_frame >= 0:
        visualize_sv_frame(args.output, args.mano_dir, args.vis_frame, args.vis_output_dir)


if __name__ == "__main__":
    main()

