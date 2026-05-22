"""
way_vis: 从 npy 轨迹数据渲染手+物体视频与抽帧图。

渲染相机和 2D 投影统一使用 npy 中的真实相机内参 K（不再支持虚拟内参兜底，
若 npy 未提供 K 会直接报错）。

用例:
  # 默认：手+物体，输出视频与抽帧图（路径见文件内 NPY_PATH / OUTPUT_VIDEO / OUTPUT_FRAME_DIR）
  python utils/way_vis.py

  # 自动用 npy 自带的 imgpath + imgnames 作为背景图（推荐，无需手动指定路径）
  python utils/way_vis.py --overlay

  # 显式指定背景图目录（优先级高于 --overlay；按帧号匹配，匹配不到按排序顺序兜底）
  python utils/way_vis.py --overlay-dir /home/pt/fbs/dataset/mrc-net-6d-pose/demo_video/DA5298464/images/Video_0

  # 只渲染手
  python utils/way_vis.py --show hand

  # 只渲染物体
  python utils/way_vis.py --show object

  # 同时导出每帧 3D 场景 mesh 到 debug_viz_output
  python utils/way_vis.py --save-scene-mesh

  # 仅物体 + 导出 3D 场景
  python utils/way_vis.py --show object --save-scene-mesh

  # 只可视化指定帧区间（左闭右开，0-based），如 100 到 200 帧
  python utils/way_vis.py --frame-range 100 200

  # 帧区间 + overlay 组合使用
  python utils/way_vis.py --frame-range 100 200 --overlay-dir /path/to/overlay_frames

  # 在画面上叠加 21 个 MANO 手部 joint 与骨架连线（右手黄/左手青）
  # 注意：开启该参数后会跳过 3D 手 mesh 渲染，仅保留物体 mesh + 2D 骨架（更便于核对 2D 对齐）；
  #       --save-scene-mesh 导出的 3D 场景文件不受影响，仍包含手 mesh
  python utils/way_vis.py --draw-hand-joints

  # joint 叠加 + 自动 overlay 背景，便于核对 2D 对齐
  python utils/way_vis.py --draw-hand-joints --overlay
"""
import os
# 推荐使用 EGL 进行离屏渲染 (如果没有 GPU 或者报错，可以注释掉这行，使用 xvfb-run 运行脚本)
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import subprocess
import shutil
import sys
import re
import numpy as np
import cv2
import torch
import smplx
import trimesh
import pyrender
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# --

# 兼容由不同 numpy 版本保存的 pickle 路径差异（numpy._core vs numpy.core）
try:
    import numpy.core as _np_core
    sys.modules.setdefault("numpy._core", _np_core)
    if hasattr(_np_core, "multiarray"):
        sys.modules.setdefault("numpy._core.multiarray", _np_core.multiarray)
except Exception:
    pass

# ================= 配置路径 =================
# NPY_PATH = '/data2/fubingshuai/golf/golf-hand-object/out/35_wood_8_01_fbs_post_concat.npy'          # 你提供的npy文件路径
# NPY_PATH = '/data2/fubingshuai/golf/golf-hand-object/out/35_wood_8_01_fbs_opt_contact.npy'          # 你提供的npy文件路径
# NPY_PATH = '/data2/fubingshuai/golf/golf-hand-object/out/35_wood_8_01_fbs_opt_force_closure.npy'
NPY_PATH = '/data2/fubingshuai/golf/golf-hand-object/out/35_wood_8_01_fbs.npy'
# NPY_PATH = '/data2/fubingshuai/golf/standard_pose/35_wood_8_01_baseball.npy'
# NPY_PATH = '/data2/fubingshuai/golf/test/npy_debug/fc_iter001000.npy'
OBJ_PATH = '/data2/fubingshuai/golf/data/club-assets/35_wood_8_01/20260421142122_meter.stl'        # 物体的 .obj 模型路径
# OBJ_PATH = '/home/pt/fbs/dataset/golf_asset/golf_simple.obj'
MANO_MODEL_DIR = '/data2/fubingshuai/golf/golf-hand-object'     # MANO模型的根目录 (包含 MANO_RIGHT.pkl 等)
OUTPUT_VIDEO = '/data2/fubingshuai/golf/test/vis/trajectory_output.mp4'      # 输出视频路径
OUTPUT_FRAME_DIR = '/data2/fubingshuai/golf/test/vis/way_vis_frames'         # 每 SAVE_FRAME_INTERVAL 帧保存一张图，便于看中间结果
SAVE_FRAME_INTERVAL = 1                               # 每 N 帧保存一帧图片（0 表示不保存）
SCENE_MESH_OUTPUT_DIR = '/data2/fubingshuai/golf/test/vis/debug_viz_output'

FPS = 30               # 输出视频的帧率
# 默认一次渲染 npy 中所有 views；输出会带 _view{idx} 后缀
# 自动处理相机外参平移单位：检测到 mm 量级时自动换算到 m
AUTO_EXTRINSIC_UNIT_FIX = True
# 自动判别外参方向（解决不同标定文件里旋转/平移方向约定不一致）
AUTO_EXTRINSIC_DIRECTION_FIX = False
# 固定使用逆方向外参变换（当你明确知道当前方向相反时打开）
# run_dexycb_clips.py 生成的 npy 里 world2cam 已是标准 world→cam，无需再取逆
FORCE_INVERT_EXTRINSIC = False
# 自动处理物体单位：检测到 mm 量级时自动换算到 m，避免与手部（通常 m）尺度不一致
AUTO_OBJ_UNIT_FIX = True
# ===========================================

# npy 中左手槽位存的是右手 MANO 参数，用左手 MANO 加载时轴角沿 Y、Z 取反
MIRROR_LEFT_PARAMS = np.array([1.0, -1.0, -1.0], dtype=np.float32)

# smplx MANO 输出的 21 个 joint 的骨架连线
# LBS 16 关节: 0=wrist, 1-3=index, 4-6=middle, 7-9=pinky, 10-12=ring, 13-15=thumb
# 指尖按 VERTEX_IDS['mano'] 字典顺序追加: 16=thumb, 17=index, 18=middle, 19=ring, 20=pinky
MANO_JOINT_BONES = [
    (0, 13), (13, 14), (14, 15), (15, 16),  # thumb
    (0, 1),  (1, 2),   (2, 3),   (3, 17),   # index
    (0, 4),  (4, 5),   (5, 6),   (6, 18),   # middle
    (0, 10), (10, 11), (11, 12), (12, 19),  # ring
    (0, 7),  (7, 8),   (8, 9),   (9, 20),   # pinky
]
# MANO 指尖在 778 个顶点上的索引（来自 smplx VERTEX_IDS['mano']，顺序 thumb/index/middle/ring/pinky）
# 用于在 smplx 没有自动追加指尖时（例如某些环境里 vertex_joint_selector 被注释掉）从 vertices 取出补齐
MANO_TIP_VERTEX_IDS = [744, 320, 443, 554, 671]


def _append_fingertips_if_needed(joints_arr: np.ndarray, verts_arr: np.ndarray) -> np.ndarray:
    """若 smplx 返回的 joints 不足 21 个（缺指尖），从 verts 按 MANO_TIP_VERTEX_IDS 取出补齐。"""
    if joints_arr is None or joints_arr.shape[1] >= 21:
        return joints_arr
    tips = verts_arr[:, MANO_TIP_VERTEX_IDS, :]
    return np.concatenate([joints_arr, tips], axis=1)


def _draw_hand_skeleton(frame, joints_cam, K_proj, w, h,
                        joint_color=(0, 255, 255), bone_color=(0, 200, 200), radius=4, thickness=2):
    """将一只手的 21 个 joint 投影并绘制点 + 骨架连线。joints_cam 已是目标相机坐标系。"""
    if joints_cam is None or joints_cam.shape[0] < 21:
        return
    pts2d = []
    for j in range(21):
        p = joints_cam[j]
        if not np.isfinite(p).all() or p[2] <= 1e-6:
            pts2d.append(None)
            continue
        u = K_proj[0, 0] * p[0] / p[2] + K_proj[0, 2]
        v = K_proj[1, 1] * p[1] / p[2] + K_proj[1, 2]
        pts2d.append((int(round(u)), int(round(v))))

    for a, b in MANO_JOINT_BONES:
        pa, pb = pts2d[a], pts2d[b]
        if pa is None or pb is None:
            continue
        cv2.line(frame, pa, pb, bone_color, thickness, lineType=cv2.LINE_AA)

    for p in pts2d:
        if p is None:
            continue
        if 0 <= p[0] < w and 0 <= p[1] < h:
            cv2.circle(frame, p, radius, joint_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, p, radius, (255, 255, 255), 1, lineType=cv2.LINE_AA)


def right_to_left_mano_params(rot: np.ndarray, pose: np.ndarray):
    """将右手 MANO 参数转为左手 MANO 输入：轴角 (x,y,z) 取 (x, -y, -z)，平移不改。"""
    rot_l = rot * MIRROR_LEFT_PARAMS
    pose_l = (pose.reshape(-1, 3) * MIRROR_LEFT_PARAMS).reshape(pose.shape)
    return rot_l, pose_l


def _resolve_camera_bundle(camera_dict: dict, target_view: str):
    """
    兼容新旧 npy 相机字段：
      - 新格式: K(list), world2cam(list), views(list)
      - 旧格式: k_use(3x3)
    返回:
      K_target(3x3), T_target_from_world(4x4), resolved_target
    """
    if not isinstance(camera_dict, dict):
        # 无相机信息时兜底
        return None, np.eye(4, dtype=np.float64), target_view

    K_field = camera_dict.get("K", None)
    W2C_field = camera_dict.get("world2cam", None)
    views = camera_dict.get("views", None)

    # 旧格式: 仅 k_use
    if (not isinstance(K_field, list) or len(K_field) == 0) and "k_use" in camera_dict:
        k_use = np.asarray(camera_dict["k_use"], dtype=np.float64)
        return k_use, np.eye(4, dtype=np.float64), target_view

    # 新格式缺失时兜底
    if not isinstance(K_field, list) or len(K_field) == 0:
        return None, np.eye(4, dtype=np.float64), target_view

    n_cam = len(K_field)
    if not isinstance(views, list) or len(views) != n_cam:
        views = [f"cam{i}" for i in range(n_cam)]

    def _find_idx(view_name: str, default_idx: int):
        if view_name in views:
            return views.index(view_name), view_name
        return default_idx, views[default_idx]

    target_idx, resolved_target = _find_idx(target_view, 0)

    K_target = np.asarray(K_field[target_idx], dtype=np.float64)

    # 没有外参时默认同一坐标系
    if not isinstance(W2C_field, list) or len(W2C_field) != n_cam:
        return K_target, np.eye(4, dtype=np.float64), resolved_target

    W2C_all = [np.asarray(x, dtype=np.float64).copy() for x in W2C_field]
    if AUTO_EXTRINSIC_UNIT_FIX:
        t_norms = [float(np.linalg.norm(T[:3, 3])) for T in W2C_all]
        # 经验规则：外参平移 > 20 通常是 mm（例如 1257 mm），与手/物体(m)混用会导致投影全空
        if max(t_norms) > 20.0:
            for T in W2C_all:
                T[:3, 3] /= 1000.0
            print(f"[外参单位修正] 检测到 mm 量级平移，已将 world2cam.t 从 mm -> m: {t_norms}")

    T_target_from_world = W2C_all[target_idx]
    return K_target, T_target_from_world, resolved_target


def _resolve_all_cameras(camera_dict: dict):
    """
    解析 npy 中所有视角，返回 [{K, W2C, name}, ...]。
      - 新格式: K(list), world2cam(list), views(list)
      - 旧格式 k_use: 返回单视角 (W2C=I)
    """
    out = []
    if not isinstance(camera_dict, dict):
        return out

    K_field = camera_dict.get("K", None)
    W2C_field = camera_dict.get("world2cam", None)
    views = camera_dict.get("views", None)

    if (not isinstance(K_field, list) or len(K_field) == 0):
        if "k_use" in camera_dict:
            out.append({
                "K": np.asarray(camera_dict["k_use"], dtype=np.float64),
                "W2C": np.eye(4, dtype=np.float64),
                "name": "cam0",
            })
        return out

    n_cam = len(K_field)
    if not isinstance(views, list) or len(views) != n_cam:
        views = [f"cam{i}" for i in range(n_cam)]

    W2C_all = None
    if isinstance(W2C_field, list) and len(W2C_field) == n_cam:
        W2C_all = [np.asarray(x, dtype=np.float64).copy() for x in W2C_field]
        if AUTO_EXTRINSIC_UNIT_FIX:
            t_norms = [float(np.linalg.norm(T[:3, 3])) for T in W2C_all]
            if max(t_norms) > 20.0:
                for T in W2C_all:
                    T[:3, 3] /= 1000.0
                print(f"[外参单位修正] 检测到 mm 量级平移，world2cam.t mm -> m: {t_norms}")

    for i in range(n_cam):
        K_i = np.asarray(K_field[i], dtype=np.float64)
        W2C_i = W2C_all[i] if W2C_all is not None else np.eye(4, dtype=np.float64)
        if FORCE_INVERT_EXTRINSIC:
            W2C_i = np.linalg.inv(W2C_i)
        out.append({"K": K_i, "W2C": W2C_i, "name": views[i]})
    return out


def _build_view_overlay_sources(data: dict, seq_key: str, view_idx: int,
                                 user_overlay_dir: str, overlay_auto: bool):
    """
    为指定 view 构造 overlay 背景源：
      - 显式 --overlay-dir: 所有视角都使用该目录
      - --overlay (自动): view0 用 npy 的 imgpath + imgnames；
                          view1+ 从相邻序列 (e.g. 12-1 → 12-2) 推导
      - 否则: 返回 None (纯渲染背景)
    """
    if user_overlay_dir is not None:
        return _prepare_overlay_sources(user_overlay_dir)
    if not overlay_auto:
        return None

    if view_idx == 0:
        return _overlay_sources_from_npy(data)

    base_imgpath = data.get("imgpath", None)
    imgnames = data.get("imgnames", None) or []
    if not base_imgpath or "-" not in seq_key:
        print(f"[overlay] view{view_idx}: 无法从 seq_key={seq_key!r} 推导相邻序列，使用纯渲染背景")
        return None
    prefix, cam_str = seq_key.rsplit("-", 1)
    try:
        cam_id = int(cam_str)
    except ValueError:
        print(f"[overlay] view{view_idx}: seq_key cam 非整数 ({cam_str})，使用纯渲染背景")
        return None
    # 简单约定: cam_id 1 ↔ 2 (dexycb_clips 双相机)
    other_cam = 2 if cam_id == 1 else 1
    other_seq = f"{prefix}-{other_cam}"
    other_dir = os.path.join(os.path.dirname(base_imgpath), other_seq)
    if not os.path.isdir(other_dir):
        print(f"[overlay] view{view_idx}: 相邻序列目录不存在 {other_dir}，使用纯渲染背景")
        return None

    # 优先复用 imgnames (dexycb 两相机同步，文件名一致)
    ordered = [os.path.join(other_dir, n) for n in imgnames
               if os.path.isfile(os.path.join(other_dir, n))]
    if len(ordered) != len(imgnames) or not ordered:
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ordered = sorted(
            os.path.join(other_dir, f) for f in os.listdir(other_dir)
            if os.path.splitext(f)[1].lower() in valid_ext
        )
    by_frame = {i: p for i, p in enumerate(ordered)}
    for p in ordered:
        stem = os.path.splitext(os.path.basename(p))[0]
        m = re.search(r"(\d+)$", stem)
        if m:
            by_frame.setdefault(int(m.group(1)), p)
    print(f"[overlay] view{view_idx}: 使用相邻序列 {other_dir}，共 {len(ordered)} 张")
    return {"ordered": ordered, "by_frame": by_frame}


def _transform_points(T_dst_from_src: np.ndarray, pts: np.ndarray):
    """将点从 src 相机坐标系变换到 dst 相机坐标系。"""
    Rm = T_dst_from_src[:3, :3]
    tv = T_dst_from_src[:3, 3]
    return np.einsum("ij,...j->...i", Rm, pts) + tv


def _score_transform_visibility(T_dst_from_src, obj_rot, obj_trans, K, w, h, max_frames=80):
    """
    评估给定相机变换在目标相机中的可见性分数（越大越好）。
    使用物体局部小十字点作为 probe，统计投影落入画面比例。
    """
    if K is None or obj_rot is None or obj_trans is None:
        return -1.0
    n = min(len(obj_rot), len(obj_trans), max_frames)
    if n <= 0:
        return -1.0

    probe = np.array([
        [0.0, 0.0, 0.0],
        [0.08, 0.0, 0.0],
        [0.0, 0.08, 0.0],
        [0.0, 0.0, 0.08],
    ], dtype=np.float64)

    scores = []
    for i in range(n):
        Rm = R.from_rotvec(obj_rot[i]).as_matrix()
        pts_src = (Rm @ probe.T).T + obj_trans[i]
        pts_dst = _transform_points(T_dst_from_src, pts_src)
        z = pts_dst[:, 2]
        valid = z > 1e-6
        if not np.any(valid):
            scores.append(0.0)
            continue
        u = K[0, 0] * pts_dst[valid, 0] / z[valid] + K[0, 2]
        v = K[1, 1] * pts_dst[valid, 1] / z[valid] + K[1, 2]
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        scores.append(float(np.mean(inside)))
    return float(np.mean(scores))


def _prepare_overlay_sources(overlay_dir: str):
    """
    扫描 overlay 目录中的图片并构建两种索引：
      1) 按文件名末尾数字匹配帧号（优先）
      2) 按文件名排序后按顺序匹配（兜底）
    """
    if overlay_dir is None:
        return None
    if not os.path.isdir(overlay_dir):
        raise FileNotFoundError(f"overlay 目录不存在: {overlay_dir}")

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    entries = []
    for name in sorted(os.listdir(overlay_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext in valid_ext:
            entries.append(os.path.join(overlay_dir, name))

    if not entries:
        raise RuntimeError(f"overlay 目录内没有可用图片: {overlay_dir}")

    index_by_frame = {}
    for p in entries:
        stem = os.path.splitext(os.path.basename(p))[0]
        m = re.search(r"(\d+)$", stem)
        if m:
            frame_id = int(m.group(1))
            # 保留第一次出现的同帧号文件，避免覆盖造成不可预期
            index_by_frame.setdefault(frame_id, p)

    print(
        f"[overlay] 已加载 {len(entries)} 张背景图；可按帧号精确匹配 {len(index_by_frame)} 张"
    )
    return {"ordered": entries, "by_frame": index_by_frame}


def _overlay_sources_from_npy(data: dict):
    """从 npy 自带的 imgpath + imgnames 字段构建 overlay_sources。"""
    base = data.get("imgpath", None)
    names = data.get("imgnames", None)
    if not base or not names:
        raise RuntimeError(
            "--overlay 需要 npy 中提供 'imgpath'(目录) 与 'imgnames'(文件名列表)，但未找到"
        )
    if not os.path.isdir(base):
        raise FileNotFoundError(f"npy 中的 imgpath 不存在: {base}")

    ordered = [os.path.join(base, n) for n in names]
    by_frame = {i: p for i, p in enumerate(ordered)}
    print(f"[overlay] 自 npy 自动加载 {len(ordered)} 张背景图，base={base}")
    return {"ordered": ordered, "by_frame": by_frame}


def _get_overlay_path(overlay_sources, global_i: int, local_i: int):
    """优先按 global 帧号匹配，其次按 local 帧号，最后按顺序兜底。"""
    if overlay_sources is None:
        return None

    by_frame = overlay_sources["by_frame"]
    if global_i in by_frame:
        return by_frame[global_i]
    if local_i in by_frame:
        return by_frame[local_i]

    ordered = overlay_sources["ordered"]
    if 0 <= local_i < len(ordered):
        return ordered[local_i]
    return None

def main(save_scene_mesh=False, show="both", frame_range=None, overlay_dir=None,
         draw_hand_joints=False, overlay_auto=False):
    """
    show: "both" | "hand" | "object"
      - both: 同时显示手和物体（默认）
      - hand:  只显示手
      - object: 只显示物体
    frame_range: (start, end) 左闭右开帧区间（0-based），None 表示全部帧
    overlay_dir: 显式指定背景图片目录（优先级高于 overlay_auto）。
                 文件名末尾数字可与帧号匹配；若未匹配则按排序顺序使用
    overlay_auto: True 时直接用 npy 中的 imgpath + imgnames 作为背景，无需手动指定路径
    draw_hand_joints: True 时在画面上叠加 21 个 MANO joint 与骨架连线
    """
    show_hand = show in ("both", "hand")
    show_object = show in ("both", "object")
    print("正在加载数据...")
    data = np.load(NPY_PATH, allow_pickle=True).item()

    # 动态获取 sequence 的 key (例如 '12-1')
    seq_key = list(data['data_dict'].keys())[0]
    params = data['data_dict'][seq_key]['params']

    # 手部参数
    rot_r = params['right hand']['rot_r']
    pose_r = params['right hand']['pose_r']
    trans_r = params['right hand']['trans_r']
    shape_r = params['right hand']['shape_r']
    has_left_hand = 'left hand' in params
    if has_left_hand:
        rot_l = params['left hand']['rot_l']
        pose_l = params['left hand']['pose_l']
        trans_l = params['left hand']['trans_l']
        shape_l = params['left hand']['shape_l']

    # 物体参数
    obj_rot = params['object']['obj_rot']
    obj_trans = params['object']['obj_trans']

    # 解析 npy 中所有相机视角
    cam_dict = params.get('camera', {})
    cam_bundles = _resolve_all_cameras(cam_dict)
    if not cam_bundles:
        raise RuntimeError("npy 中未找到可用相机")
    print(f"[相机] 将渲染 {len(cam_bundles)} 个视角: {[c['name'] for c in cam_bundles]}")

    # 帧区间切片（所有视角共享）
    frame_start = 0
    if frame_range is not None:
        frame_start, frame_end = frame_range
        rot_r = rot_r[frame_start:frame_end]
        pose_r = pose_r[frame_start:frame_end]
        trans_r = trans_r[frame_start:frame_end]
        shape_r = shape_r[frame_start:frame_end]
        if has_left_hand:
            rot_l = rot_l[frame_start:frame_end]
            pose_l = pose_l[frame_start:frame_end]
            trans_l = trans_l[frame_start:frame_end]
            shape_l = shape_l[frame_start:frame_end]
        obj_rot = obj_rot[frame_start:frame_end]
        obj_trans = obj_trans[frame_start:frame_end]
        print(f"[帧区间] 仅可视化第 {frame_start}～{frame_end - 1} 帧（共 {frame_end - frame_start} 帧）")

    print("正在初始化 MANO 与物体模型...")

    # 物体基础模型
    base_obj_mesh = trimesh.load(OBJ_PATH, process=False)
    if isinstance(base_obj_mesh, trimesh.Scene):
        base_obj_mesh = base_obj_mesh.dump()[0]

    # MANO 模型
    # ⚠️ flat_hand_mean=True: 与 HaMER 对齐——HaMER 内部用 smplx.MANOLayer.forward,
    # 它根本不加 hand_mean (pose2rot=False 路径), 所以 HaMER 输出的 pose 是 "绝对
    # 关节角". 这里也用 True 让 smplx.MANO 不再多加一份 mean. multiview 路径需要
    # 在 parse_mano_json_to_arrays 里把 manopth 的 hand_mean 加回去再写 npy,
    # 这样所有消费者看到的 pose 语义都是 "绝对角" 一致的.
    mano_layer = smplx.create(MANO_MODEL_DIR, 'MANO', use_pca=False, is_rhand=True, flat_hand_mean=True)
    mano_faces = mano_layer.faces
    mano_layer_l = None
    mano_faces_l = None
    if has_left_hand:
        mano_layer_l = smplx.create(MANO_MODEL_DIR, 'MANO', use_pca=False, is_rhand=False, flat_hand_mean=True)
        mano_faces_l = mano_layer_l.faces

    # 材质 (所有视角共享)
    hand_mat = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.1, alphaMode='OPAQUE', baseColorFactor=(0.8, 0.6, 0.5, 1.0))
    hand_mat_l = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.1, alphaMode='OPAQUE', baseColorFactor=(0.5, 0.6, 0.8, 1.0))
    obj_mat = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.3, alphaMode='OPAQUE', baseColorFactor=(0.2, 0.8, 0.2, 1.0))

    # OpenCV → OpenGL 相机坐标系转换
    camera_pose = np.array([
        [1.0,  0.0,  0.0, 0.0],
        [0.0, -1.0,  0.0, 0.0],
        [0.0,  0.0, -1.0, 0.0],
        [0.0,  0.0,  0.0, 1.0],
    ])

    # 帧数对齐
    n_candidates = [len(obj_rot), len(obj_trans), len(rot_r), len(pose_r), len(trans_r), len(shape_r)]
    if has_left_hand:
        n_candidates.extend([len(rot_l), len(pose_l), len(trans_l), len(shape_l)])
    num_frames = min(n_candidates)
    if len(set(n_candidates)) != 1:
        print(f"[帧数对齐] 检测到长度不一致 {n_candidates}，将按最小帧数 {num_frames} 渲染")

    # ==================== 预计算：世界坐标系下的 MANO + 物体（所有视角共享） ====================
    print(f"批量计算 MANO 顶点/关节（{num_frames} 帧，世界坐标系）...")
    with torch.no_grad():
        out_r = mano_layer(
            global_orient=torch.tensor(rot_r[:num_frames], dtype=torch.float32),
            hand_pose=torch.tensor(pose_r[:num_frames], dtype=torch.float32),
            betas=torch.tensor(shape_r[:num_frames], dtype=torch.float32),
            transl=torch.tensor(trans_r[:num_frames], dtype=torch.float32),
        )
    all_hand_verts_world = out_r.vertices.numpy()
    all_hand_joints_world = out_r.joints.numpy()
    all_hand_joints_world = _append_fingertips_if_needed(all_hand_joints_world, all_hand_verts_world)
    print(f"[hand-joints] right hand joints shape={all_hand_joints_world.shape}")

    all_hand_verts_l_world = None
    all_hand_joints_l_world = None
    if has_left_hand:
        rot_l_m, pose_l_m = right_to_left_mano_params(rot_l[:num_frames], pose_l[:num_frames])
        with torch.no_grad():
            out_l = mano_layer_l(
                global_orient=torch.tensor(rot_l_m, dtype=torch.float32),
                hand_pose=torch.tensor(pose_l_m, dtype=torch.float32),
                betas=torch.tensor(shape_l[:num_frames], dtype=torch.float32),
                transl=torch.tensor(trans_l[:num_frames], dtype=torch.float32),
            )
        all_hand_verts_l_world = out_l.vertices.numpy()
        all_hand_joints_l_world = out_l.joints.numpy()
        all_hand_joints_l_world = _append_fingertips_if_needed(all_hand_joints_l_world, all_hand_verts_l_world)
        print(f"[hand-joints] left hand joints shape={all_hand_joints_l_world.shape}")

    print("批量计算物体变换矩阵...")
    obj_rot_used = obj_rot[:num_frames].astype(np.float64)
    obj_trans_used = obj_trans[:num_frames].astype(np.float64)
    base_verts = np.asarray(base_obj_mesh.vertices, dtype=np.float64)

    if AUTO_OBJ_UNIT_FIX:
        mesh_extent = float(np.max(np.ptp(base_verts, axis=0)))
        trans_norm_med = float(np.median(np.linalg.norm(obj_trans_used, axis=1)))
        mesh_scale = 0.001 if mesh_extent > 10.0 else 1.0
        trans_scale = 0.001 if trans_norm_med > 20.0 else 1.0
        if mesh_scale != 1.0 or trans_scale != 1.0:
            print(
                f"[单位修正] mesh_extent={mesh_extent:.3f}, trans_norm_median={trans_norm_med:.3f}, "
                f"mesh_scale={mesh_scale}, trans_scale={trans_scale}"
            )
            base_verts = base_verts * mesh_scale
            obj_trans_used = obj_trans_used * trans_scale

    obj_rot_mats = R.from_rotvec(obj_rot_used).as_matrix()

    # ==================== 场景 mesh 导出（世界系，视角无关，只做一次） ====================
    if save_scene_mesh:
        os.makedirs(SCENE_MESH_OUTPUT_DIR, exist_ok=True)
        print(f"每帧 3D 场景（世界系）将保存到: {SCENE_MESH_OUTPUT_DIR}")
        for i in tqdm(range(num_frames), desc="导出场景 mesh"):
            scene_parts = []
            if show_object:
                obj_verts_world = (obj_rot_mats[i] @ base_verts.T).T + obj_trans_used[i]
                scene_parts.append(trimesh.Trimesh(vertices=obj_verts_world, faces=base_obj_mesh.faces, process=False))
            if show_hand:
                vr = all_hand_verts_world[i]
                if np.isfinite(vr).all():
                    scene_parts.append(trimesh.Trimesh(vr, mano_faces, process=False))
                if has_left_hand:
                    vl = all_hand_verts_l_world[i]
                    if np.isfinite(vl).all():
                        scene_parts.append(trimesh.Trimesh(vl, mano_faces_l, process=False))
            if scene_parts:
                combined = trimesh.util.concatenate(scene_parts)
                global_i = frame_start + i
                combined.export(os.path.join(SCENE_MESH_OUTPUT_DIR, f"frame_{global_i:04d}_viz.obj"))

    # ==================== 逐视角渲染 ====================
    video_base, video_ext = os.path.splitext(OUTPUT_VIDEO)
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

    for v_idx, cam_info in enumerate(cam_bundles):
        K = cam_info["K"]
        W2C = cam_info["W2C"]
        view_name = cam_info["name"]
        print(f"\n===== view{v_idx} ({view_name}) =====")

        # 该视角的 overlay 源 (view0 用 npy imgpath，view1+ 推导相邻序列)
        view_bg_sources = _build_view_overlay_sources(
            data=data, seq_key=seq_key, view_idx=v_idx,
            user_overlay_dir=overlay_dir, overlay_auto=overlay_auto,
        )

        # 画布尺寸: 优先本视角背景图，其次按 K 推算
        h = w = None
        if view_bg_sources is not None and view_bg_sources.get("ordered"):
            _img0 = cv2.imread(view_bg_sources["ordered"][0])
            if _img0 is not None:
                h, w = _img0.shape[:2]
                print(f"[画布] view{v_idx}: 使用背景图尺寸 {w}x{h}")
        if h is None or w is None:
            w = int(round(K[0, 2] * 2))
            h = int(round(K[1, 2] * 2))
            print(f"[画布] view{v_idx}: 按 K 推算 {w}x{h}")

        # 把世界系点变换到本视角相机坐标系
        hand_verts_cam = _transform_points(W2C, all_hand_verts_world)
        hand_joints_cam = _transform_points(W2C, all_hand_joints_world)
        hand_verts_cam_l = None
        hand_joints_cam_l = None
        if has_left_hand:
            hand_verts_cam_l = _transform_points(W2C, all_hand_verts_l_world)
            hand_joints_cam_l = _transform_points(W2C, all_hand_joints_l_world)

        # 输出路径
        view_video = f"{video_base}_view{v_idx}{video_ext}"
        view_frame_dir = os.path.join(OUTPUT_FRAME_DIR, f"view{v_idx}")
        if SAVE_FRAME_INTERVAL > 0:
            os.makedirs(view_frame_dir, exist_ok=True)
            print(f"每 {SAVE_FRAME_INTERVAL} 帧保存图片到: {view_frame_dir}")

        renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
        camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(view_video, fourcc, FPS, (w, h))

        last_valid_hand_mesh = None
        last_valid_hand_mesh_l = None
        warned_overlay_resize = False
        warned_overlay_missing = False
        render_hand_mesh = show_hand and not draw_hand_joints

        print(f"开始渲染 view{v_idx}，共 {num_frames} 帧...")
        for i in tqdm(range(num_frames), desc=f"view{v_idx}"):
            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.6, 0.6, 0.6])
            scene.add(camera, pose=camera_pose)
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
            scene.add(light, pose=camera_pose)

            # 物体
            if show_object:
                obj_verts_world = (obj_rot_mats[i] @ base_verts.T).T + obj_trans_used[i]
                obj_verts_cam = _transform_points(W2C, obj_verts_world)
                curr_obj_mesh = trimesh.Trimesh(vertices=obj_verts_cam, faces=base_obj_mesh.faces, process=False)
                scene.add(pyrender.Mesh.from_trimesh(curr_obj_mesh, material=obj_mat))

            # 右手
            vr_cam = hand_verts_cam[i]
            valid = show_hand and vr_cam.size > 0 and np.isfinite(vr_cam).all()
            if valid:
                trimesh_hand = trimesh.Trimesh(vr_cam, mano_faces, process=False)
                if render_hand_mesh:
                    pr_hand_mesh = pyrender.Mesh.from_trimesh(trimesh_hand, material=hand_mat)
                    last_valid_hand_mesh = pr_hand_mesh
                    scene.add(pr_hand_mesh)
            elif render_hand_mesh and last_valid_hand_mesh is not None:
                scene.add(last_valid_hand_mesh)

            # 左手
            if show_hand and has_left_hand:
                vl_cam = hand_verts_cam_l[i]
                valid_l = vl_cam.size > 0 and np.isfinite(vl_cam).all()
                if valid_l:
                    trimesh_hand_l = trimesh.Trimesh(vl_cam, mano_faces_l, process=False)
                    if render_hand_mesh:
                        pr_hand_mesh_l = pyrender.Mesh.from_trimesh(trimesh_hand_l, material=hand_mat_l)
                        last_valid_hand_mesh_l = pr_hand_mesh_l
                        scene.add(pr_hand_mesh_l)
                elif render_hand_mesh and last_valid_hand_mesh_l is not None:
                    scene.add(last_valid_hand_mesh_l)

            # 渲染
            color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            render_bgr = cv2.cvtColor(color[:, :, :3], cv2.COLOR_RGB2BGR)
            alpha = (color[:, :, 3:4].astype(np.float32) / 255.0)
            final_frame = render_bgr

            if view_bg_sources is not None:
                global_i = frame_start + i
                overlay_path = _get_overlay_path(view_bg_sources, global_i=global_i, local_i=i)
                if overlay_path is not None:
                    bg = cv2.imread(overlay_path, cv2.IMREAD_COLOR)
                    if bg is None:
                        if not warned_overlay_missing:
                            print(f"[overlay] view{v_idx}: 读取失败 {overlay_path}")
                            warned_overlay_missing = True
                    else:
                        if bg.shape[1] != w or bg.shape[0] != h:
                            if not warned_overlay_resize:
                                print(
                                    f"[overlay] view{v_idx}: 背景 {bg.shape[1]}x{bg.shape[0]} != {w}x{h}，已 resize"
                                )
                                warned_overlay_resize = True
                            bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
                        final_frame = (render_bgr.astype(np.float32) * alpha
                                       + bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
                elif not warned_overlay_missing:
                    print(f"[overlay] view{v_idx}: 部分帧找不到对应背景图")
                    warned_overlay_missing = True

            # 2D joint 骨架
            if show_hand and draw_hand_joints:
                _draw_hand_skeleton(
                    final_frame, hand_joints_cam[i], K, w, h,
                    joint_color=(0, 255, 255), bone_color=(0, 200, 200),
                )
                if has_left_hand and hand_joints_cam_l is not None:
                    _draw_hand_skeleton(
                        final_frame, hand_joints_cam_l[i], K, w, h,
                        joint_color=(255, 255, 0), bone_color=(200, 200, 0),
                    )

            video_writer.write(final_frame)
            if SAVE_FRAME_INTERVAL > 0 and i % SAVE_FRAME_INTERVAL == 0:
                cv2.imwrite(os.path.join(view_frame_dir, f"frame_{i:05d}.jpg"), final_frame)

        video_writer.release()
        renderer.delete()
        print(f"view{v_idx} 渲染完成 (mp4v): {view_video}")

        # ffmpeg 转 H.264
        if shutil.which("ffmpeg"):
            h264_path = view_video.rsplit(".", 1)[0] + "_h264.mp4"
            cmd = [
                "ffmpeg", "-y", "-i", view_video,
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p", h264_path,
            ]
            print(f"view{v_idx}: 正在用 ffmpeg 转码 H.264: {h264_path}")
            ret = subprocess.run(cmd, capture_output=True)
            if ret.returncode == 0:
                os.replace(h264_path, view_video)
                print(f"view{v_idx}: 已转码覆盖 {view_video}")
            else:
                print(f"view{v_idx}: ffmpeg 失败 (exit {ret.returncode})，保留 mp4v")
                print(ret.stderr.decode(errors="replace")[-500:])
        else:
            print("未找到 ffmpeg，跳过转码")

    print("\n全部视角渲染完成。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-scene-mesh', action='store_true', default=False,
                        help='将每帧手物 3D 场景导出到 debug_viz_output 目录')
    parser.add_argument('--show', type=str, default='both',
                        choices=['both', 'hand', 'object'],
                        help='生成内容：both=手+物体（默认），hand=仅手，object=仅物体')
    parser.add_argument('--frame-range', type=int, nargs=2, default=None, metavar=('START', 'END'),
                        help='只可视化该帧区间 [START, END)（0-based 左闭右开），不传则全部帧')
    parser.add_argument('--overlay-dir', type=str, default=None,
                        help='显式指定背景图目录（优先级高于 --overlay）。优先按文件名末尾数字匹配帧号（如 frame_000123.jpg -> 123），否则按文件名排序顺序使用')
    parser.add_argument('--overlay', action='store_true', default=False,
                        help='直接使用 npy 中的 imgpath + imgnames 作为背景图（无需手动指定路径）')
    parser.add_argument('--draw-hand-joints', action='store_true', default=False,
                        help='在画面上叠加 21 个 MANO 手部 joint 与骨架连线（右手黄/左手青）')
    args = parser.parse_args()
    main(
        save_scene_mesh=args.save_scene_mesh,
        show=args.show,
        frame_range=args.frame_range,
        overlay_dir=args.overlay_dir,
        overlay_auto=args.overlay,
        draw_hand_joints=args.draw_hand_joints,
    )