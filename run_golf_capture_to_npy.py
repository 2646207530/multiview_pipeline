"""
Golf capture -> HaMER + object pose -> optimization -> npy

支持两种 capture 目录布局（自动探测）：

布局 A（旧, 例: /data2/fubingshuai/golf/data/3-1-wood-03/20260415152908498）：

    <club_dir>/<capture_id>/
        camera_params.json
        trajectory_output/trajectory.csv
        <color_cam_a>_w1440_h1080_pBayerRG8_f120/frame_NNNNNN.jpg
        <color_cam_b>_w1440_h1080_pBayerRG8_f120/frame_NNNNNN.jpg

布局 B（新, 例: /data2/fubingshuai/golf/data/35_wood_8_01_fbs/20260424170424563）：

    <club_dir>/<capture_id>/
        camera_params.json
        trajectory_output/trajectory.csv
        .tmp_images/
            <color_cam_a>_w1440_h1080_pBayerRG8_f120/frame_NNNNNN.jpg
            <color_cam_b>_w1440_h1080_pBayerRG8_f120/frame_NNNNNN.jpg

输出 npy 的 schema 与 /data2/fubingshuai/golf/output/12-1.npy 对齐：

    {
      "imgnames": [...],                      # HaMER 推理相机的帧文件名
      "imgpath":  ".../<hamer_cam>_.../",
      "data_dict": {
          <seq_name>: {
              "params": {
                  "right hand": {rot_r, pose_r, trans_r, shape_r},
                  "left hand" : {rot_l, pose_l, trans_l, shape_l},
                  "object"    : {obj_rot, obj_trans},
                  "camera"    : {
                      "world2cam": [w2c_cam0_4x4, w2c_cam1_4x4],
                      "K":         [K_cam0_3x3,   K_cam1_3x3],
                      "views":     ["cam0", "cam1"],
                  },
              },
          },
      },
    }

    --init_method auto/hamer/multiview 选项控制手部位姿初始化用 HaMER 单视角还是多视角 Hand_Estimation。
    
所有平移单位为「米」。world 坐标系 = HaMER 推理相机 (cam0) 的 sensor 系，
因此 world2cam[cam0] = I，world2cam[cam1] = sensor2rig_cam1^{-1} ∘ sensor2rig_cam0。

用法（仅手 + 物体位姿注入, 新布局）:
  python run_golf_capture_to_npy.py \
      --capture_dir /data2/fubingshuai/golf/data/35_wood_8_01_fbs/20260424170424563 \
      --output /data2/fubingshuai/golf/output/out/35_wood_8_01_fbs.npy \
      --seq_name 20260424170424563

用法（加上 concat + 优化 + 重放, 对齐 run_hamer_to_npy.py）:
  python run_golf_capture_to_npy.py \
      --capture_dir /data2/fubingshuai/golf/data/35_wood_8_01_fbs/20260424170424563 \
      --init_method hamer \
      --output /data2/fubingshuai/golf/golf-hand-object/out/35_wood_8_01_fbs.npy \
      --seq_name 20260424170424563 \
      --point_r 0.0 0.82 0.0 --point_l 0.0 0.90 0.0 \
      --contact_config /data2/fubingshuai/golf/golf-hand-object/config/baseball_golf.json \
      --force_closure_range 1 50 \
      --ref_frame 25 \
      --opt_debug_dir /data2/fubingshuai/golf/test/npy_debug \
      --opt_debug_every 500


用法（加上 concat + 优化 + 重放, 对齐 run_hamer_to_npy.py）:
  python run_golf_capture_to_npy.py \
      --capture_dir /data2/fubingshuai/golf/data/35_wood_8_01_fbs/20260424170424563 \
      --init_method auto \
      --mv_finetune_epochs 1 \
      --mv_finetune_lr 1e-5 \
      --mv_finetune_bs 1 \
      --output /data2/fubingshuai/golf/golf-hand-object/out/35_wood_8_01_fbs.npy \
      --seq_name 20260424170424563 \
      --point_r 0.0 0.82 0.0 --point_l 0.0 0.90 0.0 \
      --contact_config /data2/fubingshuai/golf/golf-hand-object/config/baseball_golf.json \
      --opt_range 1 50 \
      --force_closure_range 1 50 \
      --ref_frame 25
"""

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import sys
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'model'))
sys.path.insert(0, os.path.join(_ROOT, 'model', 'hamer'))

import argparse
import glob
import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from model.hamer.infer import hamer_inference, matrix_to_axis_angle
from config.hamer_config import hamer_opt
from config.yolo_config import yolo_opt
from yolo.detector import Detector

# Re-use helpers from the single-camera pipeline so the optimization steps
# and NaN-handling behave identically.
from run_hamer_to_npy import (
    parse_detections,
    extract_hand_params,
    NAN_RIGHT,
    NAN_LEFT,
    _interpolate_nan,
    _slerp_interpolate_nan,
    concat_hand_to_object,
    rehand_by_object,
    smooth_object_pose_temporal,
)


# -------------------------------------------------------- capture parsing ----
FRAME_DIR_SUFFIX = "_w1440_h1080_pBayerRG8_f120"
FRAME_RE = re.compile(r"frame_(\d+)\.(?:jpg|jpeg|png|bmp)$", re.IGNORECASE)


def _resolve_club_stl(capture_dir: Path, club_stl_override: str = None) -> Path:
    """
    定位本次采集对应的球杆 STL (单位: mm)。

    支持两种 data 目录布局:
      旧:  <data_root>/club_asset/<club_name>.stl
      新:  <data_root>/club-assets/<club_name>/<id>.stl
            club_name 也允许去掉 _fbs / -fbs 等后缀后再匹配，
            例如 capture 在 35_wood_8_01_fbs/ 下，资产在 club-assets/35_wood_8_01/。
    """
    if club_stl_override:
        p = Path(club_stl_override).expanduser().resolve()
        if not p.exists():
            raise RuntimeError(f"--club_stl 指向的文件不存在: {p}")
        return p

    club_name = capture_dir.parent.name
    data_root = capture_dir.parent.parent

    # club_name 的候选写法（用于在 club-assets/ 下找子目录）
    name_variants = [club_name]
    for suffix in ("_fbs", "-fbs"):
        if club_name.endswith(suffix):
            name_variants.append(club_name[: -len(suffix)])

    tried = []

    # 候选 1（新布局）: <data_root>/club-assets/<name>/*.stl
    for name in name_variants:
        d = data_root / "club-assets" / name
        tried.append(str(d))
        if d.is_dir():
            stls = sorted(p for p in d.glob("*.stl")
                          if not p.stem.endswith("_meter"))
            if stls:
                if len(stls) > 1:
                    print(f"[球杆STL] {d} 下有多个 .stl，使用 {stls[0].name}")
                return stls[0]

    # 候选 2（旧布局）: <data_root>/club_asset/<club_name>.stl
    legacy = data_root / "club_asset" / f"{club_name}.stl"
    tried.append(str(legacy))
    if legacy.exists():
        return legacy

    raise RuntimeError(
        "默认推断的球杆 STL 不存在。\n"
        f"  capture_dir = {capture_dir}\n"
        f"  club_name   = {club_name}\n"
        "  尝试过的位置:\n    - " + "\n    - ".join(tried) + "\n"
        "  请用 --club_stl 显式指定"
    )


def _ensure_meter_stl(mm_stl: Path) -> Path:
    """
    保证球杆模型文件夹下有一个 *_meter.stl（米为单位）。
    - 若已存在同目录下 <stem>_meter.stl，直接返回。
    - 否则读入 mm 版本，顶点乘 0.001，写出二进制 STL。
    返回米为单位的 STL 路径。
    """
    mm_stl = Path(mm_stl).resolve()
    meter_stl = mm_stl.with_name(f"{mm_stl.stem}_meter{mm_stl.suffix}")
    if meter_stl.exists():
        print(f"[米制STL] 已存在: {meter_stl}")
        return meter_stl

    import trimesh  # 延迟 import，避免无优化的最小流程也要求 trimesh
    print(f"[米制STL] 未找到 {meter_stl.name}，从 {mm_stl.name} 生成 (顶点 × 0.001)")
    mesh = trimesh.load(str(mm_stl), force="mesh", process=False)
    if mesh.is_empty:
        raise RuntimeError(f"读取 STL 失败或空网格: {mm_stl}")

    mesh_scaled = mesh.copy()
    mesh_scaled.apply_scale(1.0 / 1000.0)
    # 固定写二进制 STL；trimesh 会根据扩展名决定格式，显式 file_type 更稳。
    mesh_scaled.export(str(meter_stl), file_type="stl")
    print(f"[米制STL] 已写入: {meter_stl}  顶点数={len(mesh_scaled.vertices)}")
    return meter_stl


def _load_camera_params(capture_dir: Path):
    with open(capture_dir / "camera_params.json") as f:
        data = json.load(f)
    cams = {}
    for c in data["rig"]["cameras"]:
        cams[c["name"]] = c
    return cams


def _quat_to_R(qw, qx, qy, qz):
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ])


def _K_from_cam(c):
    ip = c["intrinsics"]
    return np.array([
        [ip["fx"], 0.0, ip["cx"]],
        [0.0, ip["fy"], ip["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def _sensor2rig(c):
    R_s2r = np.array(c["sensor2rig"]["rotation"], dtype=np.float64)
    t_s2r = np.array(c["sensor2rig"]["translation"], dtype=np.float64) / 1000.0  # mm -> m
    return R_s2r, t_s2r


def _resolve_color_cams(cams_dict, capture_dir, hamer_name=None, other_name=None):
    """
    挑出两台 1440x1080 彩色相机作为 cam0 / cam1。
    默认只从「此次采集目录里确实有帧文件夹」的相机里挑，
    避免选到 camera_params.json 中列出但本次未录制的相机。
    """
    color = [c for c in cams_dict.values()
             if c["image_width"] == 1440 and c["image_height"] == 1080]
    color.sort(key=lambda c: c["camera_id"])

    def _has_frames(name):
        return _frame_dir_for(name, capture_dir) is not None

    recorded = [c["name"] for c in color if _has_frames(c["name"])]
    all_names = [c["name"] for c in color]
    if not all_names:
        raise RuntimeError("camera_params.json 里找不到 1440x1080 的彩色相机")
    if not recorded:
        raise RuntimeError(
            "没有任何 1440x1080 彩色相机在 capture_dir 下有对应的帧目录 "
            f"({FRAME_DIR_SUFFIX})。camera_params.json 里存在的彩色相机: {all_names}"
        )

    if hamer_name is None:
        hamer_name = recorded[0]
    if hamer_name not in cams_dict:
        raise RuntimeError(f"相机 {hamer_name} 未在 camera_params.json 出现")
    if not _has_frames(hamer_name):
        raise RuntimeError(
            f"--hamer_cam={hamer_name} 在 capture_dir 下没有帧目录；"
            f"本次可用的彩色相机: {recorded}"
        )

    if other_name is None:
        candidates = [n for n in recorded if n != hamer_name]
        if candidates:
            other_name = candidates[0]
    if other_name is not None:
        if other_name not in cams_dict:
            raise RuntimeError(f"相机 {other_name} 未在 camera_params.json 出现")
        if not _has_frames(other_name):
            print(f"[相机] 警告: --other_cam={other_name} 在 capture_dir 下没有帧目录，"
                  f"cam1 仅保留在 world2cam 中不做叠加")

    return hamer_name, other_name


def _frame_dir_for(cam_name, capture_dir):
    # 目录名含括号，glob 用的 [] 要 escape；直接拼。
    # 兼容两种布局：旧的 capture_dir/<cam>...，新的 capture_dir/.tmp_images/<cam>...
    sub = f"{cam_name}{FRAME_DIR_SUFFIX}"
    for base in (capture_dir / ".tmp_images", capture_dir):
        p = base / sub
        if p.is_dir():
            return p
    return None


def _index_frames(frame_dir: Path):
    """返回 {frame_idx: Path}，以及对齐长度 = max_idx + 1。"""
    frame_map = {}
    for p in frame_dir.iterdir():
        m = FRAME_RE.match(p.name)
        if not m:
            continue
        frame_map[int(m.group(1))] = p
    if not frame_map:
        raise RuntimeError(f"{frame_dir} 没有形如 frame_NNNNNN.jpg 的文件")
    return frame_map, max(frame_map) + 1


def _load_trajectory(csv_path: Path):
    """返回 (ref_camera_name, N×7 [qw,qx,qy,qz,tx,ty,tz])。tx/ty/tz 单位为 mm。"""
    ref = None
    rows = []
    header_seen = False
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "reference_camera" in line:
                    ref = line.split(":", 1)[1].strip()
                continue
            if not header_seen:
                header_seen = True
                continue
            parts = line.split(",")
            rows.append([float(x) for x in parts[1:8]])
    if ref is None:
        raise RuntimeError(f"{csv_path} 顶部未找到 '# reference_camera: ...'")
    return ref, np.array(rows, dtype=np.float64)


# ------------------------------------------------ geometry transformations ----
def _object_poses_to_world(poses_mm, cams_dict, ref_name, world_name):
    """
    CSV 里的位姿是 obj -> ref_camera_sensor (t 单位 mm)。
    把它们变到 world = world_name 相机 sensor 系，t 单位 m。
    """
    R_ref, t_ref = _sensor2rig(cams_dict[ref_name])    # ref_sensor -> rig
    R_w,   t_w   = _sensor2rig(cams_dict[world_name])  # world_sensor -> rig

    # ref_sensor 在 world 系下的位姿:
    #   P_rig    = R_w @ P_world + t_w  = R_ref @ P_ref + t_ref
    #   P_world  = R_w^T @ (R_ref @ P_ref + t_ref - t_w)
    R_r2w = R_w.T @ R_ref
    t_r2w = R_w.T @ (t_ref - t_w)

    rot_list, trans_list = [], []
    for pose in poses_mm:
        qw, qx, qy, qz, tx, ty, tz = pose
        R_obj = _quat_to_R(qw, qx, qy, qz)          # obj -> ref_sensor
        t_obj = np.array([tx, ty, tz]) / 1000.0     # mm -> m

        R_world = R_r2w @ R_obj
        t_world = R_r2w @ t_obj + t_r2w

        rot_list.append(R.from_matrix(R_world).as_rotvec())
        trans_list.append(t_world)

    return (np.array(rot_list,   dtype=np.float32),
            np.array(trans_list, dtype=np.float32))


def _build_camera_block(cams_dict, hamer_name, other_name, new_K_per_cam=None):
    """世界系 = hamer_name 相机 sensor 系。返回与 12-1.npy 对齐的 camera dict。

    new_K_per_cam: {cam_name: newK (3x3)}, 由 prepare_undistort_dir 算好的
    去畸变后内参. 若给了, K 字段填这套 newK; 否则回退到原始 K (老行为).
    """
    R_h, t_h = _sensor2rig(cams_dict[hamer_name])

    def _pick_K(cam_name):
        if new_K_per_cam is not None and cam_name in new_K_per_cam:
            return np.asarray(new_K_per_cam[cam_name], dtype=np.float32)
        return _K_from_cam(cams_dict[cam_name])

    w2c_hamer = np.eye(4, dtype=np.float32)
    K_hamer = _pick_K(hamer_name)
    w2c_list = [w2c_hamer]
    K_list = [K_hamer]
    views = ["cam0"]

    if other_name is not None:
        R_o, t_o = _sensor2rig(cams_dict[other_name])
        # P_cam_other = R_o^T @ (P_rig - t_o) = R_o^T R_h P_world + R_o^T (t_h - t_o)
        R_w2c_other = R_o.T @ R_h
        t_w2c_other = R_o.T @ (t_h - t_o)
        w2c_other = np.eye(4, dtype=np.float32)
        w2c_other[:3, :3] = R_w2c_other.astype(np.float32)
        w2c_other[:3, 3] = t_w2c_other.astype(np.float32)
        w2c_list.append(w2c_other)
        K_list.append(_pick_K(other_name))
        views.append("cam1")

    return {
        "world2cam": w2c_list,
        "K": K_list,
        "views": views,
    }


# ----------------------------------------------------------------- runner ----
def run(capture_dir, output_npy, seq_name,
        hamer_cam=None, other_cam=None,
        obj_pose_sigma=0.0,
        point_r=None, point_l=None,
        opt_range=None, force_closure_range=None, ref_frame=None,
        club_stl=None, contact_config=None,
        init_method="auto",
        mv_finetune_epochs=0, mv_finetune_lr=1e-5, mv_finetune_bs=1,
        opt_debug_every=0, opt_debug_dir="/data2/fubingshuai/golf/test"):

    capture_dir = Path(capture_dir).resolve()
    output_npy = str(Path(output_npy).resolve())
    Path(output_npy).parent.mkdir(parents=True, exist_ok=True)

    if not capture_dir.is_dir():
        raise RuntimeError(f"capture_dir 不存在: {capture_dir}")

    cams = _load_camera_params(capture_dir)
    hamer_name, other_name = _resolve_color_cams(cams, capture_dir, hamer_cam, other_cam)
    print(f"[相机] HaMER 推理相机 (world=cam0): {hamer_name}")
    print(f"[相机] 第二视角 (cam1):             {other_name}")

    hamer_dir = _frame_dir_for(hamer_name, capture_dir)
    if hamer_dir is None:
        raise RuntimeError(f"找不到 {hamer_name} 的帧目录 {hamer_name}{FRAME_DIR_SUFFIX}")
    frame_map, total_frames = _index_frames(hamer_dir)
    print(f"[帧] HaMER 相机原始帧目录: {hamer_dir}")
    print(f"[帧] 有效帧 {len(frame_map)} / 对齐长度 {total_frames}")

    # ─── 统一去畸变: 之后所有路径 (HaMER 单视角 / 多视角 / mask / SportGS / 可视化) ─
    # 都用 .undistorted/<seq>/<cam_idx>/images_undistorted/{frame:06d}.jpg + newK
    cam_names_in_order = [hamer_name]
    if other_name is not None:
        cam_names_in_order.append(other_name)
    frame_dirs_orig_for_undist = {}
    for n in cam_names_in_order:
        d = _frame_dir_for(n, capture_dir)
        if d is None:
            raise RuntimeError(f"找不到 {n} 的帧目录, 无法统一去畸变")
        frame_dirs_orig_for_undist[n] = d
    from multiview_hand_init import prepare_undistort_dir
    undist_root, capture_id_for_undist, new_K_per_cam = prepare_undistort_dir(
        capture_dir, cams, cam_names_in_order, hamer_name,
        frame_dirs_orig_for_undist,
    )
    hamer_undist_dir = (
        undist_root / capture_id_for_undist / "0" / "images_undistorted"
    )
    if not hamer_undist_dir.is_dir():
        raise RuntimeError(f"去畸变目录未生成: {hamer_undist_dir}")
    print(f"[帧] HaMER 相机去畸变目录: {hamer_undist_dir}")

    traj_csv = capture_dir / "trajectory_output" / "trajectory.csv"
    if not traj_csv.exists():
        raise RuntimeError(f"找不到 trajectory.csv: {traj_csv}")
    ref_cam_name, poses = _load_trajectory(traj_csv)
    print(f"[物体] trajectory.csv reference_camera: {ref_cam_name}")
    print(f"[物体] 位姿行数: {len(poses)}")

    if ref_cam_name not in cams:
        raise RuntimeError(f"reference_camera {ref_cam_name} 不在 camera_params.json 中")

    # 物体位姿 -> world 系（m）
    obj_rot_arr, obj_trans_arr = _object_poses_to_world(
        poses, cams, ref_cam_name, hamer_name
    )
    # 对齐到图像帧数
    n_obj = obj_rot_arr.shape[0]
    if n_obj != total_frames:
        print(f"[物体] CSV 帧数 {n_obj} 与图像帧数 {total_frames} 不一致；"
              f"按 min={min(n_obj, total_frames)} 截断")
        total_frames = min(n_obj, total_frames)
        obj_rot_arr = obj_rot_arr[:total_frames]
        obj_trans_arr = obj_trans_arr[:total_frames]

    # 相机内参: 用 newK (去畸变后), 因为下面的所有图像读取都已切到 undistorted 版本
    k_use = new_K_per_cam[hamer_name].astype(np.float32)
    print(f"[相机] HaMER 使用 newK (post-undistort):\n{k_use}")

    # ─── 选择手部初始化方法 (HaMER 单视角 vs 多视角 Hand_Estimation) ─────────
    n_color_cams_recorded = 0
    for c in cams.values():
        if c.get("image_width") == 1440 and c.get("image_height") == 1080:
            if _frame_dir_for(c["name"], capture_dir) is not None:
                n_color_cams_recorded += 1
    if init_method == "auto":
        use_multiview = n_color_cams_recorded >= 2
    elif init_method == "multiview":
        use_multiview = True
    elif init_method == "hamer":
        use_multiview = False
    else:
        raise RuntimeError(f"不支持的 --init_method: {init_method!r}")
    print(f"[初始化] 录制中的 1440x1080 彩色相机数 = {n_color_cams_recorded}; "
          f"init_method={init_method}; "
          f"实际走 {'多视角 Hand_Estimation' if use_multiview else 'HaMER 单视角'}")

    # 不论走哪一支, HaMER + YOLO 都需要——多视角分支用 HaMER 给每个相机生成 2D 伪标。
    print("[模型] 正在加载 HaMER / YOLO ...")
    hamer = hamer_inference(hamer_opt)
    detector = Detector(yolo_opt)
    print("[模型] 加载完成。")

    stats = {'missing_file': 0, 'missing_right': 0, 'missing_left': 0}

    if use_multiview:
        # 收集 cam0/cam1 的原始帧目录（按 cam0=hamer_name, cam1=other_name 顺序）
        if other_name is None:
            raise RuntimeError("init_method=multiview 但只有一台彩色相机被录制")
        cam_names_in_order = [hamer_name, other_name]
        frame_dirs_orig = {}
        for n in cam_names_in_order:
            d = _frame_dir_for(n, capture_dir)
            if d is None:
                raise RuntimeError(f"找不到 {n} 的帧目录")
            frame_dirs_orig[n] = d

        from multiview_hand_init import init_hands_from_multiview
        (r_rot_arr, r_pose_arr, r_shape_arr, r_trans_arr,
         l_rot_arr, l_pose_arr, l_shape_arr, l_trans_arr) = init_hands_from_multiview(
            project_root=Path(_ROOT),
            capture_dir=capture_dir,
            cams_dict=cams,
            cam_names_in_order=cam_names_in_order,
            world_name=hamer_name,
            frame_dirs_orig=frame_dirs_orig,
            total_frames=total_frames,
            hamer=hamer, detector=detector, parse_detections=parse_detections,
            mv_finetune_epochs=mv_finetune_epochs,
            mv_finetune_lr=mv_finetune_lr,
            mv_finetune_bs=mv_finetune_bs,
        )
        # 多视角分支自己用 NaN 占位缺失帧；统计一下方便观察
        stats['missing_right'] = int(np.isnan(r_rot_arr).any(axis=1).sum())
        stats['missing_left']  = int(np.isnan(l_rot_arr).any(axis=1).sum())
    else:
        r_rot, r_pose, r_shape, r_trans = [], [], [], []
        l_rot, l_pose, l_shape, l_trans = [], [], [], []

        for i in tqdm(range(total_frames), desc="HaMER 推理"):
            # 读 undistorted 版本; prepare_undistort_dir 保证 frame_map 里出现过的 frame_idx
            # 都被去畸变到 hamer_undist_dir/{i:06d}.jpg
            img_path = hamer_undist_dir / f"{i:06d}.jpg" if i in frame_map else None
            image = cv2.imread(str(img_path)) if img_path is not None and img_path.is_file() else None
            if image is None:
                stats['missing_file'] += 1
                for lst in (r_rot, l_rot):   lst.append(NAN_RIGHT['rot'].copy())
                for lst in (r_pose, l_pose): lst.append(NAN_RIGHT['pose'].copy())
                for lst in (r_shape, l_shape): lst.append(NAN_RIGHT['shape'].copy())
                for lst in (r_trans, l_trans): lst.append(NAN_RIGHT['trans'].copy())
                continue

            _, dets = detector.detect(image)
            detection_list = parse_detections(dets)

            frame_result = {'right': None, 'left': None}
            for bbox in detection_list:
                # YOLO 在无检测时可能返回形如 [[]] 或 [None]，过滤掉。
                if not bbox or len(bbox) < 2:
                    continue
                hand_label = bbox[0]
                if hand_label not in frame_result:
                    continue
                try:
                    output, _ = hamer.estimate_from_rgb(image, [bbox], k_use)
                    mano_params = output['pred_mano_params']
                    hand_data = extract_hand_params(output, mano_params, hand_label)
                    frame_result[hand_label] = hand_data
                except Exception as e:
                    print(f"帧 {i} 处理 {hand_label} 手出错: {e}")

            rh = frame_result['right']
            if rh is not None:
                r_rot.append(rh['pose_global']); r_pose.append(rh['pose_hand'])
                r_shape.append(rh['betas']);     r_trans.append(rh['cam_t'])
            else:
                stats['missing_right'] += 1
                r_rot.append(NAN_RIGHT['rot'].copy());   r_pose.append(NAN_RIGHT['pose'].copy())
                r_shape.append(NAN_RIGHT['shape'].copy()); r_trans.append(NAN_RIGHT['trans'].copy())

            lh = frame_result['left']
            if lh is not None:
                l_rot.append(lh['pose_global']); l_pose.append(lh['pose_hand'])
                l_shape.append(lh['betas']);     l_trans.append(lh['cam_t'])
            else:
                stats['missing_left'] += 1
                l_rot.append(NAN_LEFT['rot'].copy());   l_pose.append(NAN_LEFT['pose'].copy())
                l_shape.append(NAN_LEFT['shape'].copy()); l_trans.append(NAN_LEFT['trans'].copy())

        r_rot_arr   = np.array(r_rot,   dtype=np.float32)
        r_pose_arr  = np.array(r_pose,  dtype=np.float32)
        r_shape_arr = np.array(r_shape, dtype=np.float32)
        r_trans_arr = np.array(r_trans, dtype=np.float32)
        l_rot_arr   = np.array(l_rot,   dtype=np.float32)
        l_pose_arr  = np.array(l_pose,  dtype=np.float32)
        l_shape_arr = np.array(l_shape, dtype=np.float32)
        l_trans_arr = np.array(l_trans, dtype=np.float32)

    interp_count = 0
    interp_count += _slerp_interpolate_nan(r_rot_arr)
    interp_count += _slerp_interpolate_nan(l_rot_arr)
    for arr in [r_pose_arr, r_shape_arr, r_trans_arr,
                l_pose_arr, l_shape_arr, l_trans_arr]:
        interp_count += _interpolate_nan(arr)
    print(f"[插值] 共填补 {interp_count} 段 NaN (rot 用 SLERP)")

    # ─── 按 12-1.npy schema 组装 ────────────────────────────────────────
    # imgpath 指 undistorted 目录, imgnames 用 "{i:06d}.jpg" 命名
    # (对应 prepare_undistort_dir 写出来的 hamer_undist_dir 里的文件名).
    # 这样 SportGS / mask / 可视化都用同一份 (undistorted + newK) 几何.
    imgnames = [(f"{i:06d}.jpg" if i in frame_map else "")
                for i in range(total_frames)]
    imgpath = str(hamer_undist_dir)

    params = {
        "right hand": {
            "rot_r":   r_rot_arr,
            "pose_r":  r_pose_arr,
            "trans_r": r_trans_arr,
            "shape_r": r_shape_arr,
        },
        "left hand": {
            "rot_l":   l_rot_arr,
            "pose_l":  l_pose_arr,
            "trans_l": l_trans_arr,
            "shape_l": l_shape_arr,
        },
        "object": {
            "obj_rot":   obj_rot_arr,
            "obj_trans": obj_trans_arr,
        },
        "camera": _build_camera_block(cams, hamer_name, other_name,
                                       new_K_per_cam=new_K_per_cam),
    }

    root = {
        "imgnames": imgnames,
        "imgpath":  imgpath,
        "data_dict": {seq_name: {"params": params}},
    }

    # 物体位姿时序平滑（可选，默认不开）
    if obj_pose_sigma and obj_pose_sigma > 0:
        smooth_object_pose_temporal(root, seq_name, sigma=obj_pose_sigma)

    # 对齐物体和手的帧数（HaMER 可能与 CSV 同步，但防御性裁到 min）
    n_frames = min(
        total_frames,
        r_rot_arr.shape[0],
        obj_rot_arr.shape[0],
    )
    if n_frames != total_frames:
        print(f"[对齐] 把所有数组裁到 {n_frames} 帧")
        for hand_key in ("right hand", "left hand"):
            for k, v in list(params[hand_key].items()):
                if isinstance(v, np.ndarray):
                    params[hand_key][k] = v[:n_frames]
        for k in ("obj_rot", "obj_trans"):
            params["object"][k] = params["object"][k][:n_frames]
        imgnames = imgnames[:n_frames]
        root["imgnames"] = imgnames

    # ─── Concat / Contact / Force-closure / Rehand（复用参考脚本）─────
    if point_r is not None and point_l is not None:
        concat_hand_to_object(root, seq_name, point_r, point_l)
        base_post, ext_post = os.path.splitext(output_npy)
        post_concat_path = os.path.abspath(f"{base_post}_post_concat{ext_post}")
        np.save(post_concat_path, root, allow_pickle=True)
        print(f"[手物Concat] 已保存 concat 后的快照: {post_concat_path}")
    elif point_r is not None or point_l is not None:
        print("[手物Concat] --point_r 和 --point_l 必须同时指定，跳过")

    # 进入优化前，挑选 SportGS 实际使用的球杆 STL。
    # 优先级:
    #   1. --club_stl 显式指定 (走 _resolve_club_stl 内部分支)
    #   2. <club_dir>/<stem>_final.stl  ← 你手动削减/调整好的最终米制模型
    #   3. <club_dir>/<stem>_meter.stl  ← _ensure_meter_stl 自动生成 (mm × 0.001)
    if opt_range is not None or force_closure_range is not None:
        mm_stl_path = _resolve_club_stl(capture_dir, club_stl)
        final_stl_path = mm_stl_path.with_name(f"{mm_stl_path.stem}_final{mm_stl_path.suffix}")
        if final_stl_path.is_file():
            meter_stl_path = final_stl_path
            print(f"[球杆STL] 检测到手动 final 模型, 直接使用 (跳过 mm→m 转换): {meter_stl_path}")
        else:
            meter_stl_path = _ensure_meter_stl(mm_stl_path)
            print(f"[米制STL] 未发现 {final_stl_path.name}, 使用自动生成的米制 STL: {meter_stl_path}")

        frames_for_mask = set()
        if opt_range is not None:
            frames_for_mask.update(range(int(opt_range[0]), int(opt_range[1])))
        if force_closure_range is not None:
            frames_for_mask.update(range(int(force_closure_range[0]), int(force_closure_range[1])))
        if frames_for_mask:
            from generate_masks_sam2 import ensure_masks as _ensure_masks
            _ensure_masks(
                root, seq_name, sorted(frames_for_mask),
                club_mesh_path=os.path.abspath(meter_stl_path),
            )

    # SportGS 子进程读取 contact 配置文件 (tip 顶点等) 的路径
    sportgs_env = os.environ.copy()
    if contact_config is not None:
        contact_config_path = os.path.abspath(contact_config)
        if not os.path.isfile(contact_config_path):
            raise RuntimeError(f"--contact_config 指向的文件不存在: {contact_config_path}")
        sportgs_env["CONTACT_CONFIG_PATH"] = contact_config_path
        print(f"[contact_config] 子进程使用: {contact_config_path}")
    # 否则交给 utils/contact_config.py 的内置默认 (项目根 config/baseball_golf.json)

    if opt_range is not None:
        opt_start, opt_end = opt_range
        print(f"\n[Contact优化] 帧区间: [{opt_start}, {opt_end})")
        np.save(output_npy, root, allow_pickle=True)
        base, ext = os.path.splitext(output_npy)
        opt_contact_path = os.path.abspath(f"{base}_opt_contact{ext}")
        abs_output_npy = os.path.abspath(output_npy)
        sportgs_dir = os.path.join(_ROOT, 'model', 'SportGS')
        cmd = [
            sys.executable, 'train_contact.py',
            f'dataset.pose_path={abs_output_npy}',
            f'+dataset.opt_frame_start={opt_start}',
            f'+dataset.opt_frame_end={opt_end}',
            f'+dataset.export_data_path={abs_output_npy}',
            f'+dataset.export_output_path={opt_contact_path}',
            f'+dataset.seq_name={seq_name}',
            f'+dataset.obj_mesh_path={os.path.abspath(meter_stl_path)}',
            'wandb_disable=true',
        ]
        if opt_debug_every and opt_debug_every > 0:
            os.makedirs(opt_debug_dir, exist_ok=True)
            cmd += [
                f'+dataset.opt_debug_every={int(opt_debug_every)}',
                f'+dataset.opt_debug_dir={os.path.abspath(opt_debug_dir)}',
                f'+dataset.opt_debug_stage=contact',
            ]
        print(f"[Contact优化] 执行: cd {sportgs_dir} && {' '.join(cmd)}")
        subprocess.run(cmd, cwd=sportgs_dir, check=True, env=sportgs_env)
        root = np.load(opt_contact_path, allow_pickle=True).item()

    if force_closure_range is not None:
        fc_start, fc_end = force_closure_range
        print(f"\n[力闭合优化] 帧区间: [{fc_start}, {fc_end})")
        np.save(output_npy, root, allow_pickle=True)
        base, ext = os.path.splitext(output_npy)
        opt_fc_path = os.path.abspath(f"{base}_opt_force_closure{ext}")
        abs_output_npy = os.path.abspath(output_npy)
        sportgs_dir = os.path.join(_ROOT, 'model', 'SportGS')
        cmd = [
            sys.executable, 'finetune_force_closure.py',
            f'dataset.pose_path={abs_output_npy}',
            f'+dataset.opt_frame_start={fc_start}',
            f'+dataset.opt_frame_end={fc_end}',
            f'+dataset.export_data_path={abs_output_npy}',
            f'+dataset.export_output_path={opt_fc_path}',
            f'+dataset.seq_name={seq_name}',
            f'+dataset.obj_mesh_path={os.path.abspath(meter_stl_path)}',
            'wandb_disable=true',
        ]
        if opt_debug_every and opt_debug_every > 0:
            os.makedirs(opt_debug_dir, exist_ok=True)
            cmd += [
                f'+dataset.opt_debug_every={int(opt_debug_every)}',
                f'+dataset.opt_debug_dir={os.path.abspath(opt_debug_dir)}',
                f'+dataset.opt_debug_stage=fc',
            ]
        print(f"[力闭合优化] 执行: cd {sportgs_dir} && {' '.join(cmd)}")
        subprocess.run(cmd, cwd=sportgs_dir, check=True, env=sportgs_env)
        root = np.load(opt_fc_path, allow_pickle=True).item()

    if ref_frame is not None:
        rehand_by_object(root, seq_name, ref_frame)

    np.save(output_npy, root, allow_pickle=True)

    obj = root["data_dict"][seq_name]["params"]["object"]
    cam = root["data_dict"][seq_name]["params"]["camera"]
    print("\n========== 完成 ==========")
    print(f"输出 npy:     {output_npy}")
    print(f"seq_name:    {seq_name}")
    print(f"总帧数:       {len(root['imgnames'])}")
    print(f"imgpath:     {root['imgpath']}")
    print(f"文件缺失:     {stats['missing_file']}")
    print(f"右手缺失插值: {stats['missing_right']}")
    print(f"左手缺失插值: {stats['missing_left']}")
    print(f"右手 pose:   {r_pose_arr.shape}")
    print(f"obj_rot:     {obj['obj_rot'].shape}")
    print(f"camera.views: {cam['views']}")
    for i, (v, w2c, K) in enumerate(zip(cam['views'], cam['world2cam'], cam['K'])):
        print(f"  {v} K=\n{K}")
        print(f"  {v} world2cam=\n{w2c}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Golf capture -> HaMER + object -> npy (与 12-1.npy schema 对齐)"
    )
    parser.add_argument('--capture_dir', type=str, required=True,
                        help="例如 /data2/fubingshuai/golf/data/35_wood_8_01_fbs/20260424170424563；"
                             "支持帧目录直接位于 capture_dir 下或位于 .tmp_images/ 下")
    parser.add_argument('--output',   type=str, required=True, help="输出 npy 路径")
    parser.add_argument('--seq_name', type=str, required=True, help="序列名")
    parser.add_argument('--hamer_cam', type=str, default=None,
                        help="HaMER 推理用的相机 name（world=cam0）；默认挑第一个 1440x1080 彩色相机")
    parser.add_argument('--other_cam', type=str, default=None,
                        help="第二视角相机 name（cam1）；默认挑另一个 1440x1080 彩色相机")
    parser.add_argument('--obj_pose_sigma', type=float, default=0.0,
                        help="物体位姿时序高斯滤波 sigma（帧），0 表示不平滑")
    parser.add_argument('--point_r', type=float, nargs=3, default=None)
    parser.add_argument('--point_l', type=float, nargs=3, default=None)
    parser.add_argument('--opt_range', type=int, nargs=2, default=None,
                        metavar=('START', 'END'))
    parser.add_argument('--force_closure_range', type=int, nargs=2, default=None,
                        metavar=('START', 'END'))
    parser.add_argument('--ref_frame', type=int, default=None)
    parser.add_argument('--club_stl', type=str, default=None,
                        help="球杆 STL 路径 (单位: mm)。默认查找顺序: "
                             "<data_root>/club-assets/<club_name>/*.stl (允许去掉 _fbs 后缀), "
                             "然后回退到旧布局 <data_root>/club_asset/<club_name>.stl；"
                             "优化前会自动生成同目录的 <stem>_meter.stl")
    parser.add_argument('--contact_config', type=str, default=None,
                        help="预定义接触点 JSON (默认: golf-hand-object/config/baseball_golf.json); "
                             "通过 CONTACT_CONFIG_PATH 环境变量传给 SportGS, 决定 force-closure "
                             "用哪些 MANO tip 顶点 (hand_attract_tips 段落)")
    parser.add_argument('--init_method', type=str,
                        choices=['auto', 'hamer', 'multiview'], default='auto',
                        help="手部位姿初始化方法。"
                             "auto: 录制中有 ≥2 个 1440x1080 彩色相机时走 multiview, 否则 hamer。"
                             "multiview: 强制用 model/Hand_Estimation 多视角网络(HaMER 仅作 2D 伪标)。"
                             "hamer: 强制用 HaMER 单视角(默认旧行为)。")
    parser.add_argument('--mv_finetune_epochs', type=int, default=0,
                        help="multiview 分支专用。>0 时先用 HaMER 伪标在当前 capture 上"
                             "对 Hand_Estimation 自监督微调 N 个 epoch, 再用微调后的"
                             "权重跑推理。0 (默认) = 不微调, 直接用预训练权重")
    parser.add_argument('--mv_finetune_lr', type=float, default=1e-5,
                        help="--mv_finetune_epochs 启用时的学习率 (默认 1e-5)")
    parser.add_argument('--mv_finetune_bs', type=int, default=1,
                        help="--mv_finetune_epochs 启用时的 batch size (默认 1)")
    parser.add_argument('--opt_debug_every', type=int, default=0,
                        help="SportGS 优化每隔 N 步把当前 pose/obj 状态导出成 npy "
                             "快照, 方便 way_vis 看中间结果. 0 (默认) = 关闭")
    parser.add_argument('--opt_debug_dir', type=str,
                        default='/data2/fubingshuai/golf/test',
                        help="--opt_debug_every 启用时的快照输出目录")
    args = parser.parse_args()

    pr = np.array(args.point_r, dtype=np.float32) if args.point_r else None
    pl = np.array(args.point_l, dtype=np.float32) if args.point_l else None

    run(args.capture_dir, args.output, args.seq_name,
        hamer_cam=args.hamer_cam, other_cam=args.other_cam,
        obj_pose_sigma=args.obj_pose_sigma,
        point_r=pr, point_l=pl,
        opt_range=args.opt_range, force_closure_range=args.force_closure_range,
        ref_frame=args.ref_frame,
        club_stl=args.club_stl,
        contact_config=args.contact_config,
        init_method=args.init_method,
        mv_finetune_epochs=args.mv_finetune_epochs,
        mv_finetune_lr=args.mv_finetune_lr,
        mv_finetune_bs=args.mv_finetune_bs,
        opt_debug_every=args.opt_debug_every,
        opt_debug_dir=args.opt_debug_dir)
