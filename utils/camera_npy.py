"""相机参数 / 几何辅助函数 (从旧 CLI ``run_golf_capture_to_npy.py`` 抽出).

被 pipeline 的 step0 (setup) / step1 (undistort) / step5 (inference) 复用:
  - 读 ``camera_params.json`` 拿相机内外参
  - 在采集目录里挑出两台 1440x1080 彩色相机 (cam0/cam1)
  - 把物体轨迹 csv 的位姿变换到世界系
  - 组装 npy 里的 camera block (world2cam / K / views)

纯几何 + numpy/scipy, 不依赖任何检测/估计模型, 可独立 import.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

# 采集帧目录后缀 (1440x1080 BayerRG8 120fps). _frame_dir_for / _resolve_color_cams 用.
FRAME_DIR_SUFFIX = "_w1440_h1080_pBayerRG8_f120"


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


def _frame_dir_for(cam_name, capture_dir):
    # 目录名含括号，glob 用的 [] 要 escape；直接拼。
    # 兼容两种布局：旧的 capture_dir/<cam>...，新的 capture_dir/.tmp_images/<cam>...
    sub = f"{cam_name}{FRAME_DIR_SUFFIX}"
    for base in (capture_dir / ".tmp_images", capture_dir):
        p = base / sub
        if p.is_dir():
            return p
    return None


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
