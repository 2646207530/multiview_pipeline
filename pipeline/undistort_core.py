"""图像去畸变核心 (从 multiview_hand_init 抽出, 去掉手部估计相关内容).

prepare_undistort_dir: 按 camera_params.json 的相机内外参, 把 .tmp_images/<cam>/frame_*.jpg
去畸变写到 <capture>/.undistorted/<capture_id>/<cam_idx>/images_undistorted/000000.jpg,
并写 calib_undistorted/<cam_idx>.yaml (新 K + world->cam 的 R/t). world = 第 0 个相机 sensor 系.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm


def _newK_and_maps(K: np.ndarray, dist: np.ndarray, hw: Tuple[int, int]):
    H, W = hw
    new_K, _ = cv2.getOptimalNewCameraMatrix(
        K, dist, (W, H), alpha=0.0, newImgSize=(W, H))
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, None, new_K, (W, H), cv2.CV_16SC2)
    return new_K.astype(np.float64), map1, map2


def _write_calib_yaml(yaml_path: Path, new_K: np.ndarray,
                      R_w2c: np.ndarray, t_w2c: np.ndarray):
    """K + (R, t) world->this_cam in undistorted image plane."""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"K": new_K.tolist(), "R": R_w2c.tolist(), "t": t_w2c.reshape(3).tolist()}
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=None)


def _world2cam_RT(cams_dict, world_name: str, cam_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """world = world_name 的 sensor 系 -> cam_name 的 sensor 系。t 单位米。"""
    def _s2r(c):
        R_s2r = np.array(c["sensor2rig"]["rotation"], dtype=np.float64)
        t_s2r = np.array(c["sensor2rig"]["translation"], dtype=np.float64) / 1000.0
        return R_s2r, t_s2r
    R_w, t_w = _s2r(cams_dict[world_name])
    R_c, t_c = _s2r(cams_dict[cam_name])
    R_w2c = R_c.T @ R_w
    t_w2c = R_c.T @ (t_w - t_c)
    return R_w2c, t_w2c


def prepare_undistort_dir(capture_dir: Path, cams_dict: dict,
                          cam_names: Sequence[str], world_name: str,
                          frame_dirs: Dict[str, Path],
                          force: bool = False) -> Tuple[Path, str, Dict[str, np.ndarray]]:
    """生成/复用 <capture_dir>/.undistorted/. cam_names[0] = world(=cam0).
    frame_dirs: cam_name -> 原始帧目录 (frame_NNNNNN.jpg). 返回 (undist_root, capture_id, new_K_per_cam)."""
    undist_root = capture_dir / ".undistorted"
    capture_id = capture_dir.name
    new_K_per_cam: Dict[str, np.ndarray] = {}

    for cam_idx, cam_name in enumerate(cam_names):
        c = cams_dict[cam_name]
        H, W = int(c["image_height"]), int(c["image_width"])
        K = np.array(
            [[c["intrinsics"]["fx"], 0, c["intrinsics"]["cx"]],
             [0, c["intrinsics"]["fy"], c["intrinsics"]["cy"]],
             [0, 0, 1]], dtype=np.float64)
        dist = np.array(c["distortion_coeffs"], dtype=np.float64).reshape(-1)
        new_K, map1, map2 = _newK_and_maps(K, dist, (H, W))
        new_K_per_cam[cam_name] = new_K

        out_img_dir = undist_root / capture_id / str(cam_idx) / "images_undistorted"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        calib_dir = undist_root / capture_id / "calib_undistorted"
        calib_dir.mkdir(parents=True, exist_ok=True)
        R_w2c, t_w2c = _world2cam_RT(cams_dict, world_name, cam_name)
        _write_calib_yaml(calib_dir / f"{cam_idx}.yaml", new_K, R_w2c, t_w2c)

        src_dir = frame_dirs[cam_name]
        srcs = sorted(p for p in src_dir.iterdir()
                      if re.match(r"frame_\d+\.(?:jpg|jpeg|png|bmp)$", p.name, re.IGNORECASE))
        skipped = 0
        for src in tqdm(srcs, desc=f"undistort {cam_name}", leave=False):
            m = re.match(r"frame_(\d+)\.", src.name, re.IGNORECASE)
            if m is None:
                continue
            dst = out_img_dir / f"{int(m.group(1)):06d}.jpg"
            if dst.exists() and not force:
                skipped += 1
                continue
            img = cv2.imread(str(src))
            if img is None:
                continue
            und = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
            cv2.imwrite(str(dst), und, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[undistort] cam{cam_idx} ({cam_name}): {len(srcs)} frames, "
              f"{skipped} cached, {len(srcs)-skipped} undistorted")

    (undist_root / "pseudo_label_wilor").mkdir(parents=True, exist_ok=True)
    return undist_root, capture_id, new_K_per_cam
