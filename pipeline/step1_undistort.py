"""Step 1: 去畸变 + 写 calib_undistorted yaml.

完全 wrap ``multiview_hand_init.prepare_undistort_dir``. 它本来就把所有事做完:
  - 算 newK (cv2.getOptimalNewCameraMatrix)
  - 写去畸变后的 jpg 到 ``.undistorted/<seq>/<cam_idx>/images_undistorted/``
  - 写 calib_undistorted/<cam_idx>.yaml (含 newK + R/t)

Web step 这边只是接进度 + 返回预览路径.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from run_golf_capture_to_npy import (  # type: ignore
    _load_camera_params,
    _resolve_color_cams,
    _frame_dir_for,
)
from multiview_hand_init import prepare_undistort_dir  # type: ignore

from .state import PipelineState
from .workspace import Workspace


def run(ws: Workspace, force: bool = False) -> Dict[str, Any]:
    cap = ws.capture_dir
    cams = _load_camera_params(cap)
    hamer_name, other_name = _resolve_color_cams(cams, cap)
    cam_names = [hamer_name] + ([other_name] if other_name else [])

    # 找每个 cam 的原始帧目录
    frame_dirs_orig = {n: _frame_dir_for(n, cap) for n in cam_names}
    missing = [n for n, fd in frame_dirs_orig.items() if fd is None]
    if missing:
        raise RuntimeError(f"以下相机找不到原始帧目录: {missing}")

    # capture_id = seq_name 兼容 (prepare_undistort_dir 用它作 .undistorted 下的子目录名)
    undist_root, capture_id, newKs = prepare_undistort_dir(
        cap, cams, cam_names, hamer_name,
        frame_dirs_orig, force=force,
    )

    # 给前端找张预览图: cam0 的第一张去畸变帧
    cam0_dir = Path(undist_root) / capture_id / "0" / "images_undistorted"
    sample_jpg = next(iter(sorted(cam0_dir.glob("*.jpg"))), None)
    n_frames_per_cam = {
        # 字符串 key (orjson 不接 int key, gradio 序列化会炸)
        str(i): sum(1 for _ in (Path(undist_root) / capture_id / str(i) /
                                 "images_undistorted").glob("*.jpg"))
        for i in range(len(cam_names))
    }

    info: Dict[str, Any] = {
        "undist_root":     str(undist_root),
        "capture_id":      capture_id,
        "cam_names":       cam_names,
        "newKs":           {n: K.tolist() for n, K in newKs.items()},
        "n_frames_per_cam": n_frames_per_cam,
        "sample_jpg":      str(sample_jpg) if sample_jpg else None,
    }

    state = PipelineState.load(ws)
    state.mark_done("undistort", **info)
    state.save(ws)
    return info
