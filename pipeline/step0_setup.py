"""Step 0: setup workspace.

输入: capture_dir + seq_name.
做的事:
  - 探测 capture_dir/camera_params.json 是否存在 + 合法
  - 选出 2 台 1440x1080 彩色相机 (cam0=master, cam1=other)
  - 创建 workspace 目录
  - 返回探测结果给前端展示
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# 把 pipeline 项目根加到 path, 让 import 老 wrapper 模块能成功
_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

# 复用 run_golf_capture_to_npy 里现成的函数
from run_golf_capture_to_npy import (  # type: ignore
    _load_camera_params,
    _resolve_color_cams,
    _frame_dir_for,
    _K_from_cam,
)

from .state import PipelineState
from .workspace import Workspace
from .raw_extract import find_raw_files, detect_status, extract_all


def run(capture_dir: str, seq_name: str,
        progress: Optional[Any] = None,
        auto_extract_raw: bool = True) -> Dict[str, Any]:
    """探测 + 初始化 workspace, 返回结构化信息给前端展示."""
    cap = Path(capture_dir).expanduser().resolve()
    if not cap.is_dir():
        raise FileNotFoundError(f"capture_dir 不存在: {cap}")
    if not seq_name:
        raise ValueError("seq_name 不能为空")

    # 防御: 用户可能误把 .tmp_images / .undistorted / trajectory_output 等
    # 子目录补进 capture_dir, 自动剥掉, 用真实 capture 根目录
    while cap.name in (".tmp_images", ".undistorted", ".pipeline",
                       "trajectory_output"):
        print(f"[setup] 剥掉 capture_dir 末尾的 '{cap.name}' → 用父目录")
        cap = cap.parent

    cam_json = cap / "camera_params.json"
    if not cam_json.exists():
        raise FileNotFoundError(f"找不到 camera_params.json: {cam_json}")

    # ─── 检测并自动解 .raw → .jpg (新加) ─────────────────────────────
    # 如果 capture_dir 直下有 .raw 但还没解过, 默认自动 extract 到 .tmp_images/
    # 让下游 _frame_dir_for 能找到 frame_*.jpg.
    raw_status = detect_status(cap)
    raw_summary: Optional[dict] = None
    if raw_status:
        any_missing = any(v["needs_extract"] for v in raw_status.values())
        if any_missing and auto_extract_raw:
            print(f"[setup] 检测到 {len(raw_status)} 个 raw, 有未解码的, 开始抽帧 → "
                  f"{cap / '.tmp_images'}")

            def _p(frac, msg):
                if progress is not None:
                    try:
                        progress(frac, desc=f"raw→jpg: {msg}")
                    except TypeError:  # 非 gradio progress 对象
                        progress(frac, msg)
            extract_result = extract_all(cap, progress=_p, force=False)
            raw_summary = {"extracted": extract_result,
                            "tmp_images_dir": str(cap / ".tmp_images")}
        else:
            raw_summary = {"skipped (already extracted)": {
                k: v["n_jpg"] for k, v in raw_status.items()}}

    # 现在再去探测相机 (此时 frame 目录应该已存在)
    cams = _load_camera_params(cap)
    hamer_name, other_name = _resolve_color_cams(cams, cap)

    # 探测各相机帧目录
    cam_info = []
    for idx, name in enumerate([hamer_name, other_name]):
        if name is None:
            continue
        fd = _frame_dir_for(name, cap)
        n_frames = 0 if fd is None else sum(
            1 for p in fd.iterdir() if p.suffix.lower() in (".jpg", ".raw"))
        K = _K_from_cam(cams[name])
        cam_info.append({
            "cam_idx":   idx,
            "role":      "master (world)" if idx == 0 else "other",
            "name":      name,
            "frame_dir": str(fd) if fd else None,
            "n_frames":  n_frames,
            "K":         K.tolist(),
        })

    # 创建 workspace
    ws = Workspace(capture_dir=cap, seq_name=seq_name)
    ws.ensure_dirs()

    info: Dict[str, Any] = {
        "capture_dir":  str(cap),
        "seq_name":     seq_name,
        "workspace":    str(ws.root),
        "camera_params_json": str(cam_json),
        "raw_extract":  raw_summary,
        "cams":         cam_info,
    }

    # 初始化 / load state
    state = PipelineState.load(ws)
    state.capture_dir = str(cap)
    state.seq_name    = seq_name
    state.mark_done("setup", **info)
    state.save(ws)

    return info
