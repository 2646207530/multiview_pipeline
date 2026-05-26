"""Step 2: 双手检测 (YOLO 逐帧 / SAM2 序列分割).

每帧每相机跑后端, 输出 ``workspace/detections.json``:

{
  "0": {                           # cam_idx
    "000000": [["right", [x1,y1,x2,y2]], ["left", [x1,y1,x2,y2]]],
    "000001": [...],
    ...
  },
  "1": {...}
}

预览: 每相机拼一段 overlay mp4 给前端展示.

后端通过 ``backend`` 参数选 ("yolo" / "sam2"). SAM2 需要用户在首帧给每只手
打 (正/负) 点 prompt; 通过 ``sam2_prompts`` 传入, 形如:
{
  "0": {  # cam_idx_str
    "right": {"pos": [[x,y], ...], "neg": [[x,y], ...]},
    "left":  {"pos": [...], "neg": [...]},
  },
  "1": {...}
}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from .detect_backends import make_backend, VALID_BACKENDS
from .state import PipelineState
from .workspace import Workspace


def _draw_overlay(img: np.ndarray, bboxes: List[list]) -> np.ndarray:
    out = img.copy()
    for entry in bboxes:
        if not entry or len(entry) < 2:
            continue
        label, bbox = entry[0], entry[1]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = (0, 255, 0) if label == "right" else (64, 128, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


def _write_overlay_video(cam_dir: Path,
                          bbox_for_cam: Dict[str, list],
                          out_mp4: Path,
                          fps: int = 60) -> Optional[str]:
    """对一个相机文件夹的所有 frame_*.jpg 顺序拼成 overlay mp4."""
    jpgs = sorted(cam_dir.glob("*.jpg"))
    if not jpgs:
        return None
    first = cv2.imread(str(jpgs[0]))
    if first is None:
        return None
    h, w = first.shape[:2]
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))
    if not vw.isOpened():
        return None
    for jpg in jpgs:
        img = cv2.imread(str(jpg))
        if img is None:
            continue
        boxes = bbox_for_cam.get(jpg.stem, [])
        vw.write(_draw_overlay(img, boxes) if boxes else img)
    vw.release()
    return str(out_mp4) if out_mp4.exists() else None


def _expand_bbox(bbox, scale: float, img_w: int, img_h: int):
    """以中心不变, 长宽各乘 scale, clip 到 [0, W/H]."""
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    nx1 = max(0.0,           cx - w / 2.0)
    ny1 = max(0.0,           cy - h / 2.0)
    nx2 = min(float(img_w),  cx + w / 2.0)
    ny2 = min(float(img_h),  cy + h / 2.0)
    return [nx1, ny1, nx2, ny2]


def _expand_bbox_data(bbox_data: Dict[str, Dict[str, list]],
                       scale: float, undist_root: Path, capture_id: str
                       ) -> Dict[str, Dict[str, list]]:
    """对 bbox_data 里所有 bbox 做 scale 扩张, clip 到原图尺寸."""
    if scale == 1.0:
        return bbox_data
    # 用每个 cam 的第一帧 jpg 拿图像尺寸 (同一序列同 cam 所有帧同尺寸)
    img_size_cache: Dict[str, tuple] = {}
    for ci_str, frame_dict in bbox_data.items():
        if not frame_dict:
            continue
        if ci_str not in img_size_cache:
            cam_dir = undist_root / capture_id / ci_str / "images_undistorted"
            first_jpg = next(iter(sorted(cam_dir.glob("*.jpg"))), None)
            if first_jpg is None:
                img_size_cache[ci_str] = (10**6, 10**6)  # no clip
            else:
                im = cv2.imread(str(first_jpg))
                img_size_cache[ci_str] = (im.shape[1], im.shape[0]) if im is not None else (10**6, 10**6)
        W, H = img_size_cache[ci_str]
        for fid_str, bboxes in frame_dict.items():
            for entry in bboxes:
                if not entry or len(entry) < 2:
                    continue
                entry[1] = _expand_bbox(entry[1], scale, W, H)
    return bbox_data


def run(ws: Workspace, progress: Optional[Callable[[float, str], None]] = None,
        force: bool = False, backend: str = "yolo",
        sam2_prompts: Optional[dict] = None,
        bbox_size: float = 1.0) -> Dict[str, Any]:
    if backend not in VALID_BACKENDS:
        raise ValueError(f"backend 必须是 {VALID_BACKENDS} 之一, 收到 {backend!r}")

    state = PipelineState.load(ws)
    undist = state.steps.get("undistort", None)
    if undist is None or undist.status != "done":
        raise RuntimeError("Step 1 (undistort) 没完成, 不能跑检测")
    undist_root = Path(undist.outputs["undist_root"])
    capture_id = undist.outputs["capture_id"]
    n_cams = len(undist.outputs["cam_names"])

    # 已存在的 detections.json + 不 force, 直接返回 (idempotent)
    # NOTE: 这条 idempotent 路径用的是已经写过的 bbox_data, 不会再二次扩张;
    #       想换 bbox_size 就勾上 Force re-run.
    if ws.detections_json.exists() and not force:
        bbox_data = json.loads(ws.detections_json.read_text())
    else:
        if progress:
            progress(0.0, f"加载检测后端 {backend}...")
        detector_backend = make_backend(backend)
        bbox_data = detector_backend.detect_all(
            undist_root, capture_id, n_cams,
            progress=progress,
            prompts=sam2_prompts if backend == "sam2" else None,
        )

        # 在写入 detections.json 之前统一加 patch (中心不变, w/h 各 * bbox_size, clip 到原图)
        if bbox_size != 1.0:
            if progress:
                progress(0.98, f"expand bbox × {bbox_size:.2f} ...")
            bbox_data = _expand_bbox_data(
                bbox_data, float(bbox_size), undist_root, capture_id)

        ws.detections_json.parent.mkdir(parents=True, exist_ok=True)
        ws.detections_json.write_text(json.dumps(bbox_data, ensure_ascii=False))

    # 每相机生成一个完整 overlay mp4
    vis_videos: Dict[str, Optional[str]] = {}
    for ci in range(n_cams):
        if progress:
            progress(0.99, f"writing overlay video cam{ci} ...")
        cam_dir = undist_root / capture_id / str(ci) / "images_undistorted"
        out_mp4 = ws.detect_vis_dir / f"detect_cam{ci}.mp4"
        # force 或者 mp4 不存在时重生成
        if force or not out_mp4.exists():
            vis_videos[str(ci)] = _write_overlay_video(
                cam_dir, bbox_data.get(str(ci), {}), out_mp4)
        else:
            vis_videos[str(ci)] = str(out_mp4)

    n_detected_frames = {
        # 字符串 key (orjson / gradio JSON 不接 int)
        str(ci): len(bbox_data.get(str(ci), {})) for ci in range(n_cams)
    }

    info = {
        "detections_json":   str(ws.detections_json),
        "n_detected_frames": n_detected_frames,
        "vis_videos":        vis_videos,
        # 保留 sample_preview 字段兼容前端; 指向 cam0 视频
        "sample_preview":    vis_videos.get("0"),
        "backend":           backend,
        "bbox_size":         float(bbox_size),
    }
    state.mark_done("detect", **info)
    state.save(ws)
    return info
