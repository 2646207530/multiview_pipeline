"""Step 3: 用 step2 的 bbox 喂 HaMER 拿 21 个 2D 关节, 写伪标 npz.

输入: workspace/detections.json + 去畸变图.
输出: ``.undistorted/pseudo_label_wilor/<seq>_<cam>_<frame>_<hand>.npz``
       字段: is_right (1,), joints_2d (21, 2)
末尾: 调 make_pseudo_video 出全帧拼接 mp4 + jpg.

注意: 跟现有 ``write_pseudo_labels_from_hamer`` 的 HaMER 部分等价, 只是不再
现场跑 YOLO (改成读 step2 的 json).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
import yaml

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from model.hamer.infer import hamer_inference  # type: ignore
from model.config.hamer_config import hamer_opt  # type: ignore
from multiview_hand_init import make_pseudo_video  # type: ignore

from .state import PipelineState
from .workspace import Workspace


def _load_newK_from_yaml(yaml_path: Path) -> np.ndarray:
    with open(yaml_path) as f:
        d = yaml.safe_load(f)
    return np.array(d["K"], dtype=np.float32)


def run(ws: Workspace, progress: Optional[Callable[[float, str], None]] = None,
        force: bool = False, make_video: bool = True) -> Dict[str, Any]:
    state = PipelineState.load(ws)
    if state.steps.get("detect").status != "done":
        raise RuntimeError("Step 2 (detect) 没完成")
    undist = state.steps["undistort"].outputs
    detect = state.steps["detect"].outputs

    undist_root = Path(undist["undist_root"])
    capture_id = undist["capture_id"]
    cam_names = undist["cam_names"]
    n_cams = len(cam_names)

    bbox_data: Dict[str, Dict[str, list]] = json.loads(
        Path(detect["detections_json"]).read_text())

    # 准备 HaMER
    if progress:
        progress(0.0, "加载 HaMER (~10-30s)...")
    hamer = hamer_inference(hamer_opt)

    # 准备 K (每个 cam 一个)
    Ks: Dict[int, np.ndarray] = {}
    for ci in range(n_cams):
        Ks[ci] = _load_newK_from_yaml(ws.undist_calib_yaml(ci))

    out_dir = ws.pseudo_label_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 总任务量
    total_hands = sum(len(frame_dict) * 2  # 每帧最多 2 只手
                      for ci, frame_dict in bbox_data.items())
    processed = 0
    n_ok = {str(ci): 0 for ci in range(n_cams)}
    n_skip = 0

    for ci_str, frame_dict in bbox_data.items():
        ci = int(ci_str)
        K = Ks[ci]
        img_dir = undist_root / capture_id / ci_str / "images_undistorted"
        for fid_str, bboxes in sorted(frame_dict.items()):
            try:
                fid = int(fid_str)
            except ValueError:
                continue
            img_path = img_dir / f"{fid:06d}.jpg"
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            seen = set()
            for entry in bboxes:
                if not entry or len(entry) < 2:
                    continue
                hand_label = entry[0]
                if hand_label not in ("right", "left") or hand_label in seen:
                    continue
                seen.add(hand_label)
                hid = 0 if hand_label == "right" else 1
                p = out_dir / f"{capture_id}_{ci}_{fid}_{hid}.npz"
                processed += 1
                if p.exists() and not force:
                    n_skip += 1
                    continue
                bbox_for_hamer = [hand_label, entry[1]]
                try:
                    out, _ = hamer.estimate_from_rgb(image, [bbox_for_hamer], K)
                except Exception as e:
                    print(f"[step3] HaMER fail cam{ci} f{fid} {hand_label}: {e}")
                    continue
                kp2d = out.get("pred_keypoints_2d_full")
                if kp2d is None:
                    continue
                kp2d = kp2d.detach().cpu().numpy().squeeze().astype(np.float32)
                if kp2d.ndim != 2 or kp2d.shape[1] != 2:
                    continue
                np.savez(p,
                         is_right=np.array([1.0 if hid == 0 else 0.0],
                                           dtype=np.float32),
                         joints_2d=kp2d)
                n_ok[str(ci)] += 1
                if progress and processed % 20 == 0:
                    progress(processed / max(total_hands, 1),
                             f"HaMER  cam{ci} f{fid:06d} {hand_label}"
                             f"  ({processed}/{total_hands}, skip={n_skip})")

    # 生成可视化视频 + 每帧 jpg (拼 cam0|cam1)
    video_path: Optional[str] = None
    if make_video:
        if progress:
            progress(0.95, "生成 _pseudo_vis 视频...")
        from pathlib import Path as _Path
        # frame_dirs_undist & cam_idx_of dict (跟 make_pseudo_video 接口对齐)
        cam_idx_of = {n: i for i, n in enumerate(cam_names)}
        frame_dirs_undist = {
            n: undist_root / capture_id / str(cam_idx_of[n]) / "images_undistorted"
            for n in cam_names
        }
        total_frames = max(undist["n_frames_per_cam"].values())
        try:
            out_path = make_pseudo_video(
                undist_root, capture_id, cam_names, cam_idx_of,
                total_frames, frame_dirs_undist,
                fps=10, downscale=2, save_per_frame_jpg=True,
            )
            video_path = str(out_path) if out_path else None
        except Exception as e:
            print(f"[step3] 生成伪标视频失败: {e}")

    info = {
        "pseudo_label_dir": str(out_dir),
        "n_pseudo_per_cam": n_ok,
        "pseudo_overlay_mp4": video_path,
    }
    state.mark_done("pseudo", **info)
    state.save(ws)
    return info
