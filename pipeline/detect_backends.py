"""Step 2 检测后端: YOLO (原逻辑) / SAM2 (序列视频分割).

接口: backend.detect_all(undist_root, capture_id, n_cams, progress, prompts)
返回 ``{cam_idx_str: {frame_id_str: [["right", [x1,y1,x2,y2]], ["left", ...]]}}``,
跟 step2 之前的 detections.json 完全同构, 上游 step3 不感知后端.

YOLO: 每帧独立检测.
SAM2: 用户在首帧给每只手标若干 (正/负) 点 → init video predictor → propagate →
每帧每手 mask → 用 bounding box of nonzero pixels 当 bbox.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))


# ── 工具 ────────────────────────────────────────────────────────────────
def _mask_to_bbox(mask: np.ndarray) -> Optional[List[float]]:
    """二值 mask (H, W) → [x1, y1, x2, y2] (含两端). 空 mask 返回 None."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return [float(xs.min()), float(ys.min()),
            float(xs.max()), float(ys.max())]


def _frames_for_cam(undist_root: Path, capture_id: str, ci: int) -> Path:
    return undist_root / capture_id / str(ci) / "images_undistorted"


# ── 接口 ────────────────────────────────────────────────────────────────
class DetectBackend:
    name: str = "base"

    def detect_all(self, undist_root: Path, capture_id: str, n_cams: int,
                   progress: Optional[Callable[[float, str], None]] = None,
                   prompts: Optional[dict] = None
                   ) -> Dict[str, Dict[str, list]]:
        raise NotImplementedError


# ── YOLO ────────────────────────────────────────────────────────────────
class YoloBackend(DetectBackend):
    name = "yolo"

    def __init__(self):
        from yolo.detector import Detector  # type: ignore
        from config.yolo_config import yolo_opt  # type: ignore
        from run_hamer_to_npy import parse_detections  # type: ignore
        self._detector = Detector(yolo_opt)
        self._parse_detections = parse_detections

    def detect_all(self, undist_root, capture_id, n_cams, progress=None,
                   prompts=None):
        bbox_data: Dict[str, Dict[str, list]] = {str(c): {} for c in range(n_cams)}

        tasks = []
        for ci in range(n_cams):
            d = _frames_for_cam(undist_root, capture_id, ci)
            for p in sorted(d.glob("*.jpg")):
                try:
                    fid = int(p.stem)
                except ValueError:
                    continue
                tasks.append((ci, fid, p))

        total = len(tasks)
        for k, (ci, fid, img_path) in enumerate(tasks):
            if progress and k % 20 == 0:
                progress(k / max(total, 1),
                         f"YOLO detect cam{ci} frame {fid:06d}  ({k}/{total})")
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            _, dets = self._detector.detect(img)
            bboxes = self._parse_detections(dets)
            clean = []
            for entry in bboxes:
                if not entry or len(entry) < 2:
                    continue
                if entry[0] not in ("right", "left"):
                    continue
                clean.append([entry[0], [float(x) for x in entry[1]]])
            if clean:
                bbox_data[str(ci)][f"{fid:06d}"] = clean
        return bbox_data


# ── SAM2 ────────────────────────────────────────────────────────────────
class Sam2Backend(DetectBackend):
    name = "sam2"

    _SAM2_ROOT = _PROJECT / "model" / "sam2"
    # hydra config 名: sam2/__init__.py 已 initialize_config_module("sam2"),
    # 所以这里相对 sam2 package 写.
    _CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    _CKPT_REL = "checkpoints/sam2.1_hiera_large.pt"
    _OBJ_ID_RIGHT = 0
    _OBJ_ID_LEFT = 1
    _LABEL_FOR_OBJ = {0: "right", 1: "left"}

    def __init__(self):
        import torch

        # 先释放 sam2_interactive 里残留的 image predictor + base model,
        # 否则 video predictor 再加载一份 backbone 会 OOM.
        try:
            from .sam2_interactive import release_all as _release_interactive
            _release_interactive()
        except Exception:
            pass

        # SAM2 的 build_sam.py 有个 shadow-check 在我们的 editable-install
        # 布局下会误报 (见 sam2_interactive._import_sam2_with_shadow_bypass).
        import os
        import sam2  # type: ignore
        shadow_target = os.path.join(sam2.__path__[0], "sam2")
        orig_isdir = os.path.isdir
        os.path.isdir = lambda p: False if p == shadow_target else orig_isdir(p)
        try:
            from sam2.build_sam import build_sam2_video_predictor  # type: ignore
        finally:
            os.path.isdir = orig_isdir

        # 确保 hydra 是为 'sam2' config module 初始化的
        from .sam2_interactive import ensure_sam2_hydra
        ensure_sam2_hydra()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = self._SAM2_ROOT / self._CKPT_REL
        if not ckpt.is_file():
            raise RuntimeError(
                f"SAM2 权重不存在: {ckpt}. 跑 model/sam2/checkpoints/download_ckpts.sh"
            )
        self.predictor = build_sam2_video_predictor(
            self._CFG, str(ckpt), device=device,
        )
        self._torch = torch

    def release(self) -> None:
        """卸 video predictor + 清 CUDA cache. detect_all 收尾会自动调."""
        if getattr(self, "predictor", None) is not None:
            del self.predictor
            self.predictor = None
        import gc
        gc.collect()
        try:
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _ensure_prompts(prompts_for_cam) -> Dict[str, Dict[str, list]]:
        """规整成 {"right": {"pos":[..], "neg":[..]}, "left": {...}}."""
        out: Dict[str, Dict[str, list]] = {
            "right": {"pos": [], "neg": []},
            "left":  {"pos": [], "neg": []},
        }
        if not prompts_for_cam:
            return out
        for hand in ("right", "left"):
            sub = prompts_for_cam.get(hand) or {}
            out[hand]["pos"] = list(sub.get("pos") or [])
            out[hand]["neg"] = list(sub.get("neg") or [])
        return out

    def _detect_one_cam(self, frames_dir: Path,
                        prompts_for_cam: dict,
                        progress: Optional[Callable[[float, str], None]],
                        progress_base: float, progress_scale: float,
                        ci: int) -> Dict[str, list]:
        torch = self._torch
        prompts = self._ensure_prompts(prompts_for_cam)

        # 必须至少一只手有正点, 否则没东西可分
        active_hands = [
            h for h in ("right", "left") if prompts[h]["pos"]
        ]
        if not active_hands:
            raise RuntimeError(
                f"cam{ci}: SAM2 至少需要给一只手标一个正点 (positive). "
                f"右手/左手都没有正点."
            )

        # 帧名 → 帧索引 (init_state 内部按文件名数值升序排; 我们也用同样规则)
        jpgs = sorted([p for p in frames_dir.glob("*.jpg")],
                      key=lambda p: int(p.stem))
        if not jpgs:
            return {}
        # SAM2 里 frame_idx = 0..N-1, 对应到我们的 frame_id_str (06d)
        frame_id_for_idx = [p.stem for p in jpgs]

        with torch.inference_mode():
            # offload_video_to_cpu: 1592 帧 * 1024^2 * 3byte ~ 5GB,
            #   放 CPU 大幅省显存. 慢 ~10-15% (PCIe 拷贝).
            # offload_state_to_cpu: per-frame inference state 也搬 CPU, 再省一些,
            #   慢 ~10%.
            state = self.predictor.init_state(
                video_path=str(frames_dir),
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
            )
            self.predictor.reset_state(state)

            for hand in active_hands:
                pos = prompts[hand]["pos"]
                neg = prompts[hand]["neg"]
                pts = pos + neg
                lbls = [1] * len(pos) + [0] * len(neg)
                obj_id = (self._OBJ_ID_RIGHT if hand == "right"
                          else self._OBJ_ID_LEFT)
                self.predictor.add_new_points_or_box(
                    state, frame_idx=0, obj_id=obj_id,
                    points=np.array(pts, dtype=np.float32),
                    labels=np.array(lbls, dtype=np.int32),
                )

            out: Dict[str, list] = {}
            total = len(jpgs)
            for frame_idx, obj_ids, video_res_masks in (
                    self.predictor.propagate_in_video(state)):
                # video_res_masks: (num_obj, 1, H, W) logits
                if progress and frame_idx % 20 == 0:
                    progress(progress_base + progress_scale * (frame_idx / max(total, 1)),
                             f"SAM2 cam{ci} frame {frame_idx:06d}  ({frame_idx}/{total})")
                per_frame: list = []
                for oid, ml in zip(obj_ids, video_res_masks):
                    label = self._LABEL_FOR_OBJ.get(int(oid))
                    if label is None:
                        continue
                    mask = (ml[0] > 0.0).detach().cpu().numpy()
                    bbox = _mask_to_bbox(mask)
                    if bbox is None:
                        continue
                    per_frame.append([label, bbox])
                if per_frame:
                    fid_str = frame_id_for_idx[frame_idx]
                    out[fid_str] = per_frame
        return out

    def detect_all(self, undist_root, capture_id, n_cams, progress=None,
                   prompts=None):
        prompts = prompts or {}
        bbox_data: Dict[str, Dict[str, list]] = {str(c): {} for c in range(n_cams)}
        per_cam = 1.0 / max(n_cams, 1)
        try:
            for ci in range(n_cams):
                cam_dir = _frames_for_cam(undist_root, capture_id, ci)
                cam_prompts = prompts.get(str(ci)) or prompts.get(ci) or {}
                if progress:
                    progress(per_cam * ci, f"SAM2 init cam{ci}...")
                bbox_data[str(ci)] = self._detect_one_cam(
                    cam_dir, cam_prompts, progress,
                    progress_base=per_cam * ci, progress_scale=per_cam,
                    ci=ci,
                )
        finally:
            self.release()
        return bbox_data


_REGISTRY = {"yolo": YoloBackend, "sam2": Sam2Backend}
VALID_BACKENDS: Tuple[str, ...] = tuple(_REGISTRY.keys())


def make_backend(name: str) -> DetectBackend:
    key = (name or "yolo").lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown detect backend: {name!r} (valid: {list(_REGISTRY)})"
        )
    return _REGISTRY[key]()
