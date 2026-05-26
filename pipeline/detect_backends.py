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

    # 两只手 mask 重叠超过这个 IoU 就认为是 SAM2 latch 错: 一只手被另一只遮挡时,
    # SAM2 经常把被遮挡那只 ghost 到 visible 那只的位置, 两只 mask 几乎重合.
    # 阈值 0.5: 真实的"手交叉/握"场景 mask IoU 也很少超过 0.3-0.4, > 0.5 几乎必错.
    _INTER_HAND_IOU_DROP = 0.5

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
    def _ensure_prompts(prompts_for_cam) -> dict:
        """规整成 {"anchors": {frame_idx: {"right": {"pos":[..],"neg":[..]}, "left": {...}}}}.

        一个 cam 可以在多帧上同时打 anchor; 都会作为 SAM2 video predictor 的
        conditioning frames, 然后 propagate 从 frame 0 一口气跑到尾."""
        out: dict = {"anchors": {}}
        if not prompts_for_cam:
            return out
        raw_anchors = prompts_for_cam.get("anchors") or {}
        norm: Dict[int, dict] = {}
        for fi, frame_data in raw_anchors.items():
            try:
                fi_int = int(fi)
            except (TypeError, ValueError):
                continue
            entry = {"right": {"pos": [], "neg": []},
                     "left":  {"pos": [], "neg": []}}
            for hand in ("right", "left"):
                sub = (frame_data or {}).get(hand) or {}
                entry[hand]["pos"] = list(sub.get("pos") or [])
                entry[hand]["neg"] = list(sub.get("neg") or [])
            # 只保留至少有一个点 (正或负) 的 entry
            if any(entry[h]["pos"] or entry[h]["neg"]
                   for h in ("right", "left")):
                norm[fi_int] = entry
        out["anchors"] = norm
        return out

    def _detect_one_cam(self, frames_dir: Path,
                        prompts_for_cam: dict,
                        progress: Optional[Callable[[float, str], None]],
                        progress_base: float, progress_scale: float,
                        ci: int) -> Dict[str, list]:
        torch = self._torch
        prompts = self._ensure_prompts(prompts_for_cam)
        anchors = prompts["anchors"]

        if not anchors:
            raise RuntimeError(
                f"cam{ci}: 没有任何 anchor 帧, 至少要在一帧上给一只手标 ≥1 个正点."
            )
        # 至少要有一只手在某一帧上有正点
        has_any_pos = any(
            anchors[fi][hand]["pos"]
            for fi in anchors
            for hand in ("right", "left")
        )
        if not has_any_pos:
            raise RuntimeError(
                f"cam{ci}: 所有 anchor 都没有正点, SAM2 没法判断对象."
            )

        jpgs = sorted([p for p in frames_dir.glob("*.jpg")],
                      key=lambda p: int(p.stem))
        if not jpgs:
            return {}
        total = len(jpgs)
        for fi in anchors:
            if fi < 0 or fi >= total:
                raise RuntimeError(
                    f"cam{ci}: anchor frame_idx={fi} 越界 (0..{total - 1})"
                )
        frame_id_for_idx = [p.stem for p in jpgs]

        out: Dict[str, list] = {}

        def _obj_score_for(inference_state, obj_idx: int, frame_idx: int) -> float:
            """从 inference_state 里拿这一帧某 obj 的 object_score_logits.
            SAM2 已经把 score<0 的 obj 的 mask clamp 成 NO_OBJ_SCORE, 所以
            mask 已经空了; 这里拿 score 主要是为了两手重叠时做 tie-break."""
            d = inference_state["output_dict_per_obj"][obj_idx]
            cur = d["cond_frame_outputs"].get(frame_idx)
            if cur is None:
                cur = d["non_cond_frame_outputs"].get(frame_idx)
            if cur is None or "object_score_logits" not in cur:
                return float("-inf")
            try:
                return float(cur["object_score_logits"].flatten()[0])
            except Exception:
                return float("-inf")

        def _consume(inference_state, frame_idx, obj_ids, video_res_masks):
            if progress and frame_idx % 20 == 0:
                progress(progress_base + progress_scale * (frame_idx / max(total, 1)),
                         f"SAM2 cam{ci} frame {frame_idx:06d}  "
                         f"({frame_idx}/{total}, anchors={sorted(anchors.keys())})")
            from .sam2_interactive import keep_largest_cc

            # per_hand: label -> (mask_bool, bbox, obj_score)
            per_hand: Dict[str, tuple] = {}
            for obj_idx_in_batch, (oid, ml) in enumerate(zip(obj_ids, video_res_masks)):
                label = self._LABEL_FOR_OBJ.get(int(oid))
                if label is None:
                    continue
                mask = (ml[0] > 0.0).detach().cpu().numpy()
                # 抹掉远处的杂散小连通块
                mask = keep_largest_cc(mask)
                bbox = _mask_to_bbox(mask)
                if bbox is None:
                    continue
                score = _obj_score_for(inference_state, obj_idx_in_batch, frame_idx)
                per_hand[label] = (mask, bbox, score)

            # 两只手都有 bbox 时, 看 mask IoU. 高于阈值说明 SAM2 把两个 obj 锁到
            # 同一个东西上 (典型: 一只手被另一只遮挡, 被遮挡那只 ghost 过去).
            # 丢掉 object_score 较低那只.
            if "right" in per_hand and "left" in per_hand:
                mr, _, sr = per_hand["right"]
                ml_, _, sl = per_hand["left"]
                inter = int((mr & ml_).sum())
                if inter > 0:
                    union = int((mr | ml_).sum())
                    iou = inter / max(union, 1)
                    if iou >= self._INTER_HAND_IOU_DROP:
                        loser = "left" if sl <= sr else "right"
                        per_hand.pop(loser)

            if per_hand:
                fid_str = frame_id_for_idx[frame_idx]
                out[fid_str] = [[label, bbox]
                                 for label, (_m, bbox, _s) in per_hand.items()]

        # SAM2 的 maskmem_features 缓存为 bf16, 模型权重是 fp32. 必须用
        # autocast("cuda", bfloat16) 让 Linear 层接受 bf16 输入, 否则
        # memory_attention 里 v_proj 报 "mat1 BFloat16 and mat2 Float".
        autocast_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                        if torch.cuda.is_available()
                        else torch.cuda.amp.autocast(enabled=False))
        with torch.inference_mode(), autocast_ctx:
            state = self.predictor.init_state(
                video_path=str(frames_dir),
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
            )
            self.predictor.reset_state(state)

            # 一次性把所有 anchor 帧的点都加进 state. SAM2 的 video predictor
            # 支持多个 cond_frame_outputs, propagate 时会全部当作 memory.
            for fi in sorted(anchors.keys()):
                fd = anchors[fi]
                for hand in ("right", "left"):
                    pos = fd[hand]["pos"]
                    neg = fd[hand]["neg"]
                    if not (pos or neg):
                        continue
                    if not pos:
                        # 只有负点没正点, 单这帧丢掉 (没法定义正样本)
                        continue
                    pts = pos + neg
                    lbls = [1] * len(pos) + [0] * len(neg)
                    obj_id = (self._OBJ_ID_RIGHT if hand == "right"
                              else self._OBJ_ID_LEFT)
                    self.predictor.add_new_points_or_box(
                        state, frame_idx=int(fi), obj_id=obj_id,
                        points=np.array(pts, dtype=np.float32),
                        labels=np.array(lbls, dtype=np.int32),
                    )

            # 从头跑 (frame 0 → 末尾), 所有 anchor 全程作为 memory
            for frame_idx, obj_ids, masks in (
                    self.predictor.propagate_in_video(
                        state, start_frame_idx=0, reverse=False)):
                _consume(state, frame_idx, obj_ids, masks)
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
