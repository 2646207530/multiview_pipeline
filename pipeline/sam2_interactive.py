"""SAM2 交互式标注 (单帧 image predictor).

跟 detect_backends.py 里的 Sam2Backend 是两种用法:
- Sam2Backend 用 video predictor, 一次 propagate 整段视频.
- 这里用 image predictor, 在首帧实时给出 mask, 每次用户加/减一个 point 都重算.

模型只装一次, 两个 cam 各自缓存自己的 set_image 结果 (省去重复 backbone 编码).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def keep_largest_cc(mask: np.ndarray) -> np.ndarray:
    """只保留最大连通区域, 抹掉 SAM2 偶尔吐在远处的几个杂散像素.

    这种杂点肉眼不易察觉, 但会把 bbox 撑得很大 (因为 bbox = nonzero 的
    xmin/ymin/xmax/ymax). 必须在跑 bbox 之前清掉, 否则 detections.json 全错.
    输入 (H,W) bool; 输出 (H,W) bool. 空 mask 原样返回."""
    if mask is None:
        return mask
    if mask.dtype == np.bool_:
        m = mask.astype(np.uint8)
    else:
        m = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:  # 只有背景 / 没东西
        return mask.astype(bool) if mask.dtype != np.bool_ else mask
    # stats[0] 是背景; 在 1..num-1 里挑面积最大的那块
    sizes = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(sizes))
    return labels == largest

_PROJECT = Path(__file__).resolve().parent.parent
_CKPT = _PROJECT / "model" / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

_MODEL = None                     # 共享的 SAM2 base model
_PREDICTORS: Dict[int, object] = {}   # cam_idx -> SAM2ImagePredictor


def ensure_sam2_hydra():
    """确保 hydra GlobalHydra 是为 sam2 的 configs 目录初始化好的.

    用 initialize_config_dir (filesystem) 而不是 initialize_config_module
    ('pkg:sam2' 走 importlib.resources). 后者在我们的 editable install 布局
    下偶尔会跑出 'Primary config module sam2 not found' (即使 sam2 import
    正常). filesystem 版本直接指 model/sam2/sam2/, 用绝对路径不会受 sys.path
    / importlib 缓存影响. 幂等: 每次先 clear 再重新 init."""
    from hydra.core.global_hydra import GlobalHydra
    from hydra import initialize_config_dir
    g = GlobalHydra.instance()
    if g.is_initialized():
        g.clear()
    cfg_root = str(_PROJECT / "model" / "sam2" / "sam2")
    initialize_config_dir(config_dir=cfg_root, version_base="1.2")


def _import_sam2_with_shadow_bypass():
    """从 sam2 拿 build_sam2 + SAM2ImagePredictor.

    SAM2 的 build_sam.py 在导入时会检查 ``isdir(sam2.__path__[0] + '/sam2')``,
    如果命中就 raise "可能从 sam2 repo 父目录运行" 的错. 但我们 editable
    install 是从 model/sam2/ 装的, 而 pipeline 的运行目录就在 model 的两级
    上, 这个布局会触发误报. 此处临时屏蔽 isdir 对那条路径的回答, 让 import
    过去, 再恢复. 不影响其他 isdir 调用."""
    import sam2  # type: ignore
    shadow_target = os.path.join(sam2.__path__[0], "sam2")
    orig_isdir = os.path.isdir

    def _patched_isdir(p):
        if p == shadow_target:
            return False
        return orig_isdir(p)

    os.path.isdir = _patched_isdir
    try:
        from sam2.build_sam import build_sam2 as _build_sam2  # type: ignore
        from sam2.sam2_image_predictor import (
            SAM2ImagePredictor as _SAM2ImagePredictor,
        )  # type: ignore
    finally:
        os.path.isdir = orig_isdir
    return _build_sam2, _SAM2ImagePredictor


def _ensure_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import torch
    if not _CKPT.is_file():
        raise RuntimeError(f"SAM2 权重不存在: {_CKPT}")
    build_sam2, _ = _import_sam2_with_shadow_bypass()
    ensure_sam2_hydra()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _MODEL = build_sam2(_CFG, str(_CKPT), device=device)
    return _MODEL


def _sam2_autocast():
    """SAM2 官方推荐 autocast bfloat16; 不加的话 image_predictor 和 video
    predictor 都会在 memory/cross-attn 处 hit 'BFloat16 vs Float' dtype 错."""
    import torch
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    # CPU 时返回个 no-op 上下文
    import contextlib
    return contextlib.nullcontext()


def set_image_for_cam(ci: int, image_rgb: np.ndarray) -> None:
    """把 cam ci 的当前帧喂给对应的 image predictor (heavy, 调一次)."""
    _, SAM2ImagePredictor = _import_sam2_with_shadow_bypass()
    if ci not in _PREDICTORS:
        _PREDICTORS[ci] = SAM2ImagePredictor(_ensure_model())
    with _sam2_autocast():
        _PREDICTORS[ci].set_image(image_rgb)


def predict_hand_mask(ci: int,
                      pos: List[List[float]],
                      neg: List[List[float]],
                      image_rgb_fallback: Optional[np.ndarray] = None
                      ) -> Optional[np.ndarray]:
    """用 cam ci 当前缓存的 image features + 给定 (pos+neg) 点跑一次预测.
    返回 (H, W) bool mask, 或 None (没正点 / 失败).

    如果 cam ci 还没被 set_image (例如用户没点 Load, 或上次 Load 时 SAM2 装
    模型失败), 而 ``image_rgb_fallback`` 不为 None, 则自动补一次 set_image —
    第一次点击会稍慢, 但保证有效."""
    if not pos:
        return None
    predictor = _PREDICTORS.get(ci)
    if predictor is None:
        if image_rgb_fallback is None:
            raise RuntimeError(
                f"cam{ci} 还没 set_image, 且没传 image_rgb_fallback"
            )
        set_image_for_cam(ci, image_rgb_fallback)
        predictor = _PREDICTORS[ci]
    pts = np.array(pos + neg, dtype=np.float32)
    lbls = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
    with _sam2_autocast():
        masks, _scores, _logits = predictor.predict(
            point_coords=pts, point_labels=lbls, multimask_output=False,
        )
    if masks is None or len(masks) == 0:
        return None
    return keep_largest_cc(masks[0])


def reset_cam(ci: int) -> None:
    """清掉某个 cam 的 predictor (用户重新 Load first frames 时调)."""
    _PREDICTORS.pop(ci, None)


def release_all() -> None:
    """彻底卸 SAM2 image predictor + 共享 base model, 清空 CUDA cache.

    在 step2 detect 真正跑视频 propagate 前调一下, 把 image predictor 占的
    backbone (一份 hiera_large ~ 1GB+ activations) 让出来给 video predictor."""
    global _MODEL, _PREDICTORS
    _PREDICTORS.clear()
    _MODEL = None
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
