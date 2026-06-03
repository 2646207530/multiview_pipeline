"""Step 2 检测后端: ViTPose (上游 preprocess 出的 COCO17 手腕直接当 bbox 中心).

接口: backend.detect_all(undist_root, capture_id, n_cams, progress)
返回 ``{cam_idx_str: {frame_id_str: [["right", [x1,y1,x2,y2]], ["left", ...]]}}``,
跟 step2 之前的 detections.json 完全同构, 上游 step3 不感知后端.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np
import torch

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))


# ── 接口 ────────────────────────────────────────────────────────────────
class DetectBackend:
    name: str = "base"

    def detect_all(self, undist_root: Path, capture_id: str, n_cams: int,
                   progress: Optional[Callable[[float, str], None]] = None
                   ) -> Dict[str, Dict[str, list]]:
        raise NotImplementedError


# ── ViTPose (上游 GVHMR preprocess 出的 COCO17 关节, 取手腕直接当 bbox 中心) ──
class VitposeBackend(DetectBackend):
    """从 ``<undist_root>/<capture_id>/preprocess/<cam>/vitpose.pt`` 取 COCO17
    手腕点 (9=L_wrist, 10=R_wrist) 当 bbox 中心.

    vitpose 在 ``.undistorted/`` 子树里, 已经是在 undistorted 图上跑的, 不需要
    再 ``cv2.undistortPoints`` (师兄 ``ours.py`` 的 ``_prepare_vit_uv_conf`` 是
    给原始 distorted 图用的, 我们这里 D=0, 略过).

    输出 bbox: 以 wrist 为中心的固定边长方框. 这里默认 80px (师兄 wilor_generate2
    用 150px 是因为他相机离得远; 我们近, 150px 在挥杆时两手挨近会把两只手框进
    同一个 bbox). 上层 step2 的 ``bbox_size`` 还能再 ×scale 微调.
    """
    name = "vitpose"
    _L_WRIST_COCO = 9
    _R_WRIST_COCO = 10
    _DEFAULT_BOX_PX = 80.0
    _CONF_THRESHOLD = 0.3   # 跟 ours.py 对齐

    @staticmethod
    def _get_image_size(undist_root: Path, capture_id: str, ci: int):
        d = undist_root / capture_id / str(ci) / "images_undistorted"
        first = next(iter(sorted(d.glob("*.jpg"))), None)
        if first is None:
            return None
        img = cv2.imread(str(first))
        if img is None:
            return None
        return img.shape[1], img.shape[0]

    def detect_all(self, undist_root, capture_id, n_cams, progress=None):
        bbox_data: Dict[str, Dict[str, list]] = {
            str(c): {} for c in range(n_cams)
        }
        wrist_overlay: Dict[str, Dict[str, list]] = {
            str(c): {} for c in range(n_cams)
        }

        for ci in range(n_cams):
            vit_path = (undist_root / capture_id / "preprocess" /
                        str(ci) / "vitpose.pt")
            if not vit_path.is_file():
                raise RuntimeError(
                    f"找不到 vitpose: {vit_path}. 先跑上游 preprocess "
                    f"(human-dataset-tools/multi_view_smpl_optimizer 的 vitpose stage)."
                )
            wh = self._get_image_size(undist_root, capture_id, ci)
            if wh is None:
                continue
            W, H = wh

            if progress:
                progress(ci / max(n_cams, 1),
                         f"加载 vitpose cam{ci} ({vit_path.name})...")
            vit = torch.load(vit_path, map_location="cpu")
            if not torch.is_tensor(vit):
                raise RuntimeError(
                    f"{vit_path} 不是 tensor, 实际 {type(vit)}"
                )
            arr = vit.detach().cpu().numpy()
            if arr.ndim != 3 or arr.shape[-1] < 2:
                raise RuntimeError(
                    f"{vit_path} 形状异常 {arr.shape}, 期望 (T, J, 3)"
                )
            if arr.shape[-1] == 2:
                arr = np.concatenate(
                    [arr, np.ones((*arr.shape[:-1], 1), dtype=arr.dtype)],
                    axis=-1,
                )
            T = int(arr.shape[0])
            J = int(arr.shape[1])
            if J <= max(self._L_WRIST_COCO, self._R_WRIST_COCO):
                raise RuntimeError(
                    f"{vit_path} 关节数 {J} 不够, 至少要 11 (COCO17 wrists)"
                )

            half = self._DEFAULT_BOX_PX / 2.0
            for fi in range(T):
                if progress and fi % 200 == 0:
                    progress(
                        (ci + fi / max(T, 1)) / max(n_cams, 1),
                        f"ViTPose cam{ci} frame {fi:06d}/{T}")
                per_frame: list = []
                wrist_per_frame: list = []
                for j_idx, hand_label in (
                        (self._L_WRIST_COCO, "left"),
                        (self._R_WRIST_COCO, "right")):
                    u = float(arr[fi, j_idx, 0])
                    v = float(arr[fi, j_idx, 1])
                    c = float(arr[fi, j_idx, 2])
                    if c < self._CONF_THRESHOLD:
                        continue
                    if not (0.0 <= u < W and 0.0 <= v < H):
                        continue
                    wrist_per_frame.append([hand_label, [u, v]])
                    x1 = max(0.0,        u - half)
                    y1 = max(0.0,        v - half)
                    x2 = min(float(W),   u + half)
                    y2 = min(float(H),   v + half)
                    if x2 - x1 < 4 or y2 - y1 < 4:
                        continue
                    per_frame.append([hand_label, [x1, y1, x2, y2]])
                if per_frame:
                    bbox_data[str(ci)][f"{fi:06d}"] = per_frame
                if wrist_per_frame:
                    wrist_overlay[str(ci)][f"{fi:06d}"] = wrist_per_frame

        self.wrist_overlay = wrist_overlay
        return bbox_data


_REGISTRY = {
    "vitpose": VitposeBackend,
}
VALID_BACKENDS: Tuple[str, ...] = tuple(_REGISTRY.keys())


def make_backend(name: str, **kwargs) -> DetectBackend:
    """构造 backend 实例. kwargs 透传到具体后端的 ``__init__``;
    不认识的 kwarg 在该 backend 里会引发 TypeError."""
    key = (name or "vitpose").lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown detect backend: {name!r} (valid: {list(_REGISTRY)})"
        )
    cls = _REGISTRY[key]
    return cls(**kwargs) if kwargs else cls()
