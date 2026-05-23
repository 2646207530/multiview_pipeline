"""伪标 2D 关节后端: HaMER / WiLoR 共用同一个接口.

接口: backend.estimate_2d(image_bgr, hand_label, bbox_xyxy, K) -> np.ndarray (21, 2)
返回真实相机内参 K 下的 21 个 2D 关节 (full image 坐标), 或 None 表示失败.
两个后端的 npz 输出格式完全一致, 上游 step3 无需感知具体后端.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))


# ---------------------------------------------------------------------------
# 共用: 用真实 K 把 (pred_kp3d, pred_cam, box_*) 投到 full image 2D.
# 等价于 HaMER 内 custom_cam_crop_to_full + 透视投影. 这里 inline 写一份,
# 避免为了一个工具函数把 hamer 的 renderer (依赖 pyrender) 拖进来.
# ---------------------------------------------------------------------------
def _project_with_K(pred_kp3d: torch.Tensor,
                    pred_cam: torch.Tensor,
                    box_center: torch.Tensor,
                    box_size: torch.Tensor,
                    is_right_flag: float,
                    K: np.ndarray) -> np.ndarray:
    """pred_kp3d: (1, 21, 3); pred_cam: (1, 3); box_center: (1, 2);
    box_size: (1,) or scalar; is_right_flag: 1.0=right, 0.0=left.
    K: (3, 3) numpy. 返回 (21, 2) numpy."""
    device = pred_kp3d.device
    # 翻转修正: 左手 (do_flip=1) 在 X 镜像
    flip = 1.0 if is_right_flag > 0.5 else -1.0
    kp3d = pred_kp3d.clone()
    kp3d[:, :, 0] = kp3d[:, :, 0] * flip
    cam = pred_cam.clone()
    cam[:, 1] = cam[:, 1] * flip

    K_t = torch.as_tensor(K, dtype=torch.float32, device=device)
    fx, fy = K_t[0, 0], K_t[1, 1]
    cx, cy = K_t[0, 2], K_t[1, 2]

    # 从 crop 坐标系恢复 full image cam_t (公式跟 HaMER 一致).
    bs = box_size.view(-1) * cam[:, 0] + 1e-9
    tz = 2 * fx / bs
    tx = (2 * (box_center[:, 0] - cx) / bs) + cam[:, 1]
    ty = (2 * (box_center[:, 1] - cy) / bs) + cam[:, 2]
    if not torch.allclose(fx, fy):
        ty = ty * (fx / fy)
    cam_t_full = torch.stack([tx, ty, tz], dim=-1)  # (1, 3)

    kp_cam = kp3d + cam_t_full.unsqueeze(1)
    depth = kp_cam[:, :, 2:3] + 1e-9
    u = fx * kp_cam[:, :, 0:1] / depth + cx
    v = fy * kp_cam[:, :, 1:2] / depth + cy
    kp2d = torch.cat([u, v], dim=-1)  # (1, 21, 2)

    arr = kp2d.detach().cpu().numpy().squeeze().astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None  # type: ignore[return-value]
    return arr


# ---------------------------------------------------------------------------
class PseudoBackend:
    name: str = "base"

    def estimate_2d(self, image, hand_label: str, bbox,
                    K: np.ndarray) -> Optional[np.ndarray]:
        raise NotImplementedError


class _HamerBackend(PseudoBackend):
    name = "hamer"

    def __init__(self):
        from model.hamer.infer import hamer_inference  # type: ignore
        from model.config.hamer_config import hamer_opt  # type: ignore
        self.hamer = hamer_inference(hamer_opt)

    def estimate_2d(self, image, hand_label, bbox, K):
        try:
            out, _ = self.hamer.estimate_from_rgb(image, [[hand_label, bbox]], K)
        except Exception as e:
            print(f"[hamer] estimate_from_rgb fail: {e}")
            return None
        kp2d = out.get("pred_keypoints_2d_full")
        if kp2d is None:
            return None
        kp2d = kp2d.detach().cpu().numpy().squeeze().astype(np.float32)
        if kp2d.ndim != 2 or kp2d.shape[1] != 2:
            return None
        return kp2d


class _WilorBackend(PseudoBackend):
    name = "wilor"

    _WILOR_ROOT = _PROJECT / "model" / "WiLoR"
    _CKPT_REL = "pretrained_models/wilor_final.ckpt"
    _CFG_REL = "pretrained_models/model_config.yaml"

    def __init__(self):
        if not self._WILOR_ROOT.exists():
            raise RuntimeError(f"WiLoR 目录不存在: {self._WILOR_ROOT}")

        # WiLoR 包路径要进 sys.path
        if str(self._WILOR_ROOT) not in sys.path:
            sys.path.insert(0, str(self._WILOR_ROOT))

        # load_wilor 会把 cfg.MANO.* 重写成 './mano_data/' 相对路径,
        # 然后构造模型时 np.load 读这些路径; 所以加载阶段必须 cwd 在 WiLoR 根.
        old_cwd = os.getcwd()
        try:
            os.chdir(self._WILOR_ROOT)
            from wilor.models import load_wilor  # type: ignore
            self.model, self.cfg = load_wilor(
                checkpoint_path=self._CKPT_REL,
                cfg_path=self._CFG_REL,
            )
        finally:
            os.chdir(old_cwd)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device).eval()

    @staticmethod
    def _recursive_to(x, device):
        if isinstance(x, dict):
            return {k: _WilorBackend._recursive_to(v, device) for k, v in x.items()}
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, list):
            return [_WilorBackend._recursive_to(v, device) for v in x]
        return x

    def estimate_2d(self, image, hand_label, bbox, K):
        from wilor.datasets.vitdet_dataset import ViTDetDataset  # type: ignore

        is_right = 1.0 if hand_label == "right" else 0.0
        boxes = np.array([bbox], dtype=np.float32)
        right = np.array([is_right], dtype=np.float32)

        try:
            dataset = ViTDetDataset(self.cfg, image, boxes, right, rescale_factor=2.0)
            sample = dataset[0]
        except Exception as e:
            print(f"[wilor] prepare batch fail: {e}")
            return None

        batch = {}
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(v).unsqueeze(0)
            elif isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0)
            else:
                batch[k] = torch.tensor([v])
        batch = self._recursive_to(batch, self.device)

        try:
            with torch.no_grad():
                out = self.model(batch)
        except Exception as e:
            print(f"[wilor] forward fail: {e}")
            return None

        pred_kp3d = out["pred_keypoints_3d"].float()      # (1, 21, 3)
        pred_cam = out["pred_cam"].float()                # (1, 3)
        box_center = batch["box_center"].float()          # (1, 2)
        box_size = batch["box_size"].float()              # (1,)

        return _project_with_K(pred_kp3d, pred_cam, box_center, box_size,
                               is_right, K)


_REGISTRY = {"hamer": _HamerBackend, "wilor": _WilorBackend}
VALID_BACKENDS = tuple(_REGISTRY.keys())


def make_backend(name: str) -> PseudoBackend:
    key = (name or "hamer").lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown pseudo backend: {name!r} (valid: {list(_REGISTRY)})"
        )
    return _REGISTRY[key]()
