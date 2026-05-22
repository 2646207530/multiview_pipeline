"""路径管理: 给一个 capture_dir, 拼出所有 pipeline 中间产物 / 输出的位置.

约定: 所有产物都落在 ``<capture_dir>/.pipeline/`` 下 (workspace), 跟原 wrapper
习惯的 ``<capture_dir>/.undistorted/`` 并排, 互不干扰.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workspace:
    capture_dir: Path
    seq_name: str

    @property
    def root(self) -> Path:
        """所有 pipeline 产物的根."""
        return self.capture_dir / ".pipeline"

    @property
    def state_file(self) -> Path:
        return self.root / "state.json"

    # ── step 1 (undistort) — 沿用 multiview_hand_init.prepare_undistort_dir 的输出位置 ──
    @property
    def undist_root(self) -> Path:
        return self.capture_dir / ".undistorted"

    @property
    def undist_seq_dir(self) -> Path:
        return self.undist_root / self.seq_name

    def undist_cam_image_dir(self, cam_idx: int) -> Path:
        return self.undist_seq_dir / str(cam_idx) / "images_undistorted"

    def undist_calib_yaml(self, cam_idx: int) -> Path:
        return self.undist_seq_dir / "calib_undistorted" / f"{cam_idx}.yaml"

    # ── step 2 (detection) ──
    @property
    def detections_json(self) -> Path:
        return self.root / "detections.json"

    @property
    def detect_vis_dir(self) -> Path:
        return self.root / "_detect_vis"

    # ── step 3 (pseudo label) — 跟现有 dataset 读法保持一致 ──
    @property
    def pseudo_label_dir(self) -> Path:
        return self.undist_root / "pseudo_label_wilor"

    @property
    def pseudo_vis_dir(self) -> Path:
        return self.undist_root / "_pseudo_vis"

    @property
    def pseudo_overlay_mp4(self) -> Path:
        return self.pseudo_vis_dir / f"{self.seq_name}_pseudo_overlay.mp4"

    # ── step 4 (finetune) ──
    @property
    def finetune_dir(self) -> Path:
        # subprocess 跑 train_ddp_sf.py 时会创建 exp/mv_ft_<seq>_<timestamp>/
        # 这里只保留预期位置, 真实 ckpt 路径由 subprocess 返回后写到 state
        return self.root / "_finetune"

    # ── step 5 (multi-view inference + npy assembly) ──
    @property
    def he_output_dir(self) -> Path:
        return self.undist_root / "_he_output"

    @property
    def he_mano_json(self) -> Path:
        return self.he_output_dir / f"{self.seq_name}_mano.json"

    def he_hand_mp4(self, hand_id: int) -> Path:
        return self.he_output_dir / f"{self.seq_name}_hand{hand_id}.mp4"

    @property
    def output_npy(self) -> Path:
        return self.root / f"{self.seq_name}.npy"

    # ── step 6 (way_vis 3D 渲染) ──
    @property
    def vis_dir(self) -> Path:
        return self.root / "vis"

    def vis_view_mp4(self, view_idx: int) -> Path:
        return self.vis_dir / f"trajectory_view{view_idx}.mp4"

    # ── 工具 ──
    def ensure_dirs(self) -> None:
        for p in [self.root, self.detect_vis_dir, self.finetune_dir, self.vis_dir]:
            p.mkdir(parents=True, exist_ok=True)
