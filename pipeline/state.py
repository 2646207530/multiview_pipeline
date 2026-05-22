"""State 管理: 把每个 step 的状态 + 输出路径持久化到 ``<workspace>/state.json``,
重启 web 之后也能恢复 (用户从上次中断的地方继续).

state schema (写到 state.json):
{
  "capture_dir": "/data2/.../20260424...",
  "seq_name":    "20260424170424563",
  "created_at":  "2026-05-21T10:00:00",
  "steps": {
     "setup":     {"status": "done",    "ts": "...", "outputs": {...}},
     "undistort": {"status": "done",    "ts": "...", "outputs": {"undist_root": "...", "n_cams": 2}},
     "detect":    {"status": "running", "ts": "...", "outputs": {...}},
     "pseudo":    {"status": "pending"},
     "finetune":  {"status": "skipped"},
     "infer":     {"status": "pending"},
     "vis":       {"status": "pending"},
  }
}
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .workspace import Workspace

# step 名字, 按执行顺序排. UI 那边也用这个顺序控制 enable/disable.
STEP_NAMES = [
    "setup",
    "undistort",
    "detect",
    "pseudo",
    "finetune",
    "infer",
    "vis",
]

# 每个 step 依赖的前置 step. 前置都 done 才能跑.
STEP_DEPS: Dict[str, list[str]] = {
    "setup":     [],
    "undistort": ["setup"],
    "detect":    ["undistort"],
    "pseudo":    ["detect"],
    "finetune":  ["pseudo"],
    "infer":     ["pseudo"],   # finetune 是 optional, infer 只依赖 pseudo
    "vis":       ["infer"],
}


@dataclass
class StepState:
    status: str = "pending"   # pending | running | done | failed | skipped
    ts: Optional[str] = None
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PipelineState:
    capture_dir: str = ""
    seq_name: str = ""
    created_at: str = ""
    steps: Dict[str, StepState] = field(
        default_factory=lambda: {name: StepState() for name in STEP_NAMES})

    # ─── 序列化 ───────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "capture_dir": self.capture_dir,
            "seq_name":    self.seq_name,
            "created_at":  self.created_at,
            "steps": {name: asdict(s) for name, s in self.steps.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineState":
        steps = {n: StepState(**v) for n, v in d.get("steps", {}).items()}
        # 补齐新加的 step name (向前兼容)
        for n in STEP_NAMES:
            steps.setdefault(n, StepState())
        return cls(
            capture_dir=d.get("capture_dir", ""),
            seq_name=d.get("seq_name", ""),
            created_at=d.get("created_at", ""),
            steps=steps,
        )

    # ─── 持久化 ───────────────────────────────────────────────
    def save(self, ws: Workspace) -> None:
        ws.ensure_dirs()
        ws.state_file.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, ws: Workspace) -> "PipelineState":
        if ws.state_file.exists():
            try:
                return cls.from_dict(json.loads(ws.state_file.read_text()))
            except Exception:
                pass
        s = cls(
            capture_dir=str(ws.capture_dir),
            seq_name=ws.seq_name,
            created_at=_dt.datetime.now().isoformat(timespec="seconds"),
        )
        return s

    # ─── step 状态读写帮助 ─────────────────────────────────────
    def mark_running(self, name: str) -> None:
        self.steps[name] = StepState(
            status="running",
            ts=_dt.datetime.now().isoformat(timespec="seconds"),
        )

    def mark_done(self, name: str, **outputs) -> None:
        self.steps[name] = StepState(
            status="done",
            ts=_dt.datetime.now().isoformat(timespec="seconds"),
            outputs=outputs,
        )

    def mark_failed(self, name: str, error: str) -> None:
        self.steps[name] = StepState(
            status="failed",
            ts=_dt.datetime.now().isoformat(timespec="seconds"),
            error=error,
        )

    def mark_skipped(self, name: str) -> None:
        self.steps[name] = StepState(status="skipped",
                                     ts=_dt.datetime.now().isoformat(timespec="seconds"))

    def can_run(self, name: str) -> bool:
        """前置 step 都 done/skipped 才能跑 ``name``."""
        for dep in STEP_DEPS.get(name, []):
            if self.steps[dep].status not in ("done", "skipped"):
                return False
        return True
