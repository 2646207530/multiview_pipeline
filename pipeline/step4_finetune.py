"""Step 4: 自监督 finetune (optional).

完全 wrap ``run_hand_estimation_finetune_subprocess``. UI 上一个 checkbox
"Enable finetune" 控制要不要跑. 关掉时 state 标 ``skipped``, infer 直接用
默认 ckpt (exp/new/checkpoints/checkpoint_30).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from multiview_hand_init import run_hand_estimation_finetune_subprocess  # type: ignore

from .state import PipelineState
from .workspace import Workspace


def _relocate_ckpt_to_workspace(ft_ckpt: Path, ws: Workspace) -> Path:
    """把 model/Hand_Estimation/exp/<exp>_<ts>/checkpoints/<inner>/ 这一坨
    移到 <workspace>/_finetune/<exp>_<ts>__<inner>/, 并删掉 exp 那边整个临时目录.
    返回新位置.
    """
    # ft_ckpt = exp/<exp_id>_<ts>/checkpoints/<inner_dir>
    if not ft_ckpt.is_dir():
        return ft_ckpt
    checkpoints_dir = ft_ckpt.parent             # checkpoints
    exp_dir = checkpoints_dir.parent             # mv_ft_<seq>_<timestamp>
    new_name = f"{exp_dir.name}__{ft_ckpt.name}"
    ws.finetune_dir.mkdir(parents=True, exist_ok=True)
    dst = ws.finetune_dir / new_name
    if dst.exists():
        # 同名冲突 (强烈不应发生, 时间戳唯一), 加 _v2 后缀
        i = 2
        while (ws.finetune_dir / f"{new_name}_v{i}").exists():
            i += 1
        dst = ws.finetune_dir / f"{new_name}_v{i}"
    print(f"[step4] 把 ckpt 从 {ft_ckpt} 搬到 {dst}")
    shutil.move(str(ft_ckpt), str(dst))
    # 清理 exp 那边整个临时目录 (含日志/tensorboard/空 checkpoints/ 等)
    try:
        shutil.rmtree(exp_dir, ignore_errors=True)
        print(f"[step4] 已删除 {exp_dir}")
    except Exception as e:
        print(f"[step4] 删除 {exp_dir} 失败 (忽略): {e}")
    return dst


def skip(ws: Workspace) -> Dict[str, Any]:
    """用户选不 finetune, 直接标 skipped."""
    state = PipelineState.load(ws)
    state.mark_skipped("finetune")
    state.save(ws)
    return {"skipped": True}


def run(ws: Workspace, epochs: int = 5, lr: float = 1e-5, batch_size: int = 1,
        gpu_id: str = "0") -> Dict[str, Any]:
    """跑 finetune 子进程. epochs=0 等同于 skip."""
    state = PipelineState.load(ws)
    if state.steps["pseudo"].status != "done":
        raise RuntimeError("Step 3 (pseudo) 没完成, 不能 finetune")

    if epochs <= 0:
        return skip(ws)

    undist = state.steps["undistort"].outputs
    undist_root = Path(undist["undist_root"])
    capture_id = undist["capture_id"]

    ft_ckpt = run_hand_estimation_finetune_subprocess(
        project_root=_PROJECT,
        undist_root=undist_root,
        capture_id=capture_id,
        epochs=epochs,
        gpu_id=gpu_id,
        lr=lr,
        batch_size=batch_size,
    )

    # 把 ckpt 从 model/Hand_Estimation/exp/... 搬到 <workspace>/_finetune/
    ft_ckpt = _relocate_ckpt_to_workspace(ft_ckpt, ws)

    info = {
        "finetuned_ckpt": str(ft_ckpt),
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
    }
    state.mark_done("finetune", **info)
    state.save(ws)
    return info
