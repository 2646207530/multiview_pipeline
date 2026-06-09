"""单序列分阶段驱动 (给批处理脚本 batch_run.sh 调用).

把 Gradio app 里的 step 函数搬到命令行: 一次跑指定的若干 step, 状态照常落到
``<capture>/.pipeline/state.json``, 中间结果 / 每步可视化都保留.

阶段名 (--stages, 逗号分隔, 按下面顺序执行):
    setup      : 探测相机 + raw 抽帧 (raw → .tmp_images/*.jpg)
    undistort  : 去畸变 + 写 calib_undistorted, 输出 <cap>/.undistorted/<seq>/
    detect     : ViTPose 手腕 → bbox (需要上游 vitpose.pt 已生成)
    pseudo     : WiLoR 21 关节伪标
    infer      : UST-hand 多视角推理 + 组装 npy (只输出手)

典型用法 (配合外部 gvhmr 人体姿态 stage):
    # 1) pipeline 环境: 抽帧 + 去畸变
    python scripts/run_pipeline_stage.py --capture_dir <cap> --seq <seq> \
        --stages setup,undistort --bbox_size 1.0
    # (中间: gvhmr 环境跑 human-dataset-tools 产出 preprocess/<cam>/vitpose.pt)
    # 2) pipeline 环境: 检测 + 伪标 + 推理
    python scripts/run_pipeline_stage.py --capture_dir <cap> --seq <seq> \
        --stages detect,pseudo,infer --bbox_size 1.0

undistort 阶段会在 stdout 打印一行 ``UNDIST_DATA=<去畸变数据目录>`` 方便 bash 捕获.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

ALL_STAGES = ["setup", "undistort", "detect", "pseudo", "infer"]


def _progress(frac, msg=""):
    try:
        print(f"    [{float(frac) * 100:5.1f}%] {msg}", flush=True)
    except Exception:
        print(f"    {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture_dir", required=True,
                    help="单个采集序列根目录 (含 camera_params.json + 2 个相机的 .raw/帧目录)")
    ap.add_argument("--seq", required=True, help="序列名 (= .undistorted 下的子目录名)")
    ap.add_argument("--stages", required=True,
                    help="逗号分隔, 取值: " + ",".join(ALL_STAGES))
    ap.add_argument("--bbox_size", type=float, default=1.0,
                    help="step detect 的 bbox patch 系数 (中心不变, 长宽各 ×k). 默认 1.0")
    ap.add_argument("--gpu_id", default="0", help="step infer 用的 GPU 相对索引 (默认 0)")
    ap.add_argument("--ckpt", default=None,
                    help="step infer 的 ckpt 目录 (默认用 exp/new/checkpoints/checkpoint_30)")
    ap.add_argument("--force", action="store_true",
                    help="undistort/detect/pseudo 强制重跑 (覆盖已有结果)")
    args = ap.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in ALL_STAGES]
    if bad:
        print(f"[stage] 未知阶段 {bad}, 合法: {ALL_STAGES}", file=sys.stderr)
        return 2

    # 延迟 import (要在 sys.path 设好之后)
    from pipeline.workspace import Workspace
    from pipeline import (
        step0_setup, step1_undistort, step2_detect,
        step3_pseudo_label, step5_inference,
    )

    cap = Path(args.capture_dir).expanduser().resolve()
    seq = args.seq

    try:
        if "setup" in stages:
            print(f"[stage] setup ({seq}) ...", flush=True)
            step0_setup.run(str(cap), seq, progress=_progress,
                            auto_extract_raw=True)

        ws = Workspace(capture_dir=cap, seq_name=seq)

        if "undistort" in stages:
            print(f"[stage] undistort ({seq}) ...", flush=True)
            info = step1_undistort.run(ws, force=args.force)
            undist_data = f"{info['undist_root']}/{info['capture_id']}"
            # 给 bash 捕获的机器可读行
            print(f"UNDIST_DATA={undist_data}", flush=True)

        if "detect" in stages:
            print(f"[stage] detect (vitpose, bbox_size={args.bbox_size}) ...", flush=True)
            step2_detect.run(ws, progress=_progress, force=args.force,
                             backend="vitpose", bbox_size=float(args.bbox_size))

        if "pseudo" in stages:
            print(f"[stage] pseudo (wilor) ...", flush=True)
            step3_pseudo_label.run(ws, progress=_progress, force=args.force,
                                   make_video=True, backend="wilor")

        if "infer" in stages:
            print(f"[stage] infer (UST-hand multi-view) ...", flush=True)
            info = step5_inference.run(ws, gpu_id=args.gpu_id,
                                       ckpt_override=args.ckpt)
            print(f"NPY={info['npy_path']}", flush=True)

    except Exception as e:
        print(f"[stage] FAILED ({seq}): {e}", file=sys.stderr)
        traceback.print_exc()
        return 1

    print(f"[stage] OK ({seq}): {stages}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
