"""单序列阶段驱动 (精简版, 只做 setup + undistort).

    python scripts/run_stage.py --capture_dir <cap> --seq <seq> --stages setup,undistort

setup    : 探测相机 + raw 抽帧 (raw -> .tmp_images/<cam>/frame_*.jpg)
undistort: 去畸变 + 写 calib_undistorted, 输出 <cap>/.undistorted/<seq>/
undistort 阶段会在 stdout 打印 ``UNDIST_DATA=<去畸变数据目录>`` 方便 bash 捕获.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

ALL_STAGES = ["setup", "undistort"]


def _progress(frac, msg=""):
    try:
        print(f"    [{float(frac) * 100:5.1f}%] {msg}", flush=True)
    except Exception:
        print(f"    {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture_dir", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--stages", default="setup,undistort",
                    help="逗号分隔, 取值: " + ",".join(ALL_STAGES))
    ap.add_argument("--force", action="store_true", help="undistort 强制重跑")
    args = ap.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in ALL_STAGES]
    if bad:
        print(f"[stage] 未知阶段 {bad}, 合法: {ALL_STAGES}", file=sys.stderr)
        return 2

    from pipeline.workspace import Workspace
    from pipeline import step0_setup, step1_undistort

    cap = Path(args.capture_dir).expanduser().resolve()
    seq = args.seq
    try:
        if "setup" in stages:
            print(f"[stage] setup ({seq}) ...", flush=True)
            step0_setup.run(str(cap), seq, progress=_progress, auto_extract_raw=True)

        ws = Workspace(capture_dir=cap, seq_name=seq)

        if "undistort" in stages:
            print(f"[stage] undistort ({seq}) ...", flush=True)
            info = step1_undistort.run(ws, force=args.force)
            print(f"UNDIST_DATA={info['undist_root']}/{info['capture_id']}", flush=True)
    except Exception as e:
        print(f"[stage] FAILED ({seq}): {e}", file=sys.stderr)
        traceback.print_exc()
        return 1

    print(f"[stage] OK ({seq}): {stages}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
