"""自动把 capture_dir 里的 .raw 文件解成 jpg 帧.

约定 (跟现有 run_golf_capture_to_npy._frame_dir_for 兼容):
  - 输出根: ``<capture_dir>/.tmp_images/``
  - 每个 .raw 文件 → 一个子目录, 目录名 = raw 文件名去掉 .raw 后缀
    e.g. MV-CS016-10UC(DA5298464)_w1440_h1080_pBayerRG8_f120.raw
         → .tmp_images/MV-CS016-10UC(DA5298464)_w1440_h1080_pBayerRG8_f120/
            frame_000000.jpg, frame_000001.jpg, ...
  - 这正好是 _frame_dir_for 期望的目录布局 (cam_name + FRAME_DIR_SUFFIX).

调用 utils/raw_to_images.py 里的 convert_raw_to_png (其实存的是 jpg).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

# 复用现成 converter (相对 import, 避免跟 model/yolo/yolov7/utils 命名冲突)
from .raw_to_images import convert_raw_to_png  # type: ignore


def find_raw_files(capture_dir: Path) -> List[Path]:
    """直接列举 capture_dir 下的 .raw (不递归子目录)."""
    return sorted(p for p in capture_dir.iterdir() if p.suffix.lower() == ".raw")


def _expected_jpg_dir(capture_dir: Path, raw_path: Path) -> Path:
    """raw 文件解码后期望落在哪个 jpg 目录."""
    stem = raw_path.stem  # 去 .raw
    return capture_dir / ".tmp_images" / stem


def detect_status(capture_dir: Path) -> Dict[str, dict]:
    """返回每个 raw 文件的状态: {raw_name: {raw_path, jpg_dir, n_jpg, needs_extract}}"""
    out: Dict[str, dict] = {}
    for raw in find_raw_files(capture_dir):
        jpg_dir = _expected_jpg_dir(capture_dir, raw)
        n_jpg = (sum(1 for p in jpg_dir.glob("frame_*.jpg"))
                 if jpg_dir.is_dir() else 0)
        out[raw.name] = {
            "raw_path":      str(raw),
            "jpg_dir":       str(jpg_dir),
            "n_jpg":         n_jpg,
            "needs_extract": n_jpg == 0,
        }
    return out


def extract_all(capture_dir: Path,
                progress: Optional[Callable[[float, str], None]] = None,
                force: bool = False) -> Dict[str, int]:
    """对 capture_dir 下所有 .raw 跑一次解码, 输出到 .tmp_images/.
    已经有 frame_*.jpg 的 raw 默认跳过 (force=True 强制重做).
    返回 {raw_name: n_jpg}.
    """
    raws = find_raw_files(capture_dir)
    if not raws:
        return {}

    (capture_dir / ".tmp_images").mkdir(parents=True, exist_ok=True)
    result: Dict[str, int] = {}
    total = len(raws)
    for i, raw in enumerate(raws):
        jpg_dir = _expected_jpg_dir(capture_dir, raw)
        existing = (sum(1 for _ in jpg_dir.glob("frame_*.jpg"))
                    if jpg_dir.is_dir() else 0)
        if existing > 0 and not force:
            print(f"[raw_extract] skip {raw.name} (已有 {existing} 帧)")
            result[raw.name] = existing
            if progress:
                progress((i + 1) / total, f"skip {raw.name} ({existing} frames)")
            continue
        if progress:
            progress(i / total, f"extracting {raw.name} ...")
        # convert_raw_to_png 第二个参数是 output_dir, 它内部会再加一层 stem 子目录
        # 我们要直接把 raw 解到 .tmp_images/<stem>/, 不再多一层, 所以传 .tmp_images
        convert_raw_to_png(str(raw), str(capture_dir / ".tmp_images"),
                           verbose=True)
        n = sum(1 for _ in jpg_dir.glob("frame_*.jpg"))
        result[raw.name] = n
        if progress:
            progress((i + 1) / total, f"done {raw.name} ({n} frames)")
    return result
