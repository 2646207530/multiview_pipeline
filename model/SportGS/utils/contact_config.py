"""Loader for the predefined contact-point JSON config.

Source of truth: ``<project_root>/config/baseball_golf.json``.

Override the path by setting the env var ``CONTACT_CONFIG_PATH`` before launching
``train_contact.py`` / ``finetune_force_closure.py``. ``run_golf_capture_to_npy.py``
forwards its ``--contact_config`` argument through this env var.

Currently consumed sections:

* ``hand_attract_tips``
    {"right": [vid, ...], "left": [vid, ...]} — MANO vertex ids that the
    force-closure stage attracts toward the nearest club vertex.

If the file or section is missing, callers fall back to the historical
hardcoded indices defined at the bottom of this module.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional


# Default location: <project_root>/config/baseball_golf.json
# This file lives at: <project_root>/model/SportGS/utils/contact_config.py
# parents[3] -> <project_root>
_DEFAULT_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "baseball_golf.json"
)

# Legacy hardcoded fingertip indices (right hand defaults; left mirrors topology).
DEFAULT_ATTRACT_TIPS: List[int] = [768, 342, 454, 565, 683, 77]


_CACHE = {"path": None, "data": None}


def _resolve_path() -> Optional[Path]:
    p = os.environ.get("CONTACT_CONFIG_PATH")
    if p:
        return Path(p).expanduser().resolve()
    if _DEFAULT_PATH.is_file():
        return _DEFAULT_PATH
    return None


def _load() -> dict:
    p = _resolve_path()
    if _CACHE["path"] == p and _CACHE["data"] is not None:
        return _CACHE["data"]
    if p is None or not p.is_file():
        if _CACHE["path"] != p:
            print(f"[contact_config] 找不到 config，使用内置默认 tips={DEFAULT_ATTRACT_TIPS}")
        _CACHE["data"] = {}
        _CACHE["path"] = p
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
        if _CACHE["path"] != p:
            print(f"[contact_config] 已加载 {p}")
        _CACHE["data"] = data
        _CACHE["path"] = p
        return data
    except Exception as e:
        print(f"[contact_config] 解析 {p} 失败 ({e})，使用内置默认")
        _CACHE["data"] = {}
        _CACHE["path"] = p
        return {}


def get_attract_tips(side: str) -> List[int]:
    """Return MANO vertex ids on `side` ('right'|'left') for force-closure attraction."""
    if side not in ("right", "left"):
        raise ValueError(f"side must be 'right' or 'left', got {side!r}")
    cfg = _load()
    section = cfg.get("hand_attract_tips") or {}
    tips = section.get(side)
    if not tips or not isinstance(tips, list):
        return list(DEFAULT_ATTRACT_TIPS)
    out: List[int] = []
    for x in tips:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            pass
    return out or list(DEFAULT_ATTRACT_TIPS)


def get_hand_hand_contacts(n_mano_verts: int = 778) -> List[dict]:
    """Return validated hand-hand contact entries from the config.

    Each item: {"right_mano_vid": int, "left_mano_vid": int,
                "weight": float, "name": str}.
    Out-of-range or malformed entries are skipped with a warning.
    """
    cfg = _load()
    raw = cfg.get("hand_hand_contacts") or []
    out: List[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            r = int(entry["right_mano_vid"])
            l = int(entry["left_mano_vid"])
        except (KeyError, TypeError, ValueError):
            print(f"[contact_config] 跳过缺字段/非整数的 hand_hand_contacts 条目: {entry}")
            continue
        if not (0 <= r < n_mano_verts and 0 <= l < n_mano_verts):
            print(f"[contact_config] 跳过 vid 越界的 hand_hand 条目: "
                  f"right={r} left={l} (must be in [0, {n_mano_verts}))")
            continue
        try:
            w = float(entry.get("weight", 1.0))
        except (TypeError, ValueError):
            w = 1.0
        out.append({"right_mano_vid": r, "left_mano_vid": l,
                    "weight": w, "name": str(entry.get("name", ""))})
    return out


def get_config() -> dict:
    """Return the full parsed config (callers may use other sections)."""
    return _load()


def reload() -> None:
    """Force a re-read on the next access (e.g. after the user edits the file)."""
    _CACHE["path"] = None
    _CACHE["data"] = None
