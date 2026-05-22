"""Step 6: way_vis 3D 轨迹渲染.

way_vis.main() 现在还是 CLI 风格, 读全局变量 (NPY_PATH / OBJ_PATH / OUTPUT_VIDEO
等). 我们这里临时 monkey-patch 这几个模块全局, 然后调 main(), 跑完拿到 mp4.
不动 way_vis.py 本身, 师兄那边的 CLI 兼容性保留.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from .state import PipelineState
from .workspace import Workspace


def run(ws: Workspace, obj_mesh: Optional[str] = None,
        draw_hand_joints: bool = False,
        overlay_auto: bool = True, fps: int = 10) -> Dict[str, Any]:
    state = PipelineState.load(ws)
    if state.steps["infer"].status != "done":
        raise RuntimeError("Step 5 (infer) 没完成")

    npy_path = Path(state.steps["infer"].outputs["npy_path"])
    if not npy_path.exists():
        raise FileNotFoundError(npy_path)

    # 检测 npy 里有没有 object 字段 (step5 跳过物体轨迹时不会写)
    # 没有就改用 hand-only 模式, 并临时塞个 zero 占位让 way_vis 跑通.
    import numpy as np
    _npy_data = np.load(npy_path, allow_pickle=True).item()
    _seq_key = list(_npy_data["data_dict"].keys())[0]
    _params = _npy_data["data_dict"][_seq_key]["params"]
    has_object = "object" in _params
    show_mode = "both" if has_object else "hand"
    if not has_object:
        n_f = len(_params["right hand"]["rot_r"])
        _params["object"] = {
            "obj_rot":   np.zeros((n_f, 3), dtype=np.float32),
            "obj_trans": np.zeros((n_f, 3), dtype=np.float32),
        }
        # 写到 vis_dir 里, 不污染 step5 输出
        ws.vis_dir.mkdir(parents=True, exist_ok=True)
        _tmp_npy = ws.vis_dir / f"{ws.seq_name}_hand_only.npy"
        np.save(_tmp_npy, _npy_data, allow_pickle=True)
        npy_path = _tmp_npy
        print(f"[step6] npy 无 object 字段, 用 hand-only 模式 + 临时 npy: {_tmp_npy}")

    # 物体 mesh: way_vis 即便 show='hand' 也会 trimesh.load(OBJ_PATH), 给个能加载的就行
    if obj_mesh is None:
        import json
        cfg = json.loads((_PROJECT / "config" / "baseball_golf.json").read_text())
        obj_mesh = cfg["club_mesh_path"]
    obj_mesh_path = Path(obj_mesh)
    if not obj_mesh_path.exists():
        raise FileNotFoundError(f"obj_mesh 不存在: {obj_mesh_path}")

    ws.vis_dir.mkdir(parents=True, exist_ok=True)
    out_video = ws.vis_dir / f"{ws.seq_name}_trajectory.mp4"
    frame_dir = ws.vis_dir / "frames"

    # monkey-patch way_vis 模块全局, 然后调 main()
    # 注意: 不能直接 `import utils.way_vis`, 因为 step3 跑过 yolov7 后
    # sys.modules['utils'] 被 model/yolo/yolov7/utils/ 霸占了.
    # 用 importlib 按文件路径直接加载.
    import importlib.util
    way_vis_path = _PROJECT / "utils" / "way_vis.py"
    spec = importlib.util.spec_from_file_location("_pipeline_way_vis", way_vis_path)
    wv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wv)
    wv.NPY_PATH               = str(npy_path)
    wv.OBJ_PATH               = str(obj_mesh_path)
    wv.MANO_MODEL_DIR         = str(_PROJECT)
    wv.OUTPUT_VIDEO           = str(out_video)
    wv.OUTPUT_FRAME_DIR       = str(frame_dir)
    wv.SAVE_FRAME_INTERVAL    = 0       # 不存中间 jpg, 加快速度
    wv.SCENE_MESH_OUTPUT_DIR  = str(ws.vis_dir / "scene_mesh")
    wv.FPS                    = fps

    # monkey-patch overlay 解析: way_vis 原版 view1+ 走 dexycb_clips 命名约定
    # (seq_key '12-1' → '12-2'), 我们 pipeline 是 <undist>/<capture_id>/<view>/images_undistorted/
    # 所以重写一下让 view1+ 也能找到原图.
    import re as _re
    _orig_overlay_from_npy = wv._overlay_sources_from_npy
    _prepare_overlay_sources = wv._prepare_overlay_sources

    def _patched_build_view_overlay_sources(data, seq_key, view_idx,
                                            user_overlay_dir, overlay_auto):
        if user_overlay_dir is not None:
            return _prepare_overlay_sources(user_overlay_dir)
        if not overlay_auto:
            return None
        if view_idx == 0:
            return _orig_overlay_from_npy(data)
        base_imgpath = data.get("imgpath", None)
        imgnames = data.get("imgnames", None) or []
        if not base_imgpath:
            print(f"[overlay] view{view_idx}: npy 没 imgpath, 用纯渲染背景")
            return None
        base = Path(base_imgpath)
        # base = <undist_root>/<capture_id>/0/images_undistorted, 换 view_idx
        other_dir = base.parent.parent / str(view_idx) / "images_undistorted"
        if not other_dir.is_dir():
            print(f"[overlay] view{view_idx}: 目录不存在 {other_dir}, 用纯渲染背景")
            return None
        ordered = [str(other_dir / n) for n in imgnames
                   if (other_dir / n).is_file()]
        if len(ordered) != len(imgnames) or not ordered:
            valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ordered = sorted(str(p) for p in other_dir.iterdir()
                             if p.suffix.lower() in valid_ext)
        by_frame = {i: p for i, p in enumerate(ordered)}
        for p in ordered:
            stem = Path(p).stem
            m = _re.search(r"(\d+)$", stem)
            if m:
                by_frame.setdefault(int(m.group(1)), p)
        print(f"[overlay] view{view_idx}: 用 {other_dir}, 共 {len(ordered)} 张")
        return {"ordered": ordered, "by_frame": by_frame}

    wv._build_view_overlay_sources = _patched_build_view_overlay_sources

    print(f"[step6] way_vis: npy={npy_path.name}  out={out_video}  show={show_mode}")
    wv.main(save_scene_mesh=False, show=show_mode,
            frame_range=None,
            draw_hand_joints=draw_hand_joints,
            overlay_auto=overlay_auto)

    # way_vis 实际生成的是 <stem>_view{i}.mp4
    video_base = out_video.with_suffix("")
    view_videos = sorted(ws.vis_dir.glob(f"{video_base.name}_view*.mp4"))

    info = {
        "view_videos": [str(v) for v in view_videos],
        "obj_mesh": str(obj_mesh_path),
    }
    state.mark_done("vis", **info)
    state.save(ws)
    return info
