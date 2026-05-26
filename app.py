"""Golf multi-view pipeline 的 Gradio web 入口.

Wizard 风格: 7 个 Tab (Setup + 6 steps), 每个 Tab 一个 Run 按钮 + 状态 + 预览.
状态写到 ``<capture_dir>/.pipeline/state.json``, 重启浏览器从这里恢复.

启动:
    python app.py [--port 7860] [--host 0.0.0.0]
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

# 避免共享机器上 /tmp/gradio 被别人创建后 PermissionDenied.
# 优先用 GRADIO_TEMP_DIR / TMPDIR; 否则 fallback 到家目录下私有目录.
# 必须在 `import gradio` 之前设, 不然 gradio 已经按 /tmp 初始化好了.
_gradio_tmp = (os.environ.get("GRADIO_TEMP_DIR")
                or os.environ.get("TMPDIR")
                or str(Path.home() / ".gradio_tmp"))
Path(_gradio_tmp).mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = _gradio_tmp

# 强制 CUDA 设备按 PCI bus 编号, 跟 nvidia-smi 对齐.
# 默认 FASTEST_FIRST 会按性能重排, 同一张物理卡在 CUDA_VISIBLE_DEVICES 里的
# 编号跟 nvidia-smi 看到的可能不一样. 必须在 import torch 之前设.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import cv2
import gradio as gr
import numpy as np

_PROJECT = Path(__file__).resolve().parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from pipeline.state import PipelineState, STEP_NAMES
from pipeline.workspace import Workspace
from pipeline import (
    step0_setup,
    step1_undistort,
    step2_detect,
    step3_pseudo_label,
    step4_finetune,
    step5_inference,
    step6_visualize,
)


# ─── 全局: 当前 workspace ───────────────────────────────────────────────
# 简化: 整个 web app 服务一个 workspace; 用户切到别的 capture_dir 就重新 init.
class AppContext:
    workspace: Workspace = None
    state:     PipelineState = None

ctx = AppContext()


def _ws_or_raise() -> Workspace:
    if ctx.workspace is None:
        raise gr.Error("先在 Setup 标签里 Initialize workspace")
    return ctx.workspace


def _status_md() -> str:
    """渲染各 step 当前状态为 markdown 列表."""
    if ctx.state is None:
        return "_(未初始化)_"
    rows = ["| Step | Status | Last update |", "|---|---|---|"]
    badge = {"pending": "⚪", "running": "🟡", "done": "🟢",
             "failed": "🔴", "skipped": "⚫"}
    for name in STEP_NAMES:
        s = ctx.state.steps[name]
        rows.append(f"| {name} | {badge.get(s.status, '')} {s.status} | {s.ts or '-'} |")
    return "\n".join(rows)


# ─── Callbacks ─────────────────────────────────────────────────────────
def _existing(path) -> Any:
    """Return path string if file exists, else None (gradio components 友好)."""
    if not path:
        return None
    p = Path(path)
    return str(p) if p.exists() else None


_DEFAULT_CKPT_LABEL = "(default: exp/new/checkpoints/checkpoint_30)"


def _ud_frame_image(cam_idx: int, frame_idx: int):
    """从当前 ctx.state 拿去畸变后的 jpg 路径; 不存在返回 None."""
    if ctx.state is None:
        return None
    ud = ctx.state.steps.get("undistort")
    if ud is None or ud.status != "done":
        return None
    o = ud.outputs
    p = (Path(o["undist_root"]) / o["capture_id"] / str(cam_idx) /
         "images_undistorted" / f"{int(frame_idx):06d}.jpg")
    return str(p) if p.exists() else None


def _ud_slider_max(o1: dict) -> int:
    """undistort outputs → frame slider 最大值 (两相机最少帧数 - 1).
    Gradio 不允许 minimum == maximum, 所以保底返回 >= 1.
    """
    npc = (o1 or {}).get("n_frames_per_cam") or {}
    if isinstance(npc, dict) and npc:
        try:
            return max(1, min(int(v) for v in npc.values()) - 1)
        except Exception:
            return 1
    return 1


# ── Step 2/3/5 单帧 viewer 辅助 ─────────────────────────────────────
def _det_load_detections() -> dict:
    """加载 detections.json (bbox_data); 失败返回空 dict."""
    if ctx.state is None:
        return {}
    det = ctx.state.steps.get("detect")
    if det is None or det.status != "done":
        return {}
    import json as _json
    p = det.outputs.get("detections_json")
    if not p or not Path(p).exists():
        return {}
    try:
        return _json.loads(Path(p).read_text())
    except Exception:
        return {}


def _det_frame_image(cam_idx: int, frame_idx: int):
    """实时渲染单帧 bbox overlay (RGB ndarray). 不存在返回 None."""
    if ctx.state is None:
        return None
    ud = ctx.state.steps.get("undistort")
    if ud is None or ud.status != "done":
        return None
    o = ud.outputs
    p = (Path(o["undist_root"]) / o["capture_id"] / str(cam_idx) /
         "images_undistorted" / f"{int(frame_idx):06d}.jpg")
    if not p.exists():
        return None
    img = cv2.imread(str(p))
    if img is None:
        return None
    bbox_data = _det_load_detections()
    bboxes = (bbox_data.get(str(cam_idx)) or {}).get(f"{int(frame_idx):06d}", [])
    for entry in bboxes:
        if not entry or len(entry) < 2:
            continue
        label, bb = entry[0], entry[1]
        try:
            x1, y1, x2, y2 = [int(v) for v in bb]
        except Exception:
            continue
        color = (0, 255, 0) if label == "right" else (64, 128, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def cb_det_browse(frame_idx):
    fi = int(frame_idx) if frame_idx is not None else 0
    return _det_frame_image(0, fi), _det_frame_image(1, fi)


def _ps_frame_image(frame_idx: int):
    """读 _pseudo_vis/<seq>_<frame:06d>.jpg (cam0|cam1 stitched). 返回 path 或 None."""
    if ctx.state is None:
        return None
    ud = ctx.state.steps.get("undistort")
    if ud is None or ud.status != "done":
        return None
    o = ud.outputs
    p = (Path(o["undist_root"]) / "_pseudo_vis" /
         f"{o['capture_id']}_{int(frame_idx):06d}.jpg")
    return str(p) if p.exists() else None


def cb_ps_browse(frame_idx):
    fi = int(frame_idx) if frame_idx is not None else 0
    return _ps_frame_image(fi)


def _inf_frame_image(hand_id: int, frame_idx: int):
    """读 _he_output/<seq>_hand{hand_id}/<frame:06d>.jpg 单帧."""
    if ctx.state is None or ctx.workspace is None:
        return None
    ws = ctx.workspace
    p = (ws.he_output_dir / f"{ws.seq_name}_hand{int(hand_id)}" /
         f"{int(frame_idx):06d}.jpg")
    return str(p) if p.exists() else None


def cb_inf_browse(frame_idx):
    fi = int(frame_idx) if frame_idx is not None else 0
    return _inf_frame_image(0, fi), _inf_frame_image(1, fi)


def _list_workspace_ckpts(ws: Workspace) -> list:
    """列出 <workspace>/_finetune/ 下含任意 *.pth.tar 的子目录.
    不写死 model 类名 (FlipModel 之类切换时 ckpt 文件名也会变)."""
    if ws is None or not ws.finetune_dir.is_dir():
        return []
    out = []
    for p in sorted(ws.finetune_dir.iterdir()):
        if p.is_dir() and any(p.glob("*.pth.tar")):
            out.append(str(p))
    return out


def _ckpt_dropdown_update(ws: Workspace, state: PipelineState):
    """构造 step5 ckpt dropdown 的 (choices, value).
    默认 value: finetune 跑过 → 它的 ckpt; 否则 _DEFAULT_CKPT_LABEL.
    """
    choices = [_DEFAULT_CKPT_LABEL] + _list_workspace_ckpts(ws)
    value = _DEFAULT_CKPT_LABEL
    if state is not None:
        ft = state.steps.get("finetune")
        if ft and ft.status == "done":
            ck = ft.outputs.get("finetuned_ckpt")
            if ck and Path(ck).is_dir() and str(ck) in choices:
                value = str(ck)
    return gr.Dropdown(choices=choices, value=value)


def _restore_previews_from_state(state: PipelineState):
    """从 state.json 已完成的 step outputs 恢复每个预览组件的值.
    返回顺序跟 cb_setup outputs 一致.
    """
    def _out(step_name):
        s = state.steps.get(step_name)
        return s.outputs if (s and s.status == "done") else {}
    o1 = _out("undistort")
    o2 = _out("detect")
    o3 = _out("pseudo")
    o4 = _out("finetune")
    o5 = _out("infer")
    o6 = _out("vis")
    det_videos = o2.get("vis_videos") or {}
    view_videos = o6.get("view_videos") or []
    ud_max = _ud_slider_max(o1)
    ud_slider = (gr.Slider(minimum=0, maximum=ud_max, value=0, step=1)
                 if o1 else gr.Slider())
    # step 2/3/5 sliders 跟 undistort 共用同一个总帧数
    frame_max = _ud_slider_max(o1)
    det_slider = (gr.Slider(minimum=0, maximum=frame_max, value=0, step=1)
                  if o2 else gr.Slider())
    ps_slider = (gr.Slider(minimum=0, maximum=frame_max, value=0, step=1)
                  if o3 else gr.Slider())
    inf_slider = (gr.Slider(minimum=0, maximum=frame_max, value=0, step=1)
                  if o5 else gr.Slider())
    return (
        # step 1 (4)
        o1, ud_slider, _ud_frame_image(0, 0), _ud_frame_image(1, 0),
        # step 2 (6: o2, vid0, vid1, slider, img0, img1)
        o2, _existing(det_videos.get("0")), _existing(det_videos.get("1")),
            det_slider,
            _det_frame_image(0, 0) if o2 else None,
            _det_frame_image(1, 0) if o2 else None,
        # step 3 (4: o3, vid, slider, img)
        o3, _existing(o3.get("pseudo_overlay_mp4")),
            ps_slider,
            _ps_frame_image(0) if o3 else None,
        # step 4 (1)
        o4,
        # step 5 (7: o5, hand0, hand1, npy, slider, img0, img1)
        o5, _existing(o5.get("hand0_mp4")), _existing(o5.get("hand1_mp4")),
            _existing(o5.get("npy_path")),
            inf_slider,
            _inf_frame_image(0, 0) if o5 else None,
            _inf_frame_image(1, 0) if o5 else None,
        # step 6 (3)
        o6,
        _existing(view_videos[0]) if len(view_videos) > 0 else None,
        _existing(view_videos[1]) if len(view_videos) > 1 else None,
    )


def cb_setup(capture_dir: str, seq_name: str, auto_extract_raw: bool,
             progress=gr.Progress(track_tqdm=False)):
    def _p(frac, desc=""):
        progress(frac, desc=desc)
    try:
        info = step0_setup.run(capture_dir, seq_name,
                                progress=_p,
                                auto_extract_raw=auto_extract_raw)
        ctx.workspace = Workspace(capture_dir=Path(info["capture_dir"]),
                                   seq_name=info["seq_name"])
        ctx.state = PipelineState.load(ctx.workspace)
        return ((info, _status_md(), "")
                + _restore_previews_from_state(ctx.state)
                + (_ckpt_dropdown_update(ctx.workspace, ctx.state),))
    except Exception as e:
        # restore 元组结构: 25 项
        # step1: o1+slider+ud0+ud1                                       (4)
        # step2: o2+vid0+vid1+det_slider+det_img0+det_img1               (6)
        # step3: o3+ps_vid+ps_slider+ps_img                              (4)
        # step4: o4                                                       (1)
        # step5: o5+hand0+hand1+npy+inf_slider+inf_img0+inf_img1         (7)
        # step6: o6+view0+view1                                          (3)
        empty = (None, gr.Slider(), None, None,
                 None, None, None, gr.Slider(), None, None,
                 None, None, gr.Slider(), None,
                 None,
                 None, None, None, None, gr.Slider(), None, None,
                 None, None, None)
        return ({"error": str(e)}, _status_md(),
                traceback.format_exc()) + empty + (gr.Dropdown(),)


def cb_undistort(force: bool):
    ws = _ws_or_raise()
    try:
        info = step1_undistort.run(ws, force=force)
        ctx.state = PipelineState.load(ws)
        max_idx = _ud_slider_max(info)
        return (info, _status_md(),
                gr.Slider(minimum=0, maximum=max_idx, value=0, step=1),
                _ud_frame_image(0, 0), _ud_frame_image(1, 0), "")
    except Exception as e:
        return ({"error": str(e)}, _status_md(),
                gr.Slider(), None, None, traceback.format_exc())


def cb_ud_browse(frame_idx):
    fi = int(frame_idx) if frame_idx is not None else 0
    return _ud_frame_image(0, fi), _ud_frame_image(1, fi)


# ── SAM2 标注辅助 ──────────────────────────────────────────────────
_SAM2_LABELS = ("right_pos", "right_neg", "left_pos", "left_neg")
_SAM2_POINT_COLOR = {
    "right_pos": (40, 220, 40),
    "right_neg": (220, 40, 40),
    "left_pos":  (40, 200, 255),
    "left_neg":  (200, 80, 255),
}
# mask 叠色 (RGB)
_SAM2_MASK_COLOR = {
    "right": (50, 220, 50),
    "left":  (50, 180, 255),
}


def _sam2_state_init(ci=None) -> dict:
    """state 结构改成多 anchor: state["anchors"][frame_idx] 才是真正的标注内容.
    顶层 right/left 不再存; 渲染时按 frame_idx 取对应 anchor."""
    return {
        "image":     None,
        "ci":        ci,
        "frame_idx": 0,    # 当前在显示哪一帧 (浏览用, 不强制是 anchor)
        "n_frames":  0,
        "anchors":   {},   # {frame_idx: {"right": {"pos":[], "neg":[], "mask": np|None},
                            #              "left":  {...}}}
    }


def _sam2_get_anchor(state: dict, frame_idx) -> dict:
    """拿当前 frame_idx 的 anchor entry, 没有就创建一个空的."""
    fi = int(frame_idx)
    if fi not in state["anchors"]:
        state["anchors"][fi] = {
            "right": {"pos": [], "neg": [], "mask": None},
            "left":  {"pos": [], "neg": [], "mask": None},
        }
    return state["anchors"][fi]


def _active_hand_from_label(active_label) -> Optional[str]:
    """active_label 形如 'right_pos' / 'left_neg'; 取前缀 → 'right' / 'left'."""
    if not active_label or "_" not in active_label:
        return None
    return active_label.split("_", 1)[0]


def _sam2_render(state: dict, active_label: Optional[str] = None):
    """渲染顺序: 原图 → 当前 frame 的 active 手 mask 半透明 → active 手的点.
    多帧 anchor: 只画 state["frame_idx"] 这帧的 anchor 内容. 切到没标过的帧
    就只显示原图."""
    if state is None or state.get("image") is None:
        return None
    img = state["image"].copy()
    fi = state.get("frame_idx", 0)
    anchor = state.get("anchors", {}).get(fi)
    if anchor is None:
        return img  # 这帧没标过, 直接出原图
    active = _active_hand_from_label(active_label)
    hands_to_show = (active,) if active in ("right", "left") else ("right", "left")
    # 1. mask 半透明
    for hand in hands_to_show:
        m = anchor[hand].get("mask")
        if m is None:
            continue
        color = _SAM2_MASK_COLOR[hand]
        overlay = img.copy()
        overlay[m] = color
        img = cv2.addWeighted(img, 0.55, overlay, 0.45, 0.0)
    # 2. 点
    for hand in hands_to_show:
        for typ in ("pos", "neg"):
            color = _SAM2_POINT_COLOR[f"{hand}_{typ}"]
            for x, y in anchor[hand][typ]:
                cv2.circle(img, (int(x), int(y)), 5, color, -1)
                cv2.circle(img, (int(x), int(y)), 5, (255, 255, 255), 1)
    return img


def _sam2_recompute_mask(state: dict, hand: str):
    """对当前 frame 的某只手, 跑 SAM2 image predict 拿 mask, 写回 anchors[fi]."""
    from pipeline.sam2_interactive import predict_hand_mask
    ci = state.get("ci")
    fi = state.get("frame_idx", 0)
    anchor = state.get("anchors", {}).get(fi)
    if anchor is None:
        return
    pos = anchor[hand]["pos"]
    neg = anchor[hand]["neg"]
    if ci is None or not pos:
        anchor[hand]["mask"] = None
        return
    try:
        anchor[hand]["mask"] = predict_hand_mask(
            ci, pos, neg,
            image_rgb_fallback=state.get("image"),
        )
    except Exception as e:
        print(f"[sam2 interactive] cam{ci} fr{fi} {hand} predict fail: {e}")
        anchor[hand]["mask"] = None


def _sam2_load_cam_frame(undist_root: Path, capture_id: str, ci: int,
                            frame_idx: int):
    """读 cam ci 的指定帧 RGB. 返回 (rgb_ndarray|None, n_frames, jpg_path|None)."""
    d = undist_root / capture_id / str(ci) / "images_undistorted"
    jpgs = sorted(d.glob("*.jpg"), key=lambda p: int(p.stem))
    if not jpgs:
        return None, 0, None
    frame_idx = max(0, min(int(frame_idx), len(jpgs) - 1))
    bgr = cv2.imread(str(jpgs[frame_idx]))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
    return rgb, len(jpgs), jpgs[frame_idx]


def cb_sam2_load_frames():
    """从 ws 的 undistort 输出读 cam0/cam1 首帧 + 给 SAM2 image predictor
    各做一次 set_image (heavy, ~5-10s 一次). 同时更新两个 anchor frame slider
    的 max 为 n_frames-1."""
    from pipeline.sam2_interactive import set_image_for_cam, reset_cam
    # placeholder slider state (Gradio 不允许 min == max, 用 1 当占位)
    # NOTE: 用 gr.Slider(...) 而不是 gr.update(...), 因为这版 Gradio 用 update
    # 改 max 不会真的刷 UI; 跟 Undistort 那边的 cb_undistort 一样直接重建组件.
    empty_slider = gr.Slider(minimum=0, maximum=1, value=0, step=1)
    try:
        ws = _ws_or_raise()
    except Exception as e:
        return (None, None, _sam2_state_init(0), _sam2_state_init(1),
                str(e), empty_slider, empty_slider)
    state = PipelineState.load(ws)
    ud = state.steps.get("undistort")
    if ud is None or ud.status != "done":
        return (None, None, _sam2_state_init(0), _sam2_state_init(1),
                "Step 1 (undistort) 没完成, 先跑去畸变.",
                empty_slider, empty_slider)
    undist_root = Path(ud.outputs["undist_root"])
    capture_id = ud.outputs["capture_id"]
    out_imgs, out_states, out_sliders, errs = [], [], [], []
    for ci in (0, 1):
        rgb, n_frames, _ = _sam2_load_cam_frame(undist_root, capture_id, ci, 0)
        if rgb is None:
            errs.append(f"cam{ci}: 找不到任何首帧 jpg")
            out_imgs.append(None)
            out_states.append(_sam2_state_init(ci))
            out_sliders.append(empty_slider)
            continue
        s = _sam2_state_init(ci)
        s["image"] = rgb
        s["frame_idx"] = 0
        s["n_frames"] = n_frames
        # heavy: 给 SAM2 image predictor 喂图
        reset_cam(ci)
        try:
            set_image_for_cam(ci, rgb)
        except Exception as e:
            errs.append(f"cam{ci} set_image fail: {e}")
            print(f"[sam2 interactive] cam{ci} set_image fail: {e}")
        out_imgs.append(rgb)
        out_states.append(s)
        # Gradio 要求 max > min, n_frames < 2 时用 1 占位 (slider 拖不动)
        out_sliders.append(gr.Slider(minimum=0, maximum=max(n_frames - 1, 1),
                                       value=0, step=1,
                                       label=f"anchor frame (cam{ci}) — 拖到你想标的帧 (共 {n_frames} 帧)"))
    if errs:
        msg = ("⚠️ 加载部分失败 (第一次点击会自动补 set_image 重试):\n- "
               + "\n- ".join(errs))
    else:
        msg = ("✅ 已加载 cam0/cam1 首帧 + SAM2 image predictor. "
               "可以拖 anchor frame slider 换到任意一帧再开始点.")
    return (out_imgs[0], out_imgs[1], out_states[0], out_states[1], msg,
            out_sliders[0], out_sliders[1])


def cb_sam2_browse_frame(state: dict, frame_idx, active_label: str):
    """切到另一帧: 只换 image + frame_idx, **保留所有已存的 anchor**.
    给 SAM2 image predictor 重新喂这一帧的图, 这样在这帧上点击加点时
    predict_hand_mask 用的是新帧的 features."""
    from pipeline.sam2_interactive import set_image_for_cam
    if state is None or state.get("ci") is None:
        return (state.get("image") if state else None), state
    ci = state["ci"]
    try:
        ws = _ws_or_raise()
    except Exception as e:
        print(f"[sam2] browse_frame: 无 workspace: {e}")
        return state.get("image"), state
    pst = PipelineState.load(ws)
    ud = pst.steps.get("undistort")
    if ud is None or ud.status != "done":
        return state.get("image"), state
    undist_root = Path(ud.outputs["undist_root"])
    capture_id = ud.outputs["capture_id"]
    rgb, n_frames, _ = _sam2_load_cam_frame(undist_root, capture_id, ci,
                                              int(frame_idx))
    if rgb is None:
        return state.get("image"), state
    state["image"] = rgb
    state["frame_idx"] = max(0, min(int(frame_idx), n_frames - 1))
    state["n_frames"] = n_frames
    # anchors 不动 — 多帧标注靠这个 dict 跨帧持久化
    try:
        set_image_for_cam(ci, rgb)
    except Exception as e:
        print(f"[sam2 interactive] cam{ci} set_image fail (browse): {e}")
    return _sam2_render(state, active_label), state


def cb_sam2_click(evt: gr.SelectData, state: dict, active_label: str):
    if state is None or state.get("image") is None:
        return (state.get("image") if state else None), state
    x, y = evt.index
    hand, typ = active_label.split("_")
    fi = state.get("frame_idx", 0)
    anchor = _sam2_get_anchor(state, fi)
    anchor[hand][typ].append([float(x), float(y)])
    # 即时重算这只手在该 frame 的 mask
    _sam2_recompute_mask(state, hand)
    return _sam2_render(state, active_label), state


def cb_sam2_clear_this_frame(state: dict, active_label: str):
    """清掉当前 frame 的 anchor (不影响其他帧)."""
    if state is None:
        return None, state
    fi = state.get("frame_idx", 0)
    state.get("anchors", {}).pop(fi, None)
    return _sam2_render(state, active_label), state


def cb_sam2_clear_all_anchors(state: dict, active_label: str):
    """清掉这个 cam 的所有 anchor (全部帧)."""
    if state is None:
        return None, state
    state["anchors"] = {}
    return _sam2_render(state, active_label), state


def cb_sam2_active_change(state: dict, active_label: str):
    """切换 active label radio 时, 重渲染图 (隐藏掉非 active 那只手)."""
    return _sam2_render(state, active_label)


def cb_detect(backend: str, force: bool,
              ann0: dict, ann1: dict,
              bbox_size: float = 1.0,
              progress=gr.Progress(track_tqdm=False)):
    ws = _ws_or_raise()
    def _p(frac, msg):
        progress(frac, desc=msg)

    sam2_prompts = None
    if backend == "sam2":
        def _pack(ann):
            anchors_raw = (ann or {}).get("anchors", {}) or {}
            out_anchors = {}
            for fi, fd in anchors_raw.items():
                out_anchors[int(fi)] = {
                    hand: {
                        "pos": list((fd.get(hand) or {}).get("pos") or []),
                        "neg": list((fd.get(hand) or {}).get("neg") or []),
                    }
                    for hand in ("right", "left")
                }
            return {"anchors": out_anchors}
        sam2_prompts = {str(ci): _pack(ann)
                         for ci, ann in ((0, ann0), (1, ann1))}
        any_pos = any(
            sam2_prompts[c]["anchors"][fi][h]["pos"]
            for c in ("0", "1")
            for fi in sam2_prompts[c]["anchors"]
            for h in ("right", "left")
        )
        if not any_pos:
            return ({"error": "SAM2 需要在任意 cam 的任意一帧给一只手标 ≥1 个正点"},
                    _status_md(), None, None, "")
    try:
        info = step2_detect.run(ws, progress=_p, force=force,
                                 backend=backend, sam2_prompts=sam2_prompts,
                                 bbox_size=float(bbox_size))
        ctx.state = PipelineState.load(ws)
        vv = info.get("vis_videos") or {}
        ud_o = ctx.state.steps.get("undistort").outputs
        frame_max = _ud_slider_max(ud_o)
        return (info, _status_md(),
                _existing(vv.get("0")), _existing(vv.get("1")), "",
                gr.Slider(minimum=0, maximum=frame_max, value=0, step=1),
                _det_frame_image(0, 0), _det_frame_image(1, 0))
    except Exception as e:
        return ({"error": str(e)}, _status_md(),
                None, None, traceback.format_exc(),
                gr.Slider(), None, None)


def cb_pseudo(backend: str, force: bool, make_video: bool,
              progress=gr.Progress(track_tqdm=False)):
    ws = _ws_or_raise()
    def _p(frac, msg):
        progress(frac, desc=msg)
    try:
        info = step3_pseudo_label.run(ws, progress=_p, force=force,
                                       make_video=make_video,
                                       backend=backend)
        ctx.state = PipelineState.load(ws)
        ud_o = ctx.state.steps.get("undistort").outputs
        frame_max = _ud_slider_max(ud_o)
        return (info, _status_md(), info.get("pseudo_overlay_mp4"), "",
                gr.Slider(minimum=0, maximum=frame_max, value=0, step=1),
                _ps_frame_image(0))
    except Exception as e:
        return ({"error": str(e)}, _status_md(), None,
                traceback.format_exc(), gr.Slider(), None)


def cb_finetune(enable: bool, epochs: int, lr: float, bs: int):
    ws = _ws_or_raise()
    try:
        if not enable or epochs <= 0:
            info = step4_finetune.skip(ws)
        else:
            info = step4_finetune.run(ws, epochs=int(epochs), lr=float(lr),
                                       batch_size=int(bs))
        ctx.state = PipelineState.load(ws)
        return (info, _status_md(), "",
                _ckpt_dropdown_update(ctx.workspace, ctx.state))
    except Exception as e:
        return ({"error": str(e)}, _status_md(),
                traceback.format_exc(), gr.Dropdown())


def cb_infer(ckpt_choice: str):
    ws = _ws_or_raise()
    try:
        # _DEFAULT_CKPT_LABEL / "" / None 都视为不指定, 走自动逻辑
        ckpt_override = None if (not ckpt_choice or
                                  ckpt_choice == _DEFAULT_CKPT_LABEL) else ckpt_choice
        info = step5_inference.run(ws, ckpt_override=ckpt_override)
        ctx.state = PipelineState.load(ws)
        ud_o = ctx.state.steps.get("undistort").outputs
        frame_max = _ud_slider_max(ud_o)
        return (info, _status_md(),
                info.get("hand0_mp4"), info.get("hand1_mp4"),
                info.get("npy_path"), "",
                gr.Slider(minimum=0, maximum=frame_max, value=0, step=1),
                _inf_frame_image(0, 0), _inf_frame_image(1, 0))
    except Exception as e:
        return ({"error": str(e)}, _status_md(),
                None, None, None, traceback.format_exc(),
                gr.Slider(), None, None)


def cb_refresh_ckpts():
    return _ckpt_dropdown_update(ctx.workspace, ctx.state)


def cb_vis(obj_mesh: str, draw_hand_joints: bool, overlay_auto: bool, fps: int):
    ws = _ws_or_raise()
    try:
        info = step6_visualize.run(ws,
                                    obj_mesh=obj_mesh or None,
                                    draw_hand_joints=draw_hand_joints,
                                    overlay_auto=overlay_auto,
                                    fps=int(fps))
        ctx.state = PipelineState.load(ws)
        view_videos = info.get("view_videos", [])
        v0 = view_videos[0] if len(view_videos) > 0 else None
        v1 = view_videos[1] if len(view_videos) > 1 else None
        return info, _status_md(), v0, v1, ""
    except Exception as e:
        return {"error": str(e)}, _status_md(), None, None, traceback.format_exc()


# ─── Gradio UI ─────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Golf Multi-View Pipeline") as demo:
        gr.Markdown("# Golf Multi-View Hand-Object Pipeline\n"
                    "wizard 顺序: Setup → Undistort → Detect → Pseudo → "
                    "(Finetune) → Infer → Visualize.  "
                    "每步状态写到 `<capture>/.pipeline/state.json`, 重启可恢复.")
        status_box = gr.Markdown(value=_status_md(), label="Pipeline status")

        with gr.Tabs():
            # ── Setup ──────────────────────────────────────────────
            with gr.Tab("0. Setup"):
                cap_dir = gr.Textbox(
                    label="Capture dir",
                    placeholder="/data2/fubingshuai/golf/data/.../<20260424...>",
                )
                seq = gr.Textbox(label="Seq name (任意, 通常是 capture 文件夹名)")
                auto_raw = gr.Checkbox(
                    label="Auto extract .raw → .jpg (如果 capture 下还没解过)",
                    value=True,
                )
                btn_setup = gr.Button("Initialize workspace", variant="primary")
                gr.Markdown("_重新点 Initialize 会从 `.pipeline/state.json` 恢复之前各 step 的预览._")
                out_setup = gr.JSON(label="Detected cameras + workspace")
                log_setup = gr.Code(label="Error / Traceback", language="markdown")
                # btn_setup.click 在所有 Tab 都创建完后才能 wire (要引用 step1-6 的组件)

            # ── Step 1: Undistort ──────────────────────────────────
            with gr.Tab("1. Undistort"):
                ud_force = gr.Checkbox(label="Force re-run (覆盖已存在的去畸变图)")
                btn_ud = gr.Button("Run undistort", variant="primary")
                out_ud = gr.JSON(label="Result")
                ud_frame = gr.Slider(minimum=0, maximum=1, value=0, step=1,
                                      label="Frame index (拖动 / 输入数字浏览)")
                with gr.Row():
                    preview_ud0 = gr.Image(label="cam0 undistorted")
                    preview_ud1 = gr.Image(label="cam1 undistorted")
                log_ud = gr.Code(label="Error / Traceback", language="markdown")
                btn_ud.click(cb_undistort, [ud_force],
                             [out_ud, status_box, ud_frame,
                              preview_ud0, preview_ud1, log_ud])
                ud_frame.change(cb_ud_browse, [ud_frame],
                                [preview_ud0, preview_ud1])

            # ── Step 2: Detection (YOLO / SAM2) ─────────────────────
            with gr.Tab("2. Hand Detection"):
                det_backend = gr.Radio(
                    choices=["yolo", "sam2"], value="yolo",
                    label="检测后端",
                    info="yolo: 逐帧 YOLO; sam2: 首帧手动标正/负点 → 自动 propagate 到全帧")

                # SAM2 标注 state (始终存在; sam2 不选用时也无害)
                ann_state0 = gr.State(_sam2_state_init(0))
                ann_state1 = gr.State(_sam2_state_init(1))

                with gr.Group(visible=False) as sam2_panel:
                    gr.Markdown(
                        "**SAM2 多 anchor 标注流程**: "
                        "点 **Load first frames** 加载 SAM2 (~5-10s) → "
                        "拖 **frame slider** 到任意一帧 → 选 **当前点类型** → 点击加点 (即时看到该帧的 mask) → "
                        "**再拖到别的帧**继续标 (anchor 跨帧持久化, 不会丢) → "
                        "在多个帧 (比如开局 / 中段 / 末段) 都标几个正负样本之后, "
                        "点 **Run hand detection** → SAM2 把所有 anchor 当作 memory, **从 frame 0 一口气 propagate 到末尾**, 整段视频都有 bbox.\n\n"
                        "**点颜色**: 绿=右手正, 红=右手负, 蓝=左手正, 紫=左手负. "
                        "**mask**: 绿色半透明=右手, 蓝色半透明=左手.\n"
                        "**约束**: 至少在一帧上, 一只手有 ≥1 个正点."
                    )
                    with gr.Row():
                        btn_sam2_load = gr.Button("Load first frames", variant="secondary")
                        sam2_log = gr.Markdown("")

                    # cam0 板块 (整个一行宽, 大图)
                    gr.Markdown("### cam0")
                    ann_frame0 = gr.Slider(
                        minimum=0, maximum=1, value=0, step=1,
                        label="frame (cam0) — 在多个帧上标 anchor (Load 后会更新到真实帧数)")
                    with gr.Row():
                        ann_active0 = gr.Radio(
                            choices=list(_SAM2_LABELS),
                            value="right_pos",
                            label="当前点类型 (cam0)",
                            scale=3)
                        btn_clear_this0 = gr.Button("Clear this frame", scale=1)
                        btn_clear_all0 = gr.Button("Clear ALL anchors (cam0)", scale=1)
                    ann_img0 = gr.Image(label="cam0 current frame (点击加点)",
                                          interactive=False, height=700)

                    # cam1 板块
                    gr.Markdown("### cam1")
                    ann_frame1 = gr.Slider(
                        minimum=0, maximum=1, value=0, step=1,
                        label="frame (cam1) — 在多个帧上标 anchor (Load 后会更新到真实帧数)")
                    with gr.Row():
                        ann_active1 = gr.Radio(
                            choices=list(_SAM2_LABELS),
                            value="right_pos",
                            label="当前点类型 (cam1)",
                            scale=3)
                        btn_clear_this1 = gr.Button("Clear this frame", scale=1)
                        btn_clear_all1 = gr.Button("Clear ALL anchors (cam1)", scale=1)
                    ann_img1 = gr.Image(label="cam1 current frame (点击加点)",
                                          interactive=False, height=700)

                    btn_sam2_load.click(
                        cb_sam2_load_frames, [],
                        [ann_img0, ann_img1, ann_state0, ann_state1, sam2_log,
                         ann_frame0, ann_frame1])
                    # slider 用 .release: 只在用户松开鼠标时触发, 避免拖动过程中频繁切帧
                    ann_frame0.release(cb_sam2_browse_frame,
                                         [ann_state0, ann_frame0, ann_active0],
                                         [ann_img0, ann_state0])
                    ann_frame1.release(cb_sam2_browse_frame,
                                         [ann_state1, ann_frame1, ann_active1],
                                         [ann_img1, ann_state1])
                    ann_img0.select(cb_sam2_click,
                                     [ann_state0, ann_active0],
                                     [ann_img0, ann_state0])
                    ann_img1.select(cb_sam2_click,
                                     [ann_state1, ann_active1],
                                     [ann_img1, ann_state1])
                    btn_clear_this0.click(cb_sam2_clear_this_frame,
                                            [ann_state0, ann_active0],
                                            [ann_img0, ann_state0])
                    btn_clear_this1.click(cb_sam2_clear_this_frame,
                                            [ann_state1, ann_active1],
                                            [ann_img1, ann_state1])
                    btn_clear_all0.click(cb_sam2_clear_all_anchors,
                                            [ann_state0, ann_active0],
                                            [ann_img0, ann_state0])
                    btn_clear_all1.click(cb_sam2_clear_all_anchors,
                                            [ann_state1, ann_active1],
                                            [ann_img1, ann_state1])
                    # 切 active label 时重渲染图 (隐藏掉非 active 那只手的点/mask)
                    ann_active0.change(cb_sam2_active_change,
                                         [ann_state0, ann_active0], [ann_img0])
                    ann_active1.change(cb_sam2_active_change,
                                         [ann_state1, ann_active1], [ann_img1])

                def _toggle_sam2(b):
                    return gr.update(visible=(b == "sam2"))
                det_backend.change(_toggle_sam2, [det_backend], [sam2_panel])

                det_force = gr.Checkbox(label="Force re-detect (重跑全部帧)")
                det_bbox_size = gr.Number(
                    value=1.0, precision=2, step=0.1,
                    label="bbox patch (=1 不扩, 1.5 中心不变宽高各 ×1.5, 通常 1.0~2.0)",
                    info="给检测出的 bbox 加 padding. 1.0 = 紧 bbox; 改了要勾 Force re-run 才生效.")
                btn_det = gr.Button("Run hand detection", variant="primary")
                out_det = gr.JSON(label="Detection summary")
                with gr.Row():
                    preview_det0 = gr.Video(label="cam0 bbox overlay mp4 (60fps)")
                    preview_det1 = gr.Video(label="cam1 bbox overlay mp4 (60fps)")
                # ── 单帧浏览 ───────────────────────────────────
                gr.Markdown("**单帧浏览** — 拖动看任意帧的 bbox")
                det_frame = gr.Slider(minimum=0, maximum=1, value=0, step=1,
                                       label="frame index")
                with gr.Row():
                    det_img0 = gr.Image(label="cam0 single frame",
                                         interactive=False, height=400)
                    det_img1 = gr.Image(label="cam1 single frame",
                                         interactive=False, height=400)
                det_frame.change(cb_det_browse, [det_frame],
                                  [det_img0, det_img1])
                log_det = gr.Code(label="Error / Traceback", language="markdown")
                btn_det.click(cb_detect,
                              [det_backend, det_force,
                               ann_state0, ann_state1, det_bbox_size],
                              [out_det, status_box,
                               preview_det0, preview_det1, log_det,
                               det_frame, det_img0, det_img1])

            # ── Step 3: Pseudo Label (HaMER / WiLoR) ──────────────
            with gr.Tab("3. Pseudo Label"):
                ps_backend = gr.Radio(
                    choices=["hamer", "wilor"], value="hamer",
                    label="伪标后端",
                    info="hamer: 原版, 稳; wilor: 新版, 通常更准, 输出 npz 格式一致")
                ps_force = gr.Checkbox(label="Force re-run (覆盖已存在的 npz)")
                ps_mp4 = gr.Checkbox(label="同时生成 _pseudo_vis mp4 + 每帧 jpg",
                                      value=True)
                btn_ps = gr.Button("Generate pseudo labels", variant="primary")
                out_ps = gr.JSON(label="Pseudo label summary")
                preview_ps = gr.Video(label="_pseudo_vis 全帧 overlay mp4 (60fps)")
                # ── 单帧浏览 ───────────────────────────────────
                gr.Markdown("**单帧浏览** — 拖动看任意帧的 21 关节伪标 (cam0 | cam1)")
                ps_frame = gr.Slider(minimum=0, maximum=1, value=0, step=1,
                                      label="frame index")
                ps_img = gr.Image(label="pseudo label single frame",
                                    interactive=False, height=400)
                ps_frame.change(cb_ps_browse, [ps_frame], [ps_img])
                log_ps = gr.Code(label="Error / Traceback", language="markdown")
                btn_ps.click(cb_pseudo, [ps_backend, ps_force, ps_mp4],
                             [out_ps, status_box, preview_ps, log_ps,
                              ps_frame, ps_img])

            # ── Step 4: Finetune ──────────────────────────────────
            with gr.Tab("4. Self-supervised Finetune (optional)"):
                ft_enable = gr.Checkbox(label="Enable finetune", value=False)
                ft_epochs = gr.Slider(0, 50, value=5, step=1, label="epochs")
                ft_lr = gr.Number(value=1e-5, label="learning rate")
                ft_bs = gr.Slider(1, 8, value=1, step=1, label="batch size")
                btn_ft = gr.Button("Run finetune / mark skipped",
                                    variant="primary")
                out_ft = gr.JSON(label="Finetune result")
                log_ft = gr.Code(label="Error / Traceback", language="markdown")
                # ckpt_dd 在 step5 tab 里定义, btn_ft 的 outputs wire 推后

            # ── Step 5: Multi-view Inference ───────────────────────
            with gr.Tab("5. Multi-view Inference + npy"):
                with gr.Row():
                    ckpt_dd = gr.Dropdown(
                        label="Checkpoint",
                        info="finetune 跑过自动默认它的 ckpt; 否则用 exp/new/checkpoints/checkpoint_30. "
                             "新 finetune 完点 Refresh 刷新列表.",
                        choices=[_DEFAULT_CKPT_LABEL],
                        value=_DEFAULT_CKPT_LABEL,
                        allow_custom_value=True,
                    )
                    btn_refresh_ckpt = gr.Button("🔄 Refresh ckpt list",
                                                  scale=0, min_width=180)
                btn_in = gr.Button("Run HE inference + assemble npy",
                                    variant="primary")
                out_in = gr.JSON(label="Inference result")
                with gr.Row():
                    preview_h0 = gr.Video(label="hand0 2D overlay (right, 60fps)")
                    preview_h1 = gr.Video(label="hand1 2D overlay (left, 60fps)")
                # ── 单帧浏览 ───────────────────────────────────
                gr.Markdown("**单帧浏览** — 拖动看任意帧的双手 3D 投影 (cam0 | cam1 拼接)")
                inf_frame = gr.Slider(minimum=0, maximum=1, value=0, step=1,
                                       label="frame index")
                with gr.Row():
                    inf_img0 = gr.Image(label="hand0 (right) single frame",
                                          interactive=False, height=400)
                    inf_img1 = gr.Image(label="hand1 (left) single frame",
                                          interactive=False, height=400)
                inf_frame.change(cb_inf_browse, [inf_frame],
                                  [inf_img0, inf_img1])
                npy_file = gr.File(label="下载 npy")
                log_in = gr.Code(label="Error / Traceback", language="markdown")
                btn_refresh_ckpt.click(cb_refresh_ckpts, [], [ckpt_dd])
                btn_in.click(cb_infer, [ckpt_dd],
                             [out_in, status_box, preview_h0, preview_h1,
                              npy_file, log_in,
                              inf_frame, inf_img0, inf_img1])

            # ── Step 6: Visualization (way_vis) ────────────────────
            with gr.Tab("6. 3D Visualization (way_vis)"):
                vis_obj = gr.Textbox(label="Object mesh (留空则从 config/baseball_golf.json 读 club_mesh_path)")
                vis_joints = gr.Checkbox(label="叠加 21 关节骨架", value=False)
                vis_overlay = gr.Checkbox(label="overlay 在原图上", value=True)
                vis_fps = gr.Slider(5, 60, value=10, step=1, label="output fps")
                btn_vis = gr.Button("Render way_vis", variant="primary")
                out_vis = gr.JSON(label="Visualization result")
                with gr.Row():
                    preview_v0 = gr.Video(label="trajectory_view0.mp4")
                    preview_v1 = gr.Video(label="trajectory_view1.mp4")
                log_vis = gr.Code(label="Error / Traceback", language="markdown")
                btn_vis.click(cb_vis,
                              [vis_obj, vis_joints, vis_overlay, vis_fps],
                              [out_vis, status_box, preview_v0, preview_v1,
                               log_vis])

        # 现在所有组件都创建好了, wire setup 的 click 把 state 里的预览全恢复
        btn_ft.click(cb_finetune, [ft_enable, ft_epochs, ft_lr, ft_bs],
                     [out_ft, status_box, log_ft, ckpt_dd])

        btn_setup.click(
            cb_setup, [cap_dir, seq, auto_raw],
            [out_setup, status_box, log_setup,
             out_ud, ud_frame, preview_ud0, preview_ud1,
             out_det, preview_det0, preview_det1, det_frame, det_img0, det_img1,
             out_ps, preview_ps, ps_frame, ps_img,
             out_ft,
             out_in, preview_h0, preview_h1, npy_file,
                inf_frame, inf_img0, inf_img1,
             out_vis, preview_v0, preview_v1,
             ckpt_dd],
        )

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="开启 gradio share link")
    args = ap.parse_args()

    demo = build_ui()
    demo.queue()  # 长时间任务用队列
    # allowed_paths: gradio 默认不允许 cwd / /tmp 之外的文件出去, 把 capture 数据
    # 根 + workspace 根都允许. 用户的 capture 一般在 /data2/fubingshuai/golf 下.
    allowed = ["/data2/fubingshuai/golf", str(Path.home())]
    demo.launch(server_name=args.host, server_port=args.port,
                share=args.share, inbrowser=False,
                theme=gr.themes.Soft(),
                allowed_paths=allowed)


if __name__ == "__main__":
    main()
