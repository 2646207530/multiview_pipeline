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
from typing import Any, Dict

# 避免共享机器上 /tmp/gradio 被别人创建后 PermissionDenied.
# 优先用 GRADIO_TEMP_DIR / TMPDIR; 否则 fallback 到家目录下私有目录.
# 必须在 `import gradio` 之前设, 不然 gradio 已经按 /tmp 初始化好了.
_gradio_tmp = (os.environ.get("GRADIO_TEMP_DIR")
                or os.environ.get("TMPDIR")
                or str(Path.home() / ".gradio_tmp"))
Path(_gradio_tmp).mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = _gradio_tmp

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


def _list_workspace_ckpts(ws: Workspace) -> list:
    """列出 <workspace>/_finetune/ 下含 TestMultiviewStereo.pth.tar 的子目录."""
    if ws is None or not ws.finetune_dir.is_dir():
        return []
    out = []
    for p in sorted(ws.finetune_dir.iterdir()):
        if p.is_dir() and (p / "TestMultiviewStereo.pth.tar").is_file():
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
    return (
        o1, ud_slider, _ud_frame_image(0, 0), _ud_frame_image(1, 0),
        o2, _existing(det_videos.get("0")), _existing(det_videos.get("1")),
        o3, _existing(o3.get("pseudo_overlay_mp4")),
        o4,
        o5, _existing(o5.get("hand0_mp4")), _existing(o5.get("hand1_mp4")),
            _existing(o5.get("npy_path")),
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
        # restore 元组结构: 17 项 (step1: o1+slider+ud0+ud1, step2-6 同前)
        empty = (None, gr.Slider(), None, None,
                 None, None, None,
                 None, None,
                 None,
                 None, None, None, None,
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
    return {
        "right": {"pos": [], "neg": [], "mask": None},
        "left":  {"pos": [], "neg": [], "mask": None},
        "image": None,
        "ci":    ci,
    }


def _sam2_render(state: dict):
    """渲染顺序: 原图 → 各手 mask 半透明叠加 → 各点圆点叠上."""
    if state is None or state.get("image") is None:
        return None
    img = state["image"].copy()
    # 1. mask 半透明
    for hand in ("right", "left"):
        m = state[hand].get("mask")
        if m is None:
            continue
        color = _SAM2_MASK_COLOR[hand]
        overlay = img.copy()
        overlay[m] = color
        img = cv2.addWeighted(img, 0.55, overlay, 0.45, 0.0)
    # 2. 点 (大一圈, 方便看清)
    for hand in ("right", "left"):
        for typ in ("pos", "neg"):
            color = _SAM2_POINT_COLOR[f"{hand}_{typ}"]
            marker = "+" if typ == "pos" else "-"
            for x, y in state[hand][typ]:
                cv2.circle(img, (int(x), int(y)), 10, color, -1)
                cv2.circle(img, (int(x), int(y)), 10, (255, 255, 255), 2)
                # 中心画 +/- 标记
                cv2.putText(img, marker, (int(x) - 5, int(y) + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 2, cv2.LINE_AA)
    return img


def _sam2_recompute_mask(state: dict, hand: str):
    """对某只手, 跑 SAM2 image predict 拿 mask, 写回 state.
    如果 set_image 没被调过 (Load 没点 / 失败), 会用 state 里的 image 自动补."""
    from pipeline.sam2_interactive import predict_hand_mask
    ci = state.get("ci")
    pos = state[hand]["pos"]
    neg = state[hand]["neg"]
    if ci is None or not pos:
        state[hand]["mask"] = None
        return
    try:
        state[hand]["mask"] = predict_hand_mask(
            ci, pos, neg,
            image_rgb_fallback=state.get("image"),
        )
    except Exception as e:
        print(f"[sam2 interactive] cam{ci} {hand} predict fail: {e}")
        state[hand]["mask"] = None


def cb_sam2_load_frames():
    """从 ws 的 undistort 输出读 cam0/cam1 首帧 + 给 SAM2 image predictor
    各做一次 set_image (heavy, ~5-10s 一次)."""
    from pipeline.sam2_interactive import set_image_for_cam, reset_cam
    try:
        ws = _ws_or_raise()
    except Exception as e:
        return (None, None, _sam2_state_init(0), _sam2_state_init(1), str(e))
    state = PipelineState.load(ws)
    ud = state.steps.get("undistort")
    if ud is None or ud.status != "done":
        return (None, None, _sam2_state_init(0), _sam2_state_init(1),
                "Step 1 (undistort) 没完成, 先跑去畸变.")
    undist_root = Path(ud.outputs["undist_root"])
    capture_id = ud.outputs["capture_id"]
    out_imgs, out_states, errs = [], [], []
    for ci in (0, 1):
        d = undist_root / capture_id / str(ci) / "images_undistorted"
        first = next(iter(sorted(d.glob("*.jpg"))), None)
        if first is None:
            errs.append(f"cam{ci}: 找不到首帧 (在 {d})")
            out_imgs.append(None)
            out_states.append(_sam2_state_init(ci))
            continue
        bgr = cv2.imread(str(first))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
        s = _sam2_state_init(ci)
        s["image"] = rgb
        # heavy: 给 SAM2 image predictor 喂图
        reset_cam(ci)
        if rgb is not None:
            try:
                set_image_for_cam(ci, rgb)
            except Exception as e:
                errs.append(f"cam{ci} set_image fail: {e}")
                print(f"[sam2 interactive] cam{ci} set_image fail: {e}")
        out_imgs.append(rgb)
        out_states.append(s)
    if errs:
        msg = "⚠️ 加载部分失败 (第一次点击会自动补 set_image 重试):\n- " + "\n- ".join(errs)
    else:
        msg = "✅ 已加载 cam0/cam1 首帧 + SAM2 image predictor, 可以开始点了."
    return out_imgs[0], out_imgs[1], out_states[0], out_states[1], msg


def cb_sam2_click(evt: gr.SelectData, state: dict, active_label: str):
    if state is None or state.get("image") is None:
        return (state.get("image") if state else None), state
    x, y = evt.index
    hand, typ = active_label.split("_")
    state[hand][typ].append([float(x), float(y)])
    # 即时重算这只手的 mask
    _sam2_recompute_mask(state, hand)
    return _sam2_render(state), state


def cb_sam2_clear(state: dict):
    """清掉这个 cam 的所有标点 + mask, 保留 image 和 ci."""
    img = state.get("image") if state else None
    ci = state.get("ci") if state else None
    s = _sam2_state_init(ci)
    s["image"] = img
    return _sam2_render(s), s


def cb_detect(backend: str, force: bool,
              ann0: dict, ann1: dict,
              progress=gr.Progress(track_tqdm=False)):
    ws = _ws_or_raise()
    def _p(frac, msg):
        progress(frac, desc=msg)

    sam2_prompts = None
    if backend == "sam2":
        sam2_prompts = {
            str(ci): {
                "right": {"pos": (ann or {}).get("right", {}).get("pos", []),
                          "neg": (ann or {}).get("right", {}).get("neg", [])},
                "left":  {"pos": (ann or {}).get("left", {}).get("pos", []),
                          "neg": (ann or {}).get("left", {}).get("neg", [])},
            }
            for ci, ann in ((0, ann0), (1, ann1))
        }
        # 至少给一个 cam 一只手有正点
        any_pos = any(
            sam2_prompts[c][h]["pos"]
            for c in ("0", "1") for h in ("right", "left")
        )
        if not any_pos:
            return ({"error": "SAM2 需要在首帧给手标至少一个正样本点"},
                    _status_md(), None, None, "")
    try:
        info = step2_detect.run(ws, progress=_p, force=force,
                                 backend=backend, sam2_prompts=sam2_prompts)
        ctx.state = PipelineState.load(ws)
        vv = info.get("vis_videos") or {}
        return (info, _status_md(),
                _existing(vv.get("0")), _existing(vv.get("1")), "")
    except Exception as e:
        return ({"error": str(e)}, _status_md(),
                None, None, traceback.format_exc())


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
        return info, _status_md(), info.get("pseudo_overlay_mp4"), ""
    except Exception as e:
        return {"error": str(e)}, _status_md(), None, traceback.format_exc()


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
        return (info, _status_md(),
                info.get("hand0_mp4"), info.get("hand1_mp4"),
                info.get("npy_path"), "")
    except Exception as e:
        return ({"error": str(e)}, _status_md(),
                None, None, None, traceback.format_exc())


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
                        "**SAM2 标注流程**: 点 **Load first frames** 拉首帧 + 加载 SAM2 (~5-10s) → "
                        "选 **当前点类型** (右手正 / 右手负 / 左手正 / 左手负) → "
                        "在图上点击加点 (每点击一次即时看到这只手的 mask, 觉得不准可以继续加点) → "
                        "点 **Run hand detection** 进入序列分割.\n\n"
                        "**点颜色**: 绿=右手正, 红=右手负, 蓝=左手正, 紫=左手负. "
                        "**mask**: 绿色半透明=右手, 蓝色半透明=左手.\n"
                        "**约束**: 每个 cam 至少给一只手 ≥1 个正点."
                    )
                    with gr.Row():
                        btn_sam2_load = gr.Button("Load first frames", variant="secondary")
                        sam2_log = gr.Markdown("")

                    # cam0 板块 (整个一行宽, 大图)
                    gr.Markdown("### cam0")
                    with gr.Row():
                        ann_active0 = gr.Radio(
                            choices=list(_SAM2_LABELS),
                            value="right_pos",
                            label="当前点类型 (cam0)",
                            scale=2)
                        btn_clear0 = gr.Button("Clear cam0 points", scale=1)
                    ann_img0 = gr.Image(label="cam0 first frame (点击加点)",
                                          interactive=False, height=700)

                    # cam1 板块
                    gr.Markdown("### cam1")
                    with gr.Row():
                        ann_active1 = gr.Radio(
                            choices=list(_SAM2_LABELS),
                            value="right_pos",
                            label="当前点类型 (cam1)",
                            scale=2)
                        btn_clear1 = gr.Button("Clear cam1 points", scale=1)
                    ann_img1 = gr.Image(label="cam1 first frame (点击加点)",
                                          interactive=False, height=700)

                    btn_sam2_load.click(
                        cb_sam2_load_frames, [],
                        [ann_img0, ann_img1, ann_state0, ann_state1, sam2_log])
                    ann_img0.select(cb_sam2_click,
                                     [ann_state0, ann_active0],
                                     [ann_img0, ann_state0])
                    ann_img1.select(cb_sam2_click,
                                     [ann_state1, ann_active1],
                                     [ann_img1, ann_state1])
                    btn_clear0.click(cb_sam2_clear, [ann_state0],
                                       [ann_img0, ann_state0])
                    btn_clear1.click(cb_sam2_clear, [ann_state1],
                                       [ann_img1, ann_state1])

                def _toggle_sam2(b):
                    return gr.update(visible=(b == "sam2"))
                det_backend.change(_toggle_sam2, [det_backend], [sam2_panel])

                det_force = gr.Checkbox(label="Force re-detect (重跑全部帧)")
                btn_det = gr.Button("Run hand detection", variant="primary")
                out_det = gr.JSON(label="Detection summary")
                with gr.Row():
                    preview_det0 = gr.Video(label="cam0 bbox overlay mp4")
                    preview_det1 = gr.Video(label="cam1 bbox overlay mp4")
                log_det = gr.Code(label="Error / Traceback", language="markdown")
                btn_det.click(cb_detect,
                              [det_backend, det_force, ann_state0, ann_state1],
                              [out_det, status_box,
                               preview_det0, preview_det1, log_det])

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
                preview_ps = gr.Video(label="_pseudo_vis 全帧 overlay mp4")
                log_ps = gr.Code(label="Error / Traceback", language="markdown")
                btn_ps.click(cb_pseudo, [ps_backend, ps_force, ps_mp4],
                             [out_ps, status_box, preview_ps, log_ps])

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
                    preview_h0 = gr.Video(label="hand0 2D overlay (right)")
                    preview_h1 = gr.Video(label="hand1 2D overlay (left)")
                npy_file = gr.File(label="下载 npy")
                log_in = gr.Code(label="Error / Traceback", language="markdown")
                btn_refresh_ckpt.click(cb_refresh_ckpts, [], [ckpt_dd])
                btn_in.click(cb_infer, [ckpt_dd],
                             [out_in, status_box, preview_h0, preview_h1,
                              npy_file, log_in])

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
             out_det, preview_det0, preview_det1,
             out_ps, preview_ps,
             out_ft,
             out_in, preview_h0, preview_h1, npy_file,
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
