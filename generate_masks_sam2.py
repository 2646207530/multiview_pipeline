"""Generate per-frame segmentation masks via SAM2 video predictor.

Mask format (per requested frame): uint8 (H, W) with
    0 = background, 1 = club, 2 = right hand, 3 = left hand

Workflow (replaces the old per-frame image-mode pipeline):
    1) Symlink the requested frame range into a tmp video dir for SAM2.
    2) On the first requested frame the user labels each of the 3 targets
       (right hand → left hand → club) with positive / negative clicks.
       SAM2 returns a live preview after every click.
    3) SAM2VideoPredictor.propagate_in_video tracks all 3 targets through
       the rest of the requested range automatically.
    4) Optional review loop: navigate frames, fix bad ones with a few
       clicks, re-propagate.
    5) Save uint8 npy + colored overlay JPG per requested frame.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


_ROOT = Path(__file__).resolve().parent

# (label_id, english_name, chinese_name)  — order = labeling order on key frame
_TARGETS: List[Tuple[int, str, str]] = [
    (2, 'right hand', '右手'),
    (3, 'left hand',  '左手'),
    (1, 'club',       '球杆'),
]
_NAME_ZH = {1: '球杆', 2: '右手', 3: '左手'}
_NAME_EN = {1: 'club', 2: 'right hand', 3: 'left hand'}
_COLOR_BGR = {1: (0, 128, 255),    # orange
              2: (0, 255,   0),    # green
              3: (255, 128, 64)}   # blue-ish


# --------------------------------------------------------------------- ckpt ---
def _resolve_sam2_ckpt_and_cfg():
    ckpt_dir = _ROOT / "model" / "sam2_ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "sam2.1_hiera_small.pt"
    cfg_name = "configs/sam2.1/sam2.1_hiera_s.yaml"
    if not ckpt_path.exists():
        import urllib.request
        url = ("https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
               "sam2.1_hiera_small.pt")
        print(f"[SAM2] 下载权重到 {ckpt_path} ...")
        urllib.request.urlretrieve(url, ckpt_path)
    return cfg_name, str(ckpt_path)


# ------------------------------------------------------- helpers / overlays ---
def _build_video_dir(imgpath_dir: str, imgnames: Sequence[str],
                     requested_frames: Sequence[int]):
    """
    Symlink a contiguous window covering ``requested_frames`` into a tmp dir,
    naming them ``000000.jpg, 000001.jpg, ...`` so SAM2's frame loader can
    sort them by integer stem.

    Returns: (tmp_dir, fi_list, fi_to_vi)
        fi_list : original frame indices in video order (i.e. video idx i = fi_list[i])
        fi_to_vi: dict mapping original frame idx -> sam2 video frame idx
    """
    rmin = min(requested_frames)
    rmax = max(requested_frames)
    tmp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
    fi_to_vi: Dict[int, int] = {}
    fi_list: List[int] = []
    vi = 0
    for fi in range(rmin, rmax + 1):
        if fi >= len(imgnames) or not imgnames[fi]:
            print(f"[Mask] 警告: 帧 {fi} imgname 为空，跳过")
            continue
        src = os.path.abspath(os.path.join(imgpath_dir, imgnames[fi]))
        if not os.path.exists(src):
            print(f"[Mask] 警告: 帧 {fi} 文件不存在: {src}")
            continue
        dst = os.path.join(tmp_dir, f"{vi:06d}.jpg")
        os.symlink(src, dst)
        fi_to_vi[fi] = vi
        fi_list.append(fi)
        vi += 1
    return tmp_dir, fi_list, fi_to_vi


def _logits_to_mask(logits, target_hw):
    """A tensor of shape (1,H,W) or (H,W) of mask logits -> bool ndarray at target_hw."""
    arr = logits.detach()
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr.float().cpu().numpy()
    if arr.shape != tuple(target_hw):
        arr = cv2.resize(arr, (target_hw[1], target_hw[0]),
                         interpolation=cv2.INTER_LINEAR)
    return arr > 0


def _draw_overlay(img_bgr, masks_by_id, points=None, header=None, sub=None,
                  active_id=None, disp_scale=1.0):
    H, W = img_bgr.shape[:2]
    v = img_bgr.copy()
    for lid, m in sorted(masks_by_id.items()):
        if m is None or not m.any():
            continue
        c = np.array(_COLOR_BGR[lid], dtype=np.float32)
        # active target a bit more opaque
        alpha = 0.40 if lid == active_id else 0.55
        v[m] = (v[m].astype(np.float32) * alpha +
                c * (1.0 - alpha)).astype(np.uint8)
    if points:
        for x, y, pos, lid in points:
            color = _COLOR_BGR.get(lid, (255, 255, 255))
            if pos:
                cv2.circle(v, (x, y), 7, color, -1)
                cv2.circle(v, (x, y), 8, (255, 255, 255), 2)
            else:
                cv2.drawMarker(v, (x, y), (0, 0, 255),
                               cv2.MARKER_CROSS, 18, 3)
                cv2.drawMarker(v, (x, y), (255, 255, 255),
                               cv2.MARKER_CROSS, 18, 1)
    if header:
        cv2.putText(v, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 2)
    if sub:
        cv2.putText(v, sub, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)
    if disp_scale != 1.0:
        v = cv2.resize(v, (int(W * disp_scale), int(H * disp_scale)))
    return v


# ----------------------------------------------------- key-frame labeling ----
def _label_target_on_keyframe(predictor, state, key_vi, key_fi, label_id,
                              img_bgr, prior_masks, win_name, disp_scale):
    """Collect clicks for one obj_id on the key frame; return final bool mask or None."""
    H, W = img_bgr.shape[:2]
    name_zh = _NAME_ZH[label_id]
    name_en = _NAME_EN[label_id]
    points: List[Tuple[int, int, bool]] = []
    cur_mask = np.zeros((H, W), dtype=bool)

    print(f"\n[Mask] >>> 关键帧标注 {name_zh} ({name_en}, obj_id={label_id})  "
          f"key fi={key_fi}")
    print("[Mask]     左键=正点  右键=负点  u=撤销  c=清空  SPACE=确认  n=跳过此目标  q=退出")

    def predict_now():
        nonlocal cur_mask
        if not points:
            cur_mask = np.zeros((H, W), dtype=bool)
            return
        coords = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
        labels = np.array([1 if p[2] else 0 for p in points], dtype=np.int32)
        _, obj_ids, video_res_masks = predictor.add_new_points_or_box(
            inference_state=state, frame_idx=key_vi, obj_id=label_id,
            points=coords, labels=labels,
        )
        idx = list(obj_ids).index(label_id)
        cur_mask = _logits_to_mask(video_res_masks[idx], (H, W))

    def render():
        masks = dict(prior_masks)
        masks[label_id] = cur_mask
        pts = [(x, y, pos, label_id) for x, y, pos in points]
        header = (f"KEYFRAME fi={key_fi}  Target: {name_en} ({name_zh})  "
                  f"px={int(cur_mask.sum())}")
        sub = "Lclick=+  Rclick=-  u=undo  c=clear  SPACE=confirm  n=skip  q=quit"
        return _draw_overlay(img_bgr, masks, pts, header, sub,
                             active_id=label_id, disp_scale=disp_scale)

    def on_mouse(event, x, y, flags, param):
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        ox, oy = int(x / disp_scale), int(y / disp_scale)
        is_pos = (event == cv2.EVENT_LBUTTONDOWN)
        points.append((ox, oy, is_pos))
        sign = '+' if is_pos else '-'
        print(f"[Mask][{name_zh}] {sign} ({ox},{oy})  total {len(points)}")
        predict_now()
        cv2.imshow(win_name, render())

    cv2.setMouseCallback(win_name, on_mouse)
    cv2.imshow(win_name, render())

    try:
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('u'):
                if points:
                    p = points.pop()
                    print(f"[Mask][{name_zh}] 撤销 ({p[0]},{p[1]})  剩 {len(points)}")
                    predict_now()
                    cv2.imshow(win_name, render())
            elif key == ord('c'):
                points.clear()
                predictor.clear_all_prompts_in_frame(
                    state, key_vi, label_id, need_output=False)
                cur_mask = np.zeros((H, W), dtype=bool)
                cv2.imshow(win_name, render())
            elif key in (ord(' '), 13):  # SPACE / ENTER
                if points:
                    print(f"[Mask][{name_zh}] ✓ 确认 {int(cur_mask.sum())} 像素")
                    return cur_mask.copy()
                print(f"[Mask][{name_zh}] 无点，视为跳过")
                predictor.clear_all_prompts_in_frame(
                    state, key_vi, label_id, need_output=False)
                return None
            elif key == ord('n'):
                print(f"[Mask][{name_zh}] 跳过此目标")
                points.clear()
                predictor.clear_all_prompts_in_frame(
                    state, key_vi, label_id, need_output=False)
                return None
            elif key == ord('q'):
                raise KeyboardInterrupt('用户在关键帧标注中退出')
    finally:
        cv2.setMouseCallback(win_name, lambda *a, **k: None)


# ----------------------------------------------------------- propagation ----
def _propagate_full(predictor, state, fi_list):
    """Run ``propagate_in_video`` over the full window; collect bool masks per (vi,obj_id)."""
    n = len(fi_list)
    per_frame: Dict[int, Dict[int, np.ndarray]] = {vi: {} for vi in range(n)}
    H = state["video_height"]
    W = state["video_width"]
    print(f"[Mask] 开始视频传播（共 {n} 帧）...")
    for vi, obj_ids, video_res_masks in predictor.propagate_in_video(state):
        for j, obj_id in enumerate(obj_ids):
            per_frame[vi][int(obj_id)] = _logits_to_mask(
                video_res_masks[j], (H, W))
    print("[Mask] 视频传播完成")
    return per_frame


# ----------------------------------------------- review / correction loop ----
def _correct_frame(predictor, state, vi, fi, img_bgr, prior_masks,
                   win_name, disp_scale):
    """Edit corrections on one frame.

    Returns True if the user pressed SPACE (apply + re-propagate); False on ESC.
    On cancel, prompts added during this edit are cleared from the SAM2 state.
    """
    H, W = img_bgr.shape[:2]
    active_id = 2  # default = right hand
    points_per_obj: Dict[int, List[Tuple[int, int, bool]]] = {1: [], 2: [], 3: []}
    masks_per_obj: Dict[int, np.ndarray] = {
        lid: prior_masks.get(lid, np.zeros((H, W), bool)).copy()
        for lid in (1, 2, 3)
    }

    print(f"\n[Mask] *** 编辑 fi={fi} (vi={vi}) ***")
    print("[Mask]     1=右手  2=左手  3=球杆 (切换目标)")
    print("[Mask]     左键=正点  右键=负点  u=撤销当前目标  c=清空当前目标")
    print("[Mask]     SPACE=确认&重新传播   ESC=取消编辑")

    key_to_obj = {ord('1'): 2, ord('2'): 3, ord('3'): 1}

    def predict_for(lid):
        pts = points_per_obj[lid]
        if not pts:
            predictor.clear_all_prompts_in_frame(
                state, vi, lid, need_output=False)
            masks_per_obj[lid] = prior_masks.get(lid,
                                                 np.zeros((H, W), bool)).copy()
            return
        coords = np.array([[p[0], p[1]] for p in pts], dtype=np.float32)
        labels = np.array([1 if p[2] else 0 for p in pts], dtype=np.int32)
        _, obj_ids, video_res_masks = predictor.add_new_points_or_box(
            inference_state=state, frame_idx=vi, obj_id=lid,
            points=coords, labels=labels,
        )
        idx = list(obj_ids).index(lid)
        masks_per_obj[lid] = _logits_to_mask(video_res_masks[idx], (H, W))

    def render():
        all_pts = []
        for lid, lst in points_per_obj.items():
            all_pts.extend((x, y, pos, lid) for x, y, pos in lst)
        npx = int(masks_per_obj.get(active_id, np.zeros(0)).sum())
        header = (f"EDIT fi={fi}  active={_NAME_EN[active_id]} "
                  f"({_NAME_ZH[active_id]}, obj_id={active_id})  px={npx}")
        sub = "1=R 2=L 3=club  L+ R-  u=undo  c=clear  SPACE=apply  ESC=cancel"
        return _draw_overlay(img_bgr, masks_per_obj, all_pts, header, sub,
                             active_id=active_id, disp_scale=disp_scale)

    def on_mouse(event, x, y, flags, param):
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        ox, oy = int(x / disp_scale), int(y / disp_scale)
        is_pos = (event == cv2.EVENT_LBUTTONDOWN)
        points_per_obj[active_id].append((ox, oy, is_pos))
        sign = '+' if is_pos else '-'
        print(f"[Mask][edit][{_NAME_ZH[active_id]}] {sign} ({ox},{oy})")
        predict_for(active_id)
        cv2.imshow(win_name, render())

    cv2.setMouseCallback(win_name, on_mouse)
    cv2.imshow(win_name, render())

    def cancel_all_local_prompts():
        for lid in (1, 2, 3):
            if points_per_obj[lid]:
                predictor.clear_all_prompts_in_frame(
                    state, vi, lid, need_output=False)

    try:
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in key_to_obj:
                active_id = key_to_obj[key]
                cv2.imshow(win_name, render())
            elif key == ord('u'):
                if points_per_obj[active_id]:
                    p = points_per_obj[active_id].pop()
                    print(f"[Mask][edit] 撤销 {_NAME_ZH[active_id]} ({p[0]},{p[1]})")
                    predict_for(active_id)
                    cv2.imshow(win_name, render())
            elif key == ord('c'):
                points_per_obj[active_id].clear()
                predict_for(active_id)
                cv2.imshow(win_name, render())
            elif key in (ord(' '), 13):  # SPACE / ENTER -> apply
                any_changed = any(points_per_obj[lid] for lid in (1, 2, 3))
                if not any_changed:
                    print("[Mask][edit] 无新增 prompt，相当于取消")
                    return False
                print("[Mask][edit] ✓ 应用并重新传播")
                return True
            elif key == 27:  # ESC
                cancel_all_local_prompts()
                print("[Mask][edit] 取消，已清除本次新增 prompt")
                return False
    finally:
        cv2.setMouseCallback(win_name, lambda *a, **k: None)


def _review_loop(predictor, state, fi_list, fi_to_vi, imgnames, imgpath_dir,
                 per_frame_masks, win_name, max_disp_w):
    """Navigate / correct propagated masks. Mutates per_frame_masks in place."""
    n = len(fi_list)
    cur_idx = 0  # video index

    print("\n[Mask] === 复查阶段 ===")
    print("[Mask]   j/k = 前进/后退一帧   J/K = ±10 帧   g = 跳到指定 fi")
    print("[Mask]   e = 在当前帧上修正    s = 保存全部并退出   q = 放弃保存退出")

    def load_img(vi):
        fi = fi_list[vi]
        return cv2.imread(os.path.join(imgpath_dir, imgnames[fi]))

    def show(vi):
        fi = fi_list[vi]
        img = load_img(vi)
        if img is None:
            print(f"[Mask] 警告: 读不到帧 fi={fi}")
            return None
        H, W = img.shape[:2]
        disp_scale = min(1.0, max_disp_w / W)
        masks = per_frame_masks.get(vi, {})
        header = f"REVIEW fi={fi}  vi={vi+1}/{n}"
        sub = "j/k=±1  J/K=±10  g=goto  e=edit  s=save  q=quit"
        v = _draw_overlay(img, masks, None, header, sub,
                          active_id=None, disp_scale=disp_scale)
        cv2.imshow(win_name, v)
        return img, disp_scale

    out = show(cur_idx)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('j'):
            cur_idx = min(n - 1, cur_idx + 1); out = show(cur_idx)
        elif key == ord('k'):
            cur_idx = max(0, cur_idx - 1); out = show(cur_idx)
        elif key == ord('J'):
            cur_idx = min(n - 1, cur_idx + 10); out = show(cur_idx)
        elif key == ord('K'):
            cur_idx = max(0, cur_idx - 10); out = show(cur_idx)
        elif key == ord('g'):
            try:
                target = int(input("[Mask] 跳到原始帧号 fi: ").strip())
            except Exception:
                print("[Mask] 输入无效")
                continue
            if target in fi_to_vi:
                cur_idx = fi_to_vi[target]; out = show(cur_idx)
            else:
                print(f"[Mask] fi={target} 不在视频窗口内")
        elif key == ord('e'):
            if out is None:
                continue
            img, disp_scale = out
            fi = fi_list[cur_idx]
            applied = _correct_frame(
                predictor, state, cur_idx, fi, img,
                per_frame_masks.get(cur_idx, {}), win_name, disp_scale,
            )
            if applied:
                # re-propagate from scratch over the whole window
                new_per_frame = _propagate_full(predictor, state, fi_list)
                per_frame_masks.clear()
                per_frame_masks.update(new_per_frame)
            out = show(cur_idx)
        elif key == ord('s'):
            return True
        elif key == ord('q'):
            return False


# --------------------------------------------------------------- public API ---
def ensure_masks(
    root: dict,
    seq_name: str,
    frames: Iterable[int],
    club_mesh_path: str = None,   # kept for API compat (no longer used)
    vis_dir: str = "/data2/fubingshuai/golf/test",
    force: bool = True,
    max_disp_w: int = 1280,
):
    """SAM2 video-mode mask generation.

    Label one key frame, propagate across the requested window, then optionally
    fix bad frames in a review loop. Outputs uint8 (H,W) npy
    (0=bg, 1=club, 2=right, 3=left) plus colored overlay JPGs for QA.
    """
    imgnames: Sequence[str] = list(root["imgnames"])
    imgpath_dir = root["imgpath"]
    masks_dir = imgpath_dir + "_masks"
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    requested = sorted({int(f) for f in frames if 0 <= int(f) < len(imgnames)})
    if not requested:
        print("[Mask] 没有可处理的帧")
        return

    todo: List[Tuple[int, str, str, str]] = []
    for fi in requested:
        imgname = imgnames[fi]
        if not imgname:
            continue
        base, _ = os.path.splitext(imgname)
        out_npy = os.path.join(masks_dir, base + ".npy")
        if force or not os.path.exists(out_npy):
            todo.append((fi, imgname, out_npy, base))

    if not todo:
        print(f"[Mask] 所有 {len(requested)} 帧 mask 已存在，跳过 (force=True 可重做)")
        return

    print(f"[Mask] 视频模式: 共需 {len(todo)} 帧 mask, 窗口 {requested[0]}..{requested[-1]}")

    # --- 1) symlink frames into a tmp video dir SAM2 can load ---
    tmp_dir, fi_list, fi_to_vi = _build_video_dir(imgpath_dir, imgnames, requested)
    if not fi_list:
        print("[Mask] 没有可用的帧文件，退出")
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return
    print(f"[Mask] 已为 {len(fi_list)} 帧建立 symlink: {tmp_dir}")

    # --- 2) build SAM2 video predictor + state ---
    from sam2.build_sam import build_sam2_video_predictor
    cfg, ckpt = _resolve_sam2_ckpt_and_cfg()
    predictor = build_sam2_video_predictor(cfg, ckpt, device='cuda')
    state = predictor.init_state(video_path=tmp_dir, offload_video_to_cpu=True)
    print(f"[Mask] SAM2 state 初始化完成 ({state['num_frames']} 帧)")

    # --- 3) interactive label on the key (= first requested) frame ---
    key_fi = todo[0][0]
    key_vi = fi_to_vi[key_fi]
    key_img_path = os.path.join(imgpath_dir, imgnames[key_fi])
    key_img = cv2.imread(key_img_path)
    if key_img is None:
        raise RuntimeError(f"读不到关键帧: {key_img_path}")
    H, W = key_img.shape[:2]
    disp_scale = min(1.0, max_disp_w / W)

    win_name = "SAM2 Mask Editor (Video)"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    saved = False
    try:
        keyframe_masks: Dict[int, np.ndarray] = {}
        for label_id, _, _ in _TARGETS:
            m = _label_target_on_keyframe(
                predictor, state, key_vi, key_fi, label_id,
                key_img, keyframe_masks, win_name, disp_scale,
            )
            if m is not None:
                keyframe_masks[label_id] = m

        if not keyframe_masks:
            print("[Mask] 关键帧上没有任何 prompt，退出（不写文件）")
            return

        # --- 4) propagate across the whole window ---
        per_frame_masks = _propagate_full(predictor, state, fi_list)

        # --- 5) review / correction loop ---
        saved = _review_loop(
            predictor, state, fi_list, fi_to_vi, imgnames, imgpath_dir,
            per_frame_masks, win_name, max_disp_w,
        )
        if not saved:
            print("[Mask] 用户放弃保存，未写出任何 mask")
            return

        # --- 6) save requested frames ---
        for fi, imgname, out_npy, base in todo:
            if fi not in fi_to_vi:
                print(f"[Mask][{fi}] 跳过（不在视频窗口内）")
                continue
            vi = fi_to_vi[fi]
            img = cv2.imread(os.path.join(imgpath_dir, imgname))
            if img is None:
                print(f"[Mask][{fi}] 读不到图像，跳过")
                continue
            Hf, Wf = img.shape[:2]
            combined = np.zeros((Hf, Wf), dtype=np.uint8)
            masks_this = per_frame_masks.get(vi, {})
            # club first (lowest priority); hands paint on top
            for lid in [1, 2, 3]:
                m = masks_this.get(lid)
                if m is not None and m.any():
                    combined[m] = lid
            np.save(out_npy, combined)

            viz = img.copy()
            for lid in [1, 2, 3]:
                m = masks_this.get(lid)
                if m is None or not m.any():
                    continue
                c = np.array(_COLOR_BGR[lid], dtype=np.float32)
                viz[m] = (viz[m].astype(np.float32) * 0.55 +
                          c * 0.45).astype(np.uint8)
            cv2.imwrite(os.path.join(vis_dir, f"{seq_name}_{base}_mask.jpg"), viz)
            print(f"[Mask][{fi:05d}] 已写: {out_npy}")
        print(f"[Mask] 全部完成. 可视化 JPG: {vis_dir}")
    except KeyboardInterrupt as e:
        print(f"[Mask] 中断: {e}")
    finally:
        cv2.destroyAllWindows()
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


# -------------------------------------------------------------------- main ---
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--mesh", required=False, default=None,
                    help="(unused; kept for backward compat)")
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--vis-dir", default="/data2/fubingshuai/golf/test")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    root = np.load(args.npy, allow_pickle=True).item()
    ensure_masks(root, args.seq, args.frames, args.mesh, args.vis_dir, args.force)
