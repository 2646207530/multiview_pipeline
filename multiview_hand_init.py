"""Multi-view MANO initialization via model/Hand_Estimation/.

Replaces the per-frame HaMER MANO output when ≥2 color cameras are available.
HaMER is still used internally as a 2D-keypoint detector to fabricate WiLoR-style
pseudo labels; the actual MANO comes from Hand_Estimation's multi-view network.

Side-effect / cache layout (idempotent across runs):

    <capture_dir>/.undistorted/
        <capture_id>/
            <cam_idx>/images_undistorted/frame_NNNNNN.jpg     # cam_idx 是 0/1, 顺序与 cam_names 对齐
            calib_undistorted/<cam_idx>.yaml                  # K, R, t  (R/t 把 world=cam0 转到本相机)
        pseudo_label_wilor/<seq>_<cam_idx>_<frame_id>_<hand_id>.npz   # is_right, joints_2d (21,2) in undist coords

调 Hand_Estimation 走 ``visualize_mano.py`` 子进程, 输出
``<output_dir>/<seq>_mano.json``, 我们解析后回填进 wrapper 的 root["data_dict"][...]["params"]。

注意:
  * 第一次跑会比较慢, 因为要 (a) 把每帧每相机的 jpg 都 undistort 一遍, 然后
    (b) 每相机每帧每只手都跑一次 HaMER. 之后两者都会被 cache.
  * Hand_Estimation 只在两个相机都检出某帧某手时才能给出位姿, 缺失会留 NaN
    给上游的 SLERP 插值兜底。
  * 这个模块还没在 end-to-end 跑通过, 只通过单元静态测试。第一次跑出错时按提示
    定位 (yaml 字段缺失 / cam_idx 命名不匹配 / 子进程 reload 路径等)。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm


_PROJECT = Path(__file__).resolve().parent
_HE_REL = Path("model") / "Hand_Estimation"
_HE_CFG_REL = "config/release/GOLF_Inference.yaml"
_HE_FT_TEMPLATE_REL = "config/release/WORK_GOLF_DINO.yaml"   # 用作 self-supervised FT 的模板
_HE_CKPT_REL = "exp/new/checkpoints/checkpoint_30"


# ---------------------------------------------- pseudo-label visualization ---
# 21-keypoint MANO/HaMER skeleton edges (root=wrist=0, then 5 fingers tip→…→base)
_HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # little
]


def _draw_hand_skeleton(img, joints_2d, color_bgr):
    """In-place draw 21-keypoint skeleton on a BGR image."""
    pts = np.round(joints_2d).astype(np.int32)
    for a, b in _HAND_EDGES:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), color_bgr, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(p), 3, (0, 0, 255), -1)
        cv2.circle(img, tuple(p), 4, (255, 255, 255), 1)


# --------------------------------------------------------------- undistort ---
def _newK_and_maps(K: np.ndarray, dist: np.ndarray, hw: Tuple[int, int]):
    H, W = hw
    new_K, _ = cv2.getOptimalNewCameraMatrix(
        K, dist, (W, H), alpha=0.0, newImgSize=(W, H))
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, None, new_K, (W, H), cv2.CV_16SC2)
    return new_K.astype(np.float64), map1, map2


def _write_calib_yaml(yaml_path: Path, new_K: np.ndarray,
                      R_w2c: np.ndarray, t_w2c: np.ndarray):
    """K + (R, t) world->this_cam in undistorted image plane."""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "K": new_K.tolist(),
        "R": R_w2c.tolist(),
        "t": t_w2c.reshape(3).tolist(),
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=None)


def _world2cam_RT(cams_dict, world_name: str, cam_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """world = world_name 的 sensor 系 -> cam_name 的 sensor 系。t 单位米。"""
    def _s2r(c):
        R_s2r = np.array(c["sensor2rig"]["rotation"], dtype=np.float64)
        t_s2r = np.array(c["sensor2rig"]["translation"], dtype=np.float64) / 1000.0
        return R_s2r, t_s2r
    R_w, t_w = _s2r(cams_dict[world_name])
    R_c, t_c = _s2r(cams_dict[cam_name])
    R_w2c = R_c.T @ R_w
    t_w2c = R_c.T @ (t_w - t_c)
    return R_w2c, t_w2c


def prepare_undistort_dir(capture_dir: Path, cams_dict: dict,
                          cam_names: Sequence[str], world_name: str,
                          frame_dirs: Dict[str, Path],
                          force: bool = False) -> Tuple[Path, str, Dict[str, np.ndarray]]:
    """
    生成/复用 <capture_dir>/.undistorted/。

    cam_names: 顺序与 cam_idx (0,1,...) 对齐, 第 0 个就是 hamer_cam (= world)。
    frame_dirs: cam_name -> 该相机原始帧目录 (含 frame_NNNNNN.jpg)。
    返回: (undist_root, capture_id, new_K_per_cam)。
    """
    undist_root = capture_dir / ".undistorted"
    capture_id = capture_dir.name
    new_K_per_cam: Dict[str, np.ndarray] = {}

    for cam_idx, cam_name in enumerate(cam_names):
        c = cams_dict[cam_name]
        H, W = int(c["image_height"]), int(c["image_width"])
        K = np.array(
            [[c["intrinsics"]["fx"], 0, c["intrinsics"]["cx"]],
             [0, c["intrinsics"]["fy"], c["intrinsics"]["cy"]],
             [0, 0, 1]], dtype=np.float64)
        dist = np.array(c["distortion_coeffs"], dtype=np.float64).reshape(-1)
        new_K, map1, map2 = _newK_and_maps(K, dist, (H, W))
        new_K_per_cam[cam_name] = new_K

        out_img_dir = undist_root / capture_id / str(cam_idx) / "images_undistorted"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        calib_dir = undist_root / capture_id / "calib_undistorted"
        calib_dir.mkdir(parents=True, exist_ok=True)
        R_w2c, t_w2c = _world2cam_RT(cams_dict, world_name, cam_name)
        _write_calib_yaml(calib_dir / f"{cam_idx}.yaml", new_K, R_w2c, t_w2c)

        src_dir = frame_dirs[cam_name]
        srcs = sorted(p for p in src_dir.iterdir()
                      if re.match(r"frame_\d+\.(?:jpg|jpeg|png|bmp)$",
                                  p.name, re.IGNORECASE))
        skipped = 0
        for src in tqdm(srcs, desc=f"undistort {cam_name}", leave=False):
            m = re.match(r"frame_(\d+)\.", src.name, re.IGNORECASE)
            if m is None:
                continue
            frame_idx = int(m.group(1))
            dst = out_img_dir / f"{frame_idx:06d}.jpg"
            if dst.exists() and not force:
                skipped += 1
                continue
            img = cv2.imread(str(src))
            if img is None:
                continue
            und = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
            cv2.imwrite(str(dst), und, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[mvinit] cam{cam_idx} ({cam_name}): {len(srcs)} frames, "
              f"{skipped} cached, {len(srcs)-skipped} undistorted")

    (undist_root / "pseudo_label_wilor").mkdir(parents=True, exist_ok=True)
    return undist_root, capture_id, new_K_per_cam


# ------------------------------------------------------- pseudo labels ----
def _hand_id_for(label: str) -> int:
    """Hand_Estimation 约定: 0=right, 1=left."""
    return 0 if label.startswith("r") else 1


def write_pseudo_labels_from_hamer(undist_root: Path, capture_id: str,
                                   cam_names: Sequence[str], cam_idx_of: Dict[str, int],
                                   total_frames: int, frame_dirs_undist: Dict[str, Path],
                                   new_K_per_cam: Dict[str, np.ndarray],
                                   hamer, detector, parse_detections, force: bool = False,
                                   vis_n_frames: int = 0):
    """对每个相机每一帧每只检出的手, 用 HaMER 拿到 21 个关节 2D, 写成
    ``<undist_root>/pseudo_label_wilor/<seq>_<cam_idx>_<frame>_<hand_id>.npz``.
    fields: is_right (1.0 / 0.0), joints_2d (21,2) float32.

    另外可把每相机前 ``vis_n_frames`` 个有检出的帧画一张 21-keypoint overlay
    JPG 到 ``<undist_root>/_pseudo_vis/<seq>_<cam_idx>_<frame:06d>.jpg``.
    默认 vis_n_frames=0 (关闭), 因为 ``make_pseudo_video`` 现在会输出每帧拼接
    JPG (``<seq>_{frame:06d}.jpg``) + mp4, 整段覆盖更完整.
    """
    out_dir = undist_root / "pseudo_label_wilor"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = undist_root / "_pseudo_vis"
    if vis_n_frames > 0:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for cam_name in cam_names:
        cam_idx = cam_idx_of[cam_name]
        K = new_K_per_cam[cam_name].astype(np.float32)
        d = frame_dirs_undist[cam_name]
        written_vis: set = set()  # 本相机已写出 overlay 的 frame_idx

        for frame_idx in tqdm(range(total_frames),
                              desc=f"HaMER 2D cam{cam_idx}", leave=False):
            img_path = d / f"{frame_idx:06d}.jpg"
            if not img_path.exists():
                continue

            need = []
            for hid_label, hid_int in (("right", 0), ("left", 1)):
                p = out_dir / f"{capture_id}_{cam_idx}_{frame_idx}_{hid_int}.npz"
                if force or not p.exists():
                    need.append((hid_label, hid_int, p))
            should_vis = (vis_n_frames > 0 and len(written_vis) < vis_n_frames)
            if not need and not should_vis:
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue
            _, dets = detector.detect(image)
            detection_list = parse_detections(dets)
            seen = set()
            this_frame_kp2d: Dict[str, np.ndarray] = {}
            for bbox in detection_list:
                if not bbox or len(bbox) < 2:
                    continue
                hand_label = bbox[0]
                if hand_label not in ("right", "left"):
                    continue
                if hand_label in seen:
                    continue
                seen.add(hand_label)
                try:
                    output, _ = hamer.estimate_from_rgb(image, [bbox], K)
                except Exception as e:
                    print(f"[mvinit] HaMER 失败 cam{cam_idx} f{frame_idx} {hand_label}: {e}")
                    continue
                kp2d = output.get('pred_keypoints_2d_full', None)
                if kp2d is None:
                    continue
                kp2d = kp2d.detach().cpu().numpy().squeeze().astype(np.float32)
                if kp2d.ndim != 2 or kp2d.shape[1] != 2:
                    continue
                this_frame_kp2d[hand_label] = kp2d
                hid_int = 0 if hand_label == "right" else 1
                p = out_dir / f"{capture_id}_{cam_idx}_{frame_idx}_{hid_int}.npz"
                if force or not p.exists():
                    np.savez(p,
                             is_right=np.array([1.0 if hid_int == 0 else 0.0],
                                               dtype=np.float32),
                             joints_2d=kp2d)

            # 前 N 个有检测的帧画 overlay JPG
            if should_vis and this_frame_kp2d:
                viz = image.copy()
                for hand_label, kp2d in this_frame_kp2d.items():
                    color = (0, 255, 0) if hand_label == "right" else (255, 128, 64)
                    _draw_hand_skeleton(viz, kp2d, color)
                cv2.putText(viz,
                            f"cam{cam_idx} f{frame_idx} {'+'.join(sorted(this_frame_kp2d))}",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)
                vis_path = vis_dir / f"{capture_id}_{cam_idx}_{frame_idx:06d}.jpg"
                cv2.imwrite(str(vis_path), viz, [cv2.IMWRITE_JPEG_QUALITY, 90])
                written_vis.add(frame_idx)
                if len(written_vis) == vis_n_frames:
                    print(f"[mvinit] cam{cam_idx} 前 {vis_n_frames} 帧伪标 overlay 已写入 "
                          f"{vis_dir}")


def make_pseudo_video(undist_root: Path, capture_id: str,
                      cam_names: Sequence[str], cam_idx_of: Dict[str, int],
                      total_frames: int, frame_dirs_undist: Dict[str, Path],
                      fps: int = 10, downscale: int = 2,
                      out_filename: Optional[str] = None,
                      save_per_frame_jpg: bool = True) -> Optional[Path]:
    """把所有相机所有帧的伪标 overlay 拼成一个左右拼接的 mp4, 与 Hand_Estimation
    的 ``_he_output/<seq>_hand0.mp4`` 同样的 fps + 半分辨率, 方便并排对比.

    输出:
      * ``<undist_root>/_pseudo_vis/<capture_id>_pseudo_overlay.mp4`` — 整段视频
      * ``<undist_root>/_pseudo_vis/<capture_id>_{frame:06d}.jpg``    — 全部帧
        每帧拼好的 JPG (cam0 | cam1, 跟视频一样的内容), 方便单帧查看. 缺伪标
        的帧也会写入 (那帧的 overlay 就是空白没骨架), 保证全帧覆盖.

    每帧布局: cam0 | cam1 (左右拼接), 右手骨架=绿色, 左手=蓝色, 文字标 cam/frame.

    set ``save_per_frame_jpg=False`` 关掉 JPG 输出 (只生成 mp4).
    """
    pseudo_dir = undist_root / "pseudo_label_wilor"
    out_dir = undist_root / "_pseudo_vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_filename is None:
        out_filename = f"{capture_id}_pseudo_overlay.mp4"
    out_path = out_dir / out_filename

    # 探一下原始帧尺寸
    sample_img = None
    for cam_name in cam_names:
        d = frame_dirs_undist[cam_name]
        for fi in range(total_frames):
            p = d / f"{fi:06d}.jpg"
            if p.exists():
                sample_img = cv2.imread(str(p))
                if sample_img is not None:
                    break
        if sample_img is not None:
            break
    if sample_img is None:
        print(f"[mvinit] 找不到任何 undistorted 帧, 跳过伪标视频")
        return None

    H0, W0 = sample_img.shape[:2]
    H, W = max(1, H0 // downscale), max(1, W0 // downscale)
    n_cams = len(cam_names)
    out_W, out_H = W * n_cams, H

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (out_W, out_H))
    if not vw.isOpened():
        print(f"[mvinit] cv2.VideoWriter 打开失败: {out_path}")
        return None

    print(f"[mvinit] 开始生成伪标视频 ({out_W}x{out_H} @ {fps}fps): {out_path}")
    if save_per_frame_jpg:
        print(f"[mvinit] 同时保存每帧 JPG 到: {out_dir}/{capture_id}_{{frame:06d}}.jpg")
    n_written = 0
    for frame_idx in tqdm(range(total_frames), desc="pseudo video", leave=False):
        tiles = []
        for cam_name in cam_names:
            cam_idx = cam_idx_of[cam_name]
            d = frame_dirs_undist[cam_name]
            img_path = d / f"{frame_idx:06d}.jpg"
            img = cv2.imread(str(img_path)) if img_path.exists() else None
            if img is None:
                tile = np.zeros((H0, W0, 3), dtype=np.uint8)
            else:
                tile = img.copy()
                for hid_label, hid_int in (("right", 0), ("left", 1)):
                    p = pseudo_dir / f"{capture_id}_{cam_idx}_{frame_idx}_{hid_int}.npz"
                    if not p.exists():
                        continue
                    try:
                        d_npz = np.load(p)
                        kp2d = d_npz["joints_2d"]
                    except Exception:
                        continue
                    if kp2d.shape != (21, 2):
                        continue
                    color = (0, 255, 0) if hid_int == 0 else (255, 128, 64)
                    _draw_hand_skeleton(tile, kp2d, color)
                cv2.putText(tile, f"cam{cam_idx} f{frame_idx}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                            cv2.LINE_AA)
            if downscale > 1 and tile.shape[:2] != (H, W):
                tile = cv2.resize(tile, (W, H))
            tiles.append(tile)
        stitched = np.concatenate(tiles, axis=1)
        vw.write(stitched)
        if save_per_frame_jpg:
            jpg_path = out_dir / f"{capture_id}_{frame_idx:06d}.jpg"
            cv2.imwrite(str(jpg_path), stitched, [cv2.IMWRITE_JPEG_QUALITY, 90])
        n_written += 1
    vw.release()
    msg = f"[mvinit] 伪标视频写入完成: {out_path}  ({n_written} 帧)"
    if save_per_frame_jpg:
        msg += f", 每帧 JPG 也已写入 {out_dir}"
    print(msg)
    return out_path


# ------------------- multi-view MANO overlay video (统一风格的对比可视化) --

# smplx MANO 16 LBS 关节 + 5 个手指尖 vertex id (右手, 来自 smplx.VERTEX_IDS['mano'])
_MANO_TIP_VIDS = [745, 317, 444, 556, 673]   # thumb, index, middle, ring, pinky
# 16 LBS joints + 5 tips (附加后 indices 16..20) → OpenPose 21-joint 顺序
# (与 HaMER mano_wrapper 内的 mano_to_openpose 一致)
_MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
# 右手轴角 (x,y,z) 镜像 → 左手轴角 (x,-y,-z); 与 run_hamer_to_npy._MIRROR_LEFT
# / way_vis.right_to_left_mano_params 一致
_MIRROR_LR_AXIS = np.array([1.0, -1.0, -1.0], dtype=np.float32)


def make_mv_mano_video(undist_root: Path, capture_id: str,
                      cam_names: Sequence[str], cam_idx_of: Dict[str, int],
                      total_frames: int, frame_dirs_undist: Dict[str, Path],
                      cams_dict: dict, world_name: str,
                      new_K_per_cam: Dict[str, np.ndarray],
                      r_rot_arr: np.ndarray, r_pose_arr: np.ndarray,
                      r_shape_arr: np.ndarray, r_trans_arr: np.ndarray,
                      l_rot_arr: np.ndarray, l_pose_arr: np.ndarray,
                      l_shape_arr: np.ndarray, l_trans_arr: np.ndarray,
                      mano_root: str,
                      fps: int = 10, downscale: int = 2,
                      out_filename: Optional[str] = None) -> Optional[Path]:
    """把 Hand_Estimation 多视角 MANO 输出投影到每个相机并画 21-关节骨架, 与
    ``make_pseudo_video`` 同样的样式 (绿色右手 / 蓝色左手 / 红色关节点) 输出
    左右拼接 mp4. 这样和 ``_pseudo_vis/<seq>_pseudo_overlay.mp4`` 可以直接对照
    --- 同一个绘制器、同一个分辨率、同一个 fps、同一份图像底图.

    输入 r_*/l_* 数组的语义: pose_r/pose_l 是 "右手坐标系下的绝对角" (HE 解析时
    已 + right_hand_mean), trans_r/trans_l 是世界系 (= cam0 sensor 系) 米.
    左手画的时候参数会被镜像 (x,-y,-z) 喂给左手 smplx layer (flat=True).
    """
    import smplx
    import torch

    out_dir = undist_root / "_pseudo_vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_filename is None:
        out_filename = f"{capture_id}_mv_mano_overlay.mp4"
    out_path = out_dir / out_filename

    # 探一下原始帧尺寸 (同 make_pseudo_video)
    sample_img = None
    for cam_name in cam_names:
        d = frame_dirs_undist[cam_name]
        for fi in range(total_frames):
            p = d / f"{fi:06d}.jpg"
            if p.exists():
                sample_img = cv2.imread(str(p))
                if sample_img is not None:
                    break
        if sample_img is not None:
            break
    if sample_img is None:
        print(f"[mvinit] 找不到 undistorted 帧, 跳过 mv mano 视频")
        return None

    H0, W0 = sample_img.shape[:2]
    H, W = max(1, H0 // downscale), max(1, W0 // downscale)
    out_W, out_H = W * len(cam_names), H

    # 建 smplx MANO layer (flat=True, 与 npy 约定一致)
    mano_r = smplx.create(mano_root, 'MANO', use_pca=False, is_rhand=True,
                          flat_hand_mean=True)
    mano_l = smplx.create(mano_root, 'MANO', use_pca=False, is_rhand=False,
                          flat_hand_mean=True)
    tip_vids = torch.tensor(_MANO_TIP_VIDS, dtype=torch.long)
    perm = torch.tensor(_MANO_TO_OPENPOSE, dtype=torch.long)
    mirror = torch.tensor(_MIRROR_LR_AXIS, dtype=torch.float32)

    # 预计算所有相机外参 (world=cam0 → 各相机的 R, t)
    cam_w2c: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for cn in cam_names:
        R_w2c, t_w2c = _world2cam_RT(cams_dict, world_name, cn)
        cam_w2c[cn] = (R_w2c.astype(np.float32), t_w2c.astype(np.float32))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (out_W, out_H))
    if not vw.isOpened():
        print(f"[mvinit] cv2.VideoWriter 打开失败: {out_path}")
        return None

    print(f"[mvinit] 开始生成多视角 MANO overlay 视频 "
          f"({out_W}x{out_H} @ {fps}fps): {out_path}")

    def _joints21_world(rot_arr, pose_arr, shape_arr, trans_arr, layer,
                        is_right: bool, fi: int) -> Optional[np.ndarray]:
        rot_i = rot_arr[fi]
        pose_i = pose_arr[fi]
        shape_i = shape_arr[fi]
        trans_i = trans_arr[fi]
        if (np.isnan(rot_i).any() or np.isnan(pose_i).any()
                or np.isnan(trans_i).any() or np.isnan(shape_i).any()):
            return None
        rot_t = torch.from_numpy(rot_i.astype(np.float32)).reshape(1, 3)
        pose_t = torch.from_numpy(pose_i.astype(np.float32)).reshape(1, 45)
        shape_t = torch.from_numpy(shape_i.astype(np.float32)).reshape(1, 10)
        trans_t = torch.from_numpy(trans_i.astype(np.float32)).reshape(1, 3)
        if not is_right:
            # 左手: 把 "右手坐标系" 的 rot/pose 镜像后再喂给左手 layer
            rot_t = rot_t * mirror
            pose_t = (pose_t.reshape(-1, 3) * mirror).reshape(1, 45)
        with torch.no_grad():
            out = layer(global_orient=rot_t, hand_pose=pose_t,
                        betas=shape_t, transl=trans_t)
        joints_lbs = out.joints[0]                 # (≥16, 3)
        verts = out.vertices[0]                    # (778, 3)
        tips = verts[tip_vids, :]                  # (5, 3)
        # 取前 16 个 LBS 关节 + 5 个 tip → 21, 再按 OpenPose 顺序排
        joints21 = torch.cat([joints_lbs[:16], tips], dim=0)[perm]
        return joints21.cpu().numpy()              # (21, 3) world

    n_written = 0
    for fi in tqdm(range(total_frames), desc="mv mano video", leave=False):
        joints_world = {
            'right': _joints21_world(r_rot_arr, r_pose_arr, r_shape_arr,
                                     r_trans_arr, mano_r, True,  fi),
            'left':  _joints21_world(l_rot_arr, l_pose_arr, l_shape_arr,
                                     l_trans_arr, mano_l, False, fi),
        }

        tiles = []
        for cam_name in cam_names:
            cam_idx = cam_idx_of[cam_name]
            d = frame_dirs_undist[cam_name]
            img_path = d / f"{fi:06d}.jpg"
            img = cv2.imread(str(img_path)) if img_path.exists() else None
            tile = img.copy() if img is not None \
                                else np.zeros((H0, W0, 3), dtype=np.uint8)

            K = new_K_per_cam[cam_name].astype(np.float32)
            R_w2c, t_w2c = cam_w2c[cam_name]

            for hand_label in ('right', 'left'):
                J = joints_world[hand_label]
                if J is None:
                    continue
                J_cam = (R_w2c @ J.T).T + t_w2c        # (21, 3)
                z = np.maximum(J_cam[:, 2], 1e-6)
                u = K[0, 0] * J_cam[:, 0] / z + K[0, 2]
                v = K[1, 1] * J_cam[:, 1] / z + K[1, 2]
                kp2d = np.stack([u, v], axis=-1)
                color = (0, 255, 0) if hand_label == 'right' else (255, 128, 64)
                _draw_hand_skeleton(tile, kp2d, color)

            cv2.putText(tile, f"cam{cam_idx} f{fi}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                        cv2.LINE_AA)
            if downscale > 1 and tile.shape[:2] != (H, W):
                tile = cv2.resize(tile, (W, H))
            tiles.append(tile)

        stitched = np.concatenate(tiles, axis=1)
        vw.write(stitched)
        n_written += 1

    vw.release()
    print(f"[mvinit] 多视角 MANO 视频写入完成: {out_path}  ({n_written} 帧)")
    return out_path


# -------------------------------------------------- Hand_Estimation runner --
def run_hand_estimation_subprocess(project_root: Path, undist_root: Path,
                                   output_dir: Path, gpu_id: str = "0",
                                   cfg_rel: str = _HE_CFG_REL,
                                   ckpt_rel: str = _HE_CKPT_REL,
                                   ckpt_override: Optional[Path] = None,
                                   extra_env: Optional[dict] = None) -> Path:
    """
    起 Hand_Estimation/visualize_mano.py 子进程, 等它产出
    ``<output_dir>/<seq>_mano.json``。

    ckpt_override: 如果给了, 临时把 he_dir/'checkpoints' 软链到这个目录,
    visualize_mano.py 内置的 fallback 会优先用 'checkpoints/' 加载权重
    (见 visualize_mano.py:188-198), 这样可以用 finetune 后的权重而不动默认.
    跑完会自动撤掉软链.
    """
    he_dir = project_root / _HE_REL
    if not he_dir.is_dir():
        raise RuntimeError(f"Hand_Estimation 目录不存在: {he_dir}")
    cfg_path = he_dir / cfg_rel
    if not cfg_path.is_file():
        raise RuntimeError(f"Hand_Estimation 配置文件不存在: {cfg_path}")
    ckpt_path = he_dir / ckpt_rel
    if not ckpt_path.exists():
        raise RuntimeError(f"Hand_Estimation 检查点不存在: {ckpt_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    env.setdefault("CUDA_VISIBLE_DEVICES", gpu_id)

    # ckpt_override 路径要用绝对路径做软链, 避免 cwd 变化
    symlink_path = he_dir / "checkpoints"
    created_symlink = False
    if ckpt_override is not None:
        ckpt_override = Path(ckpt_override).resolve()
        if not ckpt_override.is_dir():
            raise RuntimeError(f"ckpt_override 不是目录: {ckpt_override}")
        if symlink_path.exists() or symlink_path.is_symlink():
            # 防御: 如果之前残留, 先清掉
            if symlink_path.is_symlink():
                symlink_path.unlink()
            else:
                raise RuntimeError(
                    f"{symlink_path} 已存在且不是软链, 怕覆盖真实数据, 主动报错")
        symlink_path.symlink_to(ckpt_override)
        created_symlink = True
        print(f"[mvinit] 临时软链 {symlink_path} -> {ckpt_override}")

    # 注意: 不要传 --reload。Hand_Estimation 的 lib/utils/config.py:103 会把 --reload
    # 同步写到 cfg.MODEL.PRETRAINED, 而 init_weights() 期望那是个文件; 但 ckpt_path 是
    # 目录(里面装 *.pth.tar), 会触发 FileNotFoundError。
    # visualize_mano.py:188-198 自带的 fallback 会在 cwd=Hand_Estimation/ 时自动找到
    # 默认权重目录 (现在是 exp/new/checkpoints/checkpoint_30), 所以不传反而更稳。
    cmd = [
        sys.executable, "visualize_mano.py",
        "--cfg", str(cfg_path),
        "--input_dir", str(undist_root),
        "--output_dir", str(output_dir),
        "--gpu_id", gpu_id,
        "--workers", "0",
    ]
    print(f"[mvinit] 调 Hand_Estimation: cd {he_dir} && {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=str(he_dir), check=True, env=env)
    finally:
        if created_symlink and symlink_path.is_symlink():
            symlink_path.unlink()
            print(f"[mvinit] 已撤掉临时软链: {symlink_path}")

    cands = list(output_dir.glob("*_mano.json"))
    if not cands:
        raise RuntimeError(f"Hand_Estimation 未输出 *_mano.json 到 {output_dir}")
    if len(cands) > 1:
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"[mvinit] 多个 mano json, 使用最新的: {cands[0].name}")
    return cands[0]


# ----------------------------- self-supervised finetune (训练) ----
def run_hand_estimation_finetune_subprocess(project_root: Path, undist_root: Path,
                                            capture_id: str,
                                            epochs: int,
                                            gpu_id: str = "0",
                                            lr: float = 1e-5,
                                            batch_size: int = 1,
                                            pretrained_ckpt_rel: str = _HE_CKPT_REL,
                                            extra_env: Optional[dict] = None) -> Path:
    """对当前 capture 在线 self-supervised 微调 Hand_Estimation 权重, 返回新
    checkpoint 目录的绝对路径 (含 ``<ModelClass>.pth.tar`` 等文件, 供 inference
    时通过软链当作 ``checkpoints/`` 使用)。

    流程:
      1. 复制 WORK_GOLF_DINO.yaml 作为模板, 把
         DATASET.TRAIN.DATA_ROOT / DATASET.TEST.DATA_ROOT 改成我们的
         ``<undist_root>``, EPOCH / LR / BATCH_SIZE 改成 finetune 友好值,
         AUG 关掉. 写到 ``<undist_root>/_ft_cfg.yaml``.
      2. 调 ``train_ddp_sf.py --cfg <ft yaml> --ft --reload <pretrained_dir>
         --exp_id mv_ft_<seq> --gpu_id <gpu> -b <bs> -w 0``
      3. 训练结束扫 ``exp/mv_ft_<seq>/checkpoints/`` 找最新 checkpoint 目录,
         返回它的绝对路径.
    """
    he_dir = project_root / _HE_REL
    if not he_dir.is_dir():
        raise RuntimeError(f"Hand_Estimation 目录不存在: {he_dir}")
    template_path = he_dir / _HE_FT_TEMPLATE_REL
    if not template_path.is_file():
        raise RuntimeError(f"FT 模板 yaml 不存在: {template_path}")
    pretrained_path = (he_dir / pretrained_ckpt_rel).resolve()
    if not pretrained_path.is_dir():
        raise RuntimeError(f"预训练 ckpt 目录不存在: {pretrained_path}")

    # ── 1) 生成 finetune YAML ────────────────────────────────────────
    with open(template_path) as f:
        ft_cfg = yaml.safe_load(f)
    undist_root_abs = str(Path(undist_root).resolve())
    for split in ("TRAIN", "TEST"):
        if "DATASET" not in ft_cfg or split not in ft_cfg["DATASET"]:
            raise RuntimeError(f"FT 模板里缺 DATASET.{split} 段")
        ft_cfg["DATASET"][split]["DATA_ROOT"] = undist_root_abs
    # finetune 调小学习率 / 少跑几个 epoch / 单 batch / 关 AUG
    ft_cfg.setdefault("TRAIN", {})["EPOCH"] = int(epochs)
    ft_cfg["TRAIN"]["LR"] = float(lr)
    ft_cfg["TRAIN"]["BATCH_SIZE"] = int(batch_size)
    ft_cfg["TRAIN"]["SAVE_EPOCH"] = [int(epochs) - 1] if epochs > 0 else [0]
    # 关 augmentation 让 self-supervised 信号更"干净" (尽量贴近测试分布)
    for split in ("TRAIN", "TEST"):
        tf = ft_cfg.get("DATASET", {}).get(split, {}).get("TRANSFORM")
        if isinstance(tf, dict):
            tf["AUG"] = False
            tf["OCCLUSION"] = False
            tf["OCCLUSION_PROB"] = 0.0

    ft_yaml_path = Path(undist_root) / "_ft_cfg.yaml"
    with open(ft_yaml_path, "w") as f:
        yaml.safe_dump(ft_cfg, f, sort_keys=False)
    print(f"[mvinit] 已写 FT yaml: {ft_yaml_path}")

    # ── 2) 调 train_ddp_sf.py ────────────────────────────────────────
    exp_id = f"mv_ft_{capture_id}"
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    env.setdefault("CUDA_VISIBLE_DEVICES", gpu_id)
    cmd = [
        sys.executable, "train_ddp_sf.py",
        "--cfg", str(ft_yaml_path),
        "--ft",                             # 启用 ft 分支
        "--reload", str(pretrained_path),   # 经过我们 patch, 指向预训练 ckpt 目录
        "--exp_id", exp_id,
        "--gpu_id", gpu_id,
        "-b", str(batch_size),
        "-w", "0",
    ]
    print(f"[mvinit] 调 Hand_Estimation 自监督 FT: cd {he_dir} && {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(he_dir), check=True, env=env)

    # ── 3) 定位最新的 checkpoint 目录 ────────────────────────────────
    # Recorder.dump_path = f"{exp_id}_{timestamp}", 所以实际目录是
    #   exp/<exp_id>_<YYYY_MMDD_HHMM_SS>/checkpoints/...
    # 不是直接的 exp/<exp_id>/checkpoints/. 用 glob 模式找带时间戳的目录,
    # 取 mtime 最新的那一个.
    exp_root = he_dir / "exp"
    candidate_exp_dirs = sorted(
        [p for p in exp_root.glob(f"{exp_id}_*") if (p / "checkpoints").is_dir()],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    # 兼容: 旧式可能直接是 exp/<exp_id>/checkpoints/ (无时间戳)
    legacy_root = exp_root / exp_id
    if (legacy_root / "checkpoints").is_dir():
        candidate_exp_dirs.append(legacy_root)
    if not candidate_exp_dirs:
        raise RuntimeError(
            f"未找到 FT 实验目录: 搜索 {exp_root}/{exp_id}_* 和 {exp_root}/{exp_id}"
        )
    chosen_exp_dir = candidate_exp_dirs[0]
    ckpts_root = chosen_exp_dir / "checkpoints"
    print(f"[mvinit] FT 输出实验目录: {chosen_exp_dir}")

    cand_dirs = [p for p in ckpts_root.iterdir() if p.is_dir()]
    if not cand_dirs:
        raise RuntimeError(f"{ckpts_root} 下无检查点子目录, FT 失败?")
    cand_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    new_ckpt = cand_dirs[0].resolve()
    # 校验里面有 TestMultiviewStereo.pth.tar
    if not (new_ckpt / "TestMultiviewStereo.pth.tar").is_file():
        # 找下一个候选
        for cd in cand_dirs[1:]:
            if (cd / "TestMultiviewStereo.pth.tar").is_file():
                new_ckpt = cd.resolve()
                break
        else:
            raise RuntimeError(f"FT 输出目录里没有 TestMultiviewStereo.pth.tar: "
                               f"{[str(p) for p in cand_dirs]}")
    print(f"[mvinit] FT 完成, 新 checkpoint: {new_ckpt}")
    return new_ckpt


# ----------------------------------------- merge HE output back into root ---
NAN3   = np.full((3,),  np.nan, dtype=np.float32)
NAN10  = np.full((10,), np.nan, dtype=np.float32)
NAN45  = np.full((45,), np.nan, dtype=np.float32)


_HAND_MEAN_CACHE = {"right": None, "left": None}


def _get_hand_mean(side: str, mano_root: str) -> np.ndarray:
    """Load `hands_mean` (45,) for one side from the MANO pkl. 用 smplx 加载
    出来的与 manopth 加载是同一份数 (来自 MANO_*.pkl), 所以可以代表
    Hand_Estimation/manopth 的 hand_mean."""
    if _HAND_MEAN_CACHE.get(side) is not None:
        return _HAND_MEAN_CACHE[side]
    import smplx
    layer = smplx.create(mano_root, 'MANO',
                         use_pca=False,
                         is_rhand=(side == "right"),
                         flat_hand_mean=False)
    mean = layer.hand_mean.detach().cpu().numpy().astype(np.float32).reshape(45)
    _HAND_MEAN_CACHE[side] = mean
    return mean


def parse_mano_json_to_arrays(json_path: Path, total_frames: int,
                              mano_root: Optional[str] = None,
                              add_hand_mean: bool = True):
    """Hand_Estimation 输出格式 (visualize_mano.py 末尾):
       {right_hand: {rot_r, pose_r, trans_r, shape_r}, left_hand: {rot_l, ...}}
       每个数组按 sorted(frame_id) 排列, 长度可能 < total_frames。
       这里按缺帧填 NaN, 再交给上游 SLERP / 线性插值。

    Pose 约定对齐:
      Hand_Estimation 网络后接 manopth.ManoLayer(flat_hand_mean=False), 它在
      forward 里会把 hand_mean 加到 pose 上, 所以网络学到的 pose_euler 是
      "绝对关节角 - hand_mean" (差量).

      HaMER 用 smplx.MANOLayer (rotmat path), forward **不加** hand_mean,
      所以它存的 pose 是 "绝对关节角".

      为了让 npy 里两条路径产出的 pose 语义一致 (= 绝对关节角, 等价于
      smplx.MANO + flat_hand_mean=True 的输入), 默认在这里给 multiview pose
      加上 hand_mean. 设 add_hand_mean=False 可以保留原始 HE 输出.

    缺陷: visualize_mano.py 当前导出时**没有**把 frame_id 一同写出, 只
    "按顺序写"——所以这里假设第 i 行就是第 i 帧. Windowing 一旦跳帧, 对齐
    会错位, 这是已知 TODO.
    """
    with open(json_path) as f:
        data = json.load(f)

    def _stack_or_nan(lst, n, shape):
        out = np.full((total_frames, *shape), np.nan, dtype=np.float32)
        if not lst:
            return out
        arr = np.array(lst, dtype=np.float32)
        if arr.ndim != 1 + len(shape):
            arr = arr.reshape((-1, *shape))
        n_use = min(arr.shape[0], total_frames)
        out[:n_use] = arr[:n_use]
        return out

    rh = data.get("right_hand") or {}
    lh = data.get("left_hand")  or {}

    r_rot   = _stack_or_nan(rh.get("rot_r"),   total_frames, (3,))
    r_pose  = _stack_or_nan(rh.get("pose_r"),  total_frames, (45,))
    r_trans = _stack_or_nan(rh.get("trans_r"), total_frames, (3,))
    r_shape = _stack_or_nan(rh.get("shape_r"), total_frames, (10,))
    l_rot   = _stack_or_nan(lh.get("rot_l"),   total_frames, (3,))
    l_pose  = _stack_or_nan(lh.get("pose_l"),  total_frames, (45,))
    l_trans = _stack_or_nan(lh.get("trans_l"), total_frames, (3,))
    l_shape = _stack_or_nan(lh.get("shape_l"), total_frames, (10,))

    if add_hand_mean:
        if mano_root is None:
            mano_root = str(_PROJECT)  # see top: project root
        try:
            # ⚠️ 左右手都加 RIGHT hand_mean (不是各自的):
            # Hand_Estimation 内部 ManoDecoder 不论 hand_id 都用 MANO('right').layer
            # (右手 manopth.ManoLayer, flat_hand_mean=False), 所以学到的 pose 是
            # "absolute_right_hand_format - right_hand_mean" (右手坐标系下的差量).
            # 左手数据走的是"图像水平镜像 + 同一个右手 layer 推理"路径, 输出 pose 仍
            # 在右手坐标系; npy 里 pose_l 因此存的是"右手格式"(消费者如 way_vis
            # 通过 right_to_left_mano_params 镜像 (x,-y,-z) 再喂给左手 layer).
            # 想把 pose 转成"右手坐标系下的绝对值"-- 不论左右手都加 right hand_mean.
            r_mean = _get_hand_mean("right", mano_root)
            # 只加到非 NaN 帧上, NaN 帧保持 NaN 让上游插值兜底
            r_valid = ~np.isnan(r_pose).any(axis=1)
            l_valid = ~np.isnan(l_pose).any(axis=1)
            r_pose[r_valid] = r_pose[r_valid] + r_mean[None, :]
            l_pose[l_valid] = l_pose[l_valid] + r_mean[None, :]  # ← 也用 right
            print(f"[mvinit] 已把 right hand_mean 加到 multiview pose 上 "
                  f"(左右手都用 right, 因为 HE 内部对两只手都用 right ManoLayer): "
                  f"±{float(np.abs(r_mean).max()):.3f}")
        except Exception as e:
            print(f"[mvinit] 加 hand_mean 失败 ({e}), 保留原始 HE 输出")

    # ──── trans 约定补偿: HE 存的是 Middle MCP 的世界位置 → 转成 smplx transl ────
    # HE 在 visualize_mano.py:verify_mano 里:
    #   output_joint[i] = (J_canonical[i] - J_canonical[9]) + mano_trans
    # 所以 mano_trans = "joint 9 (OpenPose Middle MCP) 在世界系下应当的位置".
    #
    # 左手专用:
    #   HE dataset (lib/datasets/golf.py:262-280, getitem_test) 对 hand_id==1 (左手)
    #   把整张图 X-flip 后再喂模型, 但相机外参/内参没有同步翻转. 模型在镜像图像上
    #   输出 right-format pose + trans, 这个 trans 处于 "X-flip 后的世界系" 里, 不
    #   是真实世界. HE 自己的可视化 (visualize_mano.py:123) 用 2D X-flip 把它打回正
    #   像; way_vis 走的是真 3D 渲染, 所以必须先把 trans_l 的 X 分量取反, 才能落到
    #   真实世界. (HaMER 的 infer.py:411 显式做了 pred_cam[:,1] *= flip_correction
    #   这步 X-flip, 所以 HaMER 写出的 npy 不需要这步.)
    #
    # way_vis 用 smplx.MANO 渲染时:
    #   output_joint[i] = J_posed_smplx[i] + transl
    # 注意 J_posed_smplx[i] 在 transl=0 时不是原点 (zero-pose 下 wrist ≈ (0.10, 0.006, 0.006)).
    # 要让 output_joint[Middle_MCP] = HE_trans (HE 想要的 Middle MCP 位置):
    #   transl = HE_trans - J_posed_smplx[Middle_MCP_idx]
    # 左手再多一步 X-flip:
    #   transl_l = X_FLIP(HE_trans_l) - J_posed_smplx_left[Middle_MCP_idx]
    #
    # smplx LBS 16-joint 排序: 0=wrist, 1-3=index, 4-6=middle, 7-9=pinky,
    # 10-12=ring, 13-15=thumb.
    # 左手要在镜像后 (左手 layer + (1,-1,-1) 镜像 rot/pose) 上算 J_posed, 因为 way_vis 渲染
    # 左手就走这条镜像路径.
    try:
        import smplx
        import torch as _torch
        mano_r_smplx = smplx.create(mano_root, 'MANO',
                                    use_pca=False, is_rhand=True,
                                    flat_hand_mean=True)
        mano_l_smplx = smplx.create(mano_root, 'MANO',
                                    use_pca=False, is_rhand=False,
                                    flat_hand_mean=True)
        _MIRROR = _torch.tensor([1.0, -1.0, -1.0], dtype=_torch.float32)
        _XFLIP = np.array([-1.0, 1.0, 1.0], dtype=np.float32)
        _MIDDLE_MCP_IDX = 4   # smplx LBS joint idx for Middle_1 (Middle MCP)

        def _correct(rot_arr, pose_arr, shape_arr, trans_arr, layer, side):
            n_fixed = 0
            for i in range(total_frames):
                if (np.isnan(rot_arr[i]).any() or np.isnan(pose_arr[i]).any()
                        or np.isnan(shape_arr[i]).any() or np.isnan(trans_arr[i]).any()):
                    continue
                rot_t = _torch.from_numpy(rot_arr[i].astype(np.float32)).reshape(1, 3)
                pose_t = _torch.from_numpy(pose_arr[i].astype(np.float32)).reshape(1, 45)
                shape_t = _torch.from_numpy(shape_arr[i].astype(np.float32)).reshape(1, 10)
                if side == "left":
                    rot_t = rot_t * _MIRROR
                    pose_t = (pose_t.reshape(-1, 3) * _MIRROR).reshape(1, 45)
                with _torch.no_grad():
                    out = layer(global_orient=rot_t, hand_pose=pose_t, betas=shape_t)
                J = out.joints[0]                                  # (≥16, 3)
                mcp_local = J[_MIDDLE_MCP_IDX].cpu().numpy()       # J_posed[4] in local frame
                # 左手 trans 的 X-flip (Route B 下理论一致, 必须开):
                #   dataset 用 Route B (E → M·E·M 共轭) 训练后, 模型 master_joints_mvf
                #   输出位于 X-mirrored 世界系 (X_model = M·X_real). npy / way_vis 这边
                #   使用 cam0 真实 K + 真实 world2cam 渲染, 需要把模型输出 unmirror 回
                #   真实世界: trans_real = M · trans_model = (-x, y, z).
                #   (rot/pose 由 way_vis 的 right_to_left_mano_params 通过 MIRROR_LEFT_PARAMS
                #   自动处理, 不在 parse 里翻.)
                if side == "left":
                    trans_arr[i] = trans_arr[i] * _XFLIP
                trans_arr[i] = trans_arr[i] - mcp_local            # X_FLIP(HE_trans) - J_posed[4]
                n_fixed += 1
            return n_fixed

        n_r = _correct(r_rot, r_pose, r_shape, r_trans, mano_r_smplx, "right")
        n_l = _correct(l_rot, l_pose, l_shape, l_trans, mano_l_smplx, "left")
        print(f"[mvinit] 已把 trans 修正为 smplx transl 语义 "
              f"(right: HE_trans - J_posed[4]; left: X_FLIP(HE_trans) - J_posed_l[4])"
              f": right {n_r} 帧 / left {n_l} 帧")
    except Exception as e:
        print(f"[mvinit] trans 修正失败 ({e}), 保留原 trans (way_vis 渲染会偏)")

    return (r_rot, r_pose, r_shape, r_trans,
            l_rot, l_pose, l_shape, l_trans)


# --------------------------------------------------------- top-level entry ---
def init_hands_from_multiview(*, project_root: Path, capture_dir: Path,
                              cams_dict: dict, cam_names_in_order: Sequence[str],
                              world_name: str,
                              frame_dirs_orig: Dict[str, Path],
                              total_frames: int,
                              hamer, detector, parse_detections,
                              gpu_id: str = "0",
                              force_undistort: bool = False,
                              force_pseudo: bool = False,
                              pseudo_video: bool = True,
                              pseudo_video_fps: int = 10,
                              pseudo_video_downscale: int = 2,
                              mv_finetune_epochs: int = 0,
                              mv_finetune_lr: float = 1e-5,
                              mv_finetune_bs: int = 1):
    """
    Orchestrator. cam_names_in_order[0] 必须是 hamer_cam (= world frame)。
    返回 (r_rot_arr, r_pose_arr, r_shape_arr, r_trans_arr,
          l_rot_arr, l_pose_arr, l_shape_arr, l_trans_arr).

    pseudo_video=True (默认): 把所有帧的伪标 overlay 也拼成一个左右拼接的 mp4
    到 ``<undist_root>/_pseudo_vis/<seq>_pseudo_overlay.mp4``, 与
    Hand_Estimation 自己的 mano video 同 fps 同分辨率方便对比.
    """
    if len(cam_names_in_order) < 2:
        raise RuntimeError("init_hands_from_multiview 需要至少 2 个相机")

    # 1) undistort + calib
    undist_root, capture_id, newKs = prepare_undistort_dir(
        capture_dir, cams_dict, cam_names_in_order, world_name,
        frame_dirs_orig, force=force_undistort,
    )

    # cam_name -> 整数 cam_idx ('0', '1', ...)
    cam_idx_of = {n: i for i, n in enumerate(cam_names_in_order)}
    frame_dirs_undist = {
        n: undist_root / capture_id / str(cam_idx_of[n]) / "images_undistorted"
        for n in cam_names_in_order
    }

    # 2) 每相机每帧每手跑 HaMER 拿 21 个 2D 关节
    write_pseudo_labels_from_hamer(
        undist_root, capture_id, cam_names_in_order, cam_idx_of,
        total_frames, frame_dirs_undist, newKs,
        hamer=hamer, detector=detector, parse_detections=parse_detections,
        force=force_pseudo,
    )

    # 2.5) 整段伪标 overlay 视频 (与 Hand_Estimation 输出的视频对比用)
    if pseudo_video:
        try:
            make_pseudo_video(
                undist_root, capture_id, cam_names_in_order, cam_idx_of,
                total_frames, frame_dirs_undist,
                fps=pseudo_video_fps, downscale=pseudo_video_downscale,
            )
        except Exception as e:
            print(f"[mvinit] 生成伪标视频失败 (不影响后续流程): {e}")

    # 3) (可选) 自监督微调 Hand_Estimation 权重
    finetuned_ckpt: Optional[Path] = None
    if mv_finetune_epochs and mv_finetune_epochs > 0:
        try:
            finetuned_ckpt = run_hand_estimation_finetune_subprocess(
                project_root, undist_root, capture_id,
                epochs=mv_finetune_epochs,
                gpu_id=gpu_id, lr=mv_finetune_lr, batch_size=mv_finetune_bs,
            )
        except Exception as e:
            print(f"[mvinit] 自监督微调失败 ({e}), 回退用原始权重")
            finetuned_ckpt = None

    # 4) Hand_Estimation 推理 (用原始权重或微调后权重)
    he_output_dir = undist_root / "_he_output"
    if he_output_dir.is_dir():
        # 清掉上次的 *_mano.json, 避免拿到旧结果
        for p in he_output_dir.glob("*_mano.json"):
            p.unlink()
    json_path = run_hand_estimation_subprocess(
        project_root, undist_root, he_output_dir, gpu_id=gpu_id,
        ckpt_override=finetuned_ckpt,
    )
    print(f"[mvinit] Hand_Estimation 输出: {json_path}")

    # 4) JSON -> arrays (NaN 占位让上游 SLERP 接管缺失帧)
    arrs = parse_mano_json_to_arrays(json_path, total_frames,
                                     mano_root=str(project_root))
    (r_rot_a, r_pose_a, r_shape_a, r_trans_a,
     l_rot_a, l_pose_a, l_shape_a, l_trans_a) = arrs

    # 4.5) 多视角 MANO 视频回退使用 Hand_Estimation 自带可视化:
    #   visualize_mano.py 子进程跑完会自动在 _he_output/ 下产出
    #     <seq>_hand0/  + <seq>_hand0.mp4  (右手)
    #     <seq>_hand1/  + <seq>_hand1.mp4  (左手)
    #   我们这边不再自渲染 _mv_mano_overlay.mp4. make_mv_mano_video()
    #   函数仍保留, 如需对比可手动调用。

    return arrs
