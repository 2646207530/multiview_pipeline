"""body_vis: 独立的「完整人体」overlay 渲染器 (不依赖 / 不改动 way_vis.py).

把 step5b IK 之后的 SMPLX body (世界顶点, 来自 ``body_verts_world.npz``) + 两只 MANO
手 + 物体一起, 投影到 npy 里每个相机视角, overlay 到原始去畸变图上, 逐视角出 mp4.

相机 / overlay 背景逻辑与 step6 保持一致:
  * view0 用 npy 自带 imgpath + imgnames;
  * view_i (i>0) 用 ``<undist_root>/<capture_id>/<i>/images_undistorted``.

可单独 CLI 调:
    python utils/body_vis.py --npy <hand.npy> --body-verts <body_verts_world.npz> \
        --out /tmp/body.mp4 --mano-root <pipeline_release> --obj <club.stl>
"""
from __future__ import annotations

import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import trimesh
import pyrender

# numpy._core / numpy.core pickle 兼容 (同 way_vis)
try:
    import numpy.core as _np_core
    sys.modules.setdefault("numpy._core", _np_core)
    if hasattr(_np_core, "multiarray"):
        sys.modules.setdefault("numpy._core.multiarray", _np_core.multiarray)
except Exception:
    pass

_MIRROR_LEFT = np.array([1.0, -1.0, -1.0], dtype=np.float32)
_OPENCV2GL = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)


def _transform(W2C, pts):
    return np.einsum("ij,...j->...i", W2C[:3, :3], pts) + W2C[:3, 3]


def _composite_roi(color, bg_img, w, h):
    """把渲染 RGBA overlay 到背景上, 只在 mesh 覆盖的 bounding-box 内做 float 混合.
    mesh 通常只占画面一小块, 整帧混合 (~60ms) 改 ROI 混合后 ~10ms."""
    if bg_img.shape[1] != w or bg_img.shape[0] != h:
        bg_img = cv2.resize(bg_img, (w, h))
    a = color[:, :, 3]
    rows = np.where(a.any(1))[0]
    out = bg_img.copy()
    if len(rows) == 0:
        return out
    cols = np.where(a.any(0))[0]
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    af = a[y0:y1, x0:x1, None].astype(np.float32) / 255.0
    rb = cv2.cvtColor(color[y0:y1, x0:x1, :3], cv2.COLOR_RGB2BGR)
    out[y0:y1, x0:x1] = (rb.astype(np.float32) * af
                         + out[y0:y1, x0:x1].astype(np.float32) * (1.0 - af)).astype(np.uint8)
    return out


def _resolve_cameras(cam_dict: dict):
    """解析 npy 相机块 -> [{K, W2C, name}]. mm 量级平移自动 ->m (同 way_vis)."""
    K_field = cam_dict.get("K")
    W2C_field = cam_dict.get("world2cam")
    views = cam_dict.get("views")
    n = len(K_field)
    if not isinstance(views, list) or len(views) != n:
        views = [f"cam{i}" for i in range(n)]
    W2C_all = [np.asarray(x, np.float64).copy() for x in W2C_field]
    t_norms = [float(np.linalg.norm(T[:3, 3])) for T in W2C_all]
    if max(t_norms) > 20.0:
        for T in W2C_all:
            T[:3, 3] /= 1000.0
    return [{"K": np.asarray(K_field[i], np.float64), "W2C": W2C_all[i],
             "name": views[i]} for i in range(n)]


def _overlay_sources(data: dict, view_idx: int):
    """view0 用 npy imgpath+imgnames; view_i 用同级 <i>/images_undistorted."""
    base = data.get("imgpath")
    names = data.get("imgnames") or []
    if not base:
        return None
    if view_idx == 0:
        ordered = [os.path.join(base, n) for n in names]
        return {"ordered": ordered, "by_frame": {i: p for i, p in enumerate(ordered)}}
    other = Path(base).parent.parent / str(view_idx) / "images_undistorted"
    if not other.is_dir():
        print(f"[body_vis] view{view_idx}: 目录不存在 {other}, 纯渲染背景")
        return None
    ordered = [str(other / n) for n in names if (other / n).is_file()]
    if len(ordered) != len(names) or not ordered:
        ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ordered = sorted(str(p) for p in other.iterdir() if p.suffix.lower() in ext)
    by_frame = {i: p for i, p in enumerate(ordered)}
    for p in ordered:
        m = re.search(r"(\d+)$", Path(p).stem)
        if m:
            by_frame.setdefault(int(m.group(1)), p)
    return {"ordered": ordered, "by_frame": by_frame}


def _mano_hand_verts(params: dict, mano_root: str, num_frames: int):
    """复刻 way_vis: 算两手世界顶点 (右手原参, 左手镜像). 返回 (vr, fr, vl, fl)."""
    import torch
    import smplx
    rh = params["right hand"]
    ml = smplx.create(mano_root, "MANO", use_pca=False, is_rhand=True, flat_hand_mean=True)
    with torch.no_grad():
        o = ml(global_orient=torch.tensor(np.asarray(rh["rot_r"], np.float32)[:num_frames]),
               hand_pose=torch.tensor(np.asarray(rh["pose_r"], np.float32)[:num_frames]),
               betas=torch.tensor(np.asarray(rh["shape_r"], np.float32)[:num_frames]),
               transl=torch.tensor(np.asarray(rh["trans_r"], np.float32)[:num_frames]))
    vr, fr = o.vertices.numpy(), ml.faces
    vl = fl = None
    if "left hand" in params:
        lh = params["left hand"]
        rot_l = np.asarray(lh["rot_l"], np.float32)[:num_frames] * _MIRROR_LEFT
        pose_l = (np.asarray(lh["pose_l"], np.float32)[:num_frames].reshape(-1, 3)
                  * _MIRROR_LEFT).reshape(num_frames, -1)
        ml_l = smplx.create(mano_root, "MANO", use_pca=False, is_rhand=False, flat_hand_mean=True)
        with torch.no_grad():
            ol = ml_l(global_orient=torch.tensor(rot_l),
                      hand_pose=torch.tensor(pose_l),
                      betas=torch.tensor(np.asarray(lh["shape_l"], np.float32)[:num_frames]),
                      transl=torch.tensor(np.asarray(lh["trans_l"], np.float32)[:num_frames]))
        vl, fl = ol.vertices.numpy(), ml_l.faces
    return vr, fr, vl, fl


def render(npy_path: str, body_verts_npz: str, out_video: str, mano_root: str,
           obj_path: Optional[str] = None, fps: int = 10,
           show_hands: bool = True, show_object: bool = True,
           max_frames: Optional[int] = None, stride: int = 1,
           crf: int = 28,
           club_pts: Optional[np.ndarray] = None,
           club_R: Optional[np.ndarray] = None,
           club_t: Optional[np.ndarray] = None,
           club_obs: Optional[np.ndarray] = None,
           club_color=(40, 220, 40)):
    """club_pts (N,3, 杆-local/rig 系) + club_R (T,3,3) + club_t (T,3, 米, rig->world)
    时, 把球杆当采样点云投影成 2D 点画到 overlay 上 (比渲精细 mesh 快很多)."""
    data = np.load(npy_path, allow_pickle=True).item()
    seq_key = list(data["data_dict"].keys())[0]
    params = data["data_dict"][seq_key]["params"]
    cams = _resolve_cameras(params["camera"])
    print(f"[body_vis] {len(cams)} 视角: {[c['name'] for c in cams]}")

    bd = np.load(body_verts_npz)
    body_verts = bd["verts"].astype(np.float64)        # (T,V,3) 世界系
    body_faces = bd["faces"]
    T = body_verts.shape[0]
    if max_frames is not None:
        T = min(T, int(max_frames))
        body_verts = body_verts[:T]

    # 手 / 物体
    vr = fr = vl = fl = None
    if show_hands:
        vr, fr, vl, fl = _mano_hand_verts(params, mano_root, T)
    obj_base = obj_rotmats = obj_trans = obj_faces = None
    if show_object and obj_path and Path(obj_path).exists() and "object" in params:
        from scipy.spatial.transform import Rotation as R
        m = trimesh.load(obj_path, process=False)
        if isinstance(m, trimesh.Scene):
            m = m.dump()[0]
        obj_base = np.asarray(m.vertices, np.float64)
        obj_faces = m.faces
        orot = np.asarray(params["object"]["obj_rot"], np.float64)[:T]
        otr = np.asarray(params["object"]["obj_trans"], np.float64)[:T]
        if float(np.max(np.ptp(obj_base, 0))) > 10.0:
            obj_base = obj_base * 0.001
        if float(np.median(np.linalg.norm(otr, axis=1))) > 20.0:
            otr = otr * 0.001
        if not (np.allclose(orot, 0) and np.allclose(otr, 0)):
            obj_rotmats = R.from_rotvec(orot).as_matrix()
            obj_trans = otr
        else:
            obj_base = None  # 占位 object, 不渲染

    body_mat = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.1, alphaMode="OPAQUE", baseColorFactor=(0.65, 0.7, 0.85, 1.0))
    hand_mat_r = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.1, alphaMode="OPAQUE", baseColorFactor=(0.85, 0.6, 0.5, 1.0))
    hand_mat_l = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.1, alphaMode="OPAQUE", baseColorFactor=(0.5, 0.6, 0.85, 1.0))
    obj_mat = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.3, alphaMode="OPAQUE", baseColorFactor=(0.2, 0.8, 0.2, 1.0))

    base, ext = os.path.splitext(out_video)
    os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
    out_videos = []

    for v_idx, cam in enumerate(cams):
        K, W2C = cam["K"], cam["W2C"]
        bg = _overlay_sources(data, v_idx)
        h = w = None
        if bg and bg["ordered"]:
            im0 = cv2.imread(bg["ordered"][0])
            if im0 is not None:
                h, w = im0.shape[:2]
        if h is None:
            w, h = int(round(K[0, 2] * 2)), int(round(K[1, 2] * 2))

        body_cam = _transform(W2C, body_verts)
        vr_cam = _transform(W2C, vr) if vr is not None else None
        vl_cam = _transform(W2C, vl) if vl is not None else None

        view_path = f"{base}_view{v_idx}{ext}"
        renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
        camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
        vw = cv2.VideoWriter(view_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        idxs = list(range(0, T, stride))
        print(f"[body_vis] view{v_idx} {w}x{h} ({len(idxs)} 帧) -> {view_path}")

        # 预解码背景图 (后台线程, 与 GPU 渲染重叠, 掩掉 ~14ms/帧 的 imread)
        def _bg_path(i):
            if bg is None:
                return None
            p = bg["by_frame"].get(i)
            if p is None and i < len(bg["ordered"]):
                p = bg["ordered"][i]
            return p
        pre = ThreadPoolExecutor(max_workers=4) if bg is not None else None
        AHEAD = 6
        futs = {}
        if pre is not None:
            for j in idxs[:AHEAD]:
                futs[j] = pre.submit(cv2.imread, _bg_path(j)) if _bg_path(j) else None

        for k, i in enumerate(idxs):
            scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.6, 0.6, 0.6])
            scene.add(camera, pose=_OPENCV2GL)
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=_OPENCV2GL)

            vb = body_cam[i]
            if np.isfinite(vb).all():
                scene.add(pyrender.Mesh.from_trimesh(
                    trimesh.Trimesh(vb, body_faces, process=False), material=body_mat, smooth=False))
            if obj_base is not None:
                ow = (obj_rotmats[i] @ obj_base.T).T + obj_trans[i]
                scene.add(pyrender.Mesh.from_trimesh(
                    trimesh.Trimesh(_transform(W2C, ow), obj_faces, process=False), material=obj_mat, smooth=False))
            if vr_cam is not None and np.isfinite(vr_cam[i]).all():
                scene.add(pyrender.Mesh.from_trimesh(
                    trimesh.Trimesh(vr_cam[i], fr, process=False), material=hand_mat_r, smooth=False))
            if vl_cam is not None and np.isfinite(vl_cam[i]).all():
                scene.add(pyrender.Mesh.from_trimesh(
                    trimesh.Trimesh(vl_cam[i], fl, process=False), material=hand_mat_l, smooth=False))

            color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            img = None
            if pre is not None:
                fut = futs.pop(i, None)
                img = fut.result() if fut is not None else None
                nxt = k + AHEAD
                if nxt < len(idxs):
                    jp = _bg_path(idxs[nxt])
                    futs[idxs[nxt]] = pre.submit(cv2.imread, jp) if jp else None
            if img is not None:
                frame = _composite_roi(color, img, w, h)
            else:
                frame = cv2.cvtColor(color[:, :, :3], cv2.COLOR_RGB2BGR)

            # 球杆点云: rig 系点 -> world -> 本视角相机 -> 投影画点
            if club_pts is not None and (club_obs is None or club_obs[i]):
                pw = (club_R[i] @ club_pts.T).T + club_t[i]
                pc = (W2C[:3, :3] @ pw.T).T + W2C[:3, 3]
                z = pc[:, 2]
                m = z > 1e-6
                u = (K[0, 0] * pc[:, 0] / z + K[0, 2])
                v = (K[1, 1] * pc[:, 1] / z + K[1, 2])
                m &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
                uu = u[m].astype(np.int32); vv = v[m].astype(np.int32)
                for dy in (-1, 0, 1):
                    yy = np.clip(vv + dy, 0, h - 1)
                    for dx in (-1, 0, 1):
                        xx = np.clip(uu + dx, 0, w - 1)
                        frame[yy, xx] = club_color
            vw.write(frame)

        if pre is not None:
            pre.shutdown(wait=False)
        vw.release()
        renderer.delete()
        if shutil.which("ffmpeg"):
            h264 = view_path.rsplit(".", 1)[0] + "_h264.mp4"
            r = subprocess.run(["ffmpeg", "-y", "-i", view_path, "-c:v", "libx264",
                                "-preset", "fast", "-crf", str(int(crf)), "-pix_fmt", "yuv420p", h264],
                               capture_output=True)
            if r.returncode == 0:
                os.replace(h264, view_path)
        out_videos.append(view_path)
        print(f"[body_vis] view{v_idx} 完成: {view_path}")
    return out_videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True)
    ap.add_argument("--body-verts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mano-root", required=True)
    ap.add_argument("--obj", default=None)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--no-hands", action="store_true")
    ap.add_argument("--no-object", action="store_true")
    ap.add_argument("--stride", type=int, default=1, help="每 stride 帧渲一帧 (快速预览)")
    a = ap.parse_args()
    render(a.npy, a.body_verts, a.out, a.mano_root, obj_path=a.obj, fps=a.fps,
           show_hands=not a.no_hands, show_object=not a.no_object, stride=a.stride)


if __name__ == "__main__":
    main()
