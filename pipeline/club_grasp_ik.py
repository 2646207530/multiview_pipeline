"""Step 5: 球杆驱动的手 + 人体手臂 IK.

输入:
  * 球杆 6-DoF 轨迹 (omni-club-tracking 的 results.json, rig->cam0 位姿, 平移 mm).
  * 固定的手-杆相对抓握 (canonical_grasp.npy, 手在 rig/杆-local 系).
  * 人体姿态 (world_params/smplx_params_world.pt, cam0 世界系).

做法:
  1. 把 canonical grasp 的手按每帧球杆位姿刚性搬到世界系 -> 逐帧两手 MANO 参数 (手跟着杆动).
  2. 组装成 hand npy (way_vis 格式 + 相机块).
  3. 复用 step5b: 以这些手腕为目标对 SMPLX 人体做仅手臂 IK + 灌手指 -> 完整带手人体 + overlay.

世界系 = cam0 (= camera_params.json 里 hamer 彩色相机, 与 reference_camera 同一物理相机).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from utils.camera_npy import (  # type: ignore
    _load_camera_params, _resolve_color_cams, _build_camera_block)
from .state import PipelineState
from .workspace import Workspace
from . import step5b_body_ik

_MIRROR = np.array([1.0, -1.0, -1.0], dtype=np.float32)

# 每根球杆的「预定义标准握杆」(standard_pose) 与 mesh (club-assets, 现统一 mm).
_STANDARD_POSE_DIR = Path("/data2/fubingshuai/golf/standard_pose")
_CLUB_ASSETS_DIR = Path("/data2/fubingshuai/golf/data/club-assets")


def _resolve_club_assets(club: str):
    """按球杆型号解析 (预定义抓握 npy, 球杆 mesh).
    抓握: standard_pose/<club>_canonical_grasp.npy ; mesh: club-assets/<club>/*.stl (mm)."""
    grasp = _STANDARD_POSE_DIR / f"{club}_canonical_grasp.npy"
    if not grasp.exists():
        raise FileNotFoundError(
            f"找不到球杆 {club} 的预定义握杆 {grasp} (先用 scripts/standard_pose_bind.py 绑定)")
    # 取该型号目录下的 stl, 跳过 *_used/*_final 这类变体, 优先主 mesh
    stls = sorted((_CLUB_ASSETS_DIR / club).glob("*.stl"))
    if not stls:
        raise FileNotFoundError(f"找不到球杆 mesh: {_CLUB_ASSETS_DIR / club}/*.stl")
    main = [s for s in stls if not any(t in s.stem for t in ("_used", "_final"))]
    mesh = (main or stls)[0]
    return str(grasp), str(mesh)


def _quat_to_R(qw, qx, qy, qz):
    from scipy.spatial.transform import Rotation as R
    return R.from_quat([qx, qy, qz, qw]).as_matrix()


def _load_club_traj(results_json: str):
    """results.json -> (R_club(T,3,3), t_club(T,3) 米, observed(T,))  rig->cam0."""
    d = json.loads(Path(results_json).read_text())
    traj = d["trajectory"]
    T = len(traj)
    Rs = np.zeros((T, 3, 3)); ts = np.zeros((T, 3)); obs = np.zeros(T, bool)
    for i, f in enumerate(traj):
        Rs[i] = _quat_to_R(f["qw"], f["qx"], f["qy"], f["qz"])
        ts[i] = [f["tx"], f["ty"], f["tz"]]
        obs[i] = bool(f.get("observed", True))
    ts /= 1000.0  # mm -> m
    return Rs, ts, obs


def _mano_root_j0(mano_root, is_rhand, betas):
    import torch, smplx
    ml = smplx.create(mano_root, "MANO", use_pca=False, is_rhand=is_rhand,
                      flat_hand_mean=True, batch_size=1)
    with torch.no_grad():
        o = ml(global_orient=torch.zeros(1, 3), hand_pose=torch.zeros(1, 45),
               betas=torch.tensor(betas, dtype=torch.float32).reshape(1, 10),
               transl=torch.zeros(1, 3))
    return o.joints[0, 0].numpy().astype(np.float64)


def _propagate_side(go0, transl0, J0, Rs, ts):
    """canonical 手 (go0,transl0 in rig) 按每帧 (Rs,ts) 刚性搬到世界系.
    new_go = Rs@R(go0); new_transl = Rs@(J0+transl0)+ts - J0."""
    from scipy.spatial.transform import Rotation as R
    Rgo0 = R.from_rotvec(go0).as_matrix()
    T = Rs.shape[0]
    new_go = R.from_matrix(np.einsum("tij,jk->tik", Rs, Rgo0)).as_rotvec()
    base = (J0 + transl0)
    new_transl = np.einsum("tij,j->ti", Rs, base) + ts - J0
    return new_go.astype(np.float32), new_transl.astype(np.float32)


def build_hand_npy(capture_dir: Path, seq_name: str, results_json: str,
                   canonical_grasp_npy: str, out_npy: Path,
                   mano_root: str = str(_PROJECT)) -> int:
    """grasp ⊗ club -> 逐帧两手世界 MANO 参数, 存 way_vis 格式 npy. 返回帧数."""
    state = PipelineState.load(Workspace(capture_dir=capture_dir, seq_name=seq_name))
    undist = state.steps["undistort"].outputs
    capture_id = undist["capture_id"]
    cam_names = undist["cam_names"]
    undist_root = Path(undist["undist_root"])

    Rs, ts, obs = _load_club_traj(results_json)
    cg = np.load(canonical_grasp_npy, allow_pickle=True).item()
    cgp = cg["data_dict"][list(cg["data_dict"].keys())[0]]["params"]
    T = Rs.shape[0]

    # 右手
    rh = cgp["right hand"]
    go_r0 = np.asarray(rh["rot_r"])[0].astype(np.float64)
    pose_r0 = np.asarray(rh["pose_r"])[0].astype(np.float32)
    shape_r0 = np.asarray(rh["shape_r"])[0].astype(np.float32)
    transl_r0 = np.asarray(rh["trans_r"])[0].astype(np.float64)
    J0_r = _mano_root_j0(mano_root, True, shape_r0)
    rot_r, trans_r = _propagate_side(go_r0, transl_r0, J0_r, Rs, ts)
    pose_r = np.tile(pose_r0, (T, 1)); shape_r = np.tile(shape_r0, (T, 1))

    params = {"right hand": {"rot_r": rot_r, "pose_r": pose_r,
                             "trans_r": trans_r, "shape_r": shape_r}}

    has_left = "left hand" in cgp
    if has_left:
        lh = cgp["left hand"]
        # canonical 存的是右手约定; 镜像成左 MANO 渲染参数, 变换后再 un-mirror 存回
        go_l0_m = np.asarray(lh["rot_l"])[0].astype(np.float64) * _MIRROR
        pose_l0_raw = np.asarray(lh["pose_l"])[0].astype(np.float32)
        shape_l0 = np.asarray(lh["shape_l"])[0].astype(np.float32)
        transl_l0 = np.asarray(lh["trans_l"])[0].astype(np.float64)
        J0_l = _mano_root_j0(mano_root, False, shape_l0)
        go_l_m, trans_l = _propagate_side(go_l0_m, transl_l0, J0_l, Rs, ts)
        rot_l = (go_l_m * _MIRROR).astype(np.float32)        # un-mirror
        pose_l = np.tile(pose_l0_raw, (T, 1)); shape_l = np.tile(shape_l0, (T, 1))
        params["left hand"] = {"rot_l": rot_l, "pose_l": pose_l,
                               "trans_l": trans_l, "shape_l": shape_l}

    # 相机块 (与 hand pipeline 一致): world = hamer(cam0) 系
    cams = _load_camera_params(capture_dir)
    hamer_name, other_name = _resolve_color_cams(cams, capture_dir)
    new_K = {n: np.array(undist["newKs"][n], dtype=np.float32) for n in cam_names}
    params["camera"] = _build_camera_block(cams, hamer_name, other_name, new_K_per_cam=new_K)

    imgdir = undist_root / capture_id / "0" / "images_undistorted"
    root = {
        "imgnames": [f"{i:06d}.jpg" for i in range(T)],
        "imgpath": str(imgdir),
        "data_dict": {seq_name: {"params": params}},
    }
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, root, allow_pickle=True)
    print(f"[club_grasp] 写 hand npy {out_npy.name} (T={T}, observed={int(obs.sum())}/{T})")
    return T


def _sample_club_points(club_mesh: str, n: int = 700):
    import trimesh
    m = trimesh.load(club_mesh, process=False)
    if isinstance(m, trimesh.Scene):
        m = m.dump()[0]
    pts, _ = trimesh.sample.sample_surface(m, n)
    pts = np.asarray(pts, np.float64)
    if float(np.max(np.ptp(pts, 0))) > 10.0:   # mm -> m (与 way_vis 一致)
        pts = pts * 0.001
    return pts


def run(capture_dir: str, seq_name: str, results_json: str,
        club: Optional[str] = None,
        canonical_grasp_npy: Optional[str] = None, club_mesh: Optional[str] = None,
        render_stride: int = 1, fps: int = 10, iters: int = 250,
        crf: int = 28, draw_club: bool = True, device: Optional[str] = None):
    # 按球杆型号自动加载 standard_pose 下该杆预定义的手 + club-assets 的 mesh;
    # 也可用 canonical_grasp_npy / club_mesh 显式覆盖.
    if club and (canonical_grasp_npy is None or club_mesh is None):
        g, m = _resolve_club_assets(club)
        canonical_grasp_npy = canonical_grasp_npy or g
        club_mesh = club_mesh or m
    if not canonical_grasp_npy or not club_mesh:
        raise ValueError("需要 --club, 或同时给 --canonical_grasp 和 --club_mesh")
    print(f"[club_grasp] club={club} grasp={Path(canonical_grasp_npy).name} "
          f"mesh={Path(club_mesh).name}")
    cap = Path(capture_dir)
    ws = Workspace(capture_dir=cap, seq_name=seq_name)

    # 1) 手跟杆 -> hand npy
    hand_npy = ws.root / f"{seq_name}_clubhand.npy"
    build_hand_npy(cap, seq_name, results_json, canonical_grasp_npy, hand_npy)

    # 2) 把 hand npy 喂给 step5b (state shim), 只做 IK + 存 body 顶点, 不在这里渲染
    state = PipelineState.load(ws)
    state.mark_done("infer", npy_path=str(hand_npy))
    state.save(ws)
    info = step5b_body_ik.run(ws, iters=iters, align_wrist=True,
                              render_overlay=False, device=device)

    # 3) 自己渲染: 完整带手人体 + 球杆点云 (压画质, 体积接近 step4)
    import importlib.util
    bv_path = _PROJECT / "utils" / "body_vis.py"
    spec = importlib.util.spec_from_file_location("_pipeline_body_vis", bv_path)
    bv = importlib.util.module_from_spec(spec); spec.loader.exec_module(bv)

    club_pts = club_R = club_t = club_obs = None
    if draw_club:
        Rs, ts, obs = _load_club_traj(results_json)
        club_pts = _sample_club_points(club_mesh)
        club_R, club_t, club_obs = Rs, ts, obs

    ws.vis_dir.mkdir(parents=True, exist_ok=True)
    body_out = ws.vis_dir / f"{seq_name}_body.mp4"
    vids = bv.render(npy_path=str(hand_npy), body_verts_npz=info["body_verts_npz"],
                     out_video=str(body_out), mano_root=str(_PROJECT),
                     fps=fps, show_hands=False, show_object=False,
                     stride=render_stride, crf=crf,
                     club_pts=club_pts, club_R=club_R, club_t=club_t, club_obs=club_obs)
    info["body_view_videos"] = [str(v) for v in vids]
    state.mark_done("ik", **{**state.steps["ik"].outputs, "body_view_videos": info["body_view_videos"]})
    state.save(ws)
    print(f"[club_grasp] 完整人体+球杆 overlay: {len(vids)} 视角")
    return info


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture_dir", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--results_json", required=True)
    ap.add_argument("--club", default=None,
                    help="球杆型号: 自动加载 standard_pose/<club>_canonical_grasp.npy + club-assets/<club>/*.stl")
    ap.add_argument("--canonical_grasp", default=None, help="(可选) 显式指定预定义握杆 npy")
    ap.add_argument("--club_mesh", default=None, help="(可选) 显式指定球杆 mesh")
    ap.add_argument("--render_stride", type=int, default=1)
    ap.add_argument("--crf", type=int, default=28, help="overlay 视频压缩质量 (越大越小, 28≈step4 体积)")
    ap.add_argument("--no_club", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    info = run(a.capture_dir, a.seq, a.results_json, club=a.club,
               canonical_grasp_npy=a.canonical_grasp, club_mesh=a.club_mesh,
               render_stride=a.render_stride, crf=a.crf, draw_club=not a.no_club,
               device=a.device)
    print("INFO", info)
