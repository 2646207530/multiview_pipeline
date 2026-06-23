"""单序列: 对一段人体做「手跟杆」+ 仅手臂 IK + 灌手指 -> 完整带手 SMPLX.

给四个绝对路径:
  --club_traj    球杆 6DoF 轨迹 (omni-club-tracking 的 trajectory.csv)
  --smplx_world  这段人体 (GVHMR 的 smplx_params_world.pt)
  --grasp        该球杆预定义握杆 (standard_pose/<型号>_canonical_grasp.npy)
  --out          输出完整带手人体 (smplx_params_world_ik.pt)

把 grasp 的两手按每帧球杆位姿刚性搬到世界系 -> 作为腕目标对人体做仅手臂 IK,
并把 MANO 手指姿态灌进 SMPLX 自己的手, 存到 --out.

用法:
  python single_ik.py \
    --club_traj /abs/trajectory.csv \
    --smplx_world /abs/smplx_params_world.pt \
    --grasp /abs/3-1-wood-03_canonical_grasp.npy \
    --out /abs/smplx_params_world_ik.pt
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_PROJECT = Path(__file__).resolve().parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from pipeline.step5b_body_ik import solve_body_ik, _resolve_smplx_model_dir  # noqa: E402
from pipeline.club_grasp_ik import _propagate_side, _mano_root_j0, _MIRROR    # noqa: E402


def load_traj_csv(path: str):
    """trajectory.csv -> R_club(T,3,3), t_club(T,3, 米). 跳过 # 注释行, 按列名取 quat/trans."""
    from scipy.spatial.transform import Rotation as R
    Rs, ts = [], []
    header = None
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if header is None:
                header = {n.strip(): i for i, n in enumerate(row)}
                continue
            g = lambda n: float(row[header[n]])
            Rs.append(R.from_quat([g("qx"), g("qy"), g("qz"), g("qw")]).as_matrix())
            ts.append([g("tx"), g("ty"), g("tz")])
    Rs = np.asarray(Rs, np.float64)
    ts = np.asarray(ts, np.float64) / 1000.0   # mm -> m
    return Rs, ts


def build_hand_params(grasp_npy: str, Rs, ts, mano_root: str) -> dict:
    """grasp 第 0 帧两手 (rig 系) 按每帧球杆位姿刚性搬到世界系 -> 逐帧 way_vis 格式手参数."""
    cg = np.load(grasp_npy, allow_pickle=True).item()
    P = cg["data_dict"][list(cg["data_dict"].keys())[0]]["params"]
    T = Rs.shape[0]

    def tile(v):
        return np.tile(np.asarray(v, np.float32).reshape(1, -1), (T, 1))

    rh = P["right hand"]
    go = np.asarray(rh["rot_r"])[0].astype(np.float64)
    pose = np.asarray(rh["pose_r"])[0].astype(np.float32)
    betas = np.asarray(rh["shape_r"])[0].astype(np.float32)
    transl = np.asarray(rh["trans_r"])[0].astype(np.float64)
    ngo, ntr = _propagate_side(go, transl, _mano_root_j0(mano_root, True, betas), Rs, ts)
    params = {"right hand": {"rot_r": ngo, "pose_r": tile(pose),
                             "trans_r": ntr, "shape_r": tile(betas)}}

    if "left hand" in P:
        lh = P["left hand"]
        go_m = np.asarray(lh["rot_l"])[0].astype(np.float64) * _MIRROR   # 镜像渲染系
        pose_raw = np.asarray(lh["pose_l"])[0].astype(np.float32)
        betas_l = np.asarray(lh["shape_l"])[0].astype(np.float32)
        transl_l = np.asarray(lh["trans_l"])[0].astype(np.float64)
        ngo_m, ntr_l = _propagate_side(go_m, transl_l, _mano_root_j0(mano_root, False, betas_l), Rs, ts)
        params["left hand"] = {"rot_l": (ngo_m * _MIRROR).astype(np.float32),  # un-mirror 存回
                               "pose_l": tile(pose_raw),
                               "trans_l": ntr_l, "shape_l": tile(betas_l)}
    return params


def render_from_calib(calib_yaml: str, body_verts_npz: str, out_video: str,
                      mano_root: str, T: int, stride: int = 2, crf: int = 28):
    """给一个去畸变标定 yaml (含 K + world->cam R/t), 渲染该视角下的完整带手人体 overlay 视频.
    背景图自动从 calib_undistorted/<N>.yaml 旁边的 <N>/images_undistorted/ 找 (找不到则黑底)."""
    import yaml
    import importlib.util
    cal = yaml.safe_load(open(calib_yaml))
    K = np.asarray(cal["K"], float)
    w2c = np.eye(4)
    w2c[:3, :3] = np.asarray(cal["R"], float)
    w2c[:3, 3] = np.asarray(cal["t"], float).reshape(3)
    view = Path(calib_yaml).stem                       # 0.yaml -> "0"
    img_dir = Path(calib_yaml).resolve().parent.parent / view / "images_undistorted"
    scene = {
        "imgnames": [f"{i:06d}.jpg" for i in range(T)],
        "imgpath": str(img_dir) if img_dir.is_dir() else "",
        "data_dict": {"single": {"params": {"camera": {
            "K": [K.tolist()], "world2cam": [w2c.tolist()], "views": [f"cam{view}"]}}}},
    }
    scene_npy = Path(out_video).with_suffix(".scene.npy")
    np.save(scene_npy, scene, allow_pickle=True)
    spec = importlib.util.spec_from_file_location("_bv", str(_PROJECT / "utils" / "body_vis.py"))
    bv = importlib.util.module_from_spec(spec); spec.loader.exec_module(bv)
    return bv.render(npy_path=str(scene_npy), body_verts_npz=str(body_verts_npz),
                     out_video=str(out_video), mano_root=mano_root, fps=10,
                     show_hands=False, show_object=False, stride=stride, crf=crf)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--club_traj", required=True, help="球杆轨迹 trajectory.csv 绝对路径")
    ap.add_argument("--smplx_world", required=True, help="人体 smplx_params_world.pt 绝对路径")
    ap.add_argument("--grasp", required=True, help="球杆预定义握杆 npy 绝对路径")
    ap.add_argument("--out", required=True, help="输出 smplx_params_world_ik.pt 绝对路径")
    ap.add_argument("--calib", default=None,
                    help="(可选) 去畸变标定 yaml (如 calib_undistorted/0.yaml); 给了就出该视角可视化视频")
    ap.add_argument("--vis_out", default=None, help="(可选) 可视化视频输出路径; 默认 <out>_vis.mp4")
    ap.add_argument("--vis_stride", type=int, default=2, help="可视化每 N 帧渲一帧")
    ap.add_argument("--crf", type=int, default=28, help="可视化视频压缩 (越大越小)")
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--no_align_wrist", action="store_true", help="不把 wrist 朝向对齐 MANO 手")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    mano_root = str(_PROJECT)
    Rs, ts = load_traj_csv(a.club_traj)
    print(f"[single_ik] 球杆轨迹 {Rs.shape[0]} 帧; grasp={Path(a.grasp).name}")
    params = build_hand_params(a.grasp, Rs, ts, mano_root)
    # 给了 --calib 才需要逐帧 body 顶点 (供渲染)
    verts_npz = str(Path(a.out).with_suffix(".body_verts.npz")) if a.calib else None
    info = solve_body_ik(params, a.smplx_world, a.out,
                         smplx_model_dir=_resolve_smplx_model_dir(), mano_root=mano_root,
                         iters=a.iters, align_wrist=not a.no_align_wrist, device=a.device,
                         save_verts_npz=verts_npz)
    print("[single_ik] DONE ->", a.out)
    print("[single_ik]", info)

    if a.calib:
        vis_out = a.vis_out or (str(Path(a.out).with_suffix("")) + "_vis.mp4")
        vids = render_from_calib(a.calib, info["body_verts_npz"], vis_out,
                                 mano_root, info["n_frames"], a.vis_stride, a.crf)
        print("[single_ik] 可视化视频:", vids)


if __name__ == "__main__":
    main()
