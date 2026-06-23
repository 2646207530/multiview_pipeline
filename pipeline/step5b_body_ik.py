"""人体结果 IK (仅调整手臂) + 灌 MANO 手指 -> 完整带手 SMPLX.

输入: 一段两手的 MANO 参数 (way_vis 格式, world=cam0 系) + 人体 ``smplx_params_world.pt``
(SMPLX body, 仅 21 body 关节, 无手指). 做法:
  1. 位置: 优化 collar/shoulder/elbow 让 SMPLX 腕关节(20/21)贴合 MANO 腕世界坐标.
  2. 朝向: 把 wrist 局部姿态设成让 SMPLX 腕全局朝向 == MANO 手全局朝向.
  3. 灌手: SMPLX 的手就是 MANO, 把 45 维手指姿态设进 left/right_hand_pose.
输出: ``smplx_params_world_ik.pt`` (完整带手 SMPLX, 同结构) [+ 可选 body_verts_world.npz].

`solve_body_ik(...)` 是无 Workspace 依赖的核心 (单序列脚本 single_ik.py 也用它);
`run(ws, ...)` 是 pipeline 内的包装 (从 state 解析路径 + 可选渲染 overlay).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from .state import PipelineState
from .workspace import Workspace

# npy 左手槽位存的是右手 MANO 参数, 用左手 MANO 加载时轴角沿 Y/Z 取反 (与 way_vis 一致)
_MIRROR_LEFT = np.array([1.0, -1.0, -1.0], dtype=np.float32)

# SMPLX body_pose 槽位 = 关节 index - 1 (关节0=pelvis 由 global_orient 表示).
#   左臂: collar13 shoulder16 elbow18    右臂: collar14 shoulder17 elbow19
_ARM_JOINTS_L = [13, 16, 18]
_ARM_JOINTS_R = [14, 17, 19]
_WRIST_L, _WRIST_R = 20, 21
# 从 pelvis 到 wrist 的运动链 (含 wrist 自身), 用于 FK 求腕全局朝向
_CHAIN_L = [3, 6, 9, 13, 16, 18, 20]
_CHAIN_R = [3, 6, 9, 14, 17, 19, 21]


def _arm_cols(joints: List[int]) -> List[int]:
    cols: List[int] = []
    for j in joints:
        s = j - 1
        cols += [3 * s, 3 * s + 1, 3 * s + 2]
    return cols


def _mano_wrist_world(params: dict, mano_root: str):
    """复刻 way_vis 的 MANO 前向, 取两手腕关节世界坐标 + 腕世界朝向矩阵.

    返回 (pos_r(T,3), Rmat_r(T,3,3), pos_l(T,3), Rmat_l(T,3,3), has_left)."""
    import torch
    import smplx
    from smplx.lbs import batch_rodrigues

    rh = params["right hand"]
    rot_r = np.asarray(rh["rot_r"], np.float32)
    pose_r = np.asarray(rh["pose_r"], np.float32)
    shape_r = np.asarray(rh["shape_r"], np.float32)
    trans_r = np.asarray(rh["trans_r"], np.float32)
    T = rot_r.shape[0]

    ml_r = smplx.create(mano_root, "MANO", use_pca=False, is_rhand=True,
                        flat_hand_mean=True, batch_size=T)
    with torch.no_grad():
        out_r = ml_r(global_orient=torch.tensor(rot_r),
                     hand_pose=torch.tensor(pose_r),
                     betas=torch.tensor(shape_r),
                     transl=torch.tensor(trans_r))
    pos_r = out_r.joints[:, 0].numpy()
    Rmat_r = batch_rodrigues(torch.tensor(rot_r).reshape(-1, 3)).numpy()

    has_left = "left hand" in params
    pos_l = Rmat_l = None
    if has_left:
        lh = params["left hand"]
        rot_l = np.asarray(lh["rot_l"], np.float32) * _MIRROR_LEFT
        pose_l = (np.asarray(lh["pose_l"], np.float32).reshape(-1, 3) * _MIRROR_LEFT
                  ).reshape(np.asarray(lh["pose_l"]).shape)
        shape_l = np.asarray(lh["shape_l"], np.float32)
        trans_l = np.asarray(lh["trans_l"], np.float32)
        ml_l = smplx.create(mano_root, "MANO", use_pca=False, is_rhand=False,
                            flat_hand_mean=True, batch_size=T)
        with torch.no_grad():
            out_l = ml_l(global_orient=torch.tensor(rot_l),
                         hand_pose=torch.tensor(pose_l),
                         betas=torch.tensor(shape_l),
                         transl=torch.tensor(trans_l))
        pos_l = out_l.joints[:, 0].numpy()
        Rmat_l = batch_rodrigues(torch.tensor(rot_l).reshape(-1, 3)).numpy()

    return pos_r, Rmat_r, pos_l, Rmat_l, has_left


def _chain_global_rot(global_orient_mat, pose_mats, chain):
    """沿运动链累乘局部旋转得到链末端关节的全局旋转. pose_mats: (T,21,3,3)."""
    Rg = global_orient_mat
    for j in chain:
        Rg = Rg @ pose_mats[:, j - 1]
    return Rg


def _resolve_smplx_model_dir() -> Path:
    """找 SMPLX 模型根 (含 ``smplx/SMPLX_NEUTRAL.npz``)."""
    cands = [
        _PROJECT / "assets" / "body_models",
        _PROJECT / "pipeline" / "assets" / "body_models",
        _PROJECT.parent / "pipeline" / "assets" / "body_models",
    ]
    for c in cands:
        if (c / "smplx" / "SMPLX_NEUTRAL.npz").exists():
            return c
    raise FileNotFoundError(
        "找不到 SMPLX 模型 (smplx/SMPLX_NEUTRAL.npz), 搜索过: "
        + ", ".join(str(c) for c in cands))


def solve_body_ik(params: dict, smplx_pt, out_pt, *,
                  smplx_model_dir=None, mano_root: str = str(_PROJECT),
                  iters: int = 250, lr: float = 0.05,
                  w_pos: float = 1.0, w_reg: float = 2e-3, w_smooth: float = 1e-2,
                  align_wrist: bool = True, device: Optional[str] = None,
                  save_verts_npz=None) -> Dict[str, Any]:
    """对一段人体做仅手臂 IK + 灌 MANO 手指, 存完整带手 SMPLX 到 out_pt. 无 Workspace 依赖.

    params:  way_vis 格式的逐帧手参数 (含 'right hand'/'left hand').
    smplx_pt: 人体 world 参数 .pt (含 'smpl_params_world').
    out_pt:  输出完整带手 SMPLX .pt.
    save_verts_npz: 给了就额外存逐帧 body 世界顶点 (供渲染 overlay).
    """
    import torch
    import smplx
    from smplx.lbs import batch_rodrigues

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    smplx_pt = Path(smplx_pt); out_pt = Path(out_pt)
    if smplx_model_dir is None:
        smplx_model_dir = _resolve_smplx_model_dir()

    # 1) MANO 腕目标 (世界系) + 手指姿态
    pos_r, Rm_r, pos_l, Rm_l, has_left = _mano_wrist_world(params, mano_root)

    blob = torch.load(smplx_pt, map_location="cpu")
    sw = blob["smpl_params_world"]
    body_pose0 = sw["body_pose"].float()
    betas = sw["betas"].float()
    global_orient = sw["global_orient"].float()
    transl = sw["transl"].float()

    T = min(body_pose0.shape[0], pos_r.shape[0])
    body_pose0 = body_pose0[:T].to(dev)
    betas = betas[:T].to(dev)
    global_orient = global_orient[:T].to(dev)
    transl = transl[:T].to(dev)
    tgt_r = torch.tensor(pos_r[:T], dtype=torch.float32, device=dev)
    tgt_l = (torch.tensor(pos_l[:T], dtype=torch.float32, device=dev)
             if has_left else None)
    valid_r = torch.isfinite(tgt_r).all(1)
    valid_l = (torch.isfinite(tgt_l).all(1) if has_left
               else torch.zeros(T, dtype=torch.bool, device=dev))

    hp_r = torch.tensor(np.asarray(params["right hand"]["pose_r"], np.float32)[:T], device=dev)
    if has_left:
        pose_l_raw = np.asarray(params["left hand"]["pose_l"], np.float32)[:T]
        pose_l_m = (pose_l_raw.reshape(T, -1, 3) * _MIRROR_LEFT).reshape(T, -1)
        hp_l = torch.tensor(pose_l_m, device=dev)
    else:
        hp_l = torch.zeros(T, 45, device=dev)
    print(f"[ik] frames={T} valid_r={int(valid_r.sum())} "
          f"valid_l={int(valid_l.sum()) if has_left else 0} device={device}")

    bm = smplx.create(str(smplx_model_dir), model_type="smplx", use_pca=False,
                      flat_hand_mean=True, gender="neutral", num_betas=10,
                      ext="npz", batch_size=T).to(dev)

    # 2) 位置 IK: 优化左右臂 collar/shoulder/elbow
    arm_joints = list(_ARM_JOINTS_L) + list(_ARM_JOINTS_R)
    cols_t = torch.tensor(_arm_cols(arm_joints), dtype=torch.long, device=dev)
    body_const = body_pose0.clone()
    arm = body_pose0[:, cols_t].clone().requires_grad_(True)
    arm_init = arm.detach().clone()

    def forward_joints(arm_p):
        bp = body_const.index_copy(1, cols_t, arm_p)
        out = bm(global_orient=global_orient, body_pose=bp, betas=betas, transl=transl)
        return out.joints, bp

    opt = torch.optim.Adam([arm], lr=lr)
    for it in range(iters):
        opt.zero_grad()
        joints, _ = forward_joints(arm)
        loss_pos = joints.new_zeros(())
        if valid_r.any():
            loss_pos = loss_pos + ((joints[:, _WRIST_R] - tgt_r)[valid_r] ** 2).sum(1).mean()
        if has_left and valid_l.any():
            loss_pos = loss_pos + ((joints[:, _WRIST_L] - tgt_l)[valid_l] ** 2).sum(1).mean()
        loss_reg = ((arm - arm_init) ** 2).mean()
        loss_sm = ((arm[1:] - arm[:-1]) ** 2).mean() if T > 1 else arm.new_zeros(())
        (w_pos * loss_pos + w_reg * loss_reg + w_smooth * loss_sm).backward()
        opt.step()
        if it == 0 or (it + 1) % 50 == 0 or it == iters - 1:
            print(f"[ik] iter {it+1:3d}/{iters}  pos={float(loss_pos):.5f}  "
                  f"reg={float(loss_reg):.5f}  smooth={float(loss_sm):.5f}")

    with torch.no_grad():
        joints_f, body_pose_ik = forward_joints(arm)
        err_r = (joints_f[:, _WRIST_R] - tgt_r).norm(dim=1)
        err_r0 = (bm(global_orient=global_orient, body_pose=body_pose0, betas=betas,
                     transl=transl).joints[:, _WRIST_R] - tgt_r).norm(dim=1)
        msg = (f"[ik] 右腕位置误差 init={float(err_r0[valid_r].mean())*100:.2f}cm "
               f"-> ik={float(err_r[valid_r].mean())*100:.2f}cm")
        if has_left and valid_l.any():
            err_l = (joints_f[:, _WRIST_L] - tgt_l).norm(dim=1)
            msg += f" | 左腕 -> ik={float(err_l[valid_l].mean())*100:.2f}cm"
        print(msg)
        body_pose_ik = body_pose_ik.detach()

    # 3) 朝向: 让 SMPLX 腕全局朝向 == MANO 手全局朝向 (实测两者手系一致, 直接 wrist_global := R_mano)
    if align_wrist:
        with torch.no_grad():
            def set_wrist(body_pose, chain, wrist_joint, Rm_world, valid):
                pose_mats = batch_rodrigues(body_pose.reshape(-1, 3)).reshape(T, 21, 3, 3)
                parentR = _chain_global_rot(batch_rodrigues(global_orient), pose_mats, chain[:-1])
                Rm = torch.tensor(Rm_world[:T], dtype=torch.float32, device=dev)
                vmask = valid & torch.isfinite(Rm.reshape(T, -1)).all(1)
                if int(vmask.sum()) < 1:
                    return body_pose
                wrist_local = parentR.transpose(1, 2) @ Rm
                from scipy.spatial.transform import Rotation as Rsc
                aa = torch.tensor(Rsc.from_matrix(wrist_local.cpu().numpy()).as_rotvec(),
                                  dtype=torch.float32, device=dev)
                bp = body_pose.clone()
                s = (wrist_joint - 1) * 3
                bp[vmask, s:s + 3] = aa[vmask]
                return bp

            body_pose_ik = set_wrist(body_pose_ik, _CHAIN_R, _WRIST_R, Rm_r, valid_r)
            if has_left:
                body_pose_ik = set_wrist(body_pose_ik, _CHAIN_L, _WRIST_L, Rm_l, valid_l)

    # 4) 灌入 MANO 手指姿态 -> 完整带手 SMPLX, 存 out_pt
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    out_blob = dict(blob)
    out_blob["smpl_params_world"] = {
        "body_pose": body_pose_ik.cpu(), "betas": betas.cpu(),
        "global_orient": global_orient.cpu(), "transl": transl.cpu(),
        "left_hand_pose": hp_l.cpu(), "right_hand_pose": hp_r.cpu(),
    }
    out_blob["ik_meta"] = {"source": str(smplx_pt), "iters": iters,
                           "align_wrist": bool(align_wrist), "arm_joints": arm_joints,
                           "hands_from_mano": True}
    torch.save(out_blob, out_pt)

    info = {
        "smplx_ik_pt": str(out_pt), "n_frames": int(T),
        "wrist_err_cm_r": float(err_r[valid_r].mean()) * 100 if valid_r.any() else None,
        "align_wrist": bool(align_wrist),
    }
    if save_verts_npz is not None:
        with torch.no_grad():
            out = bm(global_orient=global_orient, body_pose=body_pose_ik, betas=betas,
                     transl=transl, left_hand_pose=hp_l, right_hand_pose=hp_r)
            verts = out.vertices.cpu().numpy().astype(np.float16)
        np.savez_compressed(save_verts_npz, verts=verts, faces=bm.faces.astype(np.int32))
        info["body_verts_npz"] = str(save_verts_npz)
        print(f"[ik] 写 {out_pt.name} (含 left/right_hand_pose) + {Path(save_verts_npz).name}")
    else:
        print(f"[ik] 写 {out_pt.name} (含 left/right_hand_pose)")
    return info


def run(ws: Workspace,
        iters: int = 250, lr: float = 0.05,
        w_pos: float = 1.0, w_reg: float = 2e-3, w_smooth: float = 1e-2,
        align_wrist: bool = True, render_overlay: bool = True,
        fps: int = 10, render_stride: int = 1,
        device: Optional[str] = None) -> Dict[str, Any]:
    state = PipelineState.load(ws)
    if state.steps["infer"].status != "done":
        raise RuntimeError("Step 5 (infer) 没完成")

    npy_path = Path(state.steps["infer"].outputs["npy_path"])
    capture_id = state.steps["undistort"].outputs.get("capture_id", ws.seq_name)
    world_dir = ws.undist_root / capture_id / "world_params"
    smplx_pt = world_dir / "smplx_params_world.pt"
    if not smplx_pt.exists():
        raise FileNotFoundError(
            f"找不到人体参数 {smplx_pt} (需先用 multi_view_smpl_optimizer 产出 world_params/)")

    data = np.load(npy_path, allow_pickle=True).item()
    seq_key = list(data["data_dict"].keys())[0]
    params = data["data_dict"][seq_key]["params"]

    world_dir.mkdir(parents=True, exist_ok=True)
    info = solve_body_ik(
        params, smplx_pt, world_dir / "smplx_params_world_ik.pt",
        smplx_model_dir=_resolve_smplx_model_dir(), mano_root=str(_PROJECT),
        iters=iters, lr=lr, w_pos=w_pos, w_reg=w_reg, w_smooth=w_smooth,
        align_wrist=align_wrist, device=device,
        save_verts_npz=world_dir / "body_verts_world.npz")

    if render_overlay:
        try:
            import importlib.util
            bv_path = _PROJECT / "utils" / "body_vis.py"
            spec = importlib.util.spec_from_file_location("_pipeline_body_vis", bv_path)
            bv = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bv)
            ws.vis_dir.mkdir(parents=True, exist_ok=True)
            body_out = ws.vis_dir / f"{ws.seq_name}_body.mp4"
            body_videos = bv.render(
                npy_path=str(npy_path), body_verts_npz=info["body_verts_npz"],
                out_video=str(body_out), mano_root=str(_PROJECT),
                obj_path=None, fps=int(fps), show_hands=False,
                show_object="object" in params, stride=int(render_stride))
            info["body_view_videos"] = [str(v) for v in body_videos]
            print(f"[ik] 人体 overlay 完成: {len(body_videos)} 视角")
        except Exception as e:
            print(f"[ik] 人体 overlay 渲染失败 (IK 结果已保存): {e}")
            info["body_view_videos"] = []

    state.mark_done("ik", **info)
    state.save(ws)
    return info
