"""Step 5: 多视角推理 + 组装最终 npy.

2 个子步:
  1. ``run_hand_estimation_subprocess`` 跑 visualize_mano.py, 出
     ``_he_output/<seq>_mano.json`` + 每帧 jpg + hand0/hand1 mp4.
     (如果 step4 finetune 跑过, 用 finetune 后的 ckpt; 否则用默认 checkpoint_30)
  2. ``parse_mano_json_to_arrays`` 把 JSON 转 8 个数组, 跟 object 位姿 + 相机
     参数一起组装成 npy.

(SportGS 优化不在 v1 范围.)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from multiview_hand_init import (  # type: ignore
    run_hand_estimation_subprocess,
    parse_mano_json_to_arrays,
)
from run_golf_capture_to_npy import (  # type: ignore
    _load_camera_params,
    _resolve_color_cams,
    _load_trajectory,
    _object_poses_to_world,
    _build_camera_block,
)
from run_hamer_to_npy import _slerp_interpolate_nan, _interpolate_nan  # type: ignore

from .state import PipelineState
from .workspace import Workspace


def run(ws: Workspace, gpu_id: str = "0",
        ckpt_override: Optional[str] = None) -> Dict[str, Any]:
    state = PipelineState.load(ws)
    if state.steps["pseudo"].status != "done":
        raise RuntimeError("Step 3 (pseudo) 没完成")

    undist = state.steps["undistort"].outputs
    undist_root = Path(undist["undist_root"])
    capture_id = undist["capture_id"]
    cam_names = undist["cam_names"]

    # ckpt 选择: 用户传了 ckpt_override 优先; 否则 finetune 跑过用 ft ckpt;
    # 否则 None → run_hand_estimation_subprocess 会用 exp/new/checkpoints/checkpoint_30.
    if ckpt_override:
        ckpt_path = Path(ckpt_override)
        if not ckpt_path.is_dir():
            raise FileNotFoundError(f"指定的 ckpt 目录不存在: {ckpt_path}")
        print(f"[step5] 用用户指定 ckpt: {ckpt_path}")
    else:
        ft_step = state.steps.get("finetune")
        ckpt_path = None
        if ft_step and ft_step.status == "done":
            ckpt_path = Path(ft_step.outputs["finetuned_ckpt"])
            print(f"[step5] 默认: finetune 后的 ckpt: {ckpt_path}")
        else:
            print(f"[step5] 默认: exp/new/checkpoints/checkpoint_30")

    # ── 1) 跑 HE 子进程, 出 mano.json + hand0/hand1.mp4 ────────────────
    he_output = ws.he_output_dir
    he_output.mkdir(parents=True, exist_ok=True)
    # 清旧 *_mano.json 避免拿到上次结果
    for p in he_output.glob("*_mano.json"):
        p.unlink()

    json_path = run_hand_estimation_subprocess(
        project_root=_PROJECT,
        undist_root=undist_root,
        output_dir=he_output,
        gpu_id=gpu_id,
        ckpt_override=ckpt_path,
    )
    print(f"[step5] HE 输出 JSON: {json_path}")

    # ── 2) 加载相机 (+ 可选物体轨迹) + 组装 npy ────────────────────────
    cap = ws.capture_dir
    cams = _load_camera_params(cap)
    hamer_name, other_name = _resolve_color_cams(cams, cap)

    # n_frames_per_cam 是 {"0": N0, "1": N1} (str key, 兼容 gradio/orjson 序列化)
    npc = undist["n_frames_per_cam"]
    if isinstance(npc, dict):
        total_frames_img = min(int(v) for v in npc.values())
    else:
        total_frames_img = int(npc[0])

    # 物体轨迹是可选的 (没有 csv 就只存手, 不带 object 字段)
    traj_csv = cap / "trajectory_output" / "trajectory.csv"
    obj_rot_arr = None
    obj_trans_arr = None
    if traj_csv.exists():
        ref_cam_name, poses = _load_trajectory(traj_csv)
        if ref_cam_name not in cams:
            raise RuntimeError(f"reference_camera {ref_cam_name} 不在 camera_params.json 中")
        obj_rot_arr, obj_trans_arr = _object_poses_to_world(
            poses, cams, ref_cam_name, hamer_name)
        total_frames = min(obj_rot_arr.shape[0], total_frames_img)
        obj_rot_arr   = obj_rot_arr[:total_frames]
        obj_trans_arr = obj_trans_arr[:total_frames]
    else:
        print(f"[step5] 未找到 {traj_csv}, 跳过物体轨迹 (只输出手)")
        total_frames = total_frames_img

    # parse JSON -> 8 个数组 (按 total_frames 长度, 缺帧 NaN)
    (r_rot, r_pose, r_shape, r_trans,
     l_rot, l_pose, l_shape, l_trans) = parse_mano_json_to_arrays(
        json_path, total_frames, mano_root=str(_PROJECT))

    # NaN 插值
    interp_n = 0
    interp_n += _slerp_interpolate_nan(r_rot)
    interp_n += _slerp_interpolate_nan(l_rot)
    for arr in [r_pose, r_shape, r_trans, l_pose, l_shape, l_trans]:
        interp_n += _interpolate_nan(arr)
    print(f"[step5] 插值填补 {interp_n} 段 NaN")

    # 用 step1 拿到的 newK 重建 camera block
    new_K_per_cam = {n: np.array(undist["newKs"][n], dtype=np.float32)
                     for n in cam_names}
    camera = _build_camera_block(cams, hamer_name, other_name,
                                  new_K_per_cam=new_K_per_cam)

    # 组装 npy
    hamer_undist_dir = undist_root / capture_id / "0" / "images_undistorted"
    imgnames = [f"{i:06d}.jpg" for i in range(total_frames)]
    params = {
        "right hand": {"rot_r": r_rot, "pose_r": r_pose,
                       "trans_r": r_trans, "shape_r": r_shape},
        "left hand":  {"rot_l": l_rot, "pose_l": l_pose,
                       "trans_l": l_trans, "shape_l": l_shape},
        "camera":     camera,
    }
    if obj_rot_arr is not None:
        params["object"] = {"obj_rot": obj_rot_arr, "obj_trans": obj_trans_arr}
    root = {
        "imgnames":  imgnames,
        "imgpath":   str(hamer_undist_dir),
        "data_dict": {ws.seq_name: {"params": params}},
    }
    np.save(ws.output_npy, root, allow_pickle=True)
    print(f"[step5] 写 npy: {ws.output_npy}")

    info = {
        "mano_json":   str(json_path),
        "npy_path":    str(ws.output_npy),
        "hand0_mp4":   str(ws.he_hand_mp4(0)) if ws.he_hand_mp4(0).exists() else None,
        "hand1_mp4":   str(ws.he_hand_mp4(1)) if ws.he_hand_mp4(1).exists() else None,
        "total_frames": total_frames,
        "interpolated_nan_segments": interp_n,
    }
    state.mark_done("infer", **info)
    state.save(ws)
    return info
