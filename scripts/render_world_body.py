"""把一个 world 系 SMPLX 参数 .pt 渲染成 body overlay (用于 step3 人体姿态可视化).

读 capture 的 undistort state 拿相机块, 前向 SMPLX 出顶点, 用 body_vis 叠到去畸变图上.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import smplx

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))
from pipeline.workspace import Workspace          # noqa: E402
from pipeline.state import PipelineState          # noqa: E402
from utils.camera_npy import (                    # noqa: E402
    _load_camera_params, _resolve_color_cams, _build_camera_block)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture_dir", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--smplx_pt", required=True)
    ap.add_argument("--out_video", required=True)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--crf", type=int, default=28)
    a = ap.parse_args()

    cap = Path(a.capture_dir)
    ws = Workspace(capture_dir=cap, seq_name=a.seq)
    und = PipelineState.load(ws).steps["undistort"].outputs
    cam_names = und["cam_names"]
    capture_id = und["capture_id"]

    cams = _load_camera_params(cap)
    hn, on = _resolve_color_cams(cams, cap)
    newK = {n: np.array(und["newKs"][n], np.float32) for n in cam_names}
    camblock = _build_camera_block(cams, hn, on, new_K_per_cam=newK)

    smplx_dir = _PROJECT / "assets" / "body_models"
    for c in [smplx_dir, _PROJECT / "pipeline" / "assets" / "body_models",
              _PROJECT.parent / "pipeline" / "assets" / "body_models"]:
        if (c / "smplx" / "SMPLX_NEUTRAL.npz").exists():
            smplx_dir = c
            break

    sw = torch.load(a.smplx_pt, map_location="cpu")["smpl_params_world"]
    T = sw["body_pose"].shape[0]
    bm = smplx.create(str(smplx_dir), model_type="smplx", use_pca=False, flat_hand_mean=True,
                      gender="neutral", num_betas=10, ext="npz", batch_size=T)
    kw = dict(global_orient=sw["global_orient"].float(), body_pose=sw["body_pose"].float(),
              betas=sw["betas"].float(), transl=sw["transl"].float())
    if "right_hand_pose" in sw:
        kw["right_hand_pose"] = sw["right_hand_pose"].float()
        kw["left_hand_pose"] = sw["left_hand_pose"].float()
    with torch.no_grad():
        verts = bm(**kw).vertices.numpy().astype(np.float16)
    npz = Path(a.out_video).with_suffix(".npz")
    np.savez_compressed(npz, verts=verts, faces=bm.faces.astype(np.int32))

    undist_root = Path(und["undist_root"])
    imgdir = undist_root / capture_id / "0" / "images_undistorted"
    scene = {"imgnames": [f"{i:06d}.jpg" for i in range(T)], "imgpath": str(imgdir),
             "data_dict": {a.seq: {"params": {"camera": camblock}}}}
    scene_npy = Path(a.out_video).with_suffix(".scene.npy")
    np.save(scene_npy, scene, allow_pickle=True)

    import importlib.util
    spec = importlib.util.spec_from_file_location("_bv", _PROJECT / "utils" / "body_vis.py")
    bv = importlib.util.module_from_spec(spec); spec.loader.exec_module(bv)
    vids = bv.render(npy_path=str(scene_npy), body_verts_npz=str(npz), out_video=a.out_video,
                     mano_root=str(_PROJECT), fps=10, show_hands=False, show_object=False,
                     stride=a.stride, crf=a.crf)
    print("WORLD_BODY_VIZ", vids)


if __name__ == "__main__":
    main()
