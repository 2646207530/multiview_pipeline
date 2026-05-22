"""把指定 npy 里指定一帧的手+物体, 变换到"物体在原点+单位旋转"的相对坐标系,
另存为一个新的单帧 npy (schema 跟输入完全一致, way_vis / 下游可直接读).

输出 npy 的关键字段:
  data_dict[seq].params.object.obj_rot   = zeros(3)
  data_dict[seq].params.object.obj_trans = zeros(3)
  data_dict[seq].params.right/left hand.rot/trans  按 P_new = obj_R^T (P_old - obj_t) 同步
  data_dict[seq].params.right/left hand.pose/shape 不变
  data_dict[seq].params.camera.world2cam = w2c_old @ obj_pose  (cam 跟着拉过来,
      让原始 2D 投影位置在新坐标系下仍然成立)
  imgnames                                 只留这一帧
  imgpath, K, views                        原样保留

Usage:
python utils/extract_relative_pose_frame.py \
    --npy /data2/fubingshuai/golf/golf-hand-object/out/35_wood_8_01_fbs_opt_force_closure.npy \
    --frame 19 \
    --out /data2/fubingshuai/golf/test/35_wood_8_01_baseball.npy

"""

import argparse
from pathlib import Path

import numpy as np
import torch
import smplx
from scipy.spatial.transform import Rotation as RR

# 跟 way_vis.py / multiview_hand_init.py 保持一致
MIRROR_LEFT_PARAMS = np.array([1.0, -1.0, -1.0], dtype=np.float32)

# smplx MANO 在哪 (找 MANO_RIGHT.pkl / MANO_LEFT.pkl)
_DEFAULT_MANO_ROOT = str(Path(__file__).resolve().parent.parent)


def _canonical_wrist_pos(shape: np.ndarray, is_left: bool, mano_root: str):
    """返回这个 betas 下手在 zero pose 时 wrist (joint 0) 的 canonical 位置 (3,).
    用于 smplx LBS 的'绕 wrist 旋转'修正项 (obj_R^T - I) @ J0."""
    layer = smplx.create(mano_root, 'MANO', use_pca=False,
                         is_rhand=(not is_left), flat_hand_mean=True)
    with torch.no_grad():
        out = layer(
            global_orient=torch.zeros(1, 3),
            hand_pose=torch.zeros(1, 45),
            betas=torch.from_numpy(np.asarray(shape, dtype=np.float32)).reshape(1, 10),
        )
    return out.joints[0, 0].cpu().numpy().astype(np.float32)


def _transform_hand(rot, pose, shape, trans, obj_R, obj_t, is_left: bool, mano_root: str):
    """让手在 (new world = object canonical) 系下保持原本的手-物相对位姿.

    smplx MANO 的 LBS 渲染右手: V_world = R(rot) @ (V_canonical - J0) + J0 + trans
                          左手: V_world = R(MIRROR·rot) @ (V_canonical_L - J0_L) + J0_L + trans
    (J0 = canonical wrist 位置, ≈ ±(0.096, 0.006, 0.006), 受 shape 影响)

    要 V_obj = obj_R^T (V_world - obj_t), 解出新参数:
      rot_new:   R(rot_new) = obj_R^T @ R(rot_old)          (左手在 MIRROR 框架下推)
      trans_new: (obj_R^T - I) @ J0 + obj_R^T @ (trans - obj_t)
    pose / shape 跟外部位姿无关, 不动.
    """
    rot   = np.asarray(rot,   dtype=np.float32).reshape(3)
    trans = np.asarray(trans, dtype=np.float32).reshape(3)
    shape = np.asarray(shape, dtype=np.float32).reshape(10)

    if is_left:
        rot_lf  = MIRROR_LEFT_PARAMS * rot
        R_world = RR.from_rotvec(rot_lf).as_matrix()
        R_new   = obj_R.T @ R_world
        rot_lf_new = RR.from_matrix(R_new).as_rotvec()
        rot_new = (MIRROR_LEFT_PARAMS * rot_lf_new).astype(np.float32)
    else:
        R_world = RR.from_rotvec(rot).as_matrix()
        R_new   = obj_R.T @ R_world
        rot_new = RR.from_matrix(R_new).as_rotvec().astype(np.float32)

    J0 = _canonical_wrist_pos(shape, is_left, mano_root)
    correction = (obj_R.T - np.eye(3, dtype=np.float32)) @ J0
    trans_new = (correction + obj_R.T @ (trans - obj_t)).astype(np.float32)
    return rot_new, trans_new


def _safe_get(arr, frame, dtype=np.float32):
    a = np.asarray(arr, dtype=dtype)
    if frame < 0 or frame >= a.shape[0]:
        raise IndexError(f"frame {frame} 超出范围 [0, {a.shape[0]})")
    return a[frame].copy()


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                 description=__doc__)
    ap.add_argument('--npy',   required=True, type=str, help='输入 npy 路径')
    ap.add_argument('--frame', required=True, type=int, help='要快照的帧号 (0-based)')
    ap.add_argument('--out',   required=True, type=str, help='输出 npy 路径')
    ap.add_argument('--seq',   default=None,  type=str,
                    help='data_dict 里的 seq 名 (默认取第一个)')
    ap.add_argument('--mano_root', default=_DEFAULT_MANO_ROOT, type=str,
                    help='含 MANO_*.pkl 的目录 (用于查 canonical wrist 位置)')
    args = ap.parse_args()

    npy_path = Path(args.npy)
    out_path = Path(args.out)
    if not npy_path.exists():
        raise SystemExit(f"npy 不存在: {npy_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(str(npy_path), allow_pickle=True).item()
    dd = d.get('data_dict', {})
    if not dd:
        raise SystemExit(f"data_dict 字段为空: {npy_path}")
    seq = args.seq if args.seq is not None else next(iter(dd.keys()))
    if seq not in dd:
        raise SystemExit(f"找不到 seq={seq}, 可选: {list(dd.keys())}")
    params = dd[seq]['params']
    f = args.frame

    # ─── 物体世界位姿 ────────────────────────────────────────────────
    obj_rot_old   = _safe_get(params['object']['obj_rot'],   f)
    obj_trans_old = _safe_get(params['object']['obj_trans'], f)
    if np.isnan(obj_rot_old).any() or np.isnan(obj_trans_old).any():
        raise SystemExit(f"frame {f} 的 object 位姿含 NaN, 无法构造相对位姿")
    obj_R = RR.from_rotvec(obj_rot_old).as_matrix().astype(np.float32)
    obj_t = obj_trans_old.astype(np.float32)

    # ─── 手参数变换 ─────────────────────────────────────────────────
    rh, lh = params['right hand'], params['left hand']

    r_rot_old   = _safe_get(rh['rot_r'],   f)
    r_pose_old  = _safe_get(rh['pose_r'],  f)
    r_shape_old = _safe_get(rh['shape_r'], f)
    r_trans_old = _safe_get(rh['trans_r'], f)
    if np.isnan(r_rot_old).any() or np.isnan(r_trans_old).any():
        print(f"[warn] right hand frame {f} 有 NaN, 这只手 rot/trans 保持原值不变换")
        r_rot_new, r_trans_new = r_rot_old, r_trans_old
    else:
        r_rot_new, r_trans_new = _transform_hand(
            r_rot_old, r_pose_old, r_shape_old, r_trans_old,
            obj_R, obj_t, is_left=False, mano_root=args.mano_root)

    l_rot_old   = _safe_get(lh['rot_l'],   f)
    l_pose_old  = _safe_get(lh['pose_l'],  f)
    l_shape_old = _safe_get(lh['shape_l'], f)
    l_trans_old = _safe_get(lh['trans_l'], f)
    if np.isnan(l_rot_old).any() or np.isnan(l_trans_old).any():
        print(f"[warn] left hand frame {f} 有 NaN, 这只手 rot/trans 保持原值不变换")
        l_rot_new, l_trans_new = l_rot_old, l_trans_old
    else:
        l_rot_new, l_trans_new = _transform_hand(
            l_rot_old, l_pose_old, l_shape_old, l_trans_old,
            obj_R, obj_t, is_left=True, mano_root=args.mano_root)

    # ─── 相机外参变换 ───────────────────────────────────────────────
    # 物体齐次位姿矩阵 (old world ← new world): P_old = obj_pose @ P_new
    obj_pose = np.eye(4, dtype=np.float32)
    obj_pose[:3, :3] = obj_R
    obj_pose[:3, 3]  = obj_t

    # ⚠️ way_vis 的 _resolve_all_cameras 用 `isinstance(K_field, list)` 区分新旧格式,
    #    所以 K / world2cam / views 必须以 Python list 保存 (跟 _build_camera_block 一致).
    cam = params['camera']
    K_old      = cam['K']
    W2C_old    = cam['world2cam']
    views_old  = cam['views']
    if isinstance(K_old, (list, tuple)):
        K_list = [np.asarray(k, dtype=np.float32).copy() for k in K_old]
    else:
        K_arr = np.asarray(K_old, dtype=np.float32)
        K_list = [K_arr.copy()] if K_arr.ndim == 2 else [K_arr[i].copy() for i in range(K_arr.shape[0])]

    if isinstance(W2C_old, (list, tuple)):
        W2C_list_old = [np.asarray(w, dtype=np.float32).copy() for w in W2C_old]
    else:
        W2C_arr = np.asarray(W2C_old, dtype=np.float32)
        W2C_list_old = [W2C_arr.copy()] if W2C_arr.ndim == 2 else [W2C_arr[i].copy() for i in range(W2C_arr.shape[0])]
    W2C_list_new = [(w @ obj_pose).astype(np.float32) for w in W2C_list_old]

    if isinstance(views_old, (list, tuple)):
        views_list = list(views_old)
    else:
        views_list = list(np.asarray(views_old).reshape(-1))

    # ─── 组装单帧 params ────────────────────────────────────────────
    new_params = {
        'right hand': {
            'rot_r':   r_rot_new[None].astype(np.float32),
            'pose_r':  r_pose_old[None].astype(np.float32),
            'trans_r': r_trans_new[None].astype(np.float32),
            'shape_r': r_shape_old[None].astype(np.float32),
        },
        'left hand': {
            'rot_l':   l_rot_new[None].astype(np.float32),
            'pose_l':  l_pose_old[None].astype(np.float32),
            'trans_l': l_trans_new[None].astype(np.float32),
            'shape_l': l_shape_old[None].astype(np.float32),
        },
        'object': {
            'obj_rot':   np.zeros((1, 3), dtype=np.float32),
            'obj_trans': np.zeros((1, 3), dtype=np.float32),
        },
        'camera': {
            'world2cam': W2C_list_new,
            'K':         K_list,
            'views':     views_list,
        },
    }

    # ─── imgnames / imgpath ────────────────────────────────────────
    imgnames_old = d.get('imgnames', None)
    if isinstance(imgnames_old, (list, tuple)):
        new_imgnames = [imgnames_old[f]] if f < len(imgnames_old) else []
    elif isinstance(imgnames_old, np.ndarray):
        new_imgnames = imgnames_old[f:f+1].copy()
    else:
        new_imgnames = imgnames_old  # None 或别的类型, 原样传

    new_d = {
        'imgnames': new_imgnames,
        'imgpath':  d.get('imgpath', None),
        'data_dict': {
            seq: {'params': new_params},
        },
    }

    np.save(str(out_path), new_d, allow_pickle=True)

    # ─── 打印摘要 ────────────────────────────────────────────────────
    print(f"[snapshot] {npy_path.name}  seq={seq}  frame={f} → {out_path}")
    print(f"  object:   rot=0, trans=0   (原 rot={obj_rot_old}, trans={obj_trans_old})")
    print(f"  right hand:")
    print(f"    rot   {r_rot_old}  →  {r_rot_new}")
    print(f"    trans {r_trans_old}  →  {r_trans_new}")
    print(f"  left hand:")
    print(f"    rot   {l_rot_old}  →  {l_rot_new}")
    print(f"    trans {l_trans_old}  →  {l_trans_new}")
    print(f"  camera: {len(W2C_list_new)} 视角, K/world2cam/views 都按 list 存 (way_vis 兼容)")


if __name__ == '__main__':
    main()
