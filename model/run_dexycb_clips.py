"""
DexYCB clips: 手部 (HaMER) + 物体 (FoundationPose) 联合参数提取 (单环境版)

对 /data2/fubingshuai/golf/data/dexycb_clips 下的每个序列:
  1. 用 HaMER 识别左右手的 MANO 参数
  2. 用 FoundationPose 追踪物体的 6D 位姿
  3. 聚合成一份 npy, 输出到 /data2/fubingshuai/golf/output/<seq>.npy
所有步骤都在同一个 foundationpose_ros conda 环境中执行,
不再通过 subprocess 切换环境。

数据集目录约定 (DexYCB clips 版式):
  dexycb_clips/
    images/<seq>/color_XXXXXX.jpg              彩色帧
    images/<seq>/aligned_depth_to_color_XXXXXX.png  uint16 mm 深度
    images/<seq>/labels_XXXXXX.npz             GT (本脚本不直接使用)
    masks/<seq>/color_XXXXXX.npy               (H,W) uint8, 0=bg, 1=hand, 2=object
    masks/<seq>/color_XXXXXX.jpg               mask 可视化 (本脚本不使用)
    meta/cam_1.yml / cam_2.yml                 相机内参
    meta/extrinsics.yml
    models/<prefix>_<name>/textured_simple.obj 物体网格 (米)

序列命名约定: "<obj_prefix>-<cam_id>"，如 "12-1" 表示 object 12 + cam_1。

用法 (在 foundationpose_ros 环境中直接运行):
  python model/run_dexycb_clips.py --all
  python model/run_dexycb_clips.py --seqs 12-1 14-1
  python model/run_dexycb_clips.py --all --skip_foundationpose     # 只做手部
  python model/run_dexycb_clips.py --all --skip_hamer              # 只做物体
  python model/run_dexycb_clips.py --all --render                  # 顺带渲染双视角

识别流程约定:
  - HaMER + FoundationPose 仅使用 cam_1 的彩色图/深度/mask，位姿估计天然
    表达在 cam_1 坐标系;
  - 随后用 extrinsics.yml 里 E_i = cam_i → world 把手/物体 pose 变到
    dexycb master 世界系,并在 npy 中存为世界系下的数值。

输出 npy 的字段约定与 /home/pt/fbs/data_fixed.npy 完全一致:
  root = {
    "imgnames": [...],
    "imgpath":  "<sequence color images abspath>",
    "data_dict": {"<seq>": {"params": {
        "right hand": {"rot_r","pose_r","trans_r","shape_r"},
        "left hand":  {"rot_l","pose_l","trans_l","shape_l"},
        "object":     {"obj_rot","obj_trans"},
        "camera":     {"world2cam":[4x4,4x4], "K":[3x3,3x3], "views":["cam0","cam1"]},
    }}}
  }
  其中 world2cam[0]=inv(E_1), world2cam[1]=inv(E_2),
      views[0]对应 cam_1 的实拍, views[1]对应 cam_2 的实拍。

渲染时 view0 反投影回 images/<prefix>-1/, view1 反投影回 images/<prefix>-2/。
"""

import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"OpenGL\.")

import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'model'))
sys.path.insert(0, os.path.join(_ROOT, 'model', 'hamer'))

import argparse
import glob
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R


# ══════════════════════════════════════════════════════════════════════
# 默认路径
# ══════════════════════════════════════════════════════════════════════

DEFAULT_DATASET_ROOT = "/data2/fubingshuai/golf/data/dexycb_clips"
DEFAULT_OUTPUT_DIR = "/data2/fubingshuai/golf/output"
DEFAULT_RENDER_DIR = "/data2/fubingshuai/golf/test"
DEFAULT_FP_PYTHON = "/data2/fubingshuai/miniconda3/envs/foundationpose_ros/bin/python"
FP_PKG_DIR = "/data2/fubingshuai/golf/FoundationPoseROS2/FoundationPose"

# 渲染双视角时使用的 MANO 模型所在目录 (smplx.create 会在此目录下寻找 MANO/MANO_RIGHT.pkl 等)
DEFAULT_MANO_MODEL_DIR = os.environ.get(
    "MANO_MODEL_DIR", "/data2/fubingshuai/golf/golf-hand-object"
)

# mask 像素取值约定 (dexycb_clips)
HAND_MASK_VALUE = 1
OBJ_MASK_VALUE = 2

# 深度编码
DEPTH_SCALE_MM_TO_M = 1.0 / 1000.0  # PNG 中 uint16 mm 除以 1000 得到米

# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════

def _safe_load_yaml(path):
    """DexYCB YAML 使用了 !!python/tuple 标签，用 UnsafeLoader 容忍。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.unsafe_load(f)


def load_cam_K(meta_dir, cam_id):
    """从 dexycb_clips/meta/cam_<id>.yml 里读取彩色相机内参 (3x3)。"""
    cam_yaml = os.path.join(meta_dir, f"cam_{cam_id}.yml")
    if not os.path.isfile(cam_yaml):
        raise FileNotFoundError(f"未找到相机标定文件: {cam_yaml}")
    data = _safe_load_yaml(cam_yaml)
    color = data["color"]
    K = np.array([
        [color["fx"], 0.0,          color["ppx"]],
        [0.0,          color["fy"], color["ppy"]],
        [0.0,          0.0,          1.0],
    ], dtype=np.float32)
    return K


def parse_seq_name(seq):
    """序列名 '12-1' → (obj_prefix='12', cam_id='1')。"""
    if "-" not in seq:
        raise ValueError(f"无法解析序列名 '{seq}'：期望 <obj_prefix>-<cam_id> 形式")
    obj_prefix, cam_id = seq.rsplit("-", 1)
    return obj_prefix, cam_id


def find_object_mesh(dataset_root, obj_prefix):
    """在 dexycb_clips/models/<prefix>_<name>/ 下找 textured_simple.obj。"""
    models_root = os.path.join(dataset_root, "models")
    candidates = sorted(glob.glob(os.path.join(models_root, f"{obj_prefix}_*")))
    if not candidates:
        raise FileNotFoundError(
            f"未找到 prefix='{obj_prefix}' 对应的物体模型目录: {models_root}"
        )
    model_dir = candidates[0]
    mesh = os.path.join(model_dir, "textured_simple.obj")
    if not os.path.isfile(mesh):
        raise FileNotFoundError(f"缺少网格文件: {mesh}")
    return mesh, os.path.basename(model_dir)


def list_color_frames(images_dir):
    """返回 [(frame_id_int, color_path), ...]，按 frame_id 升序。"""
    paths = sorted(glob.glob(os.path.join(images_dir, "color_*.jpg")))
    if not paths:
        raise FileNotFoundError(f"{images_dir} 中未找到 color_*.jpg")
    out = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]  # color_000021
        idx = int(stem.split("_")[-1])
        out.append((idx, p))
    return out


def _load_cam2world_all(meta_dir):
    """
    读取 dexycb_clips/meta/extrinsics.yml，把每个相机 12 维的 3x4
    (cam→世界/master) 外参还原成 4x4 齐次矩阵。
    返回 {cam_id_str: (4,4) ndarray}。
    """
    path = os.path.join(meta_dir, "extrinsics.yml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少外参: {path}")
    data = _safe_load_yaml(path)
    raw = data["extrinsics"]
    out = {}
    for cam_id, values in raw.items():
        if cam_id == "apriltag":
            continue
        arr = np.asarray(values, dtype=np.float64).reshape(3, 4)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = arr[:, :3]
        T[:3,  3] = arr[:,  3]
        out[str(cam_id)] = T
    return out


def compute_dual_view_cameras(dataset_root, seq):
    """
    为一个序列生成 data_fixed.npy 风格的相机包 (2 视角, 世界系 = dexycb master):
      views     = ['cam0', 'cam1']                    # cam0=序列自身, cam1=另一相机
      K         = [K_self, K_other]                   # 3x3
      world2cam = [inv(E_self), inv(E_other)]          # 4x4, world→cam_i

    其中 E_i (来自 extrinsics.yml) 是 cam_i → world 的 4x4 齐次变换。
    另外返回原始的 E_self / E_other 供上游把 cam_self 系下的位姿变到世界系。
    """
    _, self_cam_id = parse_seq_name(seq)
    meta_dir = os.path.join(dataset_root, "meta")

    cam_ids_known = ("1", "2")
    if self_cam_id not in cam_ids_known:
        raise ValueError(f"未知相机编号: cam_{self_cam_id} (仅支持 {cam_ids_known})")
    other_cam_id = "2" if self_cam_id == "1" else "1"

    K_self  = load_cam_K(meta_dir, self_cam_id).astype(np.float32)
    K_other = load_cam_K(meta_dir, other_cam_id).astype(np.float32)

    cam2world_all = _load_cam2world_all(meta_dir)
    if self_cam_id not in cam2world_all or other_cam_id not in cam2world_all:
        raise RuntimeError(
            f"extrinsics.yml 缺少相机 {self_cam_id}/{other_cam_id} 的外参"
        )
    E_self  = cam2world_all[self_cam_id]   # cam_self  → world
    E_other = cam2world_all[other_cam_id]  # cam_other → world

    world2cam = [
        np.linalg.inv(E_self).astype(np.float32),
        np.linalg.inv(E_other).astype(np.float32),
    ]
    K = [K_self, K_other]
    views = ["cam0", "cam1"]
    return {
        "world2cam": world2cam,
        "K": K,
        "views": views,
        "self_cam_id": self_cam_id,
        "other_cam_id": other_cam_id,
        "E_self": E_self.astype(np.float64),
        "E_other": E_other.astype(np.float64),
    }


def other_seq_name(seq):
    """'12-1' → '12-2'，'12-2' → '12-1'。"""
    prefix, cam = parse_seq_name(seq)
    other = "2" if cam == "1" else "1"
    return f"{prefix}-{other}"


# ══════════════════════════════════════════════════════════════════════
# 位姿 cam_self → world 变换 (写 npy 前 / 旧格式迁移用)
# ══════════════════════════════════════════════════════════════════════

_MIRROR_LEFT_VEC = np.array([1.0, -1.0, -1.0], dtype=np.float32)


def _is_identity_4x4(M, atol=1e-6):
    M = np.asarray(M)
    return M.shape == (4, 4) and np.allclose(M, np.eye(4), atol=atol)


def _transform_object_poses(obj_rot, obj_trans, E):
    """
    obj_rot: (N,3) axis-angle, obj_trans: (N,3) (都在 cam_self 系)
    E:      (4,4) cam_self → world
    返回 world 系下的 (rot, trans)。
    """
    R_E = np.asarray(E)[:3, :3]
    t_E = np.asarray(E)[:3, 3]
    R_old = R.from_rotvec(np.asarray(obj_rot, dtype=np.float64)).as_matrix()  # (N,3,3)
    R_new = np.einsum("ij,njk->nik", R_E, R_old)
    rot_new = R.from_matrix(R_new).as_rotvec()
    trans_new = np.asarray(obj_trans, dtype=np.float64) @ R_E.T + t_E
    return rot_new.astype(np.float32), trans_new.astype(np.float32)


def _mano_root_joint(shape, is_rhand, mano_model_dir):
    """
    每帧 betas 下 MANO 腕关节(第 0 号 joint) 的 rest 位置 (N,3)。
    用于把 MANO transl 从 cam_self 变到 world 时做 root-offset 校正。
    """
    import smplx
    import torch
    shape = np.asarray(shape, dtype=np.float32)
    N = shape.shape[0]
    cpu = torch.device("cpu")
    mano = smplx.create(
        mano_model_dir, "MANO", use_pca=False,
        is_rhand=is_rhand, flat_hand_mean=True,
    ).to(cpu)
    with torch.no_grad():
        out = mano(
            global_orient=torch.zeros(N, 3, dtype=torch.float32, device=cpu),
            hand_pose=torch.zeros(N, 45, dtype=torch.float32, device=cpu),
            betas=torch.tensor(shape, dtype=torch.float32, device=cpu),
            transl=torch.zeros(N, 3, dtype=torch.float32, device=cpu),
        )
    return out.joints[:, 0].detach().cpu().numpy().astype(np.float64)  # (N,3)


def _transform_mano_params(rot_stored, trans_stored, shape, E,
                            is_rhand, mano_model_dir):
    """
    把 npy 中存储的 (rot, trans) (cam_self 系) 变到 world 系。

    MANO 前向:  verts = R_g @ (V_rest(beta,pose) - J_root) + J_root + transl
    world 变换: verts_world = R_E @ verts_cam + t_E
    => R_g'    = R_E @ R_g
       transl' = R_E @ (transl + J_root) + t_E - J_root

    左手 npy 的 rot_l 存的是"右手约定镜像过的" axis-angle, 送入
    mano_l.forward 之前还要再 mirror 一次; 所以这里先 mirror 还原
    出"真正喂进 MANO 的 R_g", 变换之后再 mirror 回存储形式。
    """
    rot_stored = np.asarray(rot_stored, dtype=np.float32)
    trans_stored = np.asarray(trans_stored, dtype=np.float32)
    shape = np.asarray(shape, dtype=np.float32)

    R_E = np.asarray(E)[:3, :3].astype(np.float64)
    t_E = np.asarray(E)[:3, 3].astype(np.float64)

    mirror = _MIRROR_LEFT_VEC if not is_rhand else np.ones(3, dtype=np.float32)
    rot_eff = (rot_stored * mirror).astype(np.float64)
    R_old = R.from_rotvec(rot_eff).as_matrix()
    R_new = np.einsum("ij,njk->nik", R_E, R_old)
    rot_new_eff = R.from_matrix(R_new).as_rotvec()
    rot_new_stored = (rot_new_eff.astype(np.float32) * mirror).astype(np.float32)

    J_root = _mano_root_joint(shape, is_rhand, mano_model_dir)  # (N,3), float64
    trans_new = ((trans_stored.astype(np.float64) + J_root) @ R_E.T
                 + t_E - J_root).astype(np.float32)
    return rot_new_stored, trans_new


def _maybe_migrate_to_world_frame(root, seq, cam_bundle, mano_model_dir):
    """
    如果已有 npy 里 world2cam[0] 还是 identity (旧 cam_self-frame 约定),
    把手/物体位姿从 cam_self 系变到世界系并更新 camera 字段。
    无需迁移时直接返回 False。
    """
    params = root["data_dict"][seq]["params"]
    cam = params["camera"]
    w2c_list = cam.get("world2cam", None)
    need_migrate = (
        isinstance(w2c_list, list) and len(w2c_list) >= 1
        and _is_identity_4x4(w2c_list[0])
    )
    if not need_migrate:
        return False

    print("  [迁移] 检测到旧 cam_self-frame 格式, 位姿将变换到 world 系")
    E_self = cam_bundle["E_self"]

    obj = params["object"]
    if obj.get("obj_rot") is not None and obj.get("obj_trans") is not None:
        rot_new, trans_new = _transform_object_poses(
            obj["obj_rot"], obj["obj_trans"], E_self,
        )
        obj["obj_rot"] = rot_new
        obj["obj_trans"] = trans_new

    for side_key, rot_key, trans_key, shape_key, is_rhand in [
        ("right hand", "rot_r", "trans_r", "shape_r", True),
        ("left hand",  "rot_l", "trans_l", "shape_l", False),
    ]:
        h = params.get(side_key, None)
        if not h:
            continue
        if h.get(rot_key) is None or h.get(trans_key) is None or h.get(shape_key) is None:
            continue
        rot_new, trans_new = _transform_mano_params(
            h[rot_key], h[trans_key], h[shape_key],
            E_self, is_rhand, mano_model_dir,
        )
        h[rot_key] = rot_new
        h[trans_key] = trans_new

    params["camera"] = {
        "world2cam": cam_bundle["world2cam"],
        "K": cam_bundle["K"],
        "views": cam_bundle["views"],
    }
    return True


def list_sequences(dataset_root):
    """扫描 dexycb_clips/images/ 下所有序列文件夹名。"""
    images_root = os.path.join(dataset_root, "images")
    seqs = sorted(
        d for d in os.listdir(images_root)
        if os.path.isdir(os.path.join(images_root, d))
    )
    return seqs


# ══════════════════════════════════════════════════════════════════════
# HaMER 手部参数提取 (仅在主环境可用)
# ══════════════════════════════════════════════════════════════════════

def _make_nan_hand(n_frames):
    return {
        "rot":   np.full((n_frames, 3),  np.nan, dtype=np.float32),
        "pose":  np.full((n_frames, 45), np.nan, dtype=np.float32),
        "shape": np.full((n_frames, 10), np.nan, dtype=np.float32),
        "trans": np.full((n_frames, 3),  np.nan, dtype=np.float32),
    }


def _interpolate_nan(arr):
    if not np.isnan(arr).any():
        return 0
    N, D = arr.shape
    valid_mask = ~np.isnan(arr).any(axis=1)
    if not valid_mask.any():
        return 0
    all_idx = np.arange(N)
    valid_idx = all_idx[valid_mask]
    valid_data = arr[valid_mask]
    for d in range(D):
        arr[:, d] = np.interp(all_idx, valid_idx, valid_data[:, d])
    return int(N - valid_mask.sum())


def _slerp_interpolate_nan(rot_arr):
    from scipy.spatial.transform import Rotation, Slerp
    if not np.isnan(rot_arr).any():
        return 0
    N = rot_arr.shape[0]
    valid_mask = ~np.isnan(rot_arr).any(axis=1)
    if not valid_mask.any() or valid_mask.all():
        return 0
    valid_idx = np.where(valid_mask)[0]
    slerp = Slerp(valid_idx.astype(float),
                   Rotation.from_rotvec(rot_arr[valid_idx]))
    interp_min, interp_max = valid_idx[0], valid_idx[-1]
    all_idx = np.arange(N)
    inner_mask = (all_idx >= interp_min) & (all_idx <= interp_max)
    rot_arr[inner_mask] = slerp(all_idx[inner_mask].astype(float)) \
        .as_rotvec().astype(np.float32)
    if interp_min > 0:
        rot_arr[:interp_min] = rot_arr[interp_min]
    if interp_max < N - 1:
        rot_arr[interp_max + 1:] = rot_arr[interp_max]
    return int(N - valid_mask.sum())


def run_hamer_on_sequence(color_frames, K):
    """
    对一个序列的每一帧跑 YOLO+HaMER，返回按帧对齐的左右手 MANO 参数。
    color_frames: [(frame_id, path), ...]
    K: (3,3) 彩色相机内参

    返回:
      {'right': {'rot_r', 'pose_r', 'shape_r', 'trans_r'},
       'left':  {'rot_l', 'pose_l', 'shape_l', 'trans_l'}}
    """
    from tqdm import tqdm
    from model.hamer.infer import hamer_inference, matrix_to_axis_angle
    from model.rootnet.Model_RGB import get_model
    from config.hamer_config import hamer_opt
    from config.yolo_config import yolo_opt
    from yolo.detector import Detector

    print("[HaMER] 正在加载模型 ...")
    hamer = hamer_inference(hamer_opt)
    detector = Detector(yolo_opt)
    _ = get_model()
    print("[HaMER] 模型加载完成。")

    N = len(color_frames)
    r = _make_nan_hand(N)
    l = _make_nan_hand(N)
    stats = {"missing_right": 0, "missing_left": 0}

    for i, (_fid, img_path) in enumerate(tqdm(color_frames, desc="HaMER 推理")):
        image = cv2.imread(img_path)
        if image is None:
            continue
        _, dets = detector.detect(image)
        if isinstance(dets, list) and len(dets) > 0 \
                and isinstance(dets[0], list) and len(dets[0]) > 0 \
                and isinstance(dets[0][0], list):
            dets = dets[0]

        frame = {"right": None, "left": None}
        for bbox in dets:
            hand_label = bbox[0]
            try:
                output, _params = hamer.estimate_from_rgb(image, [bbox], K)
                mp = output["pred_mano_params"]
                betas = mp["betas"].detach().cpu().numpy().squeeze()
                hand_pose = matrix_to_axis_angle(
                    mp["hand_pose"].detach().cpu().numpy().squeeze()
                )
                glb = mp["global_orient"].detach().cpu().numpy().squeeze()
                if glb.ndim == 3:
                    glb = glb[0]
                glb_aa, _ = cv2.Rodrigues(glb)
                cam_t = output["pred_cam_t_full"].detach().cpu().numpy().squeeze()
                frame[hand_label] = dict(
                    rot=glb_aa.flatten().astype(np.float32),
                    pose=hand_pose.astype(np.float32),
                    shape=betas.astype(np.float32),
                    trans=cam_t.astype(np.float32),
                )
            except Exception as e:
                print(f"  帧 {i} {hand_label} 推理异常: {e}")

        if frame["right"] is not None:
            rh = frame["right"]
            r["rot"][i]   = rh["rot"]
            r["pose"][i]  = rh["pose"]
            r["shape"][i] = rh["shape"]
            r["trans"][i] = rh["trans"]
        else:
            stats["missing_right"] += 1

        if frame["left"] is not None:
            lh = frame["left"]
            l["rot"][i]   = lh["rot"]
            l["pose"][i]  = lh["pose"]
            l["shape"][i] = lh["shape"]
            l["trans"][i] = lh["trans"]
        else:
            stats["missing_left"] += 1

    filled = 0
    filled += _slerp_interpolate_nan(r["rot"])
    filled += _slerp_interpolate_nan(l["rot"])
    for arr in (r["pose"], r["shape"], r["trans"],
                l["pose"], l["shape"], l["trans"]):
        filled += _interpolate_nan(arr)

    print(f"[HaMER] 缺失右手 {stats['missing_right']} / 左手 {stats['missing_left']} 帧，插值填补 {filled} 段")

    return {
        "right": {
            "rot_r":   r["rot"],
            "pose_r":  r["pose"],
            "shape_r": r["shape"],
            "trans_r": r["trans"],
        },
        "left": {
            "rot_l":   l["rot"],
            "pose_l":  l["pose"],
            "shape_l": l["shape"],
            "trans_l": l["trans"],
        },
    }


# ══════════════════════════════════════════════════════════════════════
# FoundationPose 物体追踪 (subprocess 模式)
# ══════════════════════════════════════════════════════════════════════

def prepare_foundationpose_scene(seq, dataset_root, scene_dir,
                                  obj_mask_value=OBJ_MASK_VALUE):
    """
    为 FoundationPose worker 准备一份按帧对齐的 scene 目录:
      scene_dir/
        rgb/<id>.jpg       彩色原图
        depth/<id>.png     uint16 mm 深度 (从 aligned_depth_to_color 拷贝)
        masks/<id>.png     首帧物体掩码，二值 0/255
        cam_K.txt          彩色相机 3x3 内参
    这里不依赖 FoundationPose 自带的 YcbineoatReader，避免其对
    mask 文件后缀 (.jpg) 及深度 scale 的硬编码假设。
    """
    images_dir = os.path.join(dataset_root, "images", seq)
    masks_dir = os.path.join(dataset_root, "masks", seq)
    meta_dir = os.path.join(dataset_root, "meta")

    _, cam_id = parse_seq_name(seq)
    K = load_cam_K(meta_dir, cam_id)

    color_frames = list_color_frames(images_dir)

    if os.path.exists(scene_dir):
        shutil.rmtree(scene_dir)
    rgb_out = os.path.join(scene_dir, "rgb")
    depth_out = os.path.join(scene_dir, "depth")
    mask_out = os.path.join(scene_dir, "masks")
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(depth_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    np.savetxt(os.path.join(scene_dir, "cam_K.txt"), K.astype(np.float64))

    id_strs = []
    for fid, color_path in color_frames:
        id_str = f"{fid:06d}"
        id_strs.append(id_str)

        shutil.copyfile(color_path, os.path.join(rgb_out, f"{id_str}.jpg"))

        depth_src = os.path.join(
            images_dir, f"aligned_depth_to_color_{id_str}.png"
        )
        if not os.path.isfile(depth_src):
            raise FileNotFoundError(f"缺少对齐深度: {depth_src}")
        shutil.copyfile(depth_src, os.path.join(depth_out, f"{id_str}.png"))

    first_id = id_strs[0]
    first_mask_npy = os.path.join(masks_dir, f"color_{first_id}.npy")
    if not os.path.isfile(first_mask_npy):
        raise FileNotFoundError(f"缺少首帧物体掩码: {first_mask_npy}")
    mask = np.load(first_mask_npy)
    obj_mask = ((mask == obj_mask_value).astype(np.uint8)) * 255
    cv2.imwrite(os.path.join(mask_out, f"{first_id}.png"), obj_mask)

    return scene_dir, id_strs, K


def run_foundationpose_inline(seq, mesh_file, scene_dir, pose_out_path,
                               est_refine_iter, track_refine_iter):
    """
    在当前 (foundationpose_ros) 进程中直接调用 FoundationPose 追踪。
    结果 6D 位姿序列写入 pose_out_path (npz: ob_in_cam [N,4,4], id_strs [N])。

    FoundationPose/Utils.py 会把全局默认 tensor type 切到 cuda.FloatTensor,
    这会污染后续 HaMER / 渲染里不带 device 参数的 torch.tensor 调用 -
    这里在进入/退出时保存并还原 cwd + default tensor type。
    """
    import torch as _torch
    ns = argparse.Namespace(
        seq_name=seq,
        mesh_file=mesh_file,
        scene_dir=scene_dir,
        pose_out=pose_out_path,
        est_refine_iter=est_refine_iter,
        track_refine_iter=track_refine_iter,
    )
    prev_cwd = os.getcwd()
    prev_default = _torch.tensor([0.0]).type()  # e.g. 'torch.FloatTensor'
    try:
        foundationpose_worker(ns)
    finally:
        try:
            os.chdir(prev_cwd)
        except OSError:
            pass
        try:
            _torch.set_default_tensor_type(prev_default)
        except Exception:
            pass
    if not os.path.isfile(pose_out_path):
        raise RuntimeError(
            f"FoundationPose 追踪结束但未生成位姿输出: {pose_out_path}"
        )


def _load_scene_frames(scene_dir, zfar=np.inf):
    """
    读取 prepare_foundationpose_scene 生成的 scene 目录。
    返回 (K, id_strs, color_paths, depth_paths, mask_path_first)，
    深度以米为单位 (float32)。
    """
    K = np.loadtxt(os.path.join(scene_dir, "cam_K.txt")).reshape(3, 3)
    color_paths = sorted(glob.glob(os.path.join(scene_dir, "rgb", "*.jpg")))
    if not color_paths:
        raise FileNotFoundError(f"scene_dir 中未找到 rgb/*.jpg: {scene_dir}")
    id_strs = [os.path.splitext(os.path.basename(p))[0] for p in color_paths]
    depth_paths = [
        os.path.join(scene_dir, "depth", f"{s}.png") for s in id_strs
    ]
    for p in depth_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"缺少深度帧: {p}")
    mask_path_first = os.path.join(scene_dir, "masks", f"{id_strs[0]}.png")
    if not os.path.isfile(mask_path_first):
        raise FileNotFoundError(f"缺少首帧 mask: {mask_path_first}")
    return K, id_strs, color_paths, depth_paths, mask_path_first


def foundationpose_worker(args):
    """
    在 foundationpose_ros 环境中执行的 FoundationPose 追踪循环
    (仿照 FoundationPose/run_demo.py, 使用自定义读取器避免 YcbineoatReader
    对深度 scale / mask 后缀的硬编码假设)。
    """
    sys.path.insert(0, FP_PKG_DIR)
    os.chdir(FP_PKG_DIR)

    from estimater import (
        FoundationPose, ScorePredictor, PoseRefinePredictor,
        set_logging_format, set_seed,
    )
    import trimesh
    import nvdiffrast.torch as dr
    import logging

    set_logging_format()
    set_seed(0)

    print(f"[FP-worker] 序列: {args.seq_name}")
    print(f"[FP-worker] mesh: {args.mesh_file}")
    print(f"[FP-worker] scene_dir: {args.scene_dir}")

    mesh = trimesh.load(args.mesh_file)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    debug_dir = os.path.join(args.scene_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    est = FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
        mesh=mesh, scorer=scorer, refiner=refiner,
        debug_dir=debug_dir, debug=0, glctx=glctx,
    )
    logging.info("FoundationPose 初始化完成")

    K, id_strs, color_paths, depth_paths, mask_path_first = \
        _load_scene_frames(args.scene_dir)
    K = K.astype(np.float64)

    def _read_color(path):
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"无法读取彩色帧: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _read_depth(path):
        d = cv2.imread(path, -1)
        if d is None:
            raise RuntimeError(f"无法读取深度帧: {path}")
        depth = d.astype(np.float32) * DEPTH_SCALE_MM_TO_M  # mm -> m
        depth[(depth < 0.001) | (depth > 10.0)] = 0
        return depth

    def _read_mask(path):
        m = cv2.imread(path, -1)
        if m is None:
            raise RuntimeError(f"无法读取 mask: {path}")
        if m.ndim == 3:
            m = m[..., 0]
        return (m > 0).astype(bool)

    poses = []
    for i, (cp, dp) in enumerate(zip(color_paths, depth_paths)):
        color = _read_color(cp)
        depth = _read_depth(dp)

        if i == 0:
            mask = _read_mask(mask_path_first)
            pose = est.register(
                K=K, rgb=color, depth=depth, ob_mask=mask,
                iteration=args.est_refine_iter,
            )
        else:
            pose = est.track_one(
                rgb=color, depth=depth, K=K,
                iteration=args.track_refine_iter,
            )
        poses.append(np.asarray(pose, dtype=np.float32).reshape(4, 4))

    poses = np.stack(poses, axis=0).astype(np.float32)
    np.savez(
        args.pose_out,
        ob_in_cam=poses,
        id_strs=np.array(id_strs),
        K=K.astype(np.float32),
    )
    print(f"[FP-worker] 已保存 {poses.shape[0]} 帧位姿: {args.pose_out}")


# ══════════════════════════════════════════════════════════════════════
# 主流程：逐序列编排 HaMER + FoundationPose
# ══════════════════════════════════════════════════════════════════════

def ob_in_cam_to_axisangle_trans(poses):
    """(N,4,4) 物体→相机齐次矩阵  →  obj_rot (N,3) 轴角, obj_trans (N,3) 米。"""
    N = poses.shape[0]
    obj_rot = np.zeros((N, 3), dtype=np.float32)
    obj_trans = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        obj_rot[i] = R.from_matrix(poses[i, :3, :3]).as_rotvec().astype(np.float32)
        obj_trans[i] = poses[i, :3, 3].astype(np.float32)
    return obj_rot, obj_trans


def process_sequence(seq, dataset_root, output_dir, fp_python,
                     skip_hamer, skip_foundationpose,
                     est_refine_iter, track_refine_iter,
                     obj_mask_value, keep_scene_dir,
                     render=False, render_dir=DEFAULT_RENDER_DIR,
                     mano_model_dir=DEFAULT_MANO_MODEL_DIR):
    """处理单个序列，并生成 <output_dir>/<seq>.npy (格式对齐 data_fixed.npy)。"""
    print("\n" + "=" * 70)
    print(f"[序列] {seq}")
    print("=" * 70)

    obj_prefix, cam_id = parse_seq_name(seq)
    images_dir = os.path.join(dataset_root, "images", seq)

    color_frames = list_color_frames(images_dir)
    n_frames = len(color_frames)

    mesh_file, obj_model_name = find_object_mesh(dataset_root, obj_prefix)
    cam_bundle = compute_dual_view_cameras(dataset_root, seq)
    E_self = cam_bundle["E_self"]   # cam_self → world
    K = cam_bundle["K"][0]

    print(f"  相机: cam_{cam_id} (view0), 对比相机: cam_{cam_bundle['other_cam_id']} (view1)")
    print(f"  对象模型: {obj_model_name}")
    print(f"  帧数: {n_frames}, 图片目录: {images_dir}")

    os.makedirs(output_dir, exist_ok=True)
    out_npy = os.path.join(output_dir, f"{seq}.npy")

    if os.path.isfile(out_npy):
        print(f"  [恢复] 加载已有 npy: {out_npy}")
        root = np.load(out_npy, allow_pickle=True).item()
        # 清理旧格式里 object 的附加字段，严格对齐 data_fixed.npy
        obj_store = root["data_dict"][seq]["params"]["object"]
        for _k in ("obj_model", "obj_mesh"):
            obj_store.pop(_k, None)
        # 如果 npy 仍然是旧 cam_self-frame 约定 (world2cam[0]=I), 迁到世界系
        _maybe_migrate_to_world_frame(root, seq, cam_bundle, mano_model_dir)
        # 统一覆盖 camera 字段 (形状 / dtype 可能也过时)
        root["data_dict"][seq]["params"]["camera"] = {
            "world2cam": cam_bundle["world2cam"],
            "K": cam_bundle["K"],
            "views": cam_bundle["views"],
        }
    else:
        root = {
            "imgnames": [os.path.basename(p) for _, p in color_frames],
            "imgpath": os.path.abspath(images_dir),
            "data_dict": {
                seq: {
                    "params": {
                        "right hand": {
                            "rot_r": None, "pose_r": None,
                            "trans_r": None, "shape_r": None,
                        },
                        "left hand": {
                            "rot_l": None, "pose_l": None,
                            "trans_l": None, "shape_l": None,
                        },
                        "object": {
                            "obj_rot": None, "obj_trans": None,
                        },
                        "camera": {
                            "world2cam": cam_bundle["world2cam"],
                            "K": cam_bundle["K"],
                            "views": cam_bundle["views"],
                        },
                    }
                }
            },
        }

    # ── 1. HaMER 手部 (estimator 输出在 cam_self 系 → 立即变到 world) ──
    params = root["data_dict"][seq]["params"]
    if not skip_hamer:
        hands = run_hamer_on_sequence(color_frames, K)
        for src_key, side_key, rot_k, pose_k, trans_k, shape_k, is_rhand in [
            ("right", "right hand", "rot_r", "pose_r", "trans_r", "shape_r", True),
            ("left",  "left hand",  "rot_l", "pose_l", "trans_l", "shape_l", False),
        ]:
            h_src = hands[src_key]
            rot_cam   = np.asarray(h_src[rot_k],   dtype=np.float32)
            pose_arr  = np.asarray(h_src[pose_k],  dtype=np.float32)
            trans_cam = np.asarray(h_src[trans_k], dtype=np.float32)
            shape_arr = np.asarray(h_src[shape_k], dtype=np.float32)
            rot_w, trans_w = _transform_mano_params(
                rot_cam, trans_cam, shape_arr, E_self, is_rhand, mano_model_dir,
            )
            dst = params[side_key]
            dst[rot_k]   = rot_w
            dst[pose_k]  = pose_arr
            dst[trans_k] = trans_w
            dst[shape_k] = shape_arr
        np.save(out_npy, root, allow_pickle=True)
        print(f"  [保存] HaMER 中间 npy (world frame) → {out_npy}")
    else:
        print("  跳过 HaMER")

    # ── 2. FoundationPose 物体追踪 (cam_self 系 → world) ──
    if not skip_foundationpose:
        scene_dir = os.path.join(output_dir, "_fp_scenes", seq)
        prepare_foundationpose_scene(
            seq, dataset_root, scene_dir, obj_mask_value=obj_mask_value,
        )

        pose_out = os.path.join(output_dir, f"_fp_{seq}.npz")
        run_foundationpose_inline(
            seq, mesh_file, scene_dir, pose_out,
            est_refine_iter, track_refine_iter,
        )

        data = np.load(pose_out, allow_pickle=True)
        poses = data["ob_in_cam"]
        obj_rot, obj_trans = ob_in_cam_to_axisangle_trans(poses)

        if obj_rot.shape[0] != n_frames:
            print(f"  警告: FoundationPose 输出帧数 {obj_rot.shape[0]} != 预期 {n_frames}")

        obj_rot_w, obj_trans_w = _transform_object_poses(obj_rot, obj_trans, E_self)
        params["object"]["obj_rot"] = obj_rot_w
        params["object"]["obj_trans"] = obj_trans_w

        np.save(out_npy, root, allow_pickle=True)
        print(f"  [保存] FoundationPose 结果合入 (world frame) → {out_npy}")

        if not keep_scene_dir:
            shutil.rmtree(scene_dir, ignore_errors=True)
            try:
                os.remove(pose_out)
            except OSError:
                pass
    else:
        print("  跳过 FoundationPose")

    # ── 3. 最终落盘 (即便两阶段都跳过也要持久化格式迁移) ──
    np.save(out_npy, root, allow_pickle=True)

    # ── 汇总打印 ──
    hand_r = params["right hand"]
    hand_l = params["left hand"]
    obj = params["object"]
    print(f"  完成: {out_npy}")
    if hand_r.get("pose_r") is not None:
        print(f"    右手 pose: {np.asarray(hand_r['pose_r']).shape}")
    if hand_l.get("pose_l") is not None:
        print(f"    左手 pose: {np.asarray(hand_l['pose_l']).shape}")
    if obj.get("obj_rot") is not None:
        print(f"    物体 rot/trans: {obj['obj_rot'].shape}, {obj['obj_trans'].shape}")

    # ── 4. 双视角重投影渲染 (可选) ──
    if render:
        render_sequence_dual_view(
            npy_path=out_npy,
            seq=seq,
            dataset_root=dataset_root,
            mesh_file=mesh_file,
            render_dir=render_dir,
            mano_model_dir=mano_model_dir,
        )


# ══════════════════════════════════════════════════════════════════════
# 双视角重投影渲染 (参考 utils/way_vis.py)
# ══════════════════════════════════════════════════════════════════════

_MIRROR_LEFT_PARAMS = np.array([1.0, -1.0, -1.0], dtype=np.float32)


def _right_to_left_mano_params(rot, pose):
    rot_l = rot * _MIRROR_LEFT_PARAMS
    pose_l = (pose.reshape(-1, 3) * _MIRROR_LEFT_PARAMS).reshape(pose.shape)
    return rot_l, pose_l


def _transform_points(T, pts):
    return np.einsum("ij,...j->...i", T[:3, :3], pts) + T[:3, 3]


def render_sequence_dual_view(npy_path, seq, dataset_root, mesh_file,
                               render_dir, mano_model_dir, fps=15,
                               encode_h264=True):
    """
    把单个序列的手 (MANO) + 物体 (mesh) 按 npy 里的 2 个视角投影回原图叠加。
    输出:
      render_dir/<seq>/view0_cam<self>/frame_XXXXXX.jpg  + view0.mp4
      render_dir/<seq>/view1_cam<other>/frame_XXXXXX.jpg + view1.mp4

    view0 的背景图来自序列自身 images/<seq>/，
    view1 的背景图来自对侧序列 images/<other_seq>/ (若存在)，
    否则用纯黑背景。
    """
    try:
        import smplx
        import trimesh
        import pyrender
        import torch
    except Exception as e:
        print(f"[渲染] 缺少依赖 (需要 smplx/trimesh/pyrender/torch): {e}")
        return

    root = np.load(npy_path, allow_pickle=True).item()
    params = root["data_dict"][seq]["params"]
    cam = params["camera"]
    K_list = [np.asarray(x, dtype=np.float64) for x in cam["K"]]
    W2C_list = [np.asarray(x, dtype=np.float64) for x in cam["world2cam"]]
    views = cam["views"]

    hand_r = params["right hand"]
    hand_l = params.get("left hand", None)
    obj = params["object"]

    def _nan_to_zero(arr):
        arr = np.asarray(arr, dtype=np.float32).copy()
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    rot_r = _nan_to_zero(hand_r["rot_r"])
    pose_r = _nan_to_zero(hand_r["pose_r"])
    trans_r = _nan_to_zero(hand_r["trans_r"])
    shape_r = _nan_to_zero(hand_r["shape_r"])

    has_left = (
        hand_l is not None
        and hand_l.get("rot_l") is not None
        and hand_l.get("pose_l") is not None
    )
    if has_left:
        rot_l = _nan_to_zero(hand_l["rot_l"])
        pose_l = _nan_to_zero(hand_l["pose_l"])
        trans_l = _nan_to_zero(hand_l["trans_l"])
        shape_l = _nan_to_zero(hand_l["shape_l"])

    if obj["obj_rot"] is None or obj["obj_trans"] is None:
        print("[渲染] 物体位姿缺失，跳过渲染")
        return
    obj_rot_arr = _nan_to_zero(obj["obj_rot"]).astype(np.float64)
    obj_trans_arr = _nan_to_zero(obj["obj_trans"]).astype(np.float64)
    obj_rot_mats = R.from_rotvec(obj_rot_arr).as_matrix()

    base_obj_mesh = trimesh.load(mesh_file, process=False)
    if isinstance(base_obj_mesh, trimesh.Scene):
        base_obj_mesh = base_obj_mesh.dump()[0]
    base_verts = np.asarray(base_obj_mesh.vertices, dtype=np.float64)

    cpu = torch.device("cpu")
    mano_r = smplx.create(mano_model_dir, "MANO",
                           use_pca=False, is_rhand=True, flat_hand_mean=True).to(cpu)
    with torch.no_grad():
        out_r = mano_r(
            global_orient=torch.tensor(rot_r, dtype=torch.float32, device=cpu),
            hand_pose=torch.tensor(pose_r, dtype=torch.float32, device=cpu),
            betas=torch.tensor(shape_r, dtype=torch.float32, device=cpu),
            transl=torch.tensor(trans_r, dtype=torch.float32, device=cpu),
        )
    verts_r_world = out_r.vertices.detach().cpu().numpy()
    faces_r = mano_r.faces

    if has_left:
        mano_l = smplx.create(mano_model_dir, "MANO",
                               use_pca=False, is_rhand=False, flat_hand_mean=True).to(cpu)
        rot_l_m, pose_l_m = _right_to_left_mano_params(rot_l, pose_l)
        with torch.no_grad():
            out_l = mano_l(
                global_orient=torch.tensor(rot_l_m, dtype=torch.float32, device=cpu),
                hand_pose=torch.tensor(pose_l_m, dtype=torch.float32, device=cpu),
                betas=torch.tensor(shape_l, dtype=torch.float32, device=cpu),
                transl=torch.tensor(trans_l, dtype=torch.float32, device=cpu),
            )
        verts_l_world = out_l.vertices.detach().cpu().numpy()
        faces_l = mano_l.faces
    else:
        verts_l_world = None
        faces_l = None

    n_frames = min(
        len(obj_rot_arr), len(obj_trans_arr),
        verts_r_world.shape[0],
        verts_l_world.shape[0] if verts_l_world is not None else np.inf,
    )
    n_frames = int(n_frames)
    print(f"  [渲染] 帧数对齐到 {n_frames}")

    out_base = os.path.join(render_dir, seq)
    os.makedirs(out_base, exist_ok=True)

    # 读取背景图: view0 用 seq 自己, view1 用对侧序列
    self_images_dir = os.path.join(dataset_root, "images", seq)
    other_seq = other_seq_name(seq)
    other_images_dir = os.path.join(dataset_root, "images", other_seq)
    bg_dirs = [self_images_dir,
               other_images_dir if os.path.isdir(other_images_dir) else None]

    color_frames = list_color_frames(self_images_dir)

    # pyrender OpenCV→OpenGL 相机姿态
    cam_pose_gl = np.array([
        [1.0,  0.0,  0.0, 0.0],
        [0.0, -1.0,  0.0, 0.0],
        [0.0,  0.0, -1.0, 0.0],
        [0.0,  0.0,  0.0, 1.0],
    ])
    hand_r_mat = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.1, alphaMode="OPAQUE",
        baseColorFactor=(0.8, 0.6, 0.5, 1.0))
    hand_l_mat = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.1, alphaMode="OPAQUE",
        baseColorFactor=(0.5, 0.6, 0.8, 1.0))
    obj_mat = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.3, alphaMode="OPAQUE",
        baseColorFactor=(0.2, 0.8, 0.2, 1.0))

    for v_idx, (view_name, K, W2C) in enumerate(zip(views, K_list, W2C_list)):
        view_tag = f"view{v_idx}"
        view_dir = os.path.join(out_base, view_tag)
        os.makedirs(view_dir, exist_ok=True)

        # 用对应视角的真实图尺寸做画布
        bg_dir = bg_dirs[v_idx]
        if bg_dir is not None:
            # 用第一帧尺寸
            first_img = cv2.imread(color_frames[0][1])
            H, W = first_img.shape[:2]
        else:
            H, W = 480, 640

        renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
        camera = pyrender.IntrinsicsCamera(
            fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_path = os.path.join(out_base, f"{view_tag}.mp4")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))

        print(f"  [渲染] {seq} / {view_tag} ({view_name}) → {view_dir}")

        from tqdm import tqdm
        for i in tqdm(range(n_frames), desc=f"  render {view_tag}"):
            fid = color_frames[i][0] if i < len(color_frames) else i

            scene = pyrender.Scene(
                bg_color=[0.0, 0.0, 0.0, 0.0],
                ambient_light=[0.6, 0.6, 0.6],
            )
            scene.add(camera, pose=cam_pose_gl)
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
            scene.add(light, pose=cam_pose_gl)

            # 物体
            obj_verts_world = (obj_rot_mats[i] @ base_verts.T).T + obj_trans_arr[i]
            obj_verts_cam = _transform_points(W2C, obj_verts_world)
            obj_mesh = trimesh.Trimesh(
                vertices=obj_verts_cam, faces=base_obj_mesh.faces, process=False,
            )
            scene.add(pyrender.Mesh.from_trimesh(obj_mesh, material=obj_mat))

            # 右手
            vr = verts_r_world[i]
            if np.isfinite(vr).all():
                vr_cam = _transform_points(W2C, vr)
                mesh_r = trimesh.Trimesh(vr_cam, faces_r, process=False)
                scene.add(pyrender.Mesh.from_trimesh(mesh_r, material=hand_r_mat))

            # 左手
            if has_left:
                vl = verts_l_world[i]
                if np.isfinite(vl).all():
                    vl_cam = _transform_points(W2C, vl)
                    mesh_l = trimesh.Trimesh(vl_cam, faces_l, process=False)
                    scene.add(pyrender.Mesh.from_trimesh(mesh_l, material=hand_l_mat))

            color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            render_bgr = cv2.cvtColor(color[:, :, :3], cv2.COLOR_RGB2BGR)
            alpha = color[:, :, 3:4].astype(np.float32) / 255.0

            # 背景
            bg_img = None
            if bg_dir is not None:
                bg_path = os.path.join(bg_dir, f"color_{fid:06d}.jpg")
                if os.path.isfile(bg_path):
                    bg_img = cv2.imread(bg_path)
            if bg_img is None:
                bg_img = np.zeros((H, W, 3), dtype=np.uint8)
            if bg_img.shape[:2] != (H, W):
                bg_img = cv2.resize(bg_img, (W, H))

            final = (render_bgr.astype(np.float32) * alpha
                     + bg_img.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

            out_path = os.path.join(view_dir, f"frame_{fid:06d}.jpg")
            cv2.imwrite(out_path, final)
            writer.write(final)

        writer.release()
        renderer.delete()

        if encode_h264 and shutil.which("ffmpeg"):
            h264_path = video_path.rsplit(".", 1)[0] + "_h264.mp4"
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-pix_fmt", "yuv420p", h264_path,
            ]
            ret = subprocess.run(cmd, capture_output=True)
            if ret.returncode == 0:
                os.replace(h264_path, video_path)

    print(f"  [渲染] {seq} 双视角输出完成 → {out_base}")


# ══════════════════════════════════════════════════════════════════════
# 命令行
# ══════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description="DexYCB clips 手部 (HaMER) + 物体 (FoundationPose) 提取",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    p.add_argument("--output_dir",   default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--fp_python",    default=DEFAULT_FP_PYTHON,
                   help="foundationpose_ros 环境 python 可执行文件")
    p.add_argument("--seqs", nargs="*", default=None,
                   help="指定序列名 (如 12-1 14-2)，不填配合 --all")
    p.add_argument("--all", action="store_true",
                   help="处理 dataset_root/images 下的所有序列")
    p.add_argument("--skip_hamer", action="store_true",
                   help="跳过 HaMER 手部阶段 (已有 npy 时用于只跑物体)")
    p.add_argument("--skip_foundationpose", action="store_true",
                   help="跳过 FoundationPose 物体阶段")
    p.add_argument("--est_refine_iter",   type=int, default=5)
    p.add_argument("--track_refine_iter", type=int, default=2)
    p.add_argument("--obj_mask_value",    type=int, default=OBJ_MASK_VALUE,
                   help="mask npy 中用于标识物体的像素值 (dexycb_clips 默认 2)")
    p.add_argument("--keep_scene_dir", action="store_true",
                   help="保留 FoundationPose 中间 scene 目录")
    p.add_argument("--render", action="store_true",
                   help="为每个序列重投影渲染双视角视频/抽帧")
    p.add_argument("--render_dir", default=DEFAULT_RENDER_DIR,
                   help="双视角渲染输出目录")
    p.add_argument("--mano_model_dir", default=DEFAULT_MANO_MODEL_DIR,
                   help="MANO 模型根目录 (含 MANO_RIGHT.pkl / MANO_LEFT.pkl)")

    # ─ worker 模式 (不要手动触发，父进程会用 foundationpose_ros 环境调用) ─
    p.add_argument("--_fp_worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--seq_name",   default=None, help=argparse.SUPPRESS)
    p.add_argument("--mesh_file",  default=None, help=argparse.SUPPRESS)
    p.add_argument("--scene_dir",  default=None, help=argparse.SUPPRESS)
    p.add_argument("--pose_out",   default=None, help=argparse.SUPPRESS)
    return p


def main():
    args = build_parser().parse_args()

    # 兼容旧的子进程 worker 入口 (单环境模式下不再需要，但保留以防外部调用)
    if args._fp_worker:
        foundationpose_worker(args)
        return

    if args.seqs:
        seqs = list(args.seqs)
    elif args.all:
        seqs = list_sequences(args.dataset_root)
        print(f"[全部序列] {seqs}")
    else:
        print("请通过 --seqs <seq> ... 或 --all 指定要处理的序列。")
        sys.exit(1)

    # 只用 cam_1 视角做识别: 过滤/警告非 *-1 序列
    kept, dropped = [], []
    for s in seqs:
        try:
            _, cam = parse_seq_name(s)
        except Exception:
            dropped.append(s)
            continue
        (kept if cam == "1" else dropped).append(s)
    if dropped:
        print(f"[过滤] 跳过非 cam_1 序列 (识别仅用 cam_1 图片): {dropped}")
    seqs = kept
    if not seqs:
        print("没有可处理的 *-1 序列, 退出。")
        sys.exit(0)

    os.makedirs(args.output_dir, exist_ok=True)

    for seq in seqs:
        try:
            process_sequence(
                seq=seq,
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
                fp_python=args.fp_python,
                skip_hamer=args.skip_hamer,
                skip_foundationpose=args.skip_foundationpose,
                est_refine_iter=args.est_refine_iter,
                track_refine_iter=args.track_refine_iter,
                obj_mask_value=args.obj_mask_value,
                keep_scene_dir=args.keep_scene_dir,
                render=args.render,
                render_dir=args.render_dir,
                mano_model_dir=args.mano_model_dir,
            )
        except Exception as e:
            print(f"[错误] 序列 {seq} 处理失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n全部序列处理完成。")


if __name__ == "__main__":
    main()
