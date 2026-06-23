"""给一个采集序列, 按它实际存在的 .raw 相机生成 club_tracker 的 tracking config.

camera_id 从 camera_params.json 的 name(含 DA 编号) 匹配; reference_camera_id = 最低 id 的
彩色相机 (= 人体管线 cam0), 让球杆轨迹与人体在同一 world 系. dict_id=2 (DICT_4X4_250) +
放宽的 detector 参数 (应对运动模糊)."""
import argparse
import json
import re
import glob
from pathlib import Path


def detect_dict_id(tmp_images: Path):
    """自动判别 ArUco 字典: 试 4X4_250(=2) 与 6X6_250(=10), 选检测到更多 rig 低 id 的那个.
    (同一杆不同采集可能换了 marker 字典.)"""
    import cv2
    import numpy as np
    color = sorted(glob.glob(str(tmp_images / '*BayerRG8*')))
    if not color:
        return 2
    frames = sorted(glob.glob(color[0] + '/*.jpg'))
    if not frames:
        return 2
    frames = frames[::max(1, len(frames) // 40)]
    best_id, best_cnt = 2, -1
    for dv, did in [(cv2.aruco.DICT_4X4_250, 2), (cv2.aruco.DICT_6X6_250, 10)]:
        det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dv),
                                      cv2.aruco.DetectorParameters())
        cnt = 0
        for f in frames:
            _, ids, _ = det.detectMarkers(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY))
            if ids is not None:
                cnt += int((ids.flatten() < 50).sum())   # rig 用小 id
        if cnt > best_cnt:
            best_id, best_cnt = did, cnt
    print(f"[gen_club] dict auto-detect: 4X4/6X6 -> dict_id={best_id} ({best_cnt} rig dets)")
    return best_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", required=True)
    ap.add_argument("--aruco_rig", required=True)
    ap.add_argument("--out_config", required=True)
    a = ap.parse_args()

    seq_dir = Path(a.seq_dir)
    calib = seq_dir / "camera_params.json"
    cams_meta = json.loads(calib.read_text())["rig"]["cameras"]
    # DA 编号 -> camera_id, 是否彩色
    by_da = {}
    for c in cams_meta:
        m = re.search(r"DA(\d+)", c["name"])
        if m:
            by_da[m.group(1)] = (c["camera_id"], "10UC" in c["name"])  # 10UC = 彩色

    tmp = seq_dir / ".tmp_images"
    cameras = []
    color_ids = []
    for raw in sorted(seq_dir.glob("*.raw")):
        m = re.search(r"DA(\d+)", raw.name)
        if not m or m.group(1) not in by_da:
            continue
        cid, is_color = by_da[m.group(1)]
        src = tmp / raw.stem
        if not src.is_dir():
            print(f"[gen_club] 警告: 帧目录不存在 {src}, 跳过 cam{cid}")
            continue
        cameras.append({"camera_id": cid, "source": str(src)})
        if is_color:
            color_ids.append(cid)
    if not cameras:
        raise RuntimeError("没有可用相机帧目录")
    ref = min(color_ids) if color_ids else cameras[0]["camera_id"]

    cfg = {
        "calibration": str(calib),
        "cameras": cameras,
        "reference_camera_id": ref,
        "rig_prior": {"path": a.aruco_rig, "fix_rig": True, "prior_weight": 10.0},
        "aruco": {
            "dict_id": detect_dict_id(tmp), "marker_size": 99.0,
            "detector": {
                "corner_refinement_method": 3, "corner_refinement_win_size": 5,
                "adaptive_thresh_win_size_min": 3, "adaptive_thresh_win_size_max": 43,
                "adaptive_thresh_win_size_step": 8, "adaptive_thresh_constant": 7,
                "min_marker_perimeter_rate": 0.01, "max_marker_perimeter_rate": 4.0,
                "polygonal_approx_accuracy_rate": 0.06,
                "error_correction_rate": 0.9, "max_erroneous_bits_in_border_rate": 0.5,
            },
        },
        "frame_selection": {"start": 0, "end": -1, "step": 1},
        "optimization": {
            "max_iterations": 500, "w_smooth": 0.001, "w_accel": 0.0,
            "max_view_angle": 75, "num_threads": 4, "max_smooth_gap": 5,
            "reproj_huber_threshold": 2.0, "smooth_w_rot": 1.0, "smooth_w_trans": 0.001,
        },
        "filtering": {
            "min_marker_frames": 3, "min_covis_frames": 2,
            "max_marker_distance": 300.0, "verbose_filter_stats": True,
        },
        "output": {
            "directory": str(seq_dir / "trajectory_output"),
            "visualize": True, "verbose": True,
        },
    }
    Path(a.out_config).write_text(json.dumps(cfg, indent=2))
    print(f"[gen_club] {seq_dir.name}: {len(cameras)} cams "
          f"(ids={[c['camera_id'] for c in cameras]}, ref={ref}) -> {a.out_config}")


if __name__ == "__main__":
    main()
