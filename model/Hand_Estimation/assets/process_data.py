import os
import pickle
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset


def generate_dataset_index(data_root="cyc/UST-Hand/seq1", cache_path="dataset_cache.pkl", image_extension=".png"):
    """
    根据您的文件结构生成数据集索引
    只遍历rgb文件夹中的PNG文件

    文件结构：
    seq1/
        camera0/
            rgb/           # PNG图像序列
            depth/         # 深度图
            center.json    # 中心点信息
            ...其他文件
        camera1/
        camera2/
        camera3/
    """

    sample_idxs = []
    all_samples = []

    # 遍历所有camera文件夹
    for cam_dir in sorted(os.listdir(data_root)):
        if not cam_dir.startswith("camera"):
            continue

        cam_path = os.path.join(data_root, cam_dir)
        if not os.path.isdir(cam_path):
            continue

        print(f"Processing {cam_dir}...")

        # 检查rgb文件夹是否存在
        rgb_dir = os.path.join(cam_path, "rgb")
        if not os.path.exists(rgb_dir):
            print(f"Warning: {rgb_dir} not found, skipping {cam_dir}")
            continue

        # 获取rgb文件夹中的所有图片文件
        image_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(image_extension)])

        # 处理每个图片文件
        for img_file in tqdm(image_files, desc=f"Processing images in {cam_dir}"):
            # 提取帧号（去掉扩展名）
            frame_id = os.path.splitext(img_file)[0]

            # 构建样本信息
            sample_info = {
                "seq_id": os.path.basename(data_root),  # 从目录结构中提取，这里是"seq1"
                "cam_id": cam_dir,  # 完整相机目录名，如"camera2"
                "frame_id": frame_id,  # 帧号，如"0380"
                "cam_dir": cam_dir,
                "frame_path": os.path.join(cam_dir, "rgb", img_file),  # 相对路径，如"camera2/rgb/0380.png"
                "image_path": os.path.join(cam_path, "rgb", img_file),  # 完整路径
                "depth_path": os.path.join(cam_dir, "depth", f"{frame_id}.png"),  # 深度图路径
                "center_data": os.path.join(cam_path, "center.json")  # center.json路径
            }

            all_samples.append(sample_info)

    # 创建索引
    for i in range(len(all_samples)):
        sample_idxs.append(i)

    # 保存缓存
    if cache_path:
        # 确保缓存目录存在
        os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".", exist_ok=True)

        with open(cache_path, "wb") as p_f:
            pickle.dump({
                "sample_idxs": sample_idxs,
                "all_samples": all_samples,
                "data_root": data_root,
                "stats": {
                    "total_samples": len(sample_idxs),
                    "cameras": len(set([s["cam_dir"] for s in all_samples])),
                    "frames_per_camera": {cam: len([s for s in all_samples if s["cam_dir"] == cam])
                                          for cam in set([s["cam_dir"] for s in all_samples])}
                }
            }, p_f)
        print(f"Wrote cache to {cache_path}")

    print(f"Got {len(sample_idxs)} samples from {len(set([s['cam_dir'] for s in all_samples]))} cameras")
    return sample_idxs, all_samples