import numpy as np
import cv2
import argparse
import os
import sys
import glob

# 尝试导入 tqdm 显示进度条，如果没有则使用普通打印
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        return iterable

def load_intrinsics(txt_path):
    """加载内参"""
    if not os.path.exists(txt_path):
        print(f"[Error] 内参文件不存在: {txt_path}")
        return None
    try:
        K = np.loadtxt(txt_path)
        return K 
    except Exception as e:
        print(f"[Error] 加载内参失败: {e}")
        return None

def load_obj(obj_path):
    """
    手动解析 OBJ 文件
    """
    vertices = []
    faces = []
    
    if not os.path.exists(obj_path):
        # 在批量处理中，单个文件不存在不应打印 Error，而是返回 None 由主循环处理
        return None, None

    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.strip().split()
                idx = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(idx[:3])
    
    return np.array(vertices), np.array(faces)

def project_and_draw(image, vertices, faces, K, alpha=0.6, color=(0, 255, 0)):
    """
    核心函数：3D 投影绘制
    """
    h_img, w_img = image.shape[:2]
    overlay = image.copy()
    
    # 1. 投影 3D -> 2D
    Z = vertices[:, 2]
    Z[Z == 0] = 1e-5 # 防止除零
    
    # 投影: (3x3 * 3xN -> 3xN)
    projected_homo = (K @ vertices.T).T
    
    u = projected_homo[:, 0] / projected_homo[:, 2]
    v = projected_homo[:, 1] / projected_homo[:, 2]
    
    pixels = np.stack([u, v], axis=1).astype(np.int32)
    
    # 2. 画家算法 (深度排序)
    face_depths = Z[faces]
    avg_depths = np.mean(face_depths, axis=1)
    sort_idx = np.argsort(avg_depths)[::-1] # 从远到近
    
    # 3. 绘制
    for idx in sort_idx:
        face = faces[idx]
        pts = pixels[face]
        
        # 简单的视锥剔除：如果有点在图像范围内才画（可选，OpenCV fillConvexPoly 会自动处理越界，但这样稍微快点）
        # if np.any((pts[:,0] >= 0) & (pts[:,0] < w_img) & (pts[:,1] >= 0) & (pts[:,1] < h_img)):
        cv2.fillConvexPoly(overlay, pts, color)
        # cv2.polylines(overlay, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. 融合
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return result

def main():
    parser = argparse.ArgumentParser(description="批量投影 OBJ 到图像文件夹")
    
    # 修改默认路径为 文件夹路径
    default_rgb_dir = '/home/pt/fbs/FoundationPoseROS2/FoundationPose/demo_data/ours/rgb'
    default_intrinsics = '/home/pt/fbs/FoundationPoseROS2/FoundationPose/demo_data/ours/cam_K.txt'
    default_obj_dir = '/home/pt/fbs/test/manopara_obj'
    default_output_dir = '/home/pt/fbs/test/projection_results'

    parser.add_argument('--img_dir', type=str, default=default_rgb_dir, help="包含原始图片的文件夹")
    parser.add_argument('--obj_dir', type=str, default=default_obj_dir, help="包含OBJ文件的文件夹")
    parser.add_argument('--intrinsics', type=str, default=default_intrinsics, help="内参文件路径 (假设所有图片共用)")
    parser.add_argument('--out_dir', type=str, default=default_output_dir, help="结果输出文件夹")

    args = parser.parse_args()

    # 0. 准备工作
    print(f"--- 开始批量处理 ---")
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"创建输出目录: {args.out_dir}")

    print("加载内参...")
    K = load_intrinsics(args.intrinsics)
    if K is None: sys.exit(1)

    # 1. 获取所有图片路径 (支持 jpg, png, jpeg)
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_paths = []
    for ext in extensions:
        # 使用 glob 搜索
        img_paths.extend(glob.glob(os.path.join(args.img_dir, ext)))
        # 大小写兼容
        img_paths.extend(glob.glob(os.path.join(args.img_dir, ext.upper())))
    
    img_paths = sorted(list(set(img_paths))) # 去重并排序
    
    if len(img_paths) == 0:
        print(f"[Error] 在 {args.img_dir} 中未找到图片。")
        sys.exit(1)

    print(f"共发现 {len(img_paths)} 张图片，开始处理...")

    # 2. 批量循环
    success_count = 0
    
    for img_path in tqdm(img_paths, desc="Processing"):
        # 获取文件名 (不带后缀) e.g. "000000"
        file_name_with_ext = os.path.basename(img_path)
        file_name = os.path.splitext(file_name_with_ext)[0]
        
        # 构造对应的 OBJ 路径
        obj_path = os.path.join(args.obj_dir, f"{file_name}.obj")
        
        # 构造输出路径 (保持同名，后缀存为 jpg)
        save_path = os.path.join(args.out_dir, file_name + ".jpg")

        # 检查 OBJ 是否存在
        if not os.path.exists(obj_path):
            # 这种情况很正常（比如这一帧没检测到手，就没有OBJ），跳过即可
            # print(f"Skipping {file_name}: OBJ not found")
            continue

        # 加载 OBJ
        vertices, faces = load_obj(obj_path)
        if vertices is None or len(vertices) == 0:
            print(f"Skipping {file_name}: OBJ empty or invalid")
            continue

        # 加载图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {file_name}: Image load failed")
            continue

        try:
            # 投影绘制
            # 你可以在这里根据左右手修改颜色，如果文件名包含 _left 或 _right 的话
            color = (0, 255, 0) # 默认绿色
            res_img = project_and_draw(img, vertices, faces, K, alpha=0.6, color=color)
            
            # 保存
            cv2.imwrite(save_path, res_img)
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue

    print(f"--- 处理完成 ---")
    print(f"成功: {success_count}/{len(img_paths)}")
    print(f"结果保存在: {args.out_dir}")

if __name__ == "__main__":
    main()