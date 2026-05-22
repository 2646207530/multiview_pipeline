import os
import uuid

import cv2
import matplotlib
import seaborn as sns
import pyrender
import trimesh
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import torch
import numpy as np
from scipy import stats
from io import BytesIO
from ..utils.heatmap import sample_with_heatmap
from ..utils.transform import bchw_2_bhwc, denormalize
from .misc import COLOR_CONST


def fig_to_numpy(fig):
    """
    将matplotlib figure转换为numpy数组
    
    Args:
        fig: matplotlib figure对象
        
    Returns:
        np.array: 形状为 (H, W, 3) 的RGB图像数组（0-255范围，uint8类型）
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    # 读取图像并转换为0-255范围的整数
    img_array = plt.imread(buf)
    buf.close()
    
    # 处理Alpha通道（如果有）
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]  # 丢弃Alpha通道
    
    # 关键步骤：转换为0-255范围的uint8
    img_array = (img_array * 255).astype(np.uint8)
    
    # OpenCV使用BGR顺序，如需保存可以转换
    # img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    return img_array


def vis_confidence_error_scatter(confidences, errors, alpha=0.6, figsize=(12, 8)):
    """
    置信度-误差散点密度图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 左侧：散点图
    scatter = ax1.scatter(confidences, errors, alpha=alpha, s=10, c=errors, cmap='viridis')
    ax1.set_xlabel('Prediction Confidence')
    ax1.set_ylabel('Prediction Error (pixels)')
    ax1.set_title('Confidence vs Error Scatter Plot')
    plt.colorbar(scatter, ax=ax1, label='Error Magnitude')
    
    # 计算相关系数
    r, p_value = stats.pearsonr(confidences, errors)
    ax1.text(0.05, 0.95, f'Pearson r = {r:.3f}\np-value = {p_value:.2e}', 
             transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # 右侧：2D密度图
    sns.kdeplot(x=confidences, y=errors, ax=ax2, fill=True, cmap='Blues', alpha=0.7)
    ax2.set_xlabel('Prediction Confidence')
    ax2.set_ylabel('Prediction Error (pixels)')
    ax2.set_title('Confidence vs Error Density')
    
    plt.tight_layout()
    fig = fig_to_numpy(fig)
    plt.close()
    return fig


def visualize_batch_3d_hand_mixed(hand_with_skeleton_batch, hand_only_points_batch, n_sample=16, img_size=(480, 640)):
    """
    批量可视化带骨骼的手部关键点和仅散点的手部关键点，返回可用于TensorBoard的图像数组。
    
    参数:
    hand_with_skeleton_batch: numpy array或torch tensor，形状为 (B, 21, 3)，带骨骼的手部关键点批次
    hand_only_points_batch: numpy array或torch tensor，形状为 (B, 21, 3)，仅散点的手部关键点批次
    n_sample: int，最大可视化样本数（不超过批次大小）
    img_size: tuple (H, W)，输出图像的高和宽
    
    返回:
    sample_array: numpy array，形状为 (n_sample, H, W, 3)，可直接传入TensorBoard的add_images
    """
    # 骨骼连接边（仅用于带骨骼的手部）
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], 
        [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], 
        [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]
    ]
    
    # 处理输入类型（转为numpy数组）
    if hasattr(hand_with_skeleton_batch, 'cpu'):
        hand_with_skeleton_batch = hand_with_skeleton_batch.detach().cpu().numpy()
    if hasattr(hand_only_points_batch, 'cpu'):
        hand_only_points_batch = hand_only_points_batch.detach().cpu().numpy()
    
    # 确定实际可视化样本数
    batch_size = hand_with_skeleton_batch.shape[0]
    n_sample = min(n_sample, batch_size)
    
    # 提取需要可视化的样本
    hand_with_skeleton = hand_with_skeleton_batch[:n_sample]  # (n_sample, 21, 3)
    hand_only_points = hand_only_points_batch[:n_sample]      # (n_sample, 21, 3)
    
    sample_list = []
    H, W = img_size
    dpi = 100  # 控制图像分辨率（与img_size配合）
    min_alpha = 0.2
    max_alpha = 0.8
    
    for i in range(n_sample):
        # 创建3D图形（使用Agg后端避免弹出窗口）
        fig = plt.figure(figsize=(W/dpi, H/dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        canvas = FigureCanvasAgg(fig)  # 用于将图像转为数组
        
        # 获取当前样本的关键点
        skeleton_hand = hand_with_skeleton[i]  # (21, 3)
        scatter_hand = hand_only_points[i]     # (21, 3)
        
        # 绘制带骨骼的手部：先画骨骼，再画关键点（同色）
        # 绘制骨骼
        for edge in edges:
            x = [skeleton_hand[edge[0], 0], skeleton_hand[edge[1], 0]]
            y = [skeleton_hand[edge[0], 1], skeleton_hand[edge[1], 1]]
            z = [skeleton_hand[edge[0], 2], skeleton_hand[edge[1], 2]]
            ax.plot(x, y, z, color='g', linewidth=2)  # 绿色骨骼
        
        # 绘制带骨骼的关键点
        ax.scatter(
            skeleton_hand[:, 0], skeleton_hand[:, 1], skeleton_hand[:, 2],
            color='r', s=40, alpha=1.0, label='Skeleton Hand'  # 红色点（与骨骼同色）
        )

        # # 提取Z轴坐标（深度信息）
        # z_coords = scatter_hand[:, 2]
        
        # # 归一化Z值到[0,1]范围（用于映射透明度）
        # z_min, z_max = z_coords.min(), z_coords.max()
        # if z_max - z_min < 1e-6:  # 避免所有点Z值相同导致除零
        #     alphas = np.ones_like(z_coords) * ((min_alpha + max_alpha) / 2)
        # else:
        #     # 归一化后：z值越大（越近），alpha越接近max_alpha；z值越小（越远），alpha越接近min_alpha
        #     z_norm = (z_coords - z_min) / (z_max - z_min)
        #     alphas = min_alpha + (max_alpha - min_alpha) * z_norm
        
        # 绘制仅散点的手部（无骨骼）
        ax.scatter(
            scatter_hand[:, 0], scatter_hand[:, 1], scatter_hand[:, 2],
            color='b', s=15, alpha=0.5, label='Scatter Hand'  # 蓝色点（统一颜色）
        )
        
        # 设置坐标轴和图例（保持可视化清晰）
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.legend(fontsize=6)
        
        # 调整视角（固定视角确保批量图像一致性）
        ax.view_init(elev=30, azim=45)
        
        # 渲染图像并转为numpy数组
        canvas.draw()
        img_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(H, W, 3)  # (H, W, 3)
        
        # 添加边框（与参考函数风格一致）
        img_array = cv2.copyMakeBorder(
            img_array, 5, 5, 5, 5, 
            cv2.BORDER_CONSTANT, 
            value=(255, 255, 255)  # 白色边框
        )
        
        # 加入样本列表
        sample_list.append(img_array[None, ...])  # 增加批次维度
        
        # 清理当前图形，避免内存泄漏
        plt.close(fig)
    
    # 拼接所有样本为批次数组
    sample_array = np.concatenate(sample_list, axis=0)  # (n_sample, H+10, W+10, 3)（+10是边框）
    return sample_array


def visualize_batch_3d_hand(hand_with_skeleton_batch, n_sample=16, img_size=(480, 640)):
    """
    批量可视化带骨骼的手部关键点和仅散点的手部关键点，返回可用于TensorBoard的图像数组。

    参数:
    hand_with_skeleton_batch: numpy array或torch tensor，形状为 (B, 21, 3)，带骨骼的手部关键点批次
    hand_only_points_batch: numpy array或torch tensor，形状为 (B, 21, 3)，仅散点的手部关键点批次
    n_sample: int，最大可视化样本数（不超过批次大小）
    img_size: tuple (H, W)，输出图像的高和宽

    返回:
    sample_array: numpy array，形状为 (n_sample, H, W, 3)，可直接传入TensorBoard的add_images
    """
    # 骨骼连接边（仅用于带骨骼的手部）
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15],
        [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]
    ]

    # 处理输入类型（转为numpy数组）
    if hasattr(hand_with_skeleton_batch, 'cpu'):
        hand_with_skeleton_batch = hand_with_skeleton_batch.detach().cpu().numpy()

    # 确定实际可视化样本数
    batch_size = hand_with_skeleton_batch.shape[0]
    n_sample = min(n_sample, batch_size)

    # 提取需要可视化的样本
    hand_with_skeleton = hand_with_skeleton_batch[:n_sample]  # (n_sample, 21, 3)

    sample_list = []
    H, W = img_size
    dpi = 100  # 控制图像分辨率（与img_size配合）
    min_alpha = 0.2
    max_alpha = 0.8

    for i in range(n_sample):
        # 创建3D图形（使用Agg后端避免弹出窗口）
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        canvas = FigureCanvasAgg(fig)  # 用于将图像转为数组

        # 获取当前样本的关键点
        skeleton_hand = hand_with_skeleton[i]  # (21, 3)

        # 绘制带骨骼的手部：先画骨骼，再画关键点（同色）
        # 绘制骨骼
        for edge in edges:
            x = [skeleton_hand[edge[0], 0], skeleton_hand[edge[1], 0]]
            y = [skeleton_hand[edge[0], 1], skeleton_hand[edge[1], 1]]
            z = [skeleton_hand[edge[0], 2], skeleton_hand[edge[1], 2]]
            ax.plot(x, y, z, color='g', linewidth=2)  # 绿色骨骼

        # 绘制带骨骼的关键点
        ax.scatter(
            skeleton_hand[:, 0], skeleton_hand[:, 1], skeleton_hand[:, 2],
            color='r', s=40, alpha=1.0, label='Skeleton Hand'  # 红色点（与骨骼同色）
        )

        # # 提取Z轴坐标（深度信息）
        # z_coords = scatter_hand[:, 2]

        # # 归一化Z值到[0,1]范围（用于映射透明度）
        # z_min, z_max = z_coords.min(), z_coords.max()
        # if z_max - z_min < 1e-6:  # 避免所有点Z值相同导致除零
        #     alphas = np.ones_like(z_coords) * ((min_alpha + max_alpha) / 2)
        # else:
        #     # 归一化后：z值越大（越近），alpha越接近max_alpha；z值越小（越远），alpha越接近min_alpha
        #     z_norm = (z_coords - z_min) / (z_max - z_min)
        #     alphas = min_alpha + (max_alpha - min_alpha) * z_norm



        # 设置坐标轴和图例（保持可视化清晰）
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.legend(fontsize=6)

        # 调整视角（固定视角确保批量图像一致性）
        ax.view_init(elev=30, azim=45)

        # 渲染图像并转为numpy数组
        canvas.draw()
        img_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(H, W, 3)  # (H, W, 3)

        # 添加边框（与参考函数风格一致）
        img_array = cv2.copyMakeBorder(
            img_array, 5, 5, 5, 5,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255)  # 白色边框
        )

        # 加入样本列表
        sample_list.append(img_array[None, ...])  # 增加批次维度

        # 清理当前图形，避免内存泄漏
        plt.close(fig)

    # 拼接所有样本为批次数组
    sample_array = np.concatenate(sample_list, axis=0)  # (n_sample, H+10, W+10, 3)（+10是边框）
    return sample_array


def visualize_3d_hand_keypoints(hand1_points, hand2_points, save_path=None):
    """
    可视化两组手部关键点及其骨骼结构。
    
    参数:
    hand1_points: numpy array, 形状为 (21, 3)，表示第一组手部关键点的三维坐标。
    hand2_points: numpy array, 形状为 (21, 3)，表示第二组手部关键点的三维坐标。
    """
    # 骨骼连接的边
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], 
        [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], 
        [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]
    ]

    # 创建一个3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    hand1_points = hand1_points.cpu().numpy()
    hand2_points = hand2_points.cpu().numpy()

    # 绘制手部1的关键点和骨骼连接
    for edge in edges:
        x_vals = [hand1_points[edge[0], 0], hand1_points[edge[1], 0]]
        y_vals = [hand1_points[edge[0], 1], hand1_points[edge[1], 1]]
        z_vals = [hand1_points[edge[0], 2], hand1_points[edge[1], 2]]
        ax.plot(x_vals, y_vals, z_vals, color='b')  # 用蓝色绘制手部1的骨骼

    # 绘制手部2的关键点和骨骼连接
    for edge in edges:
        x_vals = [hand2_points[edge[0], 0], hand2_points[edge[1], 0]]
        y_vals = [hand2_points[edge[0], 1], hand2_points[edge[1], 1]]
        z_vals = [hand2_points[edge[0], 2], hand2_points[edge[1], 2]]
        ax.plot(x_vals, y_vals, z_vals, color='r')  # 用红色绘制手部2的骨骼

    # 绘制手部1的关键点
    ax.scatter(hand1_points[:, 0], hand1_points[:, 1], hand1_points[:, 2], color='b', label='Hand 1')

    # 绘制手部2的关键点
    ax.scatter(hand2_points[:, 0], hand2_points[:, 1], hand2_points[:, 2], color='r', label='Hand 2')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置图例
    ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # # 显示图形
    # plt.show()
    # plt.pause(60)
    plt.close()


def draw_batch_mesh_images(verts3d, gt_verts3d, face, intr, tensor_image, step_idx, n_sample=16):
    batch_size = verts3d.shape[0]
    if n_sample >= batch_size:
        n_sample = batch_size

    tensor_image = tensor_image[:n_sample, ...].detach().cpu()
    image = bchw_2_bhwc(denormalize(tensor_image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False))
    image = image.mul_(255.0).numpy().astype(np.uint8)  # (B, H, W, 3)

    verts3d = verts3d[:n_sample, ...].detach().cpu().numpy()  # (B, NJ, 2)
    gt_verts3d = gt_verts3d[:n_sample, ...].detach().cpu().numpy()  # (B, NJ, 2)
    intr = intr[:n_sample, ...].detach().cpu().numpy()  # (B, 3, 3)

    sample_list = []
    for i in range(n_sample):
        verts3d_i = verts3d[i].copy()
        gt_verts3d_i = gt_verts3d[i].copy()
        intr_i = intr[i].copy()

        pred_mesh_img = draw_mesh(image[i].copy(), intr_i, verts3d_i, face)
        gt_mesh_img = draw_mesh(image[i].copy(), intr_i, gt_verts3d_i, face)

        sample = np.hstack([pred_mesh_img, gt_mesh_img])
        sample = cv2.copyMakeBorder(sample, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        sample = cv2.cvtColor(sample, cv2.COLOR_RGBA2RGB)
        sample_list.append(sample[None, ...])

    # draw finished
    sample_array = np.concatenate(sample_list, axis=0)  # (B, H, W, C)
    return sample_array


def draw_batch_verts_images(verts2d, gt_verts2d, tensor_image, step_idx, n_sample=16):
    batch_size = verts2d.shape[0]
    if n_sample >= batch_size:
        n_sample = batch_size

    tensor_image = tensor_image[:n_sample, ...].detach().cpu()
    image = bchw_2_bhwc(denormalize(tensor_image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False))
    image = image.mul_(255.0).numpy().astype(np.uint8)  # (B, H, W, 3)

    verts2d = verts2d[:n_sample, ...].detach().cpu().numpy()  # (B, NJ, 2)
    gt_verts2d = gt_verts2d[:n_sample, ...].detach().cpu().numpy()  # (B, NJ, 2)

    sample_list = []
    for i in range(n_sample):
        sample_img = image[i].copy()
        for j in range(verts2d[i].shape[0]):
            cx = int(verts2d[i, j, 0])
            cy = int(verts2d[i, j, 1])
            cv2.circle(sample_img, (cx, cy), radius=1, thickness=-1, color=np.array([1.0, 1.0, 0.0]) * 255)

        sample_img_2 = image[i].copy()
        for j in range(gt_verts2d[i].shape[0]):
            cx = int(gt_verts2d[i, j, 0])
            cy = int(gt_verts2d[i, j, 1])
            cv2.circle(sample_img_2, (cx, cy), radius=1, thickness=-1, color=np.array([1.0, 0.0, 0.0]) * 255)

        sample = np.hstack([sample_img, sample_img_2])
        sample = cv2.copyMakeBorder(sample, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        sample_list.append(sample[None, ...])

    # draw finished
    sample_array = np.concatenate(sample_list, axis=0)  # (B, H, W, C)
    return sample_array


def draw_batch_joint_images(joints2d, gt_joints2d, tensor_image, step_idx, n_sample=16):
    batch_size = joints2d.shape[0]
    if n_sample >= batch_size:
        n_sample = batch_size

    tensor_image = tensor_image[:n_sample, ...].detach().cpu()
    image = bchw_2_bhwc(denormalize(tensor_image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False))
    image = image.mul_(255.0).numpy().astype(np.uint8)  # (B, H, W, 3)

    joints2d = joints2d[:n_sample, ...].detach().cpu().numpy()  # (B, NJ, 2)
    gt_joints2d = gt_joints2d[:n_sample, ...].detach().cpu().numpy()  # (B, NJ, 2)

    sample_list = []
    for i in range(n_sample):
        joints_img =plot_body(image[i].copy(), joints2d[i]) #plot_hand(image[i].copy(), joints2d[i])
        gt_joints_img = plot_body(image[i].copy(), gt_joints2d[i])#plot_hand(image[i].copy(), gt_joints2d[i])
        sample = np.hstack([joints_img, gt_joints_img])
        sample = cv2.copyMakeBorder(sample, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        sample_list.append(sample[None, ...])

    # draw finished
    sample_array = np.concatenate(sample_list, axis=0)  # (B, H, W, C)
    return sample_array


def draw_batch_gt_images(gt_joints2d, tensor_image, pose_xyz, step_idx, n_sample=12):
    batch_size = gt_joints2d.shape[0]
    if n_sample >= batch_size:
        n_sample = batch_size

    tensor_image = tensor_image[:n_sample, ...].detach().cpu()
    image = bchw_2_bhwc(denormalize(tensor_image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False))
    image = image.mul_(255.0).numpy().astype(np.uint8)  # (B, H, W, 3)

    gt_joints2d = gt_joints2d[:n_sample, ...].detach().cpu().numpy()  # (B, NJ, 2)

    sample_list = []
    for i in range(n_sample):
        ori_img = image[i].copy()
        skeleton_3d = draw_3d_skeleton(ori_img.shape[:2], joints_xyz=pose_xyz[i])
        gt_joints_img = plot_hand(image[i].copy(), gt_joints2d[i])
        gt_joints_img_nbg = plot_hand_on_white(image[i].copy(), gt_joints2d[i])
        sample = np.hstack([ori_img, gt_joints_img, gt_joints_img_nbg, skeleton_3d[..., :3]])
        sample = cv2.copyMakeBorder(sample, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        sample_list.append(sample[None, ...])

    # draw finished
    sample_array = np.concatenate(sample_list, axis=0)  # (B, H, W, C)
    return sample_array


def plot_image_joints_mask(image, joints2d, mask):
    joints_img = plot_hand(image.copy(), joints2d)
    mask = mask[:, :, None].repeat(3, axis=2)
    mask = cv2.resize(mask, image.shape[:2])
    img_mask = cv2.addWeighted(image, 0.3, mask, 0.7, 0)
    comb_img = np.hstack([image, joints_img, img_mask])
    return comb_img


def plot_image_heatmap_mask(image, heatmap, mask):
    img_heatmap = sample_with_heatmap(image, heatmap)

    mask = mask[:, :, None].repeat(3, axis=2)
    mask = cv2.resize(mask, image.shape[:2])
    img_mask = cv2.addWeighted(image, 0.3, mask, 0.7, 0)
    comb_img = np.hstack([img_mask, img_heatmap])
    return comb_img


def imdesc(image, desc=""):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, desc, (10, 30), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def plot_body(image, coords_hw, vis=None, linewidth=2):
    """Plots a COCO 17-keypoint human stick figure into an image."""

    # 定义左、右、中轴的颜色 (BGR 格式用于 OpenCV)
    COLOR_LEFT = (0, 0, 255)  # 红色 (左手、左腿)
    COLOR_RIGHT = (255, 0, 0)  # 蓝色 (右手、右腿)
    COLOR_CENTER = (0, 255, 255)  # 黄色 (躯干连线)

    # COCO 17 骨架连线定义及对应的绘制颜色
    bones = [
        ((15, 13), COLOR_LEFT),  # 左小腿
        ((13, 11), COLOR_LEFT),  # 左大腿
        ((16, 14), COLOR_RIGHT),  # 右小腿
        ((14, 12), COLOR_RIGHT),  # 右大腿
        ((11, 12), COLOR_CENTER),  # 骨盆连线
        ((5, 11), COLOR_LEFT),  # 左躯干
        ((6, 12), COLOR_RIGHT),  # 右躯干
        ((5, 6), COLOR_CENTER),  # 肩膀连线
        ((5, 7), COLOR_LEFT),  # 左大臂
        ((7, 9), COLOR_LEFT),  # 左小臂
        ((6, 8), COLOR_RIGHT),  # 右大臂
        ((8, 10), COLOR_RIGHT),  # 右小臂
        ((1, 2), COLOR_CENTER),  # 双眼连线
        ((0, 1), COLOR_LEFT),  # 鼻子到左眼
        ((0, 2), COLOR_RIGHT),  # 鼻子到右眼
        ((1, 3), COLOR_LEFT),  # 左眼到左耳
        ((2, 4), COLOR_RIGHT),  # 右眼到右耳
        ((3, 5), COLOR_LEFT),  # 左耳到左肩
        ((4, 6), COLOR_RIGHT)  # 右耳到右肩
    ]

    if vis is None:
        vis = np.ones_like(coords_hw[:, 0]) == 1.0

    # 1. 绘制骨架连线
    for connection, color in bones:
        pt1, pt2 = connection[0], connection[1]

        # 鲁棒性检查：防止越界或点不可见
        if pt1 >= len(vis) or pt2 >= len(vis):
            continue
        if (vis[pt1] == False) or (vis[pt2] == False):
            continue

        coord1 = coords_hw[pt1, :]
        coord2 = coords_hw[pt2, :]
        c1x, c1y = int(coord1[0]), int(coord1[1])
        c2x, c2y = int(coord2[0]), int(coord2[1])
        cv2.line(image, (c1x, c1y), (c2x, c2y), color=color, thickness=linewidth)

    # 2. 绘制关节点
    for i in range(coords_hw.shape[0]):
        if i >= len(vis) or vis[i] == False:
            continue

        cx, cy = int(coords_hw[i, 0]), int(coords_hw[i, 1])

        # 根据左右区分关节点颜色
        if i in [1, 3, 5, 7, 9, 11, 13, 15]:
            node_color = COLOR_LEFT
        elif i in [2, 4, 6, 8, 10, 12, 14, 16]:
            node_color = COLOR_RIGHT
        else:
            node_color = (0, 255, 0)  # 中心点(鼻子)用绿色高亮

        cv2.circle(image, (cx, cy), radius=3 * linewidth, thickness=-1, color=node_color)

    return image

def plot_hand(image, coords_hw, vis=None, linewidth=1):
    """Plots a hand stick figure into a matplotlib figure."""

    colors = np.array(COLOR_CONST.color_hand_joints)
    colors = colors[:, ::-1]

    # define connections and colors of the bones
    bones = [
        ((0, 1), colors[1, :]),
        ((1, 2), colors[2, :]),
        ((2, 3), colors[3, :]),
        ((3, 4), colors[4, :]),
        ((0, 5), colors[5, :]),
        ((5, 6), colors[6, :]),
        ((6, 7), colors[7, :]),
        ((7, 8), colors[8, :]),
        ((0, 9), colors[9, :]),
        ((9, 10), colors[10, :]),
        ((10, 11), colors[11, :]),
        ((11, 12), colors[12, :]),
        ((0, 13), colors[13, :]),
        ((13, 14), colors[14, :]),
        ((14, 15), colors[15, :]),
        ((15, 16), colors[16, :]),
        ((0, 17), colors[17, :]),
        ((17, 18), colors[18, :]),
        ((18, 19), colors[19, :]),
        ((19, 20), colors[20, :]),
    ]

    if vis is None:
        vis = np.ones_like(coords_hw[:, 0]) == 1.0

    for connection, color in bones:
        if (vis[connection[0]] == False) or (vis[connection[1]] == False):
            continue

        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        c1x = int(coord1[0])
        c1y = int(coord1[1])
        c2x = int(coord2[0])
        c2y = int(coord2[1])
        cv2.line(image, (c1x, c1y), (c2x, c2y), color=color * 255, thickness=linewidth)

    for i in range(coords_hw.shape[0]):
        cx = int(coords_hw[i, 0])
        cy = int(coords_hw[i, 1])
        cv2.circle(image, (cx, cy), radius=3 * linewidth, thickness=-1, color=colors[i, :] * 255)

    return image


def plot_hand_on_white(image, coords_hw, vis=None, linewidth=3, image_size=(256, 256)):
    """Plots a hand stick figure on a white background."""
    # Create a white image (255 = 纯白)
    image = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 255
    
    colors = np.array(COLOR_CONST.color_hand_joints)
    colors = colors[:, ::-1]  # BGR to RGB（如果COLOR_CONST是RGB格式，可以去掉这行）
    
    # 定义骨骼连接和颜色（和原代码一致）
    bones = [
        ((0, 1), colors[1, :]),
        ((1, 2), colors[2, :]),
        ((2, 3), colors[3, :]),
        ((3, 4), colors[4, :]),
        ((0, 5), colors[5, :]),
        ((5, 6), colors[6, :]),
        ((6, 7), colors[7, :]),
        ((7, 8), colors[8, :]),
        ((0, 9), colors[9, :]),
        ((9, 10), colors[10, :]),
        ((10, 11), colors[11, :]),
        ((11, 12), colors[12, :]),
        ((0, 13), colors[13, :]),
        ((13, 14), colors[14, :]),
        ((14, 15), colors[15, :]),
        ((15, 16), colors[16, :]),
        ((0, 17), colors[17, :]),
        ((17, 18), colors[18, :]),
        ((18, 19), colors[19, :]),
        ((19, 20), colors[20, :]),
    ]
    
    if vis is None:
        vis = np.ones_like(coords_hw[:, 0]) == 1.0
    
    # 画骨骼连接线
    for connection, color in bones:
        if (vis[connection[0]] == False) or (vis[connection[1]] == False):
            continue
        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        c1x, c1y = int(coord1[0]), int(coord1[1])
        c2x, c2y = int(coord2[0]), int(coord2[1])
        cv2.line(image, (c1x, c1y), (c2x, c2y), color=color * 255, thickness=linewidth)
    
    # 画关键点
    for i in range(coords_hw.shape[0]):
        cx, cy = int(coords_hw[i, 0]), int(coords_hw[i, 1])
        cv2.circle(image, (cx, cy), radius=3 * linewidth, thickness=-1, color=colors[i, :] * 255)
    
    return image


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def draw_mesh(image, cam_param, mesh_xyz, face):
    """
    :param image: H x W x 3
    :param cam_param: 1 x 3 x 3
    :param mesh_xyz: 778 x 3
    :param face: 1538 x 3 x 2
    :return:
    """
    vertex2uv = np.matmul(cam_param, mesh_xyz.T).T
    vertex2uv = (vertex2uv / vertex2uv[:, 2:3])[:, :2].astype(np.int)

    fig = plt.figure()
    fig.set_size_inches(float(image.shape[0]) / fig.dpi, float(image.shape[1]) / fig.dpi, forward=True)
    plt.imshow(image)
    plt.axis('off')
    if face is None:
        plt.plot(vertex2uv[:, 0], vertex2uv[:, 1], 'o', color='green', markersize=1)
    else:
        plt.triplot(vertex2uv[:, 0], vertex2uv[:, 1], face, lw=0.5, color='orange')

    plt.subplots_adjust(left=0., right=1., top=1., bottom=0, wspace=0, hspace=0)

    ret = fig2data(fig)
    plt.close(fig)

    return ret


def draw_2d_skeleton(image, joints_uv=None, corners_uv=None):
    """
    :param image: H x W x 3
    :param joints_uv: 21 x 2
    wrist,
    thumb_mcp, thumb_pip, thumb_dip, thumb_tip
    index_mcp, index_pip, index_dip, index_tip,
    middle_mcp, middle_pip, middle_dip, middle_tip,
    ring_mcp, ring_pip, ring_dip, ring_tip,
    little_mcp, little_pip, little_dip, little_tip
    :return:
    """
    skeleton_overlay = image.copy()
    # skeleton_overlay = skeleton_overlay[:, :, (2, 1, 0)]
    # skeleton_overlay = (skeleton_overlay * 255).astype("float32")
    # skeleton_overlay = skeleton_overlay.copy()

    if corners_uv is not None:
        for corner_idx in range(corners_uv.shape[0]):
            corner = corners_uv[corner_idx, 0].astype("int32"), corners_uv[corner_idx, 1].astype("int32")
            cv2.circle(
                skeleton_overlay,
                corner,
                radius=1,
                color=(255, 0, 0),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
        # draw 12 segments
        #  [0, 1, 3, 2, 0], [4, 5, 7, 6, 4], [1, 5], [2, 6], [3, 7], [0, 4]
        b_list = [0, 1, 3, 2, 0]
        for curr_id, next_id in zip(b_list[:-1], b_list[1:]):
            cv2.line(
                skeleton_overlay,
                tuple(corners_uv[curr_id, :].astype("int32")),
                tuple(corners_uv[next_id, :].astype("int32")),
                color=[255, 0, 0],
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        g_list = [4, 5, 7, 6, 4]
        for curr_id, next_id in zip(g_list[:-1], g_list[1:]):
            cv2.line(
                skeleton_overlay,
                tuple(corners_uv[curr_id, :].astype("int32")),
                tuple(corners_uv[next_id, :].astype("int32")),
                color=[0, 128, 0],
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        lb_list = [[1, 5], [2, 6], [3, 7], [0, 4]]
        for curr_id, next_id in lb_list:
            cv2.line(
                skeleton_overlay,
                tuple(corners_uv[curr_id, :].astype("int32")),
                tuple(corners_uv[next_id, :].astype("int32")),
                color=[192, 192, 0],
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    if joints_uv is not None:
        assert joints_uv.shape[0] == 21
        marker_sz = 6
        line_wd = 3
        root_ind = 0

        for joint_ind in range(joints_uv.shape[0]):
            joint = joints_uv[joint_ind, 0].astype("int32"), joints_uv[joint_ind, 1].astype("int32")
            cv2.circle(
                skeleton_overlay,
                joint,
                radius=marker_sz,
                color=COLOR_CONST.color_hand_joints[joint_ind] * np.array(255),
                thickness=-1,
                lineType=cv2.CV_AA if cv2.__version__.startswith("2") else cv2.LINE_AA,
            )
            if joint_ind == 0:
                continue
            elif joint_ind % 4 == 1:
                root_joint = joints_uv[root_ind, 0].astype("int32"), joints_uv[root_ind, 1].astype("int32")
                cv2.line(
                    skeleton_overlay,
                    root_joint,
                    joint,
                    color=COLOR_CONST.color_hand_joints[joint_ind] * np.array(255),
                    thickness=int(line_wd),
                    lineType=cv2.CV_AA if cv2.__version__.startswith("2") else cv2.LINE_AA,
                )
            else:
                joint_2 = joints_uv[joint_ind - 1, 0].astype("int32"), joints_uv[joint_ind - 1, 1].astype("int32")
                cv2.line(
                    skeleton_overlay,
                    joint_2,
                    joint,
                    color=COLOR_CONST.color_hand_joints[joint_ind] * np.array(255),
                    thickness=int(line_wd),
                    lineType=cv2.CV_AA if cv2.__version__.startswith("2") else cv2.LINE_AA,
                )

    return skeleton_overlay


def axis_equal_3d(ax, ratio=1.2):
    extents = np.array([getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz)) * ratio
    r = maxsize / 2
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, "set_{}lim".format(dim))(ctr - r, ctr + r)


def draw_3d_skeleton(image_size, joints_xyz=None, corners_xyz=None):
    """
    :param joints_xyz: 21 x 3
    :param image_size: H, W
    :return:
    """
    fig = plt.figure()
    fig.set_size_inches(float(image_size[0]) / fig.dpi, float(image_size[1]) / fig.dpi, forward=True)

    ax = plt.subplot(111, projection="3d")

    if corners_xyz is not None:
        b_list = [0, 1, 3, 2, 0]
        for curr_id, next_id in zip(b_list[:-1], b_list[1:]):
            ax.plot(
                corners_xyz[(curr_id, next_id), 0],
                corners_xyz[(curr_id, next_id), 1],
                corners_xyz[(curr_id, next_id), 2],
                color=[255 / 255, 0, 0],
                linewidth=2,
            )

        g_list = [4, 5, 7, 6, 4]
        for curr_id, next_id in zip(g_list[:-1], g_list[1:]):
            ax.plot(
                corners_xyz[(curr_id, next_id), 0],
                corners_xyz[(curr_id, next_id), 1],
                corners_xyz[(curr_id, next_id), 2],
                color=[0, 128 / 255, 0],
                linewidth=2,
            )

        lb_list = [[1, 5], [2, 6], [3, 7], [0, 4]]
        for curr_id, next_id in lb_list:
            ax.plot(
                corners_xyz[(curr_id, next_id), 0],
                corners_xyz[(curr_id, next_id), 1],
                corners_xyz[(curr_id, next_id), 2],
                color=[192 / 255, 192 / 255, 0],
                linewidth=2,
            )

    if joints_xyz is not None:
        assert joints_xyz.shape[0] == 21
        marker_sz = 11
        line_wd = 2
        for joint_ind in range(joints_xyz.shape[0]):
            ax.plot(
                joints_xyz[joint_ind:joint_ind + 1, 0],
                joints_xyz[joint_ind:joint_ind + 1, 1],
                joints_xyz[joint_ind:joint_ind + 1, 2],
                ".",
                c=COLOR_CONST.color_hand_joints[joint_ind],
                markersize=marker_sz,
            )
            if joint_ind == 0:
                continue
            elif joint_ind % 4 == 1:
                ax.plot(
                    joints_xyz[[0, joint_ind], 0],
                    joints_xyz[[0, joint_ind], 1],
                    joints_xyz[[0, joint_ind], 2],
                    color=COLOR_CONST.color_hand_joints[joint_ind],
                    linewidth=line_wd,
                )
            else:
                ax.plot(
                    joints_xyz[[joint_ind - 1, joint_ind], 0],
                    joints_xyz[[joint_ind - 1, joint_ind], 1],
                    joints_xyz[[joint_ind - 1, joint_ind], 2],
                    color=COLOR_CONST.color_hand_joints[joint_ind],
                    linewidth=line_wd,
                )

    ax.view_init(elev=50, azim=-50)
    axis_equal_3d(ax)
    # turn off ticklabels
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    plt.subplots_adjust(left=-0.06, right=0.98, top=0.93, bottom=-0.07, wspace=0, hspace=0)

    ret = fig2data(fig)
    plt.close(fig)
    return ret


def draw_3d_mesh_mayavi(image_size, hand_xyz=None, hand_face=None, obj_xyz=None, obj_face=None, ratio=400 / 224):
    from mayavi import mlab

    mlab.options.offscreen = True
    cache_path = COLOR_CONST.mayavi_cache_path
    tempfile_name = "{}.png".format(str(uuid.uuid1()))
    os.makedirs(cache_path, exist_ok=True)

    # generate 400 x 400 fig
    tmp_img_size = (int(image_size[0] * ratio), int(image_size[1] * ratio))
    mlab_fig = mlab.figure(bgcolor=tuple(np.ones(3)), size=tmp_img_size)
    if hand_xyz is not None and hand_face is not None:
        mlab.triangular_mesh(
            hand_xyz[:, 0],
            hand_xyz[:, 1],
            hand_xyz[:, 2],
            np.array(hand_face),
            figure=mlab_fig,
            color=(0.4, 0.81960784, 0.95294118),
        )
    if obj_xyz is not None and obj_face is not None:
        mlab.triangular_mesh(
            obj_xyz[:, 0],
            obj_xyz[:, 1],
            obj_xyz[:, 2],
            np.array(obj_face),
            figure=mlab_fig,
            color=(1.0, 0.63921569, 0.6745098),
        )
    mlab.view(azimuth=-50, elevation=50, distance=0.6)
    mlab.savefig(os.path.join(cache_path, tempfile_name))
    mlab.close()

    # load by opencv and resize
    img = cv2.imread(os.path.join(cache_path, tempfile_name), cv2.IMREAD_COLOR)
    # resize to 224x224
    img = cv2.resize(img, image_size)
    os.remove(os.path.join(cache_path, tempfile_name))
    return img


def save_a_image_with_joints(image, cam_param, pose_uv, pose_xyz, file_name, padding=0, ret=False):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    skeleton_overlay = draw_2d_skeleton(image, joints_uv=pose_uv)
    skeleton_3d = draw_3d_skeleton(image.shape[:2], joints_xyz=pose_xyz)

    img_list = [skeleton_overlay, skeleton_3d]
    image_height = image.shape[0]
    image_width = image.shape[1]
    num_column = len(img_list)

    grid_image = np.zeros(((image_height + padding), num_column * (image_width + padding), 3), dtype=np.uint8)

    width_begin = 0
    width_end = image_width
    for show_img in img_list:
        grid_image[:, width_begin:width_end, :] = show_img[..., :3]
        width_begin += image_width + padding
        width_end = width_begin + image_width
    if ret:
        return grid_image

    cv2.imwrite(file_name, grid_image)


def save_a_image_with_mesh_joints(
    image,
    cam_param,
    mesh_xyz,
    face,
    pose_uv,
    pose_xyz,
    file_name,
    padding=0,
    ret=False,
    with_mayavi_mesh=True,
    with_skeleton_3d=True,
    renderer=None,
):
    frame = image.copy()
    rend_img_overlay = renderer(mesh_xyz, face, cam_param, img=frame)
    rend_img_overlay = cv2.cvtColor(rend_img_overlay, cv2.COLOR_RGB2BGR)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    skeleton_overlay = draw_2d_skeleton(image, joints_uv=pose_uv)

    img_list = [skeleton_overlay, rend_img_overlay]
    if with_mayavi_mesh:
        mesh_3d = draw_3d_mesh_mayavi(image.shape[:2], hand_xyz=mesh_xyz, hand_face=face)
        img_list.append(mesh_3d)
    if with_skeleton_3d:
        skeleton_3d = draw_3d_skeleton(image.shape[:2], joints_xyz=pose_xyz)
        img_list.append(skeleton_3d)

    image_height = image.shape[0]
    image_width = image.shape[1]
    num_column = len(img_list)

    grid_image = np.zeros(((image_height + padding), num_column * (image_width + padding), 3), dtype=np.uint8)

    width_begin = 0
    width_end = image_width
    for show_img in img_list:
        grid_image[:, width_begin:width_end, :] = show_img[..., :3]
        width_begin += image_width + padding
        width_end = width_begin + image_width
    if ret:
        return grid_image

    cv2.imwrite(file_name, grid_image)


def save_a_image_with_mesh_joints_objects(
    image,
    cam_param,
    mesh_xyz,
    face,
    pose_uv,
    pose_xyz,
    obj_mesh_xyz,
    obj_face,
    corners_uv,
    corners_xyz,
    file_name,
    padding=0,
    ret=False,
    renderer=None,
):
    frame = image.copy()
    frame1 = renderer(
        [mesh_xyz, obj_mesh_xyz],
        [face, obj_face],
        cam_param,
        img=frame,
        vertex_color=[np.array([102 / 255, 209 / 255, 243 / 255]),
                      np.array([255 / 255, 163 / 255, 172 / 255])],
    )
    rend_img_overlay = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)

    skeleton_overlay = draw_2d_skeleton(image, joints_uv=pose_uv, corners_uv=corners_uv)
    skeleton_3d = draw_3d_skeleton(image.shape[:2], joints_xyz=pose_xyz, corners_xyz=corners_xyz)
    mesh_3d = draw_3d_mesh_mayavi(image.shape[:2],
                                  hand_xyz=mesh_xyz,
                                  hand_face=face,
                                  obj_xyz=obj_mesh_xyz,
                                  obj_face=obj_face)

    img_list = [skeleton_overlay, rend_img_overlay, mesh_3d, skeleton_3d]
    image_height = image.shape[0]
    image_width = image.shape[1]
    num_column = len(img_list)

    grid_image = np.zeros(((image_height + padding), num_column * (image_width + padding), 3), dtype=np.uint8)

    width_begin = 0
    width_end = image_width
    for show_img in img_list:
        grid_image[:, width_begin:width_end, :] = show_img[..., :3]
        width_begin += image_width + padding
        width_end = width_begin + image_width
    if ret:
        return grid_image

    cv2.imwrite(file_name, grid_image)


def draw_batch_joint_images_all(joints2d_hm, joints2d_sv, joints2d_pse, joints2d_gt, tensor_image, confidence_hm, confidence_sv, confidences_pse, confidences_gt, step_idx, n_view=8, cols=2):
    """
    修改说明：
    1. 添加了cols参数控制每行显示的样本数
    2. 将所有样本排列到一张大图上
    """
    # batch_size = joints2d.shape[0]
    # if n_sample >= batch_size:
    #     n_sample = batch_size
    tensor_image = tensor_image[:n_view, ...].detach().cpu()
    image = bchw_2_bhwc(denormalize(tensor_image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False))
    image = image.mul_(255.0).numpy().astype(np.uint8)  # (N, H, W, 3)
    
    # 处理关键点数据
    joints2d_hm = joints2d_hm[:n_view, ...].detach().cpu().numpy()  # (N, J, 2)
    joints2d_sv = joints2d_sv[:n_view, ...].detach().cpu().numpy()  # (N, J, 2)
    joints2d_pse = joints2d_pse[:n_view, ...].detach().cpu().numpy()  # (N, J, 2)
    joints2d_gt = joints2d_gt[:n_view, ...].detach().cpu().numpy()  # (N, J, 2)
    
    # 处理置信度数据
    confidence_hm = confidence_hm[:n_view, ...].detach().cpu().numpy()  # (N, J, 1)
    confidence_sv = confidence_sv[:n_view, ...].detach().cpu().numpy()  # (N, J, 1)
    confidences_pse = confidences_pse[:n_view, ...].detach().cpu().numpy()  # (N, J, 1)
    confidences_gt = confidences_gt[:n_view, ...].detach().cpu().numpy()  # (N, J, 1)
    
    sample_list = []
    for i in range(n_view):
        # 绘制四种方法的结果
        img_hm = plot_hand_cfd(image[i].copy(), joints2d_hm[i], confidences=confidence_hm[i])
        img_sv = plot_hand_cfd(image[i].copy(), joints2d_sv[i], confidences=confidence_sv[i])
        img_pse = plot_hand_cfd(image[i].copy(), joints2d_pse[i], confidences=confidences_pse[i])
        img_gt = plot_hand_cfd(image[i].copy(), joints2d_gt[i], confidences=confidences_gt[i])
        
        # 添加方法标签
        img_hm = add_text(img_hm, "HM", (10, 30))
        img_sv = add_text(img_sv, "SV", (10, 30))
        img_pse = add_text(img_pse, "PSE", (10, 30))
        img_gt = add_text(img_gt, "GT", (10, 30))
        
        # 水平拼接四种结果
        sample_row = np.hstack([img_hm, img_sv, img_pse, img_gt])
        
        # 添加白色边框
        sample_row = cv2.copyMakeBorder(
            sample_row, 
            5, 5, 5, 5, 
            cv2.BORDER_CONSTANT, 
            value=(255, 255, 255)
        )
        
        # # 添加样本编号
        # sample_row = add_text(sample_row, f"Sample {i+1}", (10, 20))
        sample_list.append(sample_row)
    
    # 将所有样本排列到一张大图上
    grid_image = create_image_grid(sample_list, cols=cols)
    return grid_image

def add_text(image, text, position, font_scale=0.6, color=(0, 0, 255), thickness=1):
    """在图像上添加文本
    
    参数:
        image: 输入图像 (numpy数组)
        text: 要添加的文本
        position: 文本位置 (x, y)
        font_scale: 字体大小
        color: 文本颜色 (B, G, R)
        thickness: 文本线宽
        
    返回:
        添加文本后的图像副本
    """
    # 创建图像的副本，避免修改原始图像
    img_copy = image.copy()
    
    # 使用OpenCV添加文本
    cv2.putText(
        img_copy, 
        text, 
        position, 
        cv2.FONT_HERSHEY_SIMPLEX, 
        font_scale, 
        color, 
        thickness, 
        cv2.LINE_AA
    )
    
    return img_copy

def create_image_grid(images, cols=4, padding=5, bg_color=(255, 255, 255)):
    """
    将多个图像排列成网格
    :param images: 图像列表，每个图像是numpy数组 (H, W, C)
    :param cols: 每行显示的列数
    :param padding: 图像之间的间距（像素）
    :param bg_color: 背景颜色 (B, G, R)
    :return: 组合后的大图
    """
    if not images:
        return None
    
    # 获取单个图像的尺寸
    sample_h, sample_w, sample_c = images[0].shape
    n_images = len(images)
    
    # 计算网格的行数和列数
    rows = (n_images + cols - 1) // cols
    
    # 计算大图的尺寸
    grid_w = cols * sample_w + (cols - 1) * padding
    grid_h = rows * sample_h + (rows - 1) * padding
    
    # 创建空白的大图（白色背景）
    grid_image = np.full((grid_h, grid_w, sample_c), bg_color, dtype=np.uint8)
    
    # 将每个图像放置到网格中的正确位置
    for i, img in enumerate(images):
        row_idx = i // cols
        col_idx = i % cols
        
        # 计算当前图像的起始位置
        y_start = row_idx * (sample_h + padding)
        x_start = col_idx * (sample_w + padding)
        
        # 将图像复制到大图的对应位置
        grid_image[y_start:y_start+sample_h, x_start:x_start+sample_w] = img
    
    return grid_image


def plot_hand_cfd(image, coords_hw, vis=None, confidences=None, linewidth=1):  # 默认linewidth=1
    """修改说明：
    1. 添加confidences参数
    2. 关键点画实心圈（固定大小）
    3. 置信度画空心圈（大小随置信度变化）
    """
    colors = np.array(COLOR_CONST.color_hand_joints)
    colors = colors[:, ::-1]
    
    # 定义骨骼连接和颜色
    bones = [
        ((0, 1), colors[1, :]),
        ((1, 2), colors[2, :]),
        ((2, 3), colors[3, :]),
        ((3, 4), colors[4, :]),
        ((0, 5), colors[5, :]),
        ((5, 6), colors[6, :]),
        ((6, 7), colors[7, :]),
        ((7, 8), colors[8, :]),
        ((0, 9), colors[9, :]),
        ((9, 10), colors[10, :]),
        ((10, 11), colors[11, :]),
        ((11, 12), colors[12, :]),
        ((0, 13), colors[13, :]),
        ((13, 14), colors[14, :]),
        ((14, 15), colors[15, :]),
        ((15, 16), colors[16, :]),
        ((0, 17), colors[17, :]),
        ((17, 18), colors[18, :]),
        ((18, 19), colors[19, :]),
        ((19, 20), colors[20, :]),
    ]
    
    if vis is None:
        vis = np.ones_like(coords_hw[:, 0]) == 1.0
    
    # 绘制骨骼
    for connection, color in bones:
        if (vis[connection[0]] == False) or (vis[connection[1]] == False):
            continue
        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        c1x = int(coord1[0])
        c1y = int(coord1[1])
        c2x = int(coord2[0])
        c2y = int(coord2[1])
        cv2.line(image, (c1x, c1y), (c2x, c2y), color=color * 255, thickness=linewidth)
    
    # 绘制关键点和置信度圈
    for i in range(coords_hw.shape[0]):
        cx = int(coords_hw[i, 0])
        cy = int(coords_hw[i, 1])
        
        # 1. 绘制关键点（实心圈，固定大小）
        keypoint_radius = 2 * linewidth  # 固定半径
        cv2.circle(image, (cx, cy), radius=keypoint_radius, 
                   thickness=-1, color=colors[i, :] * 255)
        
        # 2. 绘制置信度圈（空心圈，大小随置信度变化）
        if confidences is not None:
            conf = confidences[i, 0]  # 取置信度值
            # 计算浮点半径，范围在2到10之间
            fractional_radius = max(1, 1 + (1 - conf) * 9)
            # 四舍五入取整
            confidence_radius = int(round(fractional_radius))
            
            # 绘制空心圈（厚度为linewidth），使用抗锯齿
            cv2.circle(image, (cx, cy), radius=confidence_radius, 
                       thickness=linewidth, color=colors[i, :] * 255, lineType=cv2.LINE_AA)
    
    return image


def draw_cfd(batch, res, step_idx):
    img = batch['image']
    joints_hm = (res['pred_joint_uv_hmap'].reshape(-1, 8, 21, 2) + 1) * 128
    joints_2d = (res['pred_joint_uv'].reshape(-1, 8, 21, 2) + 1) * 128
    pgt_2d = (batch['target_pseudo_uv'].reshape(-1, 8, 21, 2) + 1) * 128
    gt_2d = (batch['target_joints_uvd'][:, :, :, :2].reshape(-1, 8, 21, 2) + 1) * 128
    pred_conf = res['pred_cfd'].reshape(-1, 8, 21, 1)
    pred_unc = res['pred_unc'].reshape(-1, 8, 21, 1)
    pse_conf = batch['pseudo_cfd'].reshape(-1, 8, 21, 1)
    gt_conf = torch.ones_like(pse_conf)
    for b in range(joints_2d.shape[0]):
        vis_batch = draw_batch_joint_images_all(joints_hm[b], joints_2d[b], pgt_2d[b], gt_2d[b], img[b], pred_conf[b], pred_unc[b], pse_conf[b], gt_conf[b], step_idx)
        cv2.imwrite(f'./vis_data/input_image_batch_{b}.jpg', vis_batch)


def draw_gt(batch, step_idx):
    img = batch['image']
    gt_2d = (batch['target_joints_uvd'][:, :, :, :, :2] + 1) * 128
    gt_3d = batch['target_joints_3d']
    # pse_conf = batch['pseudo_cfd'].reshape(-1, 8, 21, 1)
    # gt_conf = torch.ones_like(pse_conf)
    for b in range(gt_2d.shape[0]):
        for t in range(3):
            vis_batch = draw_batch_gt_images(gt_2d[b, :, t], img[b, :, t], gt_3d[b, :, t], step_idx)
            for v in range(8):
                cv2.imwrite(f'./vis_gt/input_image_batch_{b}_view{v}_seq_{t}.jpg', vis_batch[v])

def render_hand_mesh(verts, faces, color=[0.9, 0.7, 0.6], img_size=(512, 512), interactive=False):
    """
    用 pyrender 渲染手部 mesh
    
    参数:
    verts: (N, 3) numpy 数组，手部 3D 顶点坐标
    faces: (M, 3) numpy 数组，手部三角面片索引（每个面含3个顶点索引）
    color: (3,) 列表，手部颜色（RGB，0-1范围，默认肤色）
    img_size: (H, W) 元组，渲染图像尺寸
    interactive: bool，是否启用交互式查看（True 则弹出窗口可旋转缩放）
    
    返回:
    若 interactive=False，返回 (H, W, 3) numpy 数组（渲染的 RGB 图像）
    """
    # ------------------------------
    # 1. 创建手部 mesh 对象
    # ------------------------------
    # 用 trimesh 构建网格（pyrender 兼容 trimesh 的格式）
    hand_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    
    # 为网格添加材质（颜色）
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=color + [1.0],  # +[1.0] 表示 alpha 通道（不透明）
        metallicFactor=0.0,  # 非金属材质
        roughnessFactor=0.5  # 粗糙程度（0 为镜面，1 为完全漫反射）
    )
    
    # 转换为 pyrender 可渲染的网格
    pyrender_mesh = pyrender.Mesh.from_trimesh(hand_mesh, material=material)
    
    # ------------------------------
    # 2. 创建场景并添加组件
    # ------------------------------
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0])  # 白色背景
    scene.add(pyrender_mesh)  # 添加手部网格
    
    # ------------------------------
    # 3. 添加光源（提升渲染效果）
    # ------------------------------
    # 方向光 1（从左上方照射）
    light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light1, pose=np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.8, -0.6, 0.0],
        [0.0, 0.6, 0.8, 5.0],  # 光源位置（z轴正向5单位）
        [0.0, 0.0, 0.0, 1.0]
    ]))
    
    # 方向光 2（从右上方照射）
    light2 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
    scene.add(light2, pose=np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -0.8, -0.6, 0.0],
        [0.0, 0.6, -0.8, 5.0],
        [0.0, 0.0, 0.0, 1.0]
    ]))
    
    # # 环境光（弱化阴影）
    # ambient_light = pyrender.AmbientLight(color=[1.0, 1.0, 1.0], intensity=0.5)
    # scene.add(ambient_light)
    
    # ------------------------------
    # 4. 添加相机（设置观察视角）
    # ------------------------------
    # 相机内参（视场角 fov=60度，适合观察手部）
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=img_size[1]/img_size[0])
    
    # 相机外参（位姿矩阵：从世界坐标系到相机坐标系的变换）
    # 这里设置相机在手部前方（z轴正方向），稍向上倾斜，方便观察手掌
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],    # 旋转：无旋转
        [0.0, 0.95, -0.32, 0.0], # 轻微向上倾斜（绕x轴旋转约18度）
        [0.0, 0.32, 0.95, 1.5],  # 平移：沿z轴远离手部1.5单位
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)
    
    # ------------------------------
    # 5. 渲染或交互式查看
    # ------------------------------
    if interactive:
        # 交互式查看（可鼠标拖拽旋转、滚轮缩放）
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True, window_size=img_size)
        return None
    else:
        # 离线渲染为图像
        renderer = pyrender.OffscreenRenderer(viewport_width=img_size[1], viewport_height=img_size[0])
        color, _ = renderer.render(scene)  # color: (H, W, 3)，RGB格式
        renderer.delete()  # 释放资源
        return color