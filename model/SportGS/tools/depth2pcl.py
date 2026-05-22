import cv2
import numpy as np
import open3d as o3d

def depth2pcl_with_seg(depth_metric, seg_path, output_pcl_path, cam_intris):
    """
    Input:
        depth_metric: 用DA3的带metric depth感知的模型预测深度图
        seg_path: 像素对齐的手物分割图path
        cam_intris: 内参
    
    """
    seg = cv2.cvtColor(cv2.imread(seg_path), cv2.COLOR_BGR2RGB)
    mask_r = (seg == [255, 0, 0]).all(axis=-1)
    mask_l = (seg == [0, 255, 0]).all(axis=-1)
    mask_o = (seg == [0, 0, 255]).all(axis=-1)

    hand_mask = mask_r | mask_l
    obj_mask = mask_o
    mask = hand_mask | obj_mask

    fx = cam_intris[0][0]
    fy = cam_intris[1][1]
    cx = cam_intris[0][2]
    cy = cam_intris[1][2]

    H, W = depth_metric.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    z = depth_metric[mask]
    u = u[mask]
    v = v[mask]

    # 防止无效深度
    valid = z > 0
    z = z[valid]
    u = u[valid]
    v = v[valid]

    # 反投影回3D空间
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=1)

    # ---- save point cloud ----
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(output_pcl_path, pcl)

    print(f"Saved point cloud with {points.shape[0]} points.")

