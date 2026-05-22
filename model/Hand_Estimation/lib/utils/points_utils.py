import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points, ball_query
from .logger import logger


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def index_points_4dim(points, idx):
    """
    兼容3维（空间）和4维（时序+空间）点云的索引函数
    Input:
        points: 输入点云特征
            - 3维: [B, N, C]（batch×num_points×channels）
            - 4维: [B, T, N, C]（batch×time×num_points×channels）
        idx: 索引张量（与points维度匹配）
            - 3维: [B, S, K]（batch×sample_points×k_neighbors）
            - 4维: [B, T, S, K]（batch×time×sample_points×k_neighbors）
    Return:
        new_points: 索引后的点云特征
            - 3维输出: [B, S, K, C]
            - 4维输出: [B, T, S, K, C]
    """
    # 获取输入维度
    dim = points.dim()
    assert dim in [3, 4], f"index_points仅支持3/4维张量，输入为{dim}维"
    
    if dim == 4:
        # 处理4维时序点云 [B, T, N, C]
        B, T, N, C = points.shape
        raw_size = idx.size()  # [B, T, S, K]
        # 将idx重塑为 [B, T, S*K]（保留前两维B、T，合并后两维S、K）
        idx_reshaped = idx.reshape(B, T, -1)  # [B, T, S*K]
        # 扩展索引维度以匹配points的通道数C → [B, T, S*K, C]
        idx_expanded = idx_reshaped[..., None].expand(-1, -1, -1, C)
        # 在第2维（N的维度）上索引（4维中，N在dim=2）
        res = torch.gather(points, dim=2, index=idx_expanded)  # [B, T, S*K, C]
        # 恢复原始形状 [B, T, S, K, C]
        return res.reshape(*raw_size, C)
    else:
        # 处理3维空间点云 [B, N, C]（保持原有逻辑）
        B, N, C = points.shape
        raw_size = idx.size()  # [B, S, K]
        idx_reshaped = idx.reshape(B, -1)  # [B, S*K]
        idx_expanded = idx_reshaped[..., None].expand(-1, -1, C)  # [B, S*K, C]
        res = torch.gather(points, dim=1, index=idx_expanded)  # [B, S*K, C]
        return res.reshape(*raw_size, C)  # [B, S, K, C]



def sample_points_from_ball_query(pt_xyz, pt_feats, center_point, k, radius):
    _, ball_idx, xyz = ball_query(center_point, pt_xyz, K=k, radius=radius, return_nn=True)
    invalid = torch.sum(ball_idx == -1) > 0
    if invalid:
        logger.warning(f"ball query returns {torch.sum(ball_idx == -1)} / {torch.numel(ball_idx)} -1 in its index, "
                       f"which means you need to increase raidus or decrease K")

    points = index_points(pt_feats, ball_idx).squeeze(1)
    xyz = xyz.squeeze(1)
    return xyz, points


def sample_points_from_knn(pt_xyz, pt_feats, center_point, k):
    _, knn_idx, xyz = knn_points(center_point, pt_xyz, K=k, return_nn=True)
    points = index_points(pt_feats, knn_idx).squeeze(1)
    xyz = xyz.squeeze(1)
    return xyz, points