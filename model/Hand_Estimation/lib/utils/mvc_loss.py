import torch
import torch.nn as nn
from torch.nn import functional as F
from lib.utils.ops import batch_compute_similarity_transform_torch

# class MultiViewConsistencyLoss(nn.Module):
#     def __init__(self, interval=1):
#         super(MultiViewConsistencyLoss, self).__init__()
#         self.batch_counter = -1
#         # self.interval = interval
#         self.interval = 1
#         self.coord_loss = nn.L1Loss()

#     def forward(self, coord_xyz, view_num, R=None):
#         '''
#         coord_xyz: shape=(batch, 778+21, 3)
#         view_num:
#         R: ground-truth rotation to world coordinate system
#         '''
#         device = coord_xyz.device
#         self.batch_counter += 1
#         loss_list = []
#         avg_fuse = torch.zeros_like(coord_xyz).view(-1, view_num, 799, 3).to(device)
#         for view_idx in range(view_num):
#             # use ground-truth rotation
#             if R is not None:
#                 batch_relative_R = torch.zeros_like(R).to(device)
#                 for b in range(batch_relative_R.shape[0]):
#                     for j in range(view_num):
#                         batch_relative_R[b][j] = R[b][view_idx].matmul(R[b][j].transpose(1, 0))
#                 gt_R = batch_relative_R.clone().view(-1, 3, 3)

#             all_view = coord_xyz
#             this_view_single = coord_xyz.view(-1, view_num, 799, 3)[:, view_idx]
#             this_view = coord_xyz.view(-1, view_num, 799, 3)[:, view_idx:view_idx+1].repeat(1, view_num, 1, 1).reshape(-1, 799, 3)

#             all_view = all_view.permute(0, 2, 1)
#             this_view = this_view.permute(0, 2, 1)

#             if R is not None:
#                 aligned_to_this_view, (_, pred_R, _) = batch_compute_similarity_transform_torch(
#                     all_view, this_view, gt_R)
#             else:
#                 aligned_to_this_view, (_, pred_R, _) = batch_compute_similarity_transform_torch(
#                     all_view, this_view)
                            
#             aligned_to_this_view = aligned_to_this_view.permute(0, 2, 1)
#             this_view = this_view.permute(0, 2, 1)

#             aligned_to_this_view_mean = aligned_to_this_view.reshape(-1, view_num, 799, 3).mean(1)
#             avg_fuse[:, view_idx] = aligned_to_this_view_mean

#             if self.batch_counter // self.interval % 2 == 0:
#                 loss_list.append(self.coord_loss(aligned_to_this_view[:, :, :2], this_view.detach()[:, :, :2]))
#             else:
#                 loss_list.append(self.coord_loss(aligned_to_this_view_mean.detach(), this_view_single))
#         return loss_list, avg_fuse
    

class MultiViewConsistencyLoss(nn.Module):
    def __init__(self, cfg, interval=1):
        super(MultiViewConsistencyLoss, self).__init__()
        self.cfg = cfg
        self.batch_counter = -1
        self.interval = interval
        self.coord_loss = nn.L1Loss(reduction='none')  # 设置为不自动求平均

    def forward(self, coord_xyz, view_num, confidence=None, R=None):
        '''
        coord_xyz: shape=(batch*view_num, 778+21, 3)
        view_num: 视角数量
        confidence: 关键点置信度，shape=(batch, view_num, 21, 1)
        R: ground-truth rotation to world coordinate system
        '''
        device = coord_xyz.device
        self.batch_counter += 1
        loss_list = []
        batch_size = coord_xyz.shape[0] // view_num
        
        # 初始化avg_fuse
        avg_fuse = torch.zeros(batch_size, view_num, 799, 3).to(device)
        
        # 如果没有提供置信度，则使用均匀权重
        if confidence is None:
            confidence = torch.ones(batch_size, view_num, 21, 1).to(device)
        
        for view_idx in range(view_num):
            # use ground-truth rotation
            if R is not None:
                batch_relative_R = torch.zeros_like(R).to(device)
                for b in range(batch_relative_R.shape[0]):
                    for j in range(view_num):
                        batch_relative_R[b][j] = R[b][view_idx].matmul(R[b][j].transpose(1, 0))
                gt_R = batch_relative_R.clone().view(-1, 3, 3)

            all_view = coord_xyz
            this_view_single = coord_xyz.view(-1, view_num, 799, 3)[:, view_idx]
            this_view = coord_xyz.view(-1, view_num, 799, 3)[:, view_idx:view_idx+1].repeat(1, view_num, 1, 1).reshape(-1, 799, 3)

            all_view = all_view.permute(0, 2, 1)
            this_view = this_view.permute(0, 2, 1)

            if R is not None:
                aligned_to_this_view, (_, pred_R, _) = batch_compute_similarity_transform_torch(
                    all_view, this_view, gt_R)
            else:
                aligned_to_this_view, (_, pred_R, _) = batch_compute_similarity_transform_torch(
                    all_view, this_view)
                            
            aligned_to_this_view = aligned_to_this_view.permute(0, 2, 1)
            this_view = this_view.permute(0, 2, 1)
            
            # 将对齐后的结果reshape为(batch_size, view_num, 799, 3)
            aligned_reshaped = aligned_to_this_view.reshape(batch_size, view_num, 799, 3)
            
            # 分离关键点和其他点
            keypoints_aligned = aligned_reshaped[:, :, 778:, :]  # (batch_size, view_num, 21, 3)
            other_points_aligned = aligned_reshaped[:, :, :778, :]  # (batch_size, view_num, 778, 3)
            
            # 对关键点使用置信度加权平均 - 修复维度问题
            # 扩展置信度维度以匹配坐标维度
            confidence_expanded = confidence.unsqueeze(-1)  # (batch_size, view_num, 21, 1, 1)
            
            # 修正维度：先转置关键点，使其形状为(batch_size, view_num, 21, 3, 1)
            keypoints_aligned_expanded = keypoints_aligned.unsqueeze(-1)  # (batch_size, view_num, 21, 3, 1)
            
            # 相乘并求和
            weighted_keypoints = keypoints_aligned_expanded * confidence_expanded  # (batch_size, view_num, 21, 3, 1)
            weighted_keypoints_sum = weighted_keypoints.sum(dim=1)  # (batch_size, 21, 3, 1)
            
            confidence_sum = confidence.sum(dim=1)  # (batch_size, 21, 1)
            confidence_sum_expanded = confidence_sum.unsqueeze(-1)  # (batch_size, 21, 1, 1)
            
            # 避免除以零
            confidence_sum_expanded = torch.where(
                confidence_sum_expanded < 1e-6, 
                torch.ones_like(confidence_sum_expanded), 
                confidence_sum_expanded
            )
            
            # 计算加权平均值
            fused_keypoints = weighted_keypoints_sum / confidence_sum_expanded  # (batch_size, 21, 3, 1)
            fused_keypoints = fused_keypoints.squeeze(-1)  # (batch_size, 21, 3)
            
            # 对其他点使用简单平均
            fused_other_points = other_points_aligned.mean(dim=1)  # (batch_size, 778, 3)
            
            # 合并关键点和其他点
            aligned_to_this_view_mean = torch.cat([fused_other_points, fused_keypoints], dim=1)
            
            # 存储融合结果
            avg_fuse[:, view_idx] = aligned_to_this_view_mean
            
            # 提取关键点部分
            aligned_keypoints = aligned_to_this_view[:, 778:, :2]  # (batch*view_num, 21, 2)
            this_keypoints = this_view[:, 778:, :2]  # (batch*view_num, 21, 2)
            
            # 获取当前视角的关键点
            this_view_single_keypoints = this_view_single[:, 778:, :]  # (batch, 21, 3)
            
            # 准备置信度权重
            # 对于第一种情况，我们需要所有视角的置信度
            all_confidence = confidence.view(-1, 21, 1)  # (batch*view_num, 21, 1)
            all_confidence_xy = all_confidence.expand(-1, -1, 2)  # (batch*view_num, 21, 2)
            
            # 对于第二种情况，我们需要当前视角的置信度
            this_confidence = confidence[:, view_idx, :, :]  # (batch, 21, 1)
            this_confidence_xyz = this_confidence.expand(-1, -1, 3)  # (batch, 21, 3)
            
            if self.batch_counter // self.interval % 2 == 0:
                # 计算加权L1损失
                loss_per_point = self.coord_loss(aligned_keypoints, this_keypoints.detach())
                weighted_loss = (loss_per_point * all_confidence_xy).sum() / (all_confidence_xy.sum() + 1e-6)
                loss_list.append(weighted_loss)
            else:
                # 计算加权L1损失
                loss_per_point = self.coord_loss(
                    aligned_to_this_view_mean[:, 778:, :].detach(), 
                    this_view_single_keypoints
                )
                weighted_loss = (loss_per_point * this_confidence_xyz).sum() / (this_confidence_xyz.sum() + 1e-6)
                loss_list.append(weighted_loss)
        
        return loss_list, avg_fuse
    

class MultiViewConsistency(nn.Module):
    def __init__(self):
        super(MultiViewConsistency, self).__init__()

    def forward(self, joint_xyz, view_num, R=None):
        '''
        coord_xyz: shape=(batch, 21, 3)
        view_num:
        R: ground-truth rotation to world coordinate system
        '''
        device = joint_xyz.device
        avg_fuse = torch.zeros_like(joint_xyz).view(-1, view_num, 21, 3).to(device)
        for view_idx in range(view_num):
            # use ground-truth rotation
            if R is not None:
                batch_relative_R = torch.zeros_like(R).to(device)
                for b in range(batch_relative_R.shape[0]):
                    for j in range(view_num):
                        batch_relative_R[b][j] = R[b][view_idx].matmul(R[b][j].transpose(1, 0))
                gt_R = batch_relative_R.clone().view(-1, 3, 3)

            all_view = joint_xyz
            this_view_single = joint_xyz.view(-1, view_num, 21, 3)[:, view_idx]
            this_view = joint_xyz.view(-1, view_num, 21, 3)[:, view_idx:view_idx+1].repeat(1, view_num, 1, 1).reshape(-1, 21, 3)

            all_view = all_view.permute(0, 2, 1)
            this_view = this_view.permute(0, 2, 1)

            if R is not None:
                aligned_to_this_view, (_, pred_R, _) = batch_compute_similarity_transform_torch(
                    all_view, this_view, gt_R)
            else:
                aligned_to_this_view, (_, pred_R, _) = batch_compute_similarity_transform_torch(
                    all_view, this_view)
                            
            aligned_to_this_view = aligned_to_this_view.permute(0, 2, 1)
            this_view = this_view.permute(0, 2, 1)

            aligned_to_this_view_mean = aligned_to_this_view.reshape(-1, view_num, 21, 3).mean(1)
            avg_fuse[:, view_idx] = aligned_to_this_view_mean

        return avg_fuse