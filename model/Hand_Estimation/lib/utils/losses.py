import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn import functional as F
from lib.utils.ops import batch_compute_similarity_transform_torch


# range of motion
BOF_RIGHT = np.array([
       [ -179,-179,-179,
           0, -40, -30,
           0,   0,   0,
           0,   0, -45,
           0, -20, -20,
           0,   0,   0,
           0,   0, -20,
           0, -10, -30,
           0,   0,   0,
           0,   0, -40,
           0, -20, -20,
           0,   0,   0,
           0,   0, -10,
           0, -90, -45,
           0,-100,   0,
           0, -90,   0],
        [179, 179, 179,
           0,   5,  90,
           0,   0, 110,
           0,   0,  90,
           0,  15, 100,
           0,   0, 110,
           0,   0,  85,
           0,  45, 100,
           0,   0, 100,
           0,   0,  90,
           0,  20,  90,
           0,   0, 110,
           0,   0,  80,
           0,  20,  45,
           0,   0,   0,
           0,  45,   0]]) / 180 * np.pi


class TemporalSmoothnessLoss(nn.Module):
    def __init__(self, vel_weight=1.0, acc_weight=0.5, reduction='mean'):
        """
        Args:
            vel_weight (float): 速度损失权重
            acc_weight (float): 加速度损失权重
            reduction (str): 'mean'或'sum'，指定如何聚合batch和帧间的损失
        """
        super().__init__()
        self.vel_weight = vel_weight
        self.acc_weight = acc_weight
        self.reduction = reduction

    def forward(self, x):
        """
        Args:
            x: 输入序列 (B, T, 21, 3)
        Returns:
            loss: 平滑性损失值
        """
        B, T, N, D = x.shape
        
        # 计算速度 (一阶差分，B*(T-1)*N*D)
        velocity = x[:, 1:] - x[:, :-1]  # shape: (B, T-1, 21, 3)
        
        # 计算加速度 (二阶差分，B*(T-2)*N*D)
        acceleration = velocity[:, 1:] - velocity[:, :-1]  # shape: (B, T-2, 21, 3)
        
        # 损失计算（按batch和帧维度聚合）
        loss_vel = torch.mean(velocity.pow(2), dim=[-3, -2, -1])  # (B,)
        loss_acc = torch.mean(acceleration.pow(2), dim=[-3, -2, -1])  # (B,)
        
        # 加权总损失（每个样本独立计算）
        loss_per_sample = self.vel_weight * loss_vel + self.acc_weight * loss_acc
        
        # 聚合方式选择
        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        else:
            return loss_per_sample  # 返回每个样本的损失 (B,)


def loss_velocity(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    if predicted.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(predicted.device)
    velocity_predicted = predicted[:,1:] - predicted[:,:-1]
    velocity_target = target[:,1:] - target[:,:-1]
    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=-1))

def get_hand_angles(x):
    """
    输入: (N, T, 21, 3)  # 手部关键点坐标
    输出: (N, T, 20)      # 20个预定义角度（可根据需求调整）
    """
    eps = 1e-6
    hand_limbs_id = [
        [0,1], [1,2], [2,3], [3,4], [0,5], [5,6], [6,7], [7,8],
        [0,9], [9,10], [10,11], [11,12], [0,13], [13,14], [14,15], [15,16],
        [0,17], [17,18], [18,19], [19,20]
    ]
    hand_angle_id = [
        [0,1], [1,2], [2,3], [4,5], [5,6], [6,7],
        [8,9], [9,10], [10,11], [12,13], [13,14], [14,15],
        [16,17], [17,18], [18,19], [1,5], [5,9], [9,13], [13,17]
    ]
    
    # 计算骨骼向量（肢体方向）
    limbs = x[:, :, [i[0] for i in hand_limbs_id], :] - x[:, :, [i[1] for i in hand_limbs_id], :]  # (N,T,20,3)
    
    # 提取角度计算所需的骨骼对
    angle_limbs_a = limbs[:, :, [i[0] for i in hand_angle_id], :]  # (N,T,19,3)
    angle_limbs_b = limbs[:, :, [i[1] for i in hand_angle_id], :]  # (N,T,19,3)
    
    # 计算余弦相似度（夹角）
    angle_cos = F.cosine_similarity(angle_limbs_a, angle_limbs_b, dim=-1)  # (N,T,19)
    angles = torch.acos(angle_cos.clamp(-1+eps, 1-eps))  # 反余弦得到弧度值
    
    return angles

def loss_angle(x, gt):
    '''
        Input: (N, T, 21, 3), (N, T, 21, 3)
    '''
    limb_angles_x = get_hand_angles(x)
    limb_angles_gt = get_hand_angles(gt)
    return nn.L1Loss()(limb_angles_x, limb_angles_gt)

def loss_angle_velocity(x, gt):
    """
    Mean per-angle velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert x.shape == gt.shape
    if x.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)
    x_a = get_hand_angles(x)
    gt_a = get_hand_angles(gt)
    x_av = x_a[:,1:] - x_a[:,:-1]
    gt_av = gt_a[:,1:] - gt_a[:,:-1]
    return nn.L1Loss()(x_av, gt_av)


class TempoAlignLoss(nn.Module):
    def __init__(self):
        super(TempoAlignLoss, self).__init__()

    def forward(self, joints_pred, joints_gt):
        '''
        joints_pred: (N, T, 21, 3)
        joints_gt: (N, T, 21, 3)
        '''
        loss_total = loss_velocity(joints_pred, joints_gt) + loss_angle(joints_pred, joints_gt) + loss_angle_velocity(joints_pred, joints_gt)
        return loss_total


class LaplaceNllLoss(nn.Module):
    def __init__(self):
        super(LaplaceNllLoss, self).__init__()

    def forward(self, input, target, var, reduction="mean", eps: float = 1e-2):
        # 参数检查与原始函数相同
        if torch.any(var < 0):
            raise ValueError("var has negative entry/entries")
        
        var = var.clone()
        with torch.no_grad():
            var.clamp_(min=eps)
        
        # 使用绝对值误差替代平方误差
        abs_error = torch.abs(input - target)
        loss = abs_error / var + torch.log(2 * var)  # 拉普拉斯分布的NLL
        
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss


class GaussianNllLoss(nn.Module):
    def __init__(self):
        super(GaussianNllLoss, self).__init__()

    def forward(self, pred, target, sigma, reduction="mean", eps: float = 1e-8):
        batch, joint, _ = pred.shape
        loss = 0
        # Check validity of reduction mode
        if reduction != "none" and reduction != "mean" and reduction != "sum":
            raise ValueError(reduction + " is not valid")

        # Entries of var must be non-negative
        if torch.any(sigma < 0):
            raise ValueError("var has negative entry/entries")

        # Clamp for stability
        sigma = sigma.clone()
        # with torch.no_grad():
        #     sigma.clamp_(min=eps)
        # Calculate the loss
        for i in range(joint):
            per_joint_sigma = sigma[:, i]
            per_joint_pred = pred[:, i]
            per_joint_target = target[:, i]
            loss += (torch.log(per_joint_sigma) + ((per_joint_target - per_joint_pred) ** 2) / (2 * (per_joint_sigma ** 2) + eps)).mean()
        loss /= batch

        return loss


class FnHeatmapLoss(nn.Module):
    def __init__(self):
        super(FnHeatmapLoss, self).__init__()

    def forward(self, pred_heatmap, gt_heatmap, joint_valid=None):
        """
        pred_heatmap: [B, J, H, W]
        gt_heatmap: [B, J, H, W]
        joint_valid: [B, J], 0/1 mask, 1 for valid, 0 for invalid
        """
        assert pred_heatmap.ndim == 4
        assert gt_heatmap.ndim == 4

        # err = (pred_heatmap - gt_heatmap).square().sum((-1, -2))  # [B, J]  
        loss = (pred_heatmap - gt_heatmap).square().sum((-1, -2))  # [B, J]
        if joint_valid is not None:
            valid = joint_valid
            loss = (loss * valid).sum() / (valid.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss  # per J

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
    

class MultiViewTempoConsistencyLoss(nn.Module):
    def __init__(self, interval=1):
        super(MultiViewTempoConsistencyLoss, self).__init__()
        self.batch_counter = -1
        # self.interval = interval
        self.interval = 1
        self.coord_loss = nn.L1Loss()

    def forward(self, coord_xyz, view_num, seq_num, R=None):
        '''
        coord_xyz: shape=(batch, 778+21, 3)
        view_num:
        R: ground-truth rotation to world coordinate system
        '''
        device = coord_xyz.device
        self.batch_counter += 1
        loss_list = []
        avg_fuse = torch.zeros_like(coord_xyz).view(-1, view_num, seq_num, 799, 3).to(device)
        for view_idx in range(view_num):
            # use ground-truth rotation
            if R is not None:
                batch_relative_R = torch.zeros_like(R).to(device)
                for b in range(batch_relative_R.shape[0]):
                    for j in range(view_num):
                        for t in range(seq_num):
                            batch_relative_R[b][j][t] = R[b][view_idx][t].matmul(R[b][j][t].transpose(1, 0))
                gt_R = batch_relative_R.clone().view(-1, 3, 3)

            all_view = coord_xyz
            this_view_single = coord_xyz.view(-1, view_num, seq_num, 799, 3)[:, view_idx]
            this_view = coord_xyz.view(-1, view_num, 3, 799, 3)[:, view_idx:view_idx+1].repeat(1, view_num, 1, 1, 1).reshape(-1, 799, 3)

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

            aligned_to_this_view_mean = aligned_to_this_view.reshape(-1, view_num, seq_num, 799, 3).mean(1)
            avg_fuse[:, view_idx] = aligned_to_this_view_mean

            if self.batch_counter // self.interval % 2 == 0:
                loss_list.append(self.coord_loss(aligned_to_this_view[:, :, :2], this_view.detach()[:, :, :2]))
            else:
                loss_list.append(self.coord_loss(aligned_to_this_view_mean.detach(), this_view_single))
        return loss_list, avg_fuse
    

class MultiViewJointConsistencyLoss(nn.Module):
    def __init__(self, interval=1):
        super(MultiViewJointConsistencyLoss, self).__init__()
        # self.start = start
        self.batch_counter = -1
        self.interval = interval
        self.coord_loss = nn.L1Loss()
    
    def forward(self, joint_xyz, view_num, seq_num, confidence, R=None):
        '''
        coord_xyz: shape=(batch*view*tempo, 21, 3)
        view_num: 视图数量
        seq_num: 时序长度
        confidence: shape=(batch*view*tempo, 21, 3)
        R: 世界坐标系下的真实旋转矩阵，shape=(batch*tempo, view, 3, 3)
        '''
        device = joint_xyz.device
        self.batch_counter += 1
        mvc_mvf_loss_list = []
        
        avg_fuse = torch.zeros_like(joint_xyz).view(-1, view_num, seq_num, 21, 3).to(device)
        for view_idx in range(view_num):
            # use ground-truth rotation
            if R is not None:
                batch_relative_R = torch.zeros_like(R).to(device)
                for b in range(batch_relative_R.shape[0]):
                    for j in range(view_num):
                        for t in range(seq_num):
                            batch_relative_R[b][j][t] = R[b][view_idx][t].matmul(R[b][j][t].transpose(1, 0))
                gt_R = batch_relative_R.clone().view(-1, 3, 3)

            all_view = joint_xyz
            this_view_single = joint_xyz.view(-1, view_num, seq_num, 21, 3)[:, view_idx]
            this_view = joint_xyz.view(-1, view_num, seq_num, 21, 3)[:, view_idx:view_idx+1].repeat(1, view_num, 1, 1, 1).reshape(-1, 21, 3)

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

            aligned_to_this_view_mean = aligned_to_this_view.reshape(-1, view_num, seq_num, 21, 3).mean(1)
            avg_fuse[:, view_idx] = aligned_to_this_view_mean

            if self.batch_counter // self.interval % 2 == 0:
                mvc_mvf_loss_list.append(self.coord_loss(aligned_to_this_view[:, :, :2], \
                                         this_view.detach()[:, :, :2])) 
            else:
                mvc_mvf_loss_list.append(self.coord_loss(aligned_to_this_view_mean.detach(), this_view_single))

        avg_fuse = avg_fuse.reshape(-1, 21, 3)
        
        return mvc_mvf_loss_list, avg_fuse


class MultiViewVertConsistencyLoss(nn.Module):
    def __init__(self, interval=1):
        super(MultiViewVertConsistencyLoss, self).__init__()
        self.batch_counter = -1
        # self.interval = interval
        self.interval = 1
        self.coord_loss = nn.L1Loss()

    def forward(self, coord_xyz, view_num, seq, R=None):
        '''
        coord_xyz: shape=(batch, 778, 3)
        view_num:
        R: ground-truth rotation to world coordinate system
        '''
        device = coord_xyz.device
        self.batch_counter += 1
        loss_list = []
        avg_fuse = torch.zeros_like(coord_xyz).view(-1, view_num, seq, 778, 3).to(device)
        for view_idx in range(view_num):
            # use ground-truth rotation
            if R is not None:
                batch_relative_R = torch.zeros_like(R).to(device)
                for b in range(batch_relative_R.shape[0]):
                    for j in range(view_num):
                        for t in range(seq):
                            batch_relative_R[b][j][t] = R[b][view_idx][t].matmul(R[b][j][t].transpose(1, 0))
                gt_R = batch_relative_R.clone().view(-1, 3, 3)

            all_view = coord_xyz
            this_view_single = coord_xyz.view(-1, view_num, seq, 778, 3)[:, view_idx]
            this_view = coord_xyz.view(-1, view_num, seq, 778, 3)[:, view_idx:view_idx+1].repeat(1, view_num, 1, 1, 1).reshape(-1, 778, 3)

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

            aligned_to_this_view_mean = aligned_to_this_view.reshape(-1, view_num, seq, 778, 3).mean(1)
            avg_fuse[:, view_idx] = aligned_to_this_view_mean

            if self.batch_counter // self.interval % 2 == 0:
                loss_list.append(self.coord_loss(aligned_to_this_view[:, :, :2], this_view.detach()[:, :, :2]))
            else:
                loss_list.append(self.coord_loss(aligned_to_this_view_mean.detach(), this_view_single))
        return loss_list, avg_fuse

class MultiViewConsistencyLoss(nn.Module):
    def __init__(self, interval=1):
        super(MultiViewConsistencyLoss, self).__init__()
        self.batch_counter = -1
        # self.interval = interval
        self.interval = 1
        self.coord_loss = nn.L1Loss()

    def forward(self, coord_xyz, view_num, R=None):
        '''
        coord_xyz: shape=(batch, 778+21, 3)
        view_num:
        R: ground-truth rotation to world coordinate system
        '''
        device = coord_xyz.device
        self.batch_counter += 1
        loss_list = []
        avg_fuse = torch.zeros_like(coord_xyz).view(-1, view_num, 799, 3).to(device)
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

            aligned_to_this_view_mean = aligned_to_this_view.reshape(-1, view_num, 799, 3).mean(1)
            avg_fuse[:, view_idx] = aligned_to_this_view_mean

            if self.batch_counter // self.interval % 2 == 0:
                loss_list.append(self.coord_loss(aligned_to_this_view[:, :, :2], this_view.detach()[:, :, :2]))
            else:
                loss_list.append(self.coord_loss(aligned_to_this_view_mean.detach(), this_view_single))
        return loss_list, avg_fuse
    

class MotionModelLoss(nn.Module):
    def __init__(
        self,
        lambda_pos: float = 1.0,     # 位置平滑权重
        lambda_vel: float = 0.3,     # 速度一致性权重
        lambda_acc: float = 0.2,     # 加速度平滑权重（可选）
        reduction: str = "mean",     # 损失计算方式（"mean"或"sum"）
    ):
        super().__init__()
        self.lambda_pos = lambda_pos
        self.lambda_vel = lambda_vel
        self.lambda_acc = lambda_acc
        self.reduction = reduction

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        """
        计算时序3D关键点的运动建模损失。
        
        输入:
            keypoints: [batch_size, seq_len, num_joints, 3]
                - batch_size: 批大小
                - seq_len: 时序长度（帧数）
                - num_joints: 手部关节数（如21）
                - 3: (x, y, z)坐标
        
        返回:
            loss: 标量损失值（加权后的总损失）
        """
        if keypoints.ndim != 4:
            raise ValueError(f"输入张量形状应为[batch, seq_len, joints, 3]，但得到的是{keypoints.shape}")

        batch_size, seq_len, num_joints, _ = keypoints.shape
        
        # --- 1. 位置平滑损失（一阶差分） ---
        if seq_len < 2:
            loss_pos = torch.tensor(0.0, device=keypoints.device)
        else:
            # 计算相邻帧的关节位置差 (L1或L2损失)
            diff_pos = keypoints[:, 1:] - keypoints[:, :-1]  # [batch, seq_len-1, joints, 3]
            loss_pos = torch.mean(torch.abs(diff_pos))  # L1损失（更鲁棒）

        # --- 2. 速度一致性损失（二阶差分） ---
        if seq_len < 3:
            loss_vel = torch.tensor(0.0, device=keypoints.device)
        else:
            # 速度 = 当前帧位置 - 前一帧位置
            vel = keypoints[:, 1:] - keypoints[:, :-1]  # [batch, seq_len-1, joints, 3]
            # 速度变化 = 当前速度 - 前一速度
            diff_vel = vel[:, 1:] - vel[:, :-1]  # [batch, seq_len-2, joints, 3]
            loss_vel = torch.mean(torch.abs(diff_vel))

        # --- 3. 加速度损失（三阶差分，可选） ---
        if seq_len < 4 or self.lambda_acc == 0:
            loss_acc = torch.tensor(0.0, device=keypoints.device)
        else:
            # 加速度 = 速度变化
            acc = vel[:, 1:] - vel[:, :-1]  # [batch, seq_len-2, joints, 3]
            # 加速度变化 = 当前加速度 - 前一加速度
            diff_acc = acc[:, 1:] - acc[:, :-1]  # [batch, seq_len-3, joints, 3]
            loss_acc = torch.mean(torch.abs(diff_acc))

        # 加权总损失
        total_loss = (
            self.lambda_pos * loss_pos +
            self.lambda_vel * loss_vel +
            self.lambda_acc * loss_acc
        )

        return total_loss
    

# class RLELoss(nn.Module):
#     ''' RLE Regression Loss
#     '''

#     def __init__(self, OUTPUT_3D=False, size_average=True):
#         super(RLELoss, self).__init__()
#         self.size_average = size_average
#         self.amp = 1 / math.sqrt(2 * math.pi)

#     def logQ(self, gt_uv, pred_jts, sigma):
#         return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

#     def forward(self, output, target_uv):
#         nf_loss = output.nf_loss
#         pred_jts = output.pred_jts
#         sigma = output.sigma
#         gt_uv = target_uv.reshape(pred_jts.shape)

#         residual = True
#         if residual:
#             Q_logprob = self.logQ(gt_uv, pred_jts, sigma)
#             loss = nf_loss + Q_logprob

#         if self.size_average > 0:
#             return loss.sum() / len(loss)
#         else:
#             return loss.sum()
        

def pose_losses_uncertainty(
        pred_pose,
        pred_beta,
        gt_pose,
        gt_beta,
        pred_uncert_pose,
        loss_ver,
        uncert_type,
        criterion,
):
    device = gt_pose.device
    batch_size = gt_pose.shape[0]
    not_uncert_idx = torch.cuda.BoolTensor(batch_size).fill_(0)

    pred_pose_valid = pred_pose
    pred_pose_no_uncert = pred_pose[not_uncert_idx == 1]
    gt_pose_valid = gt_pose
    gt_pose_no_uncert = gt_pose[not_uncert_idx == 1]

    pred_beta_valid = pred_beta
    gt_beta_valid = gt_beta
    eps = 1e-8

    loss_regr_pose, loss_regr_beta = None, None
    if len(pred_pose_valid) > 0:
        # Pose Loss
        if 'pose' in uncert_type:
            pose_var = pred_uncert_pose
            if len(pose_var.shape) == 2:
                pose_var = pose_var.unsqueeze(2).repeat(1,1,2)
            elif loss_ver == 'norm_flow_res':
                amp = 1 / math.sqrt(2 * math.pi)
                var_loss = torch.log(pose_var / amp)
                pose_loss = torch.abs(pred_pose_valid - gt_pose_valid)
                logQ = var_loss + (pose_loss / (math.sqrt(2) * pose_var + 1e-9))
                loss_regr_pose = logQ.mean()
            elif loss_ver == 'norm_flow_res_gaus':
                if pose_var.shape[1] < 16: # Some parts are excluded from uncertainty
                    loss_regr_pose = criterion(pred_pose_valid, gt_pose_valid)
                else:
                    pose_loss1 = torch.pow(pred_pose_valid - gt_pose_valid, 2) / (pose_var + eps)
                    pose_loss2 = torch.log(pose_var + eps)
                    loss_regr_pose = 0.5 * (pose_loss1 + pose_loss2).mean()
            else:
                loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(pred_pose.device)

        if loss_regr_pose is None:
            loss_regr_pose = criterion(pred_pose_valid, gt_pose_valid)
        if loss_regr_beta is None:
            loss_regr_beta = criterion(pred_beta_valid, gt_beta_valid)
    else:
        loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(pred_pose.device)
        loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(pred_pose.device)

    gt_var = pred_uncert_pose[not_uncert_idx == 1]
    if len(gt_var) > 0:
        loss_regr_pose_no_uncert = criterion(pred_pose_no_uncert, gt_pose_no_uncert)
        loss_gt_var = gt_var.mean()
        loss_regr_pose += loss_regr_pose_no_uncert + loss_gt_var

    return loss_regr_pose, loss_regr_beta


def joints_losses_uncertainty(
        pred_joints,
        gt_joints,
        pred_uncert_joints,
        loss_ver,
        uncert_type,
        criterion,
):
    device = gt_joints.device
    batch_size = gt_joints.shape[0]
    not_uncert_idx = torch.cuda.BoolTensor(batch_size).fill_(0)

    pred_joints_valid = pred_joints
    pred_joints_no_uncert = pred_joints[not_uncert_idx == 1]
    gt_joints_valid = gt_joints
    gt_joints_no_uncert = gt_joints[not_uncert_idx == 1]

    eps = 1e-8

    loss_regr_joints = None
    if len(pred_joints_valid) > 0:
        # Pose Loss
        if 'joints' in uncert_type:
            joints_var = pred_uncert_joints
            if len(joints_var.shape) == 2:
                joints_var = joints_var.unsqueeze(2).repeat(1,1,2)
            elif loss_ver == 'norm_flow_res':
                amp = 1 / math.sqrt(2 * math.pi)
                var_loss = torch.log(joints_var / amp)
                joints_loss = torch.abs(pred_joints_valid - gt_joints_valid)
                logQ = var_loss + (joints_loss / (math.sqrt(2) * joints_var + 1e-9))
                loss_regr_joints = logQ.mean()
            elif loss_ver == 'norm_flow_res_gaus':
                if joints_var.shape[1] < 21: # Some parts are excluded from uncertainty
                    loss_regr_joints = criterion(pred_joints_valid, gt_joints_valid)
                else:
                    pose_loss1 = torch.pow(pred_joints_valid - gt_joints_valid, 2) / (joints_var + eps)
                    pose_loss2 = torch.log(joints_var + eps)
                    loss_regr_joints = 0.5 * (pose_loss1 + pose_loss2).mean()
            else:
                loss_regr_joints = torch.FloatTensor(1).fill_(0.).to(pred_joints.device)

        if loss_regr_joints is None:
            loss_regr_joints = criterion(pred_joints_valid, gt_joints_valid)
    else:
        loss_regr_joints = torch.FloatTensor(1).fill_(0.).to(pred_joints.device)

    gt_var = pred_uncert_joints[not_uncert_idx == 1]
    if len(gt_var) > 0:
        loss_regr_joints_no_uncert = criterion(pred_joints_no_uncert, gt_joints_no_uncert)
        loss_gt_var = gt_var.mean()
        loss_regr_joints += loss_regr_joints_no_uncert + loss_gt_var

    return loss_regr_joints


class Keypoint2DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        2D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint2DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D reprojection loss on the keypoints.
        Args:
            pred_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 2] containing projected 2D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the ground truth 2D keypoints and confidence.
        Returns:
            torch.Tensor: 2D keypoint loss.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        batch_size = conf.shape[0]
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).sum(dim=(1,2))
        return loss.sum()
    

class Keypoint3DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        3D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_3d: torch.Tensor, gt_keypoints_3d: torch.Tensor, has_conf=True):
        """
        Compute 3D keypoint loss.
        Args:
            pred_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the predicted 3D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 4] containing the ground truth 3D keypoints and confidence.
        Returns:
            torch.Tensor: 3D keypoint loss.
        """
        # batch_size = pred_keypoints_3d.shape[0]
        # gt_keypoints_3d = gt_keypoints_3d.clone()
        # pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, pelvis_id, :].unsqueeze(dim=1)
        # gt_keypoints_3d[:, :, :-1] = gt_keypoints_3d[:, :, :-1] - gt_keypoints_3d[:, pelvis_id, :-1].unsqueeze(dim=1)
        if has_conf:
            conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
            gt_keypoints_3d = gt_keypoints_3d[:, :, :-1]
            loss = (conf * self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)).sum(dim=(1,2))
        else:
            loss = self.loss_fn(pred_keypoints_3d, gt_keypoints_3d).sum(dim=(1,2))
        return loss.sum()
    

class Mesh3DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        3D mesh loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Mesh3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_mesh_3d: torch.Tensor, gt_mesh_3d: torch.Tensor, pelvis_id: int = 0):
        """
        Compute 3D keypoint loss.
        Args:
            pred_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the predicted 3D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 4] containing the ground truth 3D keypoints and confidence.
        Returns:
            torch.Tensor: 3D keypoint loss.
        """
        # batch_size = pred_keypoints_3d.shape[0]
        # gt_keypoints_3d = gt_keypoints_3d.clone()
        # pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, pelvis_id, :].unsqueeze(dim=1)
        # gt_keypoints_3d[:, :, :-1] = gt_keypoints_3d[:, :, :-1] - gt_keypoints_3d[:, pelvis_id, :-1].unsqueeze(dim=1)

        loss = self.loss_fn(pred_mesh_3d, gt_mesh_3d).sum(dim=(1,2))
        return loss.sum()
    

class ParameterLoss(nn.Module):

    def __init__(self):
        """
        MANO parameter loss module.
        """
        super(ParameterLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor):
        """
        Compute MANO parameter loss.
        Args:
            pred_param (torch.Tensor): Tensor of shape [B, S, ...] containing the predicted parameters (body pose / global orientation / betas)
            gt_param (torch.Tensor): Tensor of shape [B, S, ...] containing the ground truth MANO parameters.
        Returns:
            torch.Tensor: L2 parameter loss loss.
        """
        loss_param = self.loss_fn(pred_param, gt_param)
        return loss_param.sum()   


class RLELoss(nn.Module):
    ''' RLE Regression Loss
    '''

    def __init__(self, OUTPUT_3D=False, size_average=True):
        super(RLELoss, self).__init__()
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def forward(self, output, gt_uv, vis):
        nf_loss = output['nf_loss']
        pred_jts = output['pred_jts']
        sigma = output['sigma']
        gt_uv = gt_uv.reshape(pred_jts.shape)
        gt_uv_weight = vis.unsqueeze(-1).repeat(1, 1, 2)
        
        nf_loss = nf_loss * gt_uv_weight[:, :, :1]

        residual = True
        if residual:
            Q_logprob = self.logQ(gt_uv, pred_jts, sigma) * gt_uv_weight
            loss = nf_loss + Q_logprob

        if self.size_average and gt_uv_weight.sum() > 0:
            return loss.sum() / len(loss)
        else:
            return loss.sum()
     

class RLELoss3D(nn.Module):
    ''' RLE Regression Loss 3D
    '''

    def __init__(self, OUTPUT_3D=False, size_average=True):
        super(RLELoss3D, self).__init__()
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def forward(self, output):
        nf_loss = output['nf_loss']
        pred_jts = output['keypoints_3d_sv']
        pred = (pred_jts - pred_jts[:, 9:10]) / 0.2
        sigma = output['sigma']
        gt_xyz = output['ref_joints_in_cam'].detach()
        gt = (gt_xyz - gt_xyz[:, 9:10]) / 0.2
        # gt_uv_weight = labels['target_uvd_weight'].reshape(pred_jts.shape)
        gt_uv_weight = output['confidence'].unsqueeze(-1).repeat(1, 1, 3)
        nf_loss = nf_loss * gt_uv_weight

        residual = True
        if residual:
            Q_logprob = self.logQ(gt, pred, sigma) * gt_uv_weight
            loss = nf_loss + Q_logprob

        if self.size_average and gt_uv_weight.sum() > 0:
            return loss.sum() / len(loss)
        else:
            return loss.sum()
        

def compute_anglebound(pred_pose, angle_limits, batch_size, interjoint, device):
    # update joint range of motion given different joint angles using the function anatomy knowledge

    angle_limits_bound = angle_limits.expand(2, batch_size, 48).clone()

    if interjoint == 0:
        return angle_limits_bound
    # eliminate prediction that are outside the original bound
    pred_pose_bound = pred_pose.clone().detach()
    pred_pose_bound = torch.where(pred_pose_bound>angle_limits_bound[0],pred_pose_bound,angle_limits_bound[0])
    pred_pose_bound = torch.where(pred_pose_bound<angle_limits_bound[1],pred_pose_bound,angle_limits_bound[1])

    # DIP on PIP
    DIP_index = [11,20,38,29] # [4,7,13,10]
    PIP_index = [8,17,35,26] # [3,6,12,9]
    angle_limits_bound[0,:,PIP_index] = torch.where(angle_limits_bound[0,:,DIP_index]>0, torch.zeros([batch_size,4]).to(device), angle_limits[0,:,PIP_index])

    # MCP joint fingers
    MCP_index_beta = [4,13,31,22] # [2,5,11,8]
    MCP_index_gamma = [5,14,32,23]
    angle_limits_bound[0,:,MCP_index_beta] = angle_limits_bound[0,:,MCP_index_beta] * (1 - (pred_pose_bound[:,MCP_index_gamma].clamp(max=70/180*np.pi)/(70/180*np.pi)))
    angle_limits_bound[1,:,MCP_index_beta] = angle_limits_bound[1,:,MCP_index_beta] * (1 - (pred_pose_bound[:,MCP_index_gamma].clamp(max=70/180*np.pi)/(70/180*np.pi)))
    # MCP joint thumb
    MCP_thumb_beta = 40 # 14
    MCP_thumb_gamma = 41
    angle_limits_bound[0,:,MCP_thumb_gamma] = angle_limits_bound[0,:,MCP_thumb_gamma] * (1 + (pred_pose_bound[:,MCP_thumb_beta].clamp(min=-70/180*np.pi)/(70/180*np.pi)))
    angle_limits_bound[1,:,MCP_thumb_gamma] = angle_limits_bound[1,:,MCP_thumb_gamma] * (1 + (pred_pose_bound[:,MCP_thumb_beta].clamp(min=-70/180*np.pi)/(70/180*np.pi)))

    # eliminate new bound that are outside the original bound
    angle_limits_bound[0] = torch.where(angle_limits_bound[0]>angle_limits[0],angle_limits_bound[0],angle_limits[0])
    angle_limits_bound[1] = torch.where(angle_limits_bound[1]<angle_limits[1],angle_limits_bound[1],angle_limits[1])

    return angle_limits_bound

def pose_norm_loss(pred_euler, interjoint, device):
    # pred_euler: Nx15x3 (no root joint rotation)
    # interjoint: scalar (if 0, not consider the functional anatomy knowledge)
    
    batch_size = pred_euler.shape[0]

    # biomechanics
    # if handtype == 'RIGHT':
    angle_limits = torch.from_numpy(BOF_RIGHT).to(device).float().unsqueeze(1)
    # else:
    #     angle_limits = torch.from_numpy(constants.BOF_LEFT).to(device).float().unsqueeze(1)

    # functional anatomy
    angle_limits = compute_anglebound(pred_euler, angle_limits, batch_size, interjoint, device)

    anglegreater = pred_euler - angle_limits[1,:]
    anglesmaller = angle_limits[0,:] - pred_euler
    jointangle_loss = torch.mean(torch.sum(anglegreater.clamp(min=0)**2+anglesmaller.clamp(min=0)**2, dim=1)) # N,

    return jointangle_loss