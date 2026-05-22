import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from einops import rearrange
from pytorch3d.ops import knn_points, sample_farthest_points
from lib.utils.points_utils import index_points, index_points_4dim
import os

def anchor_points(xyz, K, device):
    # TODO use fixed anchor each time.
    anchor_dir = os.path.join("assets", "anchor.npy")
    anchor_idx_dir = os.path.join("assets", "anchor_idx.npy")
    batch_size = xyz.shape[0]
    if not os.path.exists(anchor_dir):
        # the points of xyz are be the same set of BPS points loaded from bps.npy
        bps_points = xyz[0].unsqueeze(0) 
        local_xyz, local_idx = sample_farthest_points(bps_points, K=K) # batch_size = 1
        local_xyz_dump = local_xyz.cpu().detach().numpy()
        local_idx_dump = local_idx.cpu().detach().numpy()
        np.save(anchor_dir, local_xyz_dump)    
        np.save(anchor_idx_dir, local_idx_dump)
    else:
        local_xyz_load = np.load(anchor_dir)
        local_idx_load = np.load(anchor_idx_dir)
        local_xyz = torch.Tensor(local_xyz_load).to(device)
        local_idx = torch.Tensor(local_idx_load).to(device).type(torch.int64)
        
    local_xyz = local_xyz.repeat(batch_size, 1, 1)
    local_idx = local_idx.repeat(batch_size, 1)
        
    return local_xyz, local_idx


class SpatialAttention(nn.Module):
    """第一阶段：空间邻域自注意力（处理单帧点云，输入展平的(b*t, n, d)）"""
    def __init__(self, d_model, k_spatial):
        super().__init__()
        self.k_spatial = k_spatial  # 空间邻域KNN数量
        # 空间位置编码（基于坐标差）
        self.fc_delta_spatial = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        # Q/K/V映射
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        # 注意力权重映射
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        # 输出映射
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, xyz, features):
        """
        xyz: 点云坐标，(b*t, n, 3)
        features: 点云特征，(b*t, n, d_model)
        return: 空间注意力更新后的特征，(b*t, n, d_model)
        """
        b_t, n, d = features.shape
        # 1. 计算空间邻域（KNN）

        _, knn_idx, knn_xyz = knn_points(xyz, xyz, K=self.k_spatial, return_nn=True)  # (b*t, n, k), (b*t, n, k, 3)
        
        # 2. Q/K/V计算
        q = self.w_qs(features)  # (b*t, n, d)
        k = index_points(self.w_ks(features), knn_idx)  # (b*t, n, k, d)（从邻域提取K）
        v = index_points(self.w_vs(features), knn_idx)  # (b*t, n, k, d)（从邻域提取V）
        
        # 3. 空间位置编码（当前点 - 邻域点的坐标差）
        delta_spatial = xyz.unsqueeze(-2) - knn_xyz  # (b*t, n, 1, 3) - (b*t, n, k, 3) → (b*t, n, k, 3)
        enc_spatial = self.fc_delta_spatial(delta_spatial)  # (b*t, n, k, d)
        
        # 4. 空间注意力计算
        q_expanded = q.unsqueeze(-2)  # (b*t, n, 1, d)
        attn = self.fc_gamma(q_expanded - k + enc_spatial)  # 融合特征差和空间位置差
        attn = F.softmax(attn / torch.sqrt(torch.tensor(d, dtype=torch.float32)), dim=-2)  # (b*t, n, k, d)
        
        # 5. 聚合V并输出
        res = torch.einsum('bnkd,bnkd->bnd', attn, v + enc_spatial)  # (b*t, n, d_model)
        res = self.fc_out(res) + features  # 残差连接
        return res


class TemporalAttention(nn.Module):
    """第二阶段：时序自注意力（处理时间序列，输入恢复的(b, t, n, d)）"""
    def __init__(self, d_model, max_time_steps):
        super().__init__()
        self.d_model = d_model
        # 可学习时序位置编码（每个时间步t的嵌入）
        self.temporal_embedding = nn.Embedding(max_time_steps, d_model)
        # 时序自注意力（Q/K/V基于时间维度）
        self.w_qs_temp = nn.Linear(d_model, d_model, bias=False)
        self.w_ks_temp = nn.Linear(d_model, d_model, bias=False)
        self.w_vs_temp = nn.Linear(d_model, d_model, bias=False)
        # 输出映射
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, features):
        """
        features: 空间注意力输出的特征，恢复为(b, t, n, d_model)
        return: 时序注意力更新后的特征，(b, t, n, d_model)
        """
        b, t, n, d = features.shape
        
        # 1. 嵌入时序位置编码（每个时间步t的特征加上对应的编码）
        temporal_indices = torch.arange(t, device=features.device)  # (t,)
        temp_enc = self.temporal_embedding(temporal_indices)  # (t, d)
        temp_enc = temp_enc.unsqueeze(0).unsqueeze(2)  # (1, t, 1, d) → 扩展到(b, t, n, d)
        temp_enc = temp_enc.expand(b, -1, n, -1)  # (b, t, n, d)
        x = features + temp_enc  # 特征融合时序位置信息
        
        # 2. 调整维度：将时序和关键点合并为“序列长度”，便于自注意力计算
        # 形状转换：(b, t, n, d) → (b, n, t, d) → (b*n, t, d)（每个关键点单独处理时序）
        x_reshaped = rearrange(x, 'b t n d -> (b n) t d')  # (b*n, t, d)
        
        # 3. 时序Q/K/V计算（基于时间步）
        q_temp = self.w_qs_temp(x_reshaped)  # (b*n, t, d)
        k_temp = self.w_ks_temp(x_reshaped)  # (b*n, t, d)
        v_temp = self.w_vs_temp(x_reshaped)  # (b*n, t, d)
        
        # 4. 时序自注意力计算（每个关键点的t个时间步之间做注意力）
        attn_temp = torch.matmul(q_temp, k_temp.transpose(-2, -1))  # (b*n, t, t)（时间步间的相似度）
        attn_temp = F.softmax(attn_temp / torch.sqrt(torch.tensor(d, dtype=torch.float32)), dim=-1)  # (b*n, t, t)
        res_temp = torch.matmul(attn_temp, v_temp)  # (b*n, t, d)（聚合时序信息）
        
        # 5. 恢复维度并输出
        res = rearrange(res_temp, '(b n) t d -> b t n d', b=b, n=n)  # (b, t, n, d)
        res = self.fc_out(res) + features  # 残差连接
        return res


class ptTransition(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_channel, in_channel), nn.ReLU(), nn.Linear(in_channel, out_channel))

    def forward(self, points):
        new_points = self.fc(points)
        return new_points
    

class SpatialTemporalTransformer(nn.Module):
    """整体模块：先空间注意力，后时序注意力"""
    def __init__(self, d_points, d_model, k_spatial, max_time_steps):
        super().__init__()
        # 特征映射（将输入特征维度d_points转换为d_model）
        self.fc_in = nn.Linear(d_points, d_model)
        # 第一阶段：空间邻域自注意力
        self.spatial_attn = SpatialAttention(d_model, k_spatial)
        # 第二阶段：时序自注意力
        self.temporal_attn = TemporalAttention(d_model, max_time_steps)
        # 输出映射（回到原始特征维度）
        self.fc_out = nn.Linear(d_model, d_points)

    def forward(self, xyz, features, b, t):
        """
        xyz: 点云坐标，(b*t, n, 3)
        features: 点云特征，(b*t, n, d_points)
        b: 原始batch size
        t: 时间步数量
        return: 最终输出特征，(b*t, n, d_points)
        """
        # 1. 特征映射到d_model维度
        x = self.fc_in(features)  # (b*t, n, d_model)
        
        # 2. 第一阶段：空间邻域自注意力（保持(b*t, n, d_model)）
        x_spatial = self.spatial_attn(xyz, x)  # (b*t, n, d_model)
        
        # 3. 恢复时间维度：(b*t, n, d_model) → (b, t, n, d_model)
        x_reshaped = rearrange(x_spatial, '(b t) n d -> b t n d', b=b, t=t)  # (b, t, n, d_model)
        
        # 4. 第二阶段：时序自注意力（处理时间序列）
        x_temporal = self.temporal_attn(x_reshaped)  # (b, t, n, d_model)
        
        # 5. 重新展平维度并映射回输出
        x_flat = rearrange(x_temporal, 'b t n d -> (b t) n d')  # (b*t, n, d_model)
        out = self.fc_out(x_flat)  # (b*t, n, d_points)
        return out


# *  Modified to support Iterative Farthest Point Sampling
class ptTransformerBlock(nn.Module):

    def __init__(self, d_points, d_model, k, IFPS=False) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(nn.Linear(3, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.fc_gamma = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        self.IFPS = IFPS

    def forward(self, xyz, features):
        if self.training:
            x = cp.checkpoint(self._forward, xyz, features)
        else:
            x = self._forward(xyz, features)
        return x
            

    # xyz: b x n x 3, features: b x n x f
    def _forward(self, xyz, features):
        if self.IFPS:
            # Flawed here, further testing required
            # local_xyz := [4, 32, 3]
            # local_idx := [4, 32]
            local_xyz, local_idx = anchor_points(xyz, K=self.k, device=xyz.device)
            # Here the 'NN' for each point is the same
            # So we simply unsqueeze it to match it with knn case
            local_xyz = local_xyz.unsqueeze(1).repeat(1, 21, 1, 1)
            local_idx = local_idx.unsqueeze(1).repeat(1, 21, 1)            
        else:
            # local_idx := [4, 799, 32]
            # local_xyz := [4, 799, 32, 3]
            _, local_idx, local_xyz = knn_points(xyz, xyz, K=self.k, return_nn=True)

        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), local_idx), index_points(self.w_vs(x), local_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - local_xyz)  # b x n x k x f

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn
    

class TemporalPtTransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k_spatial, k_temporal, max_time_steps):
        """
        时序点云Transformer块
        :param d_points: 点云特征维度（输入输出特征维度）
        :param d_model: 注意力内部特征维度
        :param k_spatial: 空间邻域KNN数量
        :param k_temporal: 时序邻域数量（每个时间步考虑前后k_temporal个时间步）
        :param max_time_steps: 最大时间步（用于时序编码初始化）
        """
        super().__init__()
        self.d_model = d_model
        self.k_spatial = k_spatial  # 空间邻域大小
        self.k_temporal = k_temporal  # 时序邻域大小

        # 1. 可学习时序编码（为每个时间步t创建嵌入）
        self.temporal_embedding = nn.Embedding(max_time_steps, d_model)

        # 2. 特征映射层（与原结构一致）
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)

        # 3. 位置编码（空间+时序）
        self.fc_delta_spatial = nn.Sequential(  # 空间位置差编码
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_delta_temporal = nn.Sequential(  # 时序位置差编码
            nn.Linear(1, d_model),  # 输入为时间步差（t - t'）
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

        # 4. 注意力权重映射（Q/K/V）
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)

    def forward(self, xyz, features):
        """
        前向传播（含checkpoint支持）
        :param xyz: 点云坐标，形状为 [bt, n, 3]（batch×time×num_points×3）
        :param features: 点云特征，形状为 [bt, n, f]（batch×time×num_points×features）
        :return: 更新后的特征 [bt, n, d_points] 和注意力图 [bt, n, k_total, 1]
        """
        xyz = xyz.reshape(-1, self.k_temporal, 21, 3)
        features = features.reshape(-1, self.k_temporal, 21, self.d_model)
        if self.training:
            # 训练时用checkpoint节省显存
            x, attn = cp.checkpoint(self._forward, xyz, features)
        else:
            x, attn = self._forward(xyz, features)
        x = x.reshape(-1, 21, self.d_model)
        return x, attn

    def _forward(self, xyz, features):
        b, t, n, _ = xyz.shape  # 解析维度：batch, time, num_points

        # --------------------------
        # 1. 加入时序编码
        # --------------------------
        # 生成时序索引 [0, 1, ..., t-1]，形状为 [t]
        temporal_indices = torch.arange(t, device=features.device)
        # 时序编码：[t, d_model] → 扩展为 [b, t, 1, d_model] 以匹配特征维度
        temp_enc = self.temporal_embedding(temporal_indices).unsqueeze(0).unsqueeze(2)  # [1, t, 1, d_model]
        temp_enc = temp_enc.expand(b, -1, n, -1)  # [b, t, n, d_model]

        # 特征映射 + 时序编码
        x = self.fc1(features)  # [b, t, n, d_model]
        x = x + temp_enc  # 加入时序编码

        # --------------------------
        # 2. 获取时空邻域（空间KNN + 时序邻域）
        # --------------------------
        # 空间邻域：对每个时间步单独计算KNN（保持时序独立性）
        # 将xyz重排为 [b*t, n, 3]，计算每个点的空间KNN
        xyz_flat = rearrange(xyz, 'b t n c -> (b t) n c')  # [b*t, n, 3]
        _, knn_spatial_idx, knn_spatial_xyz = knn_points(
            xyz_flat, xyz_flat, K=self.k_spatial, return_nn=True
        )  # 空间邻域索引：[b*t, n, k_spatial]，坐标：[b*t, n, k_spatial, 3]
        # 恢复时序维度：[b, t, n, k_spatial]
        knn_spatial_idx = rearrange(knn_spatial_idx, '(b t) n k -> b t n k', b=b, t=t)
        knn_spatial_xyz = rearrange(knn_spatial_xyz, '(b t) n k c -> b t n k c', b=b, t=t)

        # 时序邻域：每个时间步t考虑 [t-k_temporal, ..., t+k_temporal]（超出边界则取最近）
        # 生成时序邻域索引后，扩展到batch维度b
        knn_temporal_idx = self._get_temporal_neighbors(t, n, self.k_temporal, device=xyz.device)
        knn_temporal_idx = knn_temporal_idx.expand(b, -1, -1, -1)  # [b, t, n, k_total]（关键修正）

        # 合并时空邻域索引（总邻域数 = k_spatial + k_temporal）
        k_total = self.k_spatial + self.k_temporal
        # 空间邻域索引直接使用，时序邻域索引需要映射到全局索引（时间步+点索引）
        # 这里简化处理：将时序邻域的点特征和坐标提取出来
        # （注：更高效的实现可使用einops或索引技巧，此处为清晰起见分步处理）

        # --------------------------
        # 3. 计算Q/K/V
        # --------------------------
        q = self.w_qs(x)  # [b, t, n, d_model]

        # 空间邻域的K和V（从当前时间步的空间邻域提取）
        k_spatial = index_points_4dim(self.w_ks(x), knn_spatial_idx)  # [b, t, n, k_spatial, d_model]
        v_spatial = index_points_4dim(self.w_vs(x), knn_spatial_idx)  # [b, t, n, k_spatial, d_model]

        # 时序邻域的K和V（从相邻时间步的对应点提取）
        # 先将x重排为 [b, t, n, d_model] → 用时序索引提取邻域
        k_temporal = self._index_temporal_neighbors(self.w_ks(x), knn_temporal_idx)  # [b, t, n, k_temporal, d_model]
        v_temporal = self._index_temporal_neighbors(self.w_vs(x), knn_temporal_idx)  # [b, t, n, k_temporal, d_model]

        # 合并空间和时序的K、V
        k = torch.cat([k_spatial, k_temporal], dim=-2)  # [b, t, n, k_total, d_model]
        v = torch.cat([v_spatial, v_temporal], dim=-2)  # [b, t, n, k_total, d_model]

        # --------------------------
        # --------------------------
        # 4. 时空位置编码（修正时序位置差计算）
        # --------------------------
        # 空间位置差：当前点 - 空间邻域点 [b, t, n, k_spatial, 3]
        delta_spatial = xyz.unsqueeze(-2) - knn_spatial_xyz  # [b, t, n, 1, 3] - [b, t, n, k_spatial, 3]
        enc_spatial = self.fc_delta_spatial(delta_spatial)  # [b, t, n, k_spatial, d_model]

        # 时序位置差：当前时间步 - 时序邻域时间步（修正维度对齐）
        # 1. 生成当前时间步索引：形状 [1, t, 1, 1, 1]（5维，与knn_temporal_idx.unsqueeze(-1)维度一致）
        temporal_indices = torch.arange(t, device=xyz.device)  # [t]
        delta_temporal = temporal_indices.view(1, t, 1, 1, 1)  # 扩展为5维：[1, t, 1, 1, 1]

        # 2. knn_temporal_idx.unsqueeze(-1)的形状是 [b, t, n, k_temporal, 1]（5维）
        # 3. 计算时间差（利用广播机制）
        delta_temporal = delta_temporal - knn_temporal_idx.unsqueeze(-1).float()  # [b, t, n, k_temporal, 1]

        # 时序位置编码
        enc_temporal = self.fc_delta_temporal(delta_temporal)  # [b, t, n, k_temporal, d_model]

        # 合并位置编码
        pos_enc = torch.cat([enc_spatial, enc_temporal], dim=-2)  # [b, t, n, k_total, d_model]

        # --------------------------
        # 5. 注意力计算
        # --------------------------
        # Q扩展维度后与K+位置编码计算差异
        q_expanded = q.unsqueeze(-2)  # [b, t, n, 1, d_model]
        attn = self.fc_gamma(q_expanded - k + pos_enc)  # [b, t, n, k_total, d_model]
        # 注意力归一化（在邻域维度上做softmax）
        attn = F.softmax(attn / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32)), dim=-2)  # [b, t, n, k_total, d_model]

        # 聚合V（加权求和）
        res = torch.einsum('b t n k f, b t n k f -> b t n f', attn, v + pos_enc)  # [b, t, n, d_model]

        # --------------------------
        # 6. 输出映射 + 残差连接
        # --------------------------
        res = self.fc2(res) + features  # [b, t, n, d_points]（残差连接输入特征）
        return res, attn

    def _get_temporal_neighbors(self, t, n, k_temporal, device):
        """生成时序邻域索引：形状为 [1, t, n, k_total]（后续会扩展到batch维度）"""
        k_total = 2 * k_temporal + 1  # 每个时间步的邻域总数（前k + 后k + 当前）
        temporal_neighbors = []
        for curr_t in range(t):
            # 计算当前时间步的邻域范围（超出边界则用边界填充）
            start = max(0, curr_t - k_temporal)
            end = min(t, curr_t + k_temporal + 1)
            neighbors = list(range(start, end))
            # 填充不足的邻域（用最后一个索引补全）
            while len(neighbors) < k_total:
                neighbors.append(neighbors[-1])
            temporal_neighbors.append(neighbors[:k_total])  # [t, k_total]
        
        # 转换为tensor：[1, t, n, k_total]（先保留batch维度为1，后续扩展）
        temporal_idx = torch.tensor(temporal_neighbors, device=device).unsqueeze(0).unsqueeze(2)  # [1, t, 1, k_total]
        temporal_idx = temporal_idx.expand(-1, -1, n, -1)  # [1, t, n, k_total]
        return temporal_idx

    def _index_temporal_neighbors(self, x, temporal_idx):
        """
        x: [b, t, n, d_model]（时序特征）
        temporal_idx: [b, t, n, k_temporal]（时序邻域索引）
        """
        b, t, n, d = x.shape
        k_temporal = temporal_idx.shape[-1]
        
        # 重排x为 [b, n, t, d]（将时间步t放到第三维，便于索引）
        x_rearranged = x.permute(0, 2, 1, 3)  # [b, n, t, d]
        
        # 重排时序索引为 [b, n, t, k_temporal]（匹配x_rearranged的维度）
        temporal_idx_rearranged = temporal_idx.permute(0, 2, 1, 3)  # [b, n, t, k_temporal]
        
        # 调用index_points（此时x_rearranged是4维，temporal_idx_rearranged是4维，维度匹配）
        x_temporal = index_points_4dim(x_rearranged, temporal_idx_rearranged)  # [b, n, t, k_temporal, d]
        
        # 恢复维度顺序为 [b, t, n, k_temporal, d]
        return x_temporal.permute(0, 2, 1, 3, 4)


class ptTransformerBlock_CrossAttn(nn.Module):

    def __init__(self, d_points, d_model, k, expand_query_dim=False, IFPS=False) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(nn.Linear(3, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.fc_gamma = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.w_qs = nn.Linear(d_points, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        self.expand_query_dim = expand_query_dim
        self.IFPS = IFPS

        if expand_query_dim is True:
            self.fc_query = nn.Sequential(nn.Linear(d_points, d_points), nn.ReLU(), nn.Linear(d_points, d_points * 2))

    def forward(self, xyz, features, query):
        if self.training:
            x = cp.checkpoint(self._forward, xyz, features, query)
        else:
            x = self._forward(xyz, features, query)
        return x

    # xyz: b x n x 3, features: b x n x f, query = b x 799 x (3 + f)
    def _forward(self, xyz, features, query):
        query_xyz = query[:, :, :3]  # b x 799 x 3
        query_f = query[:, :, 3:]  # b x 799 x f

        if self.IFPS:
            local_xyz, local_idx = anchor_points(xyz, K=self.k, device=xyz.device)
            local_xyz = local_xyz.unsqueeze(1).repeat(1, 799, 1, 1)
            local_idx = local_idx.unsqueeze(1).repeat(1, 799, 1) 
        else:
            _, local_idx, local_xyz = knn_points(query_xyz, xyz, K=self.k, return_nn=True)
        # _, knn_idx, knn_xyz -> b x 799 x k, b x 799 x k, b x 799 x k x 3
        knn_features = index_points(features, local_idx)  # b x 799 x k x f

        pre = query_f  # b x 799 x f
        q = self.w_qs(query_f)  # b x 799 x d_model

        x = self.fc1(knn_features)  # b x 799 x k x d_model
        k = self.w_ks(x)  # k: b x 799 x k x d_model
        v = self.w_vs(x)  # v: b x 799 x k x d_model

        pos_enc = self.fc_delta(query_xyz[:, :, None] - local_xyz)  # b x 799 x k x d_model

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)  # b x 799 x 1 x d_model - b x 799 x k x d_model
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x 799 x k x d_model

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre  # b x 799 x f

        if self.expand_query_dim:
            res = self.fc_query(res)  # b x 799 x 2f

        return res, attn

