import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def orthgonalProj(xy, scale, transl, img_size=256):
    scale = scale * img_size
    transl = transl * img_size / 2 + img_size / 2
    return xy * scale + transl

class GraphConv(nn.Module):
    def __init__(self, num_joint, in_features, out_features):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.adj = nn.Parameter(torch.eye(num_joint).float(), requires_grad=True)

    def laplacian(self, A_hat):
        D_hat = torch.sum(A_hat, 1, keepdim=True) + 1e-5
        L = 1 / D_hat * A_hat
        return L

    def forward(self, x):
        batch = x.size(0)
        A_hat = self.laplacian(self.adj.to(x.device))
        A_hat = A_hat.unsqueeze(0).repeat(batch, 1, 1)
        out = self.fc(torch.matmul(A_hat, x))
        return out

class GraphRegression(nn.Module):
    def __init__(self, node_num, in_dim, out_dim, layer_num=2, last=True):
        super(GraphRegression, self).__init__()
        self.num_node = node_num
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = nn.LeakyReLU(0.1)
        self.reg = nn.Sequential()
        self.reg.add_module('ln', nn.LayerNorm(self.in_dim))
        for i in range(layer_num-1):
            self.reg.add_module('gcn_i', GraphConv(node_num, self.in_dim, self.in_dim))
            self.reg.add_module('activate_i', self.activation)
        self.reg.add_module(f'dp', nn.Dropout(0.1))
        self.reg.add_module(f'gcn_{layer_num-1}', GraphConv(node_num, self.in_dim, self.out_dim))
        if not last:
            self.reg.add_module(f'activate_{layer_num-1}', self.activation)

    def forward(self, graph, shortcut=False):
        in_graph = graph
        out_graph = self.reg(graph)
        if shortcut:
            assert in_graph.shape[2] == out_graph.shape[2]
            return out_graph + in_graph
        else:
            return out_graph
        
class WeightGraphRegression(nn.Module):
    def __init__(self, node_num, in_dim, out_dim, layer_num=2, last=True):
        super(WeightGraphRegression, self).__init__()
        self.num_node = node_num
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = nn.LeakyReLU(0.1)
        self.reg = nn.Sequential()
        self.reg.add_module('ln', nn.LayerNorm(self.in_dim))
        for i in range(layer_num-1):
            self.reg.add_module('gcn_i', GraphConv(node_num, self.in_dim, self.in_dim))
            self.reg.add_module('activate_i', self.activation)
        self.reg.add_module(f'dp', nn.Dropout(0.1))
        self.reg.add_module(f'gcn_{layer_num-1}', GraphConv(node_num, self.in_dim, self.out_dim))
        if not last:
            self.reg.add_module(f'activate_{layer_num-1}', self.activation)

    def forward(self, graph, weight, shortcut=False):
        in_graph = graph * weight
        out_graph = self.reg(graph)
        if shortcut:
            assert in_graph.shape[2] == out_graph.shape[2]
            return out_graph + in_graph
        else:
            return out_graph

class MLP_res_block(nn.Module):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    def __init__(self, in_dim, hid_dim, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, in_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _ff_block(self, x):
        x = self.fc2(self.dropout1(F.relu(self.fc1(x))))
        return self.dropout2(x)

    def forward(self, x):
        x = x + self._ff_block(self.layer_norm(x))
        return x

class SelfAttn(nn.Module):
    def __init__(self, f_dim, hid_dim=None, n_heads=4, d_q=None, d_v=None, num_view=8, dropout=0.1):
        super().__init__()
        if d_q is None:
            d_q = f_dim // n_heads
        if d_v is None:
            d_v = f_dim // n_heads
        if hid_dim is None:
            hid_dim = f_dim

        self.n_heads = n_heads
        self.num_view = num_view
        self.d_q = d_q
        self.d_v = d_v
        self.norm = d_q ** 0.5
        self.f_dim = f_dim

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.w_qs = nn.Linear(f_dim, n_heads * d_q)
        self.w_ks = nn.Linear(f_dim, n_heads * d_q)
        self.w_vs = nn.Linear(f_dim, n_heads * d_v)

        self.layer_norm = nn.LayerNorm(f_dim, eps=1e-6)
        self.fc = nn.Linear(n_heads * d_v, f_dim)

        self.ff = MLP_res_block(f_dim, hid_dim, dropout)

    def self_attn(self, x, valid=None, mask=False):
        BS, V, f = x.shape

        q = self.w_qs(x).view(BS, -1, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        k = self.w_ks(x).view(BS, -1, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        v = self.w_vs(x).view(BS, -1, self.n_heads, self.d_v).transpose(1, 2)  # BS x h x V x v

        attn = torch.matmul(q, k.transpose(-1, -2)) / self.norm  # bs, h, V, V

        if mask and self.training:
            # mask = torch.rand(168, 168)
            # mask = torch.where(mask >= 0.5, float(0), float('-inf'))
            use_view_num = random.randint(1, self.num_view-1)
            joint_num = 21
            mask = torch.zeros(joint_num * self.num_view, joint_num * self.num_view)
            for view_idx in range(self.num_view):
                use_view_set = random.sample([i for i in range(self.num_view) if i != view_idx], self.num_view - use_view_num)
                use_view_set.sort()
                for use_idx in use_view_set:
                    mask[view_idx*joint_num:(view_idx+1)*joint_num, use_idx*joint_num:(use_idx+1)*joint_num] = torch.zeros_like(mask[view_idx*joint_num:(view_idx+1)*joint_num, use_idx*joint_num:(use_idx+1)*joint_num]) + float('-inf')
            # mask_joint = torch.rand(joint_num * cfg.num_view, joint_num * cfg.num_view)
            # mask_joint = torch.where(mask_joint > 0.5, float(0), float('-inf'))
            # mask = mask_joint + mask
            mask = mask.unsqueeze(0).unsqueeze(1).repeat(attn.shape[0], attn.shape[1], 1, 1)
            attn = attn + mask.to(x.device)
        
        if valid is not None:
            joint_num = 21
            valid = valid.view(-1, self.num_view)
            batch = valid.shape[0]
            valid = valid.view(batch, 1, self.num_view).repeat(1, joint_num, 1).permute(0, 2, 1).reshape(batch, 1, -1)
            valid = torch.where(valid == 0, float(-2**32+1), float(0)).repeat(1, self.num_view * joint_num, 1)
            valid = valid.permute(0, 2, 1) + valid
            valid = valid.unsqueeze(1)
            attn = attn + valid

        attn = F.softmax(attn, dim=-1)  # bs, h, V, V
        attn = self.dropout1(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(BS, V, -1)
        out = self.dropout2(self.fc(out))
        return out

    def forward(self, x, valid=None, mask=False):
        BS, V, f = x.shape
        assert f == self.f_dim

        x = x + self.self_attn(self.layer_norm(x), valid, mask)
        x = self.ff(x)

        return x
    
class VAIGB(nn.Module):
    def __init__(self, backbone_channels, num_FMs, feature_size, num_kps, position_embedding=False):
        super(VAIGB, self).__init__()
        self.backbone_channels = backbone_channels
        self.feature_size = feature_size
        self.num_kps = num_kps
        self.num_FMs = num_FMs
        self.group = nn.Sequential(
            nn.Conv2d(self.backbone_channels, self.num_FMs * self.num_kps, 1),
            nn.LeakyReLU(0.1)
        )
        if position_embedding:
            self.position_embeddings = nn.Embedding(self.num_kps, self.feature_size * self.num_FMs)
        self.use_position_embedding = position_embedding

    def forward(self, x):
        init_graph = self.group(x).reshape(-1, self.num_kps, self.feature_size * self.num_FMs)
        if self.use_position_embedding:
            position_ids = torch.arange(self.num_kps, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).repeat(x.shape[0], 1)
            position_embeddings = self.position_embeddings(position_ids)
            init_graph += position_embeddings
        return init_graph
    

class TAIGB(nn.Module):
    def __init__(self, backbone_channels, num_Ts, feature_size, num_kps, position_embedding=False):
        super(TAIGB, self).__init__()
        self.backbone_channels = backbone_channels
        self.feature_size = feature_size
        self.num_kps = num_kps
        self.num_tempos = num_Ts
        self.group = nn.Sequential(
            nn.Conv2d(self.backbone_channels, self.num_tempos * self.num_kps, 1),
            nn.LeakyReLU(0.1)
        )
        if position_embedding:
            self.position_embeddings = nn.Embedding(self.num_kps, self.feature_size * self.num_tempos)
        self.use_position_embedding = position_embedding

    def forward(self, x):
        init_graph = self.group(x).reshape(-1, self.num_kps, self.feature_size * self.num_tempos)
        if self.use_position_embedding:
            position_ids = torch.arange(self.num_kps, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).repeat(x.shape[0], 1)
            position_embeddings = self.position_embeddings(position_ids)
            init_graph += position_embeddings
        return init_graph
    

class SAIGB(nn.Module):
    def __init__(self, backbone_channels, num_FMs, feature_size, num_kps, position_embedding=False):
        super(SAIGB, self).__init__()
        self.backbone_channels = backbone_channels
        self.feature_size = feature_size
        self.num_kps = num_kps
        self.num_FMs = num_FMs
        self.group = nn.Sequential(
            nn.Conv2d(self.backbone_channels, self.num_FMs * self.num_kps, 1),
            nn.LeakyReLU(0.1)
        )
        if position_embedding:
            self.position_embeddings = nn.Embedding(self.num_kps, self.feature_size * self.num_FMs)
        self.use_position_embedding = position_embedding

    def forward(self, x):
        init_graph = self.group(x).reshape(-1, self.num_kps, self.feature_size * self.num_FMs)
        if self.use_position_embedding:
            position_ids = torch.arange(self.num_kps, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).repeat(x.shape[0], 1)
            position_embeddings = self.position_embeddings(position_ids)
            init_graph += position_embeddings
        return init_graph
    

class PixelFeatureSampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, point_uv, s_feat):
        return torch.nn.functional.grid_sample(s_feat, point_uv.unsqueeze(2), align_corners=True)[..., 0]