import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F
import numpy as np

from functools import partial
from torchvision import models
from lib.utils.config import CN
from lib.utils.triangulation import batch_triangulate_dlt_cfd_torch
from lib.utils.transform import batch_cam_extr_transf
from lib.models.backbones import build_backbone
from lib.models.bricks.conv import ConvBlock
from lib.models.integal_pose import integral_heatmap2d, norm_heatmap
from rlepose.models.layers.real_nvp import RealNVP


def nets():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2), nn.Tanh())


def nett():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2))


class Linear(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super(Linear, self).__init__()
        self.bias = bias
        self.norm = norm
        self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x):
        y = x.matmul(self.linear.weight.t())

        if self.norm:
            x_norm = torch.norm(x, dim=1, keepdim=True)
            y = y / x_norm

        if self.bias:
            y = y + self.linear.bias
        return y


class KeypointPredictNet(nn.Module):
    def __init__(self, cfg: CN, cfg_preset: CN):
        super(KeypointPredictNet, self).__init__()
        self.num_verts = cfg.NUM_VERTS
        self.num_joints = cfg.NUM_JOINTS
        self.num_view = cfg.NUM_VIEW
        self.feat_dim = cfg.FEAT_DIM
        self.center_idx = cfg.ROOT_IDX
        self.image_size = cfg.IMAGE_SIZE
        self.data_preset_cfg = cfg_preset

        self.img_backbone = build_backbone(cfg.BACKBONE, data_preset=self.data_preset_cfg)
        assert self.img_backbone.name in ["resnet18", "resnet34", "resnet50"], "Wrong backbone for PETR"
        if self.img_backbone.name == "resnet18":
            self.feat_size = (512, 256, 128, 64)
        elif self.img_backbone.name == "resnet34":
            self.feat_size = (512, 256, 128, 64)
        elif self.img_backbone.name == "resnet50":
            self.feat_size = (2048, 1024, 512, 256)
        self.channel = self.feat_size[0]
        
        self.fc_coord = Linear(self.channel, self.num_joints * 2)
        self.fc_sigma = Linear(self.channel, self.num_joints * 2, norm=False)

        prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
        masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
        self.flow = RealNVP(nets, nett, masks, prior)

        self.feat_delayer = nn.ModuleList([
            ConvBlock(self.feat_size[1] + self.feat_size[0], self.feat_size[1], kernel_size=3, relu=True, norm='bn'),
            ConvBlock(self.feat_size[2] + self.feat_size[1], self.feat_size[2], kernel_size=3, relu=True, norm='bn'),
            ConvBlock(self.feat_size[3] + self.feat_size[2], self.feat_size[3], kernel_size=3, relu=True, norm='bn'),
        ])
        self.feat_in = ConvBlock(self.feat_size[3], self.feat_size[2], kernel_size=1, padding=0, relu=False, norm=None)
        
    def feat_decode(self, mlvl_feats):
        mlvl_feats_rev = list(reversed(mlvl_feats))
        x = mlvl_feats_rev[0]
        for i, fde in enumerate(self.feat_delayer):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = torch.cat((x, mlvl_feats_rev[i + 1]), dim=1)
            x = fde(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # (BxN, 64, 32, 32)
        x = self.feat_in(x)  # (BxN, 128, 32, 32)
        return x

    def extract_img_feat(self, img):
        B = img.size(0)
        if img.dim() == 5:
            if img.size(0) == 1 and img.size(1) != 1:  # (1, N, C, H, W)
                img = img.squeeze()  # (N, C, H, W)
            else:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)

        img_feats = self.img_backbone(image=img)
        global_feat = img_feats["res_layer4_mean"]  # [B*N,512]
        if isinstance(img_feats, dict):
            """img_feats for ResNet 34: 
                torch.Size([BN, 64, 64, 64])
                torch.Size([BN, 128, 32, 32])
                torch.Size([BN, 256, 16, 16])
                torch.Size([BN, 512, 8, 8])
            """
            img_feats = list([v for v in img_feats.values() if len(v.size()) == 4])

        feat_high = img_feats[-1]  # (BxN, C, Hh, Wh)

        return img_feats, feat_high, global_feat

    def forward(self, inputs):
        img = inputs['image']
        B, N, C, H, W = img.shape
        inp_img_shape = (H, W)  # H, W
        
        img_all = img.view(-1, C, H, W)  # (BN, C, H, W)

        # 1. extract feature
        img_feats, feat_high, global_feat = self.extract_img_feat(img_all)  # [(B, N, C, H, W), ...]

        mlvl_feat = self.feat_decode(img_feats)  # (BxN, 128, 32, 32)
        mlvl_feat = mlvl_feat.view(B, N, *mlvl_feat.shape[1:])  # (B, N, 128, 32, 32)

        # 2. get rle confidence
        out_coord = self.fc_coord(global_feat).reshape(B*N, self.num_joints, 2)
        assert out_coord.shape[2] == 2
        out_sigma = self.fc_sigma(global_feat).reshape(B*N, self.num_joints, -1)

        # (B, N, 2)
        pred_jts = out_coord.reshape(B*N, self.num_joints, 2)
        sigma = out_sigma.reshape(B*N, self.num_joints, -1).sigmoid()
        scores = 1 - sigma

        scores = torch.mean(scores, dim=2, keepdim=True)

        # 3. sample error
        sampled_error = self.flow.sample(B*N)

        if self.training and inputs is not None:
            gt_uv = inputs['target_pseudo_uv'].reshape(pred_jts.shape)
            bar_mu = (pred_jts - gt_uv) / sigma
            # (B, K, 2)
            log_phi = self.flow.log_prob(bar_mu.reshape(-1, 2)).reshape(B*N, self.num_joints, 1)

            nf_loss = torch.log(sigma) - log_phi
        else:
            nf_loss = None

        K = inputs['target_cam_intr']  # (B, N, 3, 3)
        T_c2m = inputs['target_cam_extr']  # (B, N, 4, 4)
        # ref_joints = batch_triangulate_dlt_torch(uv_coord_im, K, T_c2m)  # (B, J, 3)
        uv_coord_im = (pred_jts.reshape(B, N, self.num_joints, 2) + 1) * H / 2
        ref_joints = batch_triangulate_dlt_cfd_torch(uv_coord_im, K, T_c2m, scores.detach().reshape(B, N, self.num_joints))   # (B, J, 3)
        ref_proj = ref_joints.unsqueeze(1).repeat(1, N, 1, 1).float()  # (B, N, J, 3)
        ref_joints_in_cam = batch_cam_extr_transf(T_c2m, ref_proj).flatten(0, 1)  # (B*N, J, 3)

        # kpts_hm = uv_coord - 0.5  # normalize to [-0.5, 0.5]

        outputs = {
            'cam_intr': K,
            'cam_extr': torch.linalg.inv(T_c2m),
            'master_id': inputs['master_id'],
            "inp_img_shape": inp_img_shape,
            'img_feats': img_feats, 
            'mlvl_feat': mlvl_feat,  # (B, N, 128, 32, 32)
            # 'pred_hmap': uv_hmap,  # (BN, J, 32, 32)
            'feat_high': feat_high,  # (BN, C, Hh, Wh)
            'global_feat': global_feat,  # (BN, 512)
            'pred_jts': pred_jts,
            # 'pred_joints_uv': uv_coord_im,  # (B, N, J, 2)
            'ref_joints_master': ref_joints,  # (B, J, 3)
            'ref_joints_in_cam': ref_joints_in_cam,  # (B*N, J, 3)
            # 'uv_norm': uv_coord * 2 - 1,
            # 'uvc': uvc,
            # 'confidence': uv_confi  # (BN, J)
            'sigma': sigma,
            'maxvals': scores.float(),
            'nf_loss': nf_loss
        }

        return outputs