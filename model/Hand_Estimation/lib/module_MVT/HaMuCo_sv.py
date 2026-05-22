import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
import torchgeometry as tgm

from torchvision import models
from lib.utils.config import CN
from .Graph import PixelFeatureSampler
from lib.models.bricks.conv import ConvBlock
from lib.utils.triangulation import batch_triangulate_dlt_cfd_torch
from lib.utils.transform import batch_cam_extr_transf
from lib.models.backbones import build_backbone
from lib.module.ManoDecoder import ManoDecoder
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


class SingleViewNet(nn.Module):
    def __init__(self, cfg: CN, cfg_preset: CN):
        super(SingleViewNet, self).__init__()
        self.cfg = cfg
        self.channel = cfg.CHANNEL
        self.joint_num = cfg.NUM_JOINT
        self.num_FMs = cfg.NUM_FMS
        self.root_joint = cfg.ROOT_IDX
        # self.num_seq = cfg.NUM_SEQ
        self.out_dim = cfg.OUT_DIM
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

        self.feat_delayer = nn.ModuleList([
            ConvBlock(self.feat_size[1] + self.feat_size[0], self.feat_size[1], kernel_size=3, relu=True, norm='bn'),
            ConvBlock(self.feat_size[2] + self.feat_size[1], self.feat_size[2], kernel_size=3, relu=True, norm='bn'),
            ConvBlock(self.feat_size[3] + self.feat_size[2], self.feat_size[3], kernel_size=3, relu=True, norm='bn'),
        ])
        self.feat_in = ConvBlock(self.feat_size[3], self.feat_size[2], kernel_size=1, padding=0, relu=False, norm=None)

        self.layer_mano = nn.Sequential(
            nn.Linear(self.channel, self.channel),
            nn.LeakyReLU(0.1),
            nn.Linear(self.channel, 6 * 16 + 10 + 3)
        )       

        self.fc_sigma = Linear(self.channel, 21*2)

        prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
        masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
        self.flow = RealNVP(nets, nett, masks, prior)

        # Instantiate MANO decoder
        self.decoder = ManoDecoder(9, 0.4, 256)

        self.jaf_extractor_64 = PixelFeatureSampler()
        self.jaf_extractor_32 = PixelFeatureSampler()
        self.jaf_extractor_16 = PixelFeatureSampler()

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

    def forward(self, batch):
        img = batch['image']  # (B, N, 3, 256, 256)
        K = batch['target_cam_intr']  # (B, N, 3, 3)
        T_c2m = batch['target_cam_extr']  # (B, N, 4, 4)
        B, N, C, H, W = img.shape
        inp_img_shape = (H, W)  # H, W

        img_all = img.view(-1, C, H, W)  # (BN, C, H, W)
        img_feats, feat_high, global_feat = self.extract_img_feat(img_all)  # [(B, N, C, H, W), ...]\
        mlvl_feat = self.feat_decode(img_feats)  # (BxN, 128, 32, 32)
        mlvl_feat = mlvl_feat.view(B, N, *mlvl_feat.shape[1:])  # (B, N, 128, 32, 32)

        pred_mano_params = self.layer_mano(global_feat)    # (B*N, 109)

        pred_hand_pose = pred_mano_params[:, :96]    # (B*N, 96)
        pred_shape = pred_mano_params[:, 96:106]     # (B*N, 10)
        pred_cam = pred_mano_params[:, 106:]         # (B*N, 3)

        # positive scale
        pred_cam = torch.cat((F.relu(pred_cam[:, 0:1]), pred_cam[:, 1:]), dim=1).view(B*N, 3)

        out_sigma = self.fc_sigma(global_feat).reshape(B*N, self.joint_num, -1)
        sigma = out_sigma.sigmoid()
        scores = 1 - sigma
        scores = torch.mean(scores, dim=2, keepdim=True)

        coord_xyz, coord_uv, pose_euler, shape, cam = self.decoder(pred_hand_pose, pred_shape, pred_cam)

        ref_joints = (coord_uv[:, 778:].reshape(B, N, self.joint_num, 2) + 1) * (H // 2)
        ref_joints_master = batch_triangulate_dlt_cfd_torch(ref_joints, K, T_c2m, scores.detach().view(B, N, self.joint_num))   # (B, J, 3)
        ref_proj = ref_joints_master.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, J, 3)
        ref_joints_in_cam = batch_cam_extr_transf(T_c2m, ref_proj).flatten(0, 1)  # (B*N, J, 3)
        # ref_joints_in_cam = ref_joints_in_cam / (0.4 / 2)
        
        if self.training and batch is not None:
            gt_uv = batch['target_pseudo_uv'].reshape(coord_uv[:, 778:].shape)
            bar_mu = ((coord_uv[:, 778:] - gt_uv) / sigma).float()
            # (B, K, 2)
            log_phi = self.flow.log_prob(bar_mu.reshape(-1, 2)).reshape(B*N, self.joint_num, 1)

            nf_loss = torch.log(sigma) - log_phi
        else:
            nf_loss = None
        
        output = {
            'cam_intr': K,
            'cam_extr': torch.linalg.inv(T_c2m),
            'master_id': batch['master_id'],
            "inp_img_shape": inp_img_shape, 
            'mlvl_feat': mlvl_feat,  # (B, N, 128, 32, 32)
            'coord_xyz': coord_xyz,
            'coord_uv': coord_uv,
            'pose_euler': pose_euler,
            'shape': shape,
            'cam': cam,
            'ref_joints_master': ref_joints_master,
            'ref_joints_in_cam': ref_joints_in_cam,
            'pred_mano_params_sv': pred_mano_params,
            'sigma': sigma,
            'confidence': scores.float(),
            'nf_loss': nf_loss
        }

        return output