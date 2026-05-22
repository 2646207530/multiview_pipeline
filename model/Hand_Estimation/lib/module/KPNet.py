import torch
import torch.nn as nn

from functools import partial
from torchvision import models
from lib.utils.config import CN
from lib.utils.triangulation import batch_triangulate_dlt_cfd_torch
from lib.utils.transform import batch_cam_extr_transf
from lib.models.backbones import build_backbone
from lib.models.bricks.conv import ConvBlock
from lib.models.integal_pose import integral_heatmap2d, norm_heatmap
import torch.nn.functional as F


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
        
        self.uv_delayer = nn.ModuleList([
            ConvBlock(self.feat_size[1] + self.feat_size[0], self.feat_size[1], kernel_size=3, relu=True, norm='bn'),
            ConvBlock(self.feat_size[2] + self.feat_size[1], self.feat_size[2], kernel_size=3, relu=True, norm='bn'),
            ConvBlock(self.feat_size[3] + self.feat_size[2], self.feat_size[3], kernel_size=3, relu=True, norm='bn'),
        ])
        self.uv_out = ConvBlock(self.feat_size[3], self.num_joints, kernel_size=1, padding=0, relu=False, norm=None)
        self.uv_in = ConvBlock(self.num_joints, self.feat_size[2], kernel_size=1, padding=0, relu=True, norm='bn')

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

    def uv_decode(self, mlvl_feats, scale_factor):

        mlvl_feats_rev = list(reversed(mlvl_feats))
        x = mlvl_feats_rev[0]
        for i, de in enumerate(self.uv_delayer):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = torch.cat((x, mlvl_feats_rev[i + 1]), dim=1)
            x = de(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # (BxN, 64, 32, 32)
        # uv_hmap = self.uv_out(x)  # (BxN, 21, 32, 32)
        uv_hmap = torch.sigmoid(self.uv_out(x) * scale_factor)  # (BxN, 21, 32, 32)
        uv_feat = self.uv_in(uv_hmap)  # (BxN, 128, 32, 32)

        assert uv_hmap.shape[1:] == (21, 32, 32), uv_hmap.shape
        assert uv_feat.shape[1:] == (128, 32, 32), uv_feat.shape

        return uv_hmap, uv_feat

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

        # 2. get heatmap and confidence
        uv_hmap, uv_feat = self.uv_decode(img_feats, scale_factor=0.6)  # (BxN, 21, 32, 32), (BxN, 128, 32, 32)
        uv_pdf = uv_hmap.reshape(*uv_hmap.shape[:2], -1)  # (BxN, 21, 32x32)
        # uv_pdf = norm_heatmap('sigmoid', uv_hmap)  
        uv_confi = torch.max(uv_pdf, dim=-1).values  # (BxN, 21)
        uv_pdf = uv_pdf / (uv_pdf.sum(dim=-1, keepdim=True) + 1e-6)
        uv_pdf = uv_pdf.contiguous().view(B*N, self.num_joints,
                                          *uv_feat.shape[-2:])  # (BxN, 21, 32, 32)
        uv_coord = integral_heatmap2d(uv_pdf)  #(BxN, 21, 2), range 0~1
        uv_coord_im = torch.einsum("bij, j->bij", uv_coord, torch.tensor([W, H]).to(uv_coord.device))  # range 0~W,H
        uv_coord_im = uv_coord_im.view(B, N, self.num_joints, 2)  # (B, N, 21, 2)
        uvc = torch.cat(((uv_coord * 2 - 1), uv_confi.unsqueeze(-1)), dim=-1)  # (BN, J, 3)

        K = inputs['target_cam_intr']  # (B, N, 3, 3)
        T_c2m = inputs['target_cam_extr']  # (B, N, 4, 4)
        # ref_joints = batch_triangulate_dlt_torch(uv_coord_im, K, T_c2m)  # (B, J, 3)
        ref_joints = batch_triangulate_dlt_cfd_torch(uv_coord_im, K, T_c2m, uv_confi.detach().view(B, N, self.num_joints))   # (B, J, 3)
        ref_proj = ref_joints.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, J, 3)
        ref_joints_in_cam = batch_cam_extr_transf(T_c2m, ref_proj).flatten(0, 1)  # (B*N, J, 3)

        # kpts_hm = uv_coord - 0.5  # normalize to [-0.5, 0.5]

        outputs = {
            'cam_intr': K,
            'cam_extr': torch.linalg.inv(T_c2m),
            'master_id': inputs['master_id'],
            "inp_img_shape": inp_img_shape,
            'img_feats': img_feats, 
            'mlvl_feat': mlvl_feat,  # (B, N, 128, 32, 32)
            'pred_hmap': uv_hmap,  # (BN, J, 32, 32)
            'feat_high': feat_high,  # (BN, C, Hh, Wh)
            'global_feat': global_feat,  # (BN, 512)
            'pred_joints_uv': uv_coord_im,  # (B, N, J, 2)
            'ref_joints_master': ref_joints,  # (B, J, 3)
            'ref_joints_in_cam': ref_joints_in_cam,  # (B*N, J, 3)
            # 'uv_norm': uv_coord * 2 - 1,
            'uvc': uvc,
            'confidence': uv_confi  # (BN, J)
        }

        return outputs