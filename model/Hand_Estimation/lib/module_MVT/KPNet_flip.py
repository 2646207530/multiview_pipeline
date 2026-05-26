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
        self.image_size = cfg.IMAGE_SIZE
        self.data_preset_cfg = cfg_preset

        self.img_backbone = build_backbone(cfg.BACKBONE, data_preset=self.data_preset_cfg)
        # assert self.img_backbone.name in ["resnet18", "resnet34", "resnet50",'hamer'], "Wrong backbone for PETR"
        if self.img_backbone.name == "resnet18":
            self.feat_size = (512, 256, 128, 64)
        elif self.img_backbone.name == "resnet34":
            self.feat_size = (512, 256, 128, 64)
        elif self.img_backbone.name == "resnet50":
            self.feat_size = (2048, 1024, 512, 256)
        elif self.img_backbone.name == "hamer":
            self.feat_size = (1280,)
        elif "convnext" in self.img_backbone.name:
            self.feat_size = tuple(reversed(self.img_backbone.embed_dims))

        self.channel = self.feat_size[0]
        if self.img_backbone.name != "hamer":
            self.uv_delayer = nn.ModuleList([
                ConvBlock(self.feat_size[1] + self.feat_size[0], self.feat_size[1], kernel_size=3, relu=True,
                          norm='bn'),
                ConvBlock(self.feat_size[2] + self.feat_size[1], self.feat_size[2], kernel_size=3, relu=True,
                          norm='bn'),
                ConvBlock(self.feat_size[3] + self.feat_size[2], self.feat_size[3], kernel_size=3, relu=True,
                          norm='bn'),
            ])
            self.uv_out = ConvBlock(self.feat_size[3], self.num_joints, kernel_size=1, padding=0, relu=False, norm=None)
            self.uv_in = ConvBlock(self.num_joints, self.feat_size[2], kernel_size=1, padding=0, relu=True, norm='bn')

            self.feat_delayer = nn.ModuleList([
                ConvBlock(self.feat_size[1] + self.feat_size[0], self.feat_size[1], kernel_size=3, relu=True,
                          norm='bn'),
                ConvBlock(self.feat_size[2] + self.feat_size[1], self.feat_size[2], kernel_size=3, relu=True,
                          norm='bn'),
                ConvBlock(self.feat_size[3] + self.feat_size[2], self.feat_size[3], kernel_size=3, relu=True,
                          norm='bn'),
            ])
            self.feat_in = ConvBlock(self.feat_size[3], self.feat_size[2], kernel_size=1, padding=0, relu=False,
                                     norm=None)

        else:
            self.uv_decoder = nn.Sequential(
                nn.ConvTranspose2d(self.feat_size[0], 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                ConvBlock(256, 64, kernel_size=3, relu=True, norm='bn')
            )
            self.uv_out = ConvBlock(64, self.num_joints, kernel_size=1, padding=0, relu=False, norm=None)

            # 2. 同样需要为 feat_decode 准备一个单尺度的特征解码器
            self.feat_decoder = nn.Sequential(
                nn.ConvTranspose2d(self.feat_size[0], 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                # 最终输出通道设为 128，与原版 feat_in 的输出维度严格对齐
                ConvBlock(256, 128, kernel_size=3, relu=True, norm='bn')
            )

            self.feat_high_adapter = nn.Sequential(
                nn.Conv2d(self.feat_size[0], 512, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )

        self.uv_in = ConvBlock(self.num_joints, 128, kernel_size=1, padding=0, relu=True, norm='bn')

    def feat_decode(self, mlvl_feats):
        if self.img_backbone.name == "hamer":
            single_feat = mlvl_feats[0] if isinstance(mlvl_feats, (list, tuple)) else mlvl_feats
            # 输入 (BxN, 1280, 16, 16) -> 输出 (BxN, 128, 32, 32)
            x = self.feat_decoder(single_feat)
            return x
        else:
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

        if self.img_backbone.name == "hamer":
            single_feat = mlvl_feats[0] if isinstance(mlvl_feats, (list, tuple)) else mlvl_feats
            # 输入 (BxN, 1280, 16, 16) -> 输出 (BxN, 64, 32, 32)
            x = self.uv_decoder(single_feat)

            uv_hmap = torch.sigmoid(self.uv_out(x) * scale_factor)  # (BxN, 21, 32, 32)
            uv_feat = self.uv_in(uv_hmap)

        else:

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

        # assert uv_hmap.shape[1:] == (21, 32, 32), uv_hmap.shape
        # assert uv_feat.shape[1:] == (128, 32, 32), uv_feat.shape

        return uv_hmap, uv_feat

    def extract_img_feat(self, img):
        B = img.size(0)
        if img.dim() == 5:
            if img.size(0) == 1 and img.size(1) != 1:  # (1, N, C, H, W)
                img = img.squeeze()  # (N, C, H, W)
            else:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)

        if self.img_backbone.name == "hamer":
            out = self.img_backbone(image=img)
            if isinstance(out, dict):
                feat_high_raw = list(out.values())[0] if len(out) == 1 else out.get("res_layer4", list(out.values())[0])
            else:
                feat_high_raw = out[0]

            # 1. 原始的 1280 维特征交给内部的 feat_decode / uv_decode 列表使用
            img_feats = [feat_high_raw]

            # 2. 将高层特征降维为 512，适配外部所有对 ResNet34 设计的头部网络
            feat_high = self.feat_high_adapter(feat_high_raw)
            # 全局特征同样基于降维后的特征提取，使得外部也能拿到 512 维的 vector
            global_feat = feat_high.mean(dim=[2, 3]).flatten(1)

        elif "convnext" in self.img_backbone.name.lower():
            # 使用 convnext_dino 中提供的接口，提取 4 个阶段的特征和 class tokens
            # n=4 表示提取所有 4 个层级，reshape=True 确保输出为 (B, C, H, W) 形状的 2D 特征图
            outputs, class_tokens = self.img_backbone.forward_features_seq_out(
                img, n=4, reshape=True, return_class_token=True, norm=True
            )

            # img_feats 是一个包含 4 个 Tensor 的列表，分辨率从大到小，通道数从小到大
            img_feats = list(outputs)
            # 最后一层作为高层语义特征
            feat_high = img_feats[-1]
            # 获取最后一层的 class token 作为全局特征展平向量 (BxN, embed_dim)
            global_feat = class_tokens[-1]

        else:
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
        pse_vis = inputs['pse_joints_vis'].flatten(0, 2).unsqueeze(-1)
        # pse_vis = inputs['target_joints_vis'].flatten(0, 2).unsqueeze(-1)
        B, T, N, C, H, W = img.shape
        inp_img_shape = (H, W)  # H, W

        img_all = img.view(-1, C, H, W)  # (BTN, C, H, W)

        # 1. extract feature
        img_feats, feat_high, global_feat = self.extract_img_feat(img_all)  # [(B, T, N, C, H, W), ...]

        mlvl_feat = self.feat_decode(img_feats)  # (BTN, 128, 32, 32)
        mlvl_feat = mlvl_feat.view(B * T, N, *mlvl_feat.shape[1:])  # (BT, N, 128, 32, 32)

        # 2. get heatmap and confidence
        uv_hmap, uv_feat = self.uv_decode(img_feats, scale_factor=0.5)  # (BTN, 21, 32, 32), (BTN, 128, 32, 32)
        uv_pdf = uv_hmap.reshape(*uv_hmap.shape[:2], -1)  # (BTN, 21, 32x32)
        # uv_pdf = norm_heatmap('sigmoid', uv_hmap)
        uv_confi = torch.max(uv_pdf, dim=-1).values  # (BTN, 21)
        uv_confi = uv_confi.unsqueeze(-1)
        uv_pdf = uv_pdf / (uv_pdf.sum(dim=-1, keepdim=True) + 1e-6)
        uv_pdf = uv_pdf.contiguous().view(B * T * N, self.num_joints,
                                          *uv_feat.shape[-2:])  # (BTN, 21, 32, 32)
        uv_coord = integral_heatmap2d(uv_pdf)  # (BTN, 21, 2), range 0~1
        uv_coord_im = torch.einsum("bij, j->bij", uv_coord, torch.tensor([W, H]).to(uv_coord.device))  # range 0~W,H
        uv_coord_im = uv_coord_im.view(B * T, N, self.num_joints, 2)  # (BT, N, 21, 2)
        # uvc = torch.cat(((uv_coord * 2 - 1), uv_confi), dim=-1)  # (BTN, J, 3)

        K = inputs['target_cam_intr'].flatten(0, 1)  # (BT, N, 3, 3)
        T_c2m = inputs['target_cam_extr'].flatten(0, 1)  # (BT, N, 4, 4)
        #ref_joints = batch_triangulate_dlt_torch(uv_coord_im, K, T_c2m)  # (B, J, 3)
        #ref_joints = batch_triangulate_dlt_cfd_torch(uv_coord_im, K, T_c2m, (pse_vis * uv_confi.detach()).view(B * T, N,
        #                                                                                                       self.num_joints))  # (BT, J, 3)
        # ref_proj = ref_joints.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, J, 3)
        # ref_joints_in_cam = batch_cam_extr_transf(T_c2m, ref_proj).flatten(0, 1)  # (B*N, J, 3)

        # kpts_hm = uv_coord - 0.5  # normalize to [-0.5, 0.5]

        outputs = {
            'cam_intr': K,
            'cam_extr': torch.linalg.inv(T_c2m),
            'master_id': inputs['master_id'],
            "inp_img_shape": inp_img_shape,
            'img_feats': img_feats,
            'mlvl_feat': mlvl_feat,  # (BT, N, 128, 32, 32)
            'pred_hmap': uv_hmap,  # (BTN, J, 32, 32)
            'feat_high': feat_high,  # (BTN, C, Hh, Wh)
            'global_feat': global_feat,  # (BTN, 512)
            'pred_joints_uv': uv_coord_im,  # (BT, N, J, 2)
            #'ref_joints_in_cam': ref_joints_in_cam,  # (B*N, J, 3)
            # 'uv_norm': uv_coord * 2 - 1,
            'uvc': uv_coord,
            'confidence': uv_confi  # (BTN, J)
        }

        return outputs