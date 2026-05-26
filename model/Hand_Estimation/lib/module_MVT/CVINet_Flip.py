import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
import random
from torch import Tensor

from typing import Dict, Tuple, List
from lib.utils.config import CN
from lib.utils.triangulation import batch_triangulate_dlt_cfd_torch
# from lib.module.ManoDecoder import ManoDecoder
from .Graph import GraphRegression, GraphRegression, SelfAttnWithConfidence, SAIGB
from rlepose.models.layers.real_nvp import RealNVP


def nets():
    return nn.Sequential(nn.Linear(512 + 2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2),
                         nn.Tanh())


def nett():
    return nn.Sequential(nn.Linear(512 + 2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2))


class CrossViewInteractNet(nn.Module):
    def __init__(self, cfg: CN):
        super(CrossViewInteractNet, self).__init__()
        self.cfg = cfg
        self.channel = cfg.CHANNEL
        self.view_num = cfg.VIEW_NUMS
        self.joint_num = cfg.NUM_JOINT
        self.num_FMs = cfg.NUM_FMS
        self.z_dim = 2
        self.c_dim = 512
        self.noise_scale = 0.025
        # self.num_seq = cfg.NUM_SEQ
        self.selected_samples = 21
        self.out_dim = 512
        self.cfemb_dim = 128
        # self.jaf_dim = 128
        # self.jaf_dim = 512
        self.jaf_dim = 192
        self.feature_size = cfg.FEATURE_SIZE
        self.saigb_dim = self.feature_size * self.num_FMs
        self.feat_dim = self.saigb_dim + self.jaf_dim  # 可配置化
        # self.cf_embeding = nn.Sequential(
        #     nn.Linear(self.joint_num * 3, self.joint_num * self.cfemb_dim),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(self.joint_num * self.cfemb_dim, self.joint_num * self.cfemb_dim),
        # )
        self.saigb_pose = SAIGB(self.channel, self.num_FMs, self.feature_size, self.joint_num)
        self.gcn_0 = GraphRegression(self.joint_num, self.feat_dim, self.feat_dim, last=False)
        self.att_0 = SelfAttnWithConfidence(self.feat_dim, self.feat_dim, num_view=self.view_num, dropout=0)
        self.gcn_1 = GraphRegression(self.joint_num, self.feat_dim, self.feat_dim, last=False)
        self.att_1 = SelfAttnWithConfidence(self.feat_dim, self.feat_dim, num_view=self.view_num, dropout=0)
        # self.gcn_0 = GraphRegression(self.joint_num, self.feat_dim, self.feat_dim, last=False)
        # self.gcn_1 = GraphRegression(self.joint_num, self.feat_dim, self.feat_dim, last=False)
        # self.att_0 = SelfAttn(self.feat_dim, self.feat_dim, dropout=0)
        # self.att_1 = SelfAttn(self.feat_dim, self.feat_dim, dropout=0)
        self.fuse = GraphRegression(self.joint_num, self.feat_dim, self.out_dim)

        prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
        masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
        self.flow = RealNVP(nets, nett, masks, prior)

        # self.fc_reg = nn.Sequential(
        #     nn.Linear(self.out_dim*self.joint_num, self.out_dim*self.joint_num),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(self.out_dim*self.joint_num, 16 * 6 + 10 + 3),
        # )

        # # Instantiate MANO decoder
        # self.decoder = ManoDecoder(9, 0.4, (256, 256))

    def get_inputs_log_prob(self, ctx: Tensor, UV: Tensor) -> Tuple[Tensor]:
        """
        In:
            ctx: [BNJ, c_dim], input context
            UV: [BNJ, 2],
        Out:
            log_p: [B, N]
            z: [B, N, z_dim]
        """

        # inputs check
        BTNJ = UV.shape[0]
        # assert J == self.joint_num, f'Expect {self.joint_num} joints, but get {J}'
        # assert ctx.shape[1] == self.c_dim, f'Expect context features {self.c_dim}, but get {ctx.shape[1]}'

        ctx = ctx.reshape(BTNJ, 1, self.c_dim).repeat(1, self.selected_samples, 1)  # [B, N, c_dim]
        # r6d = batch_rotMat2R6d_tensor(rmt).reshape(B, N, -1) # [B, N, J*6]

        # if self.training: # add noise to relax overfit
        #     UV = UV + torch.randn_like(UV) * self.noise_scale

        log_p = self.flow.log_prob(UV.reshape(BTNJ * self.selected_samples, -1),
                                   ctx.reshape(BTNJ * self.selected_samples, -1))

        log_p = log_p.reshape(BTNJ, self.selected_samples) / self.z_dim  # mean the prob to each compenent.
        # z = z.reshape(B, N, self.z_dim)

        return log_p

    # ----------------
    def generate_random_samples(self, ctx: Tensor, num_samples: int) -> Tuple[Tensor]:
        """
        In:
            ctx: [BNJ, c_dim]
            num_samples: int
        Out:
            sample: [BNJ, ns, 2], the first sample is always z0 sample
            log_p: [BNJ, ns], the log_p of generated samples
        """

        assert ctx.shape[1] == self.c_dim, f'Expect context features {self.c_dim}, but get {ctx.shape[1]}'

        BTNJ = ctx.shape[0]

        # randomly generate inputs in flow's distribution, and the first sample is always the z0 sample.
        zs: Tensor = self.flow.prior.sample((BTNJ, num_samples - 1)).requires_grad_(False)  # zs samples
        z0: Tensor = self.flow.prior.mean.reshape(1, 1, self.z_dim).repeat(BTNJ, 1, 1)  # z0 sample, i.e. mode point.
        z = torch.cat([z0, zs], dim=1).to(device=ctx.device).requires_grad_(True)

        samples = self.flow.forward_p(z.reshape(BTNJ * num_samples, -1),
                                      ctx.unsqueeze(1).repeat(1, num_samples, 1).reshape(BTNJ * num_samples, -1))
        samples = samples.reshape(BTNJ, num_samples, 2)
        log_prob = self.flow.log_prob(z.reshape(BTNJ * num_samples, -1),
                                      ctx.unsqueeze(1).repeat(1, num_samples, 1).reshape(BTNJ * num_samples, -1))
        log_prob = log_prob.reshape(BTNJ, num_samples) / self.z_dim

        # rmt = batch_rotR6d2Mat_tensor(r6d.clone().reshape(-1, 6)).reshape(B, num_samples, -1, 3, 3)
        # r6d = r6d.reshape(B, num_samples, -1, 6)
        # log_p = log_p.reshape(BNJ, num_samples) / self.z_dim

        return samples, log_prob

    def forward(self, batch, inputs):
        img = batch['image']  # (B, T, N, 3, 256, 256)
        K = batch['target_cam_intr'].flatten(0, 1)  # (BT, N, 3, 3)
        T_c2m = batch['target_cam_extr'].flatten(0, 1)  # (BT, N, 4, 4)
        pse_vis = batch['pse_joints_vis'].flatten(0, 2).unsqueeze(-1)
        # pse_vis = batch['target_joints_vis'].flatten(0, 1).unsqueeze(-1)
        B, T, N, C, H, W = img.shape
        mlvl_feat = inputs['mlvl_feat']
        feat_high = inputs['feat_high']
        uvc = inputs['uvc']
        confidence = inputs['confidence']  # (B*T*N, J, 1)
        conf = pse_vis * confidence

        # uv_pgt = batch['target_joints_uvd'][:, :, :, :2].reshape(-1, 2)

        jaf = F.grid_sample(mlvl_feat.flatten(0, 1), uvc.detach().unsqueeze(2), align_corners=False)[..., 0].permute(0,
                                                                                                                     2,
                                                                                                                     1)
        # confidence_emb = self.cf_embeding(uvc.reshape(B*N, -1)).reshape(-1, self.joint_num, self.cfemb_dim)
        feat_pose = self.saigb_pose(feat_high)
        BTN, J, _ = feat_pose.shape

        features = torch.cat((feat_pose, jaf), dim=-1)  # (BTN, J, feat_dim)

        all_feats = features.reshape(-1, N * J, self.feat_dim)

        master_feats = self.gcn_0(all_feats.reshape(-1, J, self.feat_dim))  # (BTN, J, feat_dim)
        master_feats = master_feats.reshape(-1, N, J, self.feat_dim)[:, 0].repeat(1, N, 1)
        all_feats = all_feats + self.att_0(all_feats,
                                           conf.detach().reshape(-1, N * J, 1)) + master_feats  # (BT, NJ, feat_dim)

        avg_feats = self.gcn_1(all_feats.reshape(-1, J, self.feat_dim))
        avg_feats = avg_feats.reshape(-1, N, J, self.feat_dim).mean(1).repeat(1, N, 1)
        all_feats = all_feats + self.att_1(all_feats,
                                           conf.detach().reshape(-1, N * J, 1)) + avg_feats  # (BT, NJ, feat_dim)

        features = self.fuse(all_feats.reshape(-1, J, self.feat_dim))  # (BTN, J, out_dim)
        context = features.reshape(-1, self.out_dim)  # (BTNJ, out_dim)

        pred_uv, sampled_logp = self.generate_random_samples(ctx=context, num_samples=self.selected_samples)

        if self.training:
            uv_pgt = batch['target_pseudo_uv'].reshape(-1, 2)
            uv_pgt_NS = uv_pgt.unsqueeze(1).repeat(1, self.selected_samples, 1)

            pred_logp = self.get_inputs_log_prob(
                ctx=context,
                UV=uv_pgt_NS
            )
        else:
            pred_logp = torch.zeros(BTN * J, self.selected_samples).to(pred_uv.device)

        affine = batch['affine'].reshape(BTN, -1, 3)
        if affine.shape[1] == 2:
            pad = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=affine.dtype, device=affine.device).expand(BTN, 1, 3)
            affine = torch.cat([affine, pad], dim=1)  # (BTN, 3, 3)
        affine_inv = torch.inverse(affine)  # (BTN, 3, 3)

        anchor_uv = pred_uv[:, 0].reshape(BTN, J, 2)
        # 从 [-1, 1] 映射到 增强Crop图像像素尺度 [0, H]
        anchor_im_aug = (anchor_uv + 1) * H / 2

        # 补齐齐次坐标进行反变换: P_orig = affine_inv @ P_aug
        anchor_im_aug_homo = torch.cat([anchor_im_aug, torch.ones(BTN, J, 1, device=anchor_im_aug.device)],
                                       dim=-1)  # (BTN, J, 3)
        anchor_im_orig = torch.bmm(anchor_im_aug_homo, affine_inv.transpose(1, 2))[..., :2]  # (BTN, J, 2)

        anchor_im_orig = anchor_im_orig.reshape(B * T, N, J, 2)
        # 用校正回原图的坐标进行三角化
        ref_joints = batch_triangulate_dlt_cfd_torch(anchor_im_orig, K, T_c2m, conf.reshape(B * T, N, J))

        # --- 2. 处理 Ambig UV (Sampled points) ---
        ambig_uv = pred_uv[:, 1:self.selected_samples].reshape(BTN, J * (self.selected_samples - 1), 2)
        ambig_im_aug = (ambig_uv + 1) * H / 2

        # 同样，执行齐次反变换
        ambig_im_aug_homo = torch.cat(
            [ambig_im_aug, torch.ones(BTN, J * (self.selected_samples - 1), 1, device=ambig_im_aug.device)], dim=-1)
        ambig_im_orig = torch.bmm(ambig_im_aug_homo, affine_inv.transpose(1, 2))[..., :2]

        ambig_im_orig = ambig_im_orig.reshape(B * T, N, J * (self.selected_samples - 1), 2)
        ambig_conf = conf.unsqueeze(2).repeat(1, 1, self.selected_samples - 1, 1)
        ambig_ref_joints = batch_triangulate_dlt_cfd_torch(ambig_im_orig, K, T_c2m, ambig_conf.reshape(B * T, N, J * (
                    self.selected_samples - 1)))
        # ================================================================

        output = {
            'joints_feature': features,
            'pred_uv': pred_uv.reshape(BTN, J * self.selected_samples, 2),
            'anchor_uv': anchor_uv,
            'ref_joints': ref_joints,
            'ambig_uv': ambig_uv,
            'ambig_ref_joints': ambig_ref_joints,
            'log_p_sampled': sampled_logp,
            'log_p': pred_logp
        }

        return output