import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np

from easydict import EasyDict
from lib.utils.config import CN
from lib.models.layers.real_nvp import RealNVP
from lib.module.Graph import GraphRegression, SAIGB, SelfAttn
from lib.module.ManoDecoder import ManoDecoder


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


# cross-view interaction
class CrossViewInteraction(nn.Module):
    def __init__(self, in_dim, out_dim, cfg: CN):
        super(CrossViewInteraction, self).__init__()
        joint_num = cfg.NUM_JOINT
        self.view_num = cfg.NUM_VIEW
        self.location_embedding_dim = 64
        self.random_mask = cfg.RANDOM_MASK
        # resnet
        self.feat_dim = cfg.FEATURE_SIZE * cfg.NUM_FMS + in_dim // 8 + in_dim // 4 + in_dim // 2 + self.location_embedding_dim
        self.pose_mapping = nn.Sequential(
            nn.Linear(joint_num * 3 + 6 * 15, joint_num * self.location_embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(joint_num * self.location_embedding_dim, joint_num * self.location_embedding_dim),
        )
        self.saigb_pose = SAIGB(in_dim, cfg.NUM_FMS, cfg.FEATURE_SIZE, joint_num)
        # two-branch transformer
        self.gcn_0 = GraphRegression(joint_num, self.feat_dim, self.feat_dim, last=False)
        self.gcn_1 = GraphRegression(joint_num, self.feat_dim, self.feat_dim, last=False)
        self.att_0 = SelfAttn(self.feat_dim, self.feat_dim, dropout=0)
        self.att_1 = SelfAttn(self.feat_dim, self.feat_dim, dropout=0)
        # mano parameters regression
        self.fuse = GraphRegression(joint_num, self.feat_dim, out_dim, last=False)

        self.fc_coord = Linear(out_dim * joint_num, joint_num * 2)
        self.fc_sigma = Linear(out_dim * joint_num, joint_num * 2, norm=False)

        self.fc_layers = [self.fc_coord, self.fc_sigma]

        self.fc = nn.Sequential(
            nn.Linear(out_dim * joint_num, out_dim * joint_num),
            nn.LeakyReLU(0.1),
            nn.Linear(out_dim * joint_num, 16 * 6 + 10 + 3),
        )
        self.decoder = ManoDecoder(cfg.ROOT_IDX, cfg.BBOX_3D_SIZE, cfg.IMAGE_SIZE)

        prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
        masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))

        self.flow = RealNVP(nets, nett, masks, prior)

    def forward(self, feat_pose, jaf, joint_uvd, prev_mano_params, target_uv, view_num=None):
        if view_num is not None:
            self.view_num = view_num

        # multi-view graph building
        feat_pose = self.saigb_pose(feat_pose)
        batch, joint_num, feat_dim = feat_pose.shape
        localization_feat = self.pose_mapping(torch.cat((joint_uvd.reshape(-1, joint_num * 3), prev_mano_params[:, 6:6*16].reshape(-1, 6*15)), dim=1))
        mv_feat = torch.cat((
            feat_pose.reshape(-1, joint_num, feat_dim), 
            jaf.reshape(-1, joint_num, jaf.shape[-1]), 
            localization_feat.view(-1, joint_num, 64)), dim=2)

        new_feat = mv_feat.reshape(-1, self.view_num * joint_num, self.feat_dim)

        if self.random_mask and self.training:
            use_view_num = random.randint(1, self.view_num)
            use_view_set = random.sample([i for i in range(self.view_num)], use_view_num)
            use_view_set.sort()
        else:
            use_view_set = [i for i in range(self.view_num)]

        # view-shared feature - max
        canonical_feat = self.gcn_0(new_feat.view(-1, joint_num, self.feat_dim)) 
        canonical_feat = canonical_feat.view(-1, self.view_num, joint_num, self.feat_dim)[:, use_view_set].max(1)[0].repeat(1, self.view_num, 1) 
        # attention feature
        att_aug_feat = self.att_0(new_feat, mask=self.random_mask)
        # two-branch resisual
        new_feat = att_aug_feat + canonical_feat

        if self.random_mask and self.training:
            use_view_num = random.randint(1, self.view_num)
            use_view_set = random.sample([i for i in range(self.view_num)], use_view_num)
            use_view_set.sort()
        else:
            use_view_set = [i for i in range(self.view_num)]

        # view-shared feature - max
        canonical_feat = self.gcn_1(new_feat.view(-1, joint_num, self.feat_dim)) 
        canonical_feat = canonical_feat.view(-1, self.view_num, joint_num, self.feat_dim)[:, use_view_set].max(1)[0].repeat(1, self.view_num, 1) 

        att_aug_feat = self.att_1(new_feat, mask=self.random_mask)

        # two-branch resisual
        new_feat = att_aug_feat + canonical_feat
        # new_feat = new_feat + canonical_feat

        # mano parameters regression
        new_feat = self.fuse(new_feat.reshape(-1, joint_num, new_feat.shape[-1]))
        new_feat = new_feat.view(-1, joint_num * new_feat.shape[-1])

        out_sigma = self.fc_sigma(new_feat).reshape(batch, joint_num, -1)  # (B, N, 2)
        sigma = out_sigma.sigmoid()
        scores = 1 - sigma
        scores = torch.mean(scores, dim=2, keepdim=True)

        mano_params = self.fc(new_feat) #+ prev_mano_params.detach()

        # mano decoder
        cam = mano_params[:, 16*6+10:]
        cam = torch.cat((F.relu(cam[:, 0:1]), cam[:, 1:]), dim=1).view(batch, 3)
        pose = mano_params[:, :16*6]
        shape = prev_mano_params[:, 16*6:16*6+10].detach()
        coord_xyz, coord_uvd, pose, shape, cam = self.decoder(pose, shape, cam)

        if self.training and target_uv is not None:
            gt_uv = target_uv.reshape(pred_jts.shape)
            bar_mu = (pred_jts - gt_uv) / sigma
            # (B, K, 2)
            log_phi = self.flow.log_prob(bar_mu.reshape(-1, 2)).reshape(batch, joint_num, 1)

            nf_loss = torch.log(sigma) - log_phi
        else:
            nf_loss = None

        return RLE_output, coord_xyz, coord_uvd, pose, shape, cam