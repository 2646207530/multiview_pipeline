import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.config import CN
from .Graph import GraphRegression, GraphRegression, SelfAttn, SAIGB
from lib.module.ManoDecoder import ManoDecoder


class CrossViewInteractNet(nn.Module):
    def __init__(self, cfg: CN):
        super(CrossViewInteractNet, self).__init__()
        self.cfg = cfg
        self.channel = cfg.CHANNEL
        self.joint_num = cfg.NUM_JOINT
        self.num_FMs = cfg.NUM_FMS
        self.root_joint = cfg.ROOT_IDX
        # self.num_seq = cfg.NUM_SEQ
        self.out_dim = cfg.OUT_DIM
        self.cfemb_dim = 128
        self.jaf_dim = 128
        self.feature_size = cfg.FEATURE_SIZE
        self.saigb_dim = self.feature_size * self.num_FMs
        self.feat_dim = self.saigb_dim + self.jaf_dim + self.cfemb_dim # 可配置化
        self.cf_embeding = nn.Sequential(
            nn.Linear(self.joint_num * 3, self.joint_num * self.cfemb_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.joint_num * self.cfemb_dim, self.joint_num * self.cfemb_dim),
        )
        self.saigb_pose = SAIGB(self.channel, self.num_FMs, self.feature_size, self.joint_num)
        self.gcn_0 = GraphRegression(self.joint_num, self.feat_dim, self.feat_dim, last=False)
        self.gcn_1 = GraphRegression(self.joint_num, self.feat_dim, self.feat_dim, last=False)
        self.att_0 = SelfAttn(self.feat_dim, self.feat_dim, dropout=0)
        self.att_1 = SelfAttn(self.feat_dim, self.feat_dim, dropout=0)
        self.fuse = GraphRegression(self.joint_num, self.feat_dim, self.out_dim, last=False)
        self.fc_reg = nn.Sequential(
            nn.Linear(self.out_dim*self.joint_num, self.out_dim*self.joint_num),
            nn.LeakyReLU(0.1),
            nn.Linear(self.out_dim*self.joint_num, 16 * 6 + 10 + 3),
        )

        # Instantiate MANO decoder
        self.decoder = ManoDecoder(9, 0.4, (256, 256))

    def forward(self, batch, inputs):
        img = batch['image']  # (B, N, 3, 256, 256)
        B, N, C, H, W = img.shape
        mlvl_feat = inputs['mlvl_feat']
        feat_high = inputs['feat_high']
        uvc = inputs['uvc']

        jaf = F.grid_sample(mlvl_feat.flatten(0, 1), uvc[:, :, :2].unsqueeze(2), align_corners=True)[..., 0].permute(0, 2, 1)
        confidence_emb = self.cf_embeding(uvc.reshape(B*N, -1)).reshape(-1, self.joint_num, self.cfemb_dim)
        feat_pose = self.saigb_pose(feat_high) 
        BN, J, _ = feat_pose.shape

        features = torch.cat((feat_pose, jaf, confidence_emb), dim=-1)  # (BN, J, feat_dim)

        all_feats = features.reshape(-1, N*J, self.feat_dim)

        master_feats = self.gcn_0(all_feats.reshape(-1, J, self.feat_dim))  # (BN, J, feat_dim)
        master_feats = master_feats.reshape(-1, N, J, self.feat_dim)[:, 0].repeat(1, N, 1)
        all_feats = self.att_0(all_feats) + master_feats  # (BN, J, feat_dim)

        avg_feats = self.gcn_1(all_feats.reshape(-1, J, self.feat_dim))
        avg_feats = avg_feats.reshape(-1, N, J, self.feat_dim).mean(1).repeat(1, N, 1)
        all_feats = self.att_1(all_feats) + avg_feats  # (BN, J, feat_dim)

        features = self.fuse(all_feats.reshape(-1, J, self.feat_dim))  # (BN, J, out_dim)
        fu = features.reshape(-1, J*self.out_dim)  # (BN, J*out_dim)

        pred_mano_params = self.fc_reg(fu)    # (B*N, 109)

        pred_hand_pose = pred_mano_params[:, :96]    # (B*N, 96)
        pred_shape = pred_mano_params[:, 96:106]     # (B*N, 10)
        pred_cam = pred_mano_params[:, 106:]         # (B*N, 3)

        # positive scale
        pred_cam = torch.cat((F.relu(pred_cam[:, 0:1]), pred_cam[:, 1:]), dim=1).view(BN, 3)
        coord_xyz, coord_uv, pose_euler, shape, cam = self.decoder(pred_hand_pose, pred_shape, pred_cam)

        
        output = {
            'joints_feat': features,
            'coord_xyz': coord_xyz,
            'coord_uv': coord_uv,
            'pose_euler': pose_euler,
            'shape': shape,
            'cam': cam,
            'pred_mano_params_sv': pred_mano_params,
        }

        return output