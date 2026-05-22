import warnings
import numpy as np
import torch
import torch.nn as nn
import trimesh
from kornia.geometry import angle_axis_to_rotation_matrix

from manopth_utils.manopth.manolayer import ManoLayer
from right_hand_model import MANO
from utils.general_utils import get_jtr
from utils.loss_utils import compute_contact_loss
from utils.nets.config import cfg
from utils.nets.layer import MLP
from utils.nets.mano_head import ManoHead
from utils.nets.misc import get_mano_tgt_mask, get_mano_memory_mask
from utils.nets.transformer import Transformer, VoteTransformer


class KeypointTR(nn.Module):
    def __init__(self, config):
        super(KeypointTR, self).__init__()
        
        self.body_model_r = MANO(model_path='/mnt/sda2/lxy/arctic/unpack/body_models/mano/', flat_hand_mean=False)  # .cuda()
        self.body_model_l = MANO(model_path='/mnt/sda2/lxy/arctic/unpack/body_models/mano/',
                            is_rhand=False, flat_hand_mean=False)
        self.faces = {'right': self.body_model_r.faces, 'left':self.body_model_l.faces}

        # self.mano_head = ManoHead(self.mano_layer, coord_change_mat=coord_change_mat)
        self.pose_fan_out = 3
        self.linear_pose_l = MLP(cfg.hidden_dim, cfg.hidden_dim, self.pose_fan_out, 3)
        self.linear_pose_r = MLP(cfg.hidden_dim, cfg.hidden_dim, self.pose_fan_out, 3)

        self.linear_obj_rel_trans = MLP(cfg.hidden_dim, cfg.hidden_dim, 3, 3)
        self.linear_obj_rot = MLP(cfg.hidden_dim, cfg.hidden_dim, 9, 3)


    def forward(self, camera, hand_points_posed, obj_points_posed):

        hand_transformer_in = torch.cat([hand_points_posed, gaussian_feat_h, pixel_feat_h], dim=2)

        obj_transformer_in = torch.cat([obj_points_posed, gaussian_feat_o ,pixel_feat_o], dim=2)

        hand_transformer_in = hand_transformer_in.permute(1, 0, 2).contiguous()
        obj_transformer_in = obj_transformer_in.permute(1, 0, 2).contiguous()

        hand_positions = torch.zeros_like(hand_transformer_in).to(
            hand_transformer_in.device
        )
        obj_positions = torch.zeros_like(obj_transformer_in).to(
            obj_transformer_in.device
        )
        tgt_mask = get_mano_tgt_mask().to(hand_transformer_in.device)
        memory_mask = get_mano_memory_mask().to(hand_transformer_in.device)


        hand_transformer_out, memory, hand_encoder_out, attn_wts = (
            self.hand_transformer(
                src=hand_transformer_in,
                mask=None,
                pos_embed=hand_positions,
                src_mask=None,
                query_embed=self.mano_query_embed.weight,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
            )
        )

        obj_memory, obj_encoder_out = self.obj_transformer(
            src=obj_transformer_in, mask=None, pos_embed=obj_positions, src_mask=None
        )


        obj_rot = self.linear_obj_rot(
            obj_encoder_out[:, : cfg.num_samp_obj]
        )  # 6 x N x 3
        obj_trans = self.linear_obj_rel_trans(obj_encoder_out[:, : cfg.num_samp_obj])

        L, N ,B, _ = obj_rot.shape
        obj_rot = obj_rot.view(L, N, B, 3, 3)

        mano_pose6d = self.linear_pose(
                hand_transformer_out[:, : cfg.mano_shape_indx]
            )  # 6 x 16 x N x 3(9)

        mano_shape = self.linear_shape(
            hand_transformer_out[:, cfg.mano_shape_indx]
        )  # 6 x N x 10
        if self.pose_fan_out == 3:
            mano_pose6d = mano_pose6d+ camera.hand_param[:, :48].view(1, -1, 16, 3).permute(0, 2, 1, 3).contiguous().repeat(mano_pose6d.shape[0], 1, 1, 1)

        elif self.pose_fan_out == 9:
            mano_pose6d_init = angle_axis_to_rotation_matrix(camera.hand_param[:, :48].view(1, B, 16, 3).permute(0, 2, 1, 3).repeat(mano_pose6d.shape[0], 1, 1, 1).contiguous().view(-1, 3)).view(mano_pose6d.shape[0], 16, B, 9)
            mano_pose6d = mano_pose6d + mano_pose6d_init

        mano_shape = mano_shape + camera.hand_param[:, 48:58].view(1, -1, 10).repeat(mano_shape.shape[0], 1, 1)

        pred_mano_results, gt_mano_results = self.mano_head(
            mano_pose6d, mano_shape, mano_params=camera.hand_param_gt
        )


        root_orient, pose_hand, betas, hand_root = pred_mano_results["mano_pose6d"][-1][:, :3] ,\
                                                   pred_mano_results["mano_pose6d"][-1][:, 3:48], \
                                                   pred_mano_results["mano_shape"][-1], \
                                                   camera.hand_param[:, 58:]
        body = self.body_model(global_orient=root_orient, hand_pose=pose_hand, betas=betas, transl=hand_root)
        bone_transforms = body['bone_transforms']

        rot_mats = body['rot_mats']

        rots = torch.cat([torch.eye(3).reshape(1, 1, 3, 3).repeat(rot_mats.shape[0], 1, 1, 1).to(rot_mats.device),
                          rot_mats[:, 1:]], dim=1)
        rots = rots.reshape(rot_mats.shape[0], -1, 9).contiguous()

        bone_transforms[:, :, :3, 3] = bone_transforms[:, :, :3, 3] + hand_root.unsqueeze(1)

        Jtrs = get_jtr(body)


        updated_camera = camera.copy()

        updated_camera.update(
            rots= rots.squeeze(0),
            Jtrs= Jtrs.squeeze(0),
            bone_transforms= bone_transforms.squeeze(0),
            hand_param= torch.cat([root_orient, pose_hand, betas, hand_root], dim=-1),
            obj_rots= obj_rot[-1].permute(1, 0, 2, 3).contiguous().mean(1).view(3,3)+camera.obj_rots,
            obj_trans= obj_trans[-1].permute(1, 0, 2).contiguous().mean(1).view(3)+camera.obj_trans,
            pred_joints_mano= pred_mano_results['joints3d'][-1],
            pred_joints=pred_mano_results['joints3d'][-1],
            gt_mano_joints= gt_mano_results['joints3d']
        )

        loss = {}
        # loss["contact"], loss["penetration"], _, _ = compute_contact_loss(
        #     body['v'],
        #     obj_points,
        #     obj_triangles,
        #     # contact_thresh=5 / 1000,
        #     # collision_thresh=20 / 1000,
        #     contact_thresh=10 / 1000,
        #     #contact_mode="dist_sq",
        #     collision_thresh=20 / 1000,
        #     #collision_mode="dist_sq",
        #     contact_zones="zones",
        # )

        return updated_camera, loss

