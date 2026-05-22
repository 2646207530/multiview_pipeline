import copy

import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision
from pytorch3d import ops


from utils.pointnet_utils import index_points
from .network_utils import homoify, points3DToImg, get_skinning_mlp
from .pose_correction import get_pose_correction



class GaussianConverter(nn.Module):
    def __init__(self, cfg, save_dir, metadata, metadata_obj, smpl_data, subject_labels, obj_labels, freeze_pose=False):
        super().__init__()
        print(f"freeze_pose: {freeze_pose}")
        self.cfg = cfg
        self.metadata = metadata
        self.save_dir = save_dir
        self.metadata_obj = metadata_obj
        self.smpl_data = smpl_data
        self.obj_labels = obj_labels
        self.subject_labels = subject_labels

        self.pose_correction = get_pose_correction(cfg.model.deformer, smpl_data, freeze_pose=freeze_pose)

        self.lr_scale = 1 * self.cfg.get('batch_size', 8) / 8
        self.hand_lr_scale = 1
        
        self.optimizer, self.scheduler = None, None
        self.set_optimizer()

        self.roi_size = cfg.model.deformer.non_rigid.get('roi_size', 224)
        #self.L2Loss = torch.nn.MSELoss().cuda()
        self.L2Loss = nn.SmoothL1Loss(reduction="mean").cuda()

        #self.body_model = MANO(model_path='/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/mano/').cuda()



    def set_optimizer(self, pose_only=False):
        opt_params_pose = [{'name': 'pose_correction', 'params': [p for n, p in self.pose_correction.named_parameters()],
                           'lr': self.cfg.opt.get('pose_correction_lr', 0.) * 1}]
        
        self.optimizer = torch.optim.Adam(params=opt_params_pose, lr=0.001, eps=1e-15)
        
        gamma = self.cfg.opt.lr_ratio ** (1. / self.cfg.opt.iterations)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

   

    def forward(self, camera, iteration, compute_loss=True, prev_camera=None):
       

        loss_reg = {}
        camera, loss_reg = self.pose_correction(camera, iteration)


        return loss_reg, camera


    def optimize(self, iteration):
        grad_clip = self.cfg.opt.get('grad_clip', 0.)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        

        # gamma = self.cfg.opt.lr_ratio ** (1. / self.cfg.opt.iterations)

        # for group in self.optimizer.param_groups:
        #     name = group['name']
        #     group['lr'] = group['lr'] * gamma





    def random_sampling(self, xyz, npoint=2048):
        """
        Input:
            xyz: point cloud data, [B, N, C]
            npoint: number of points to sample
        Return:
            sampled_xyz: sampled point cloud data, [B, npoint, C]
        """
        B, N, C = xyz.shape
        # 生成随机索引
        indices = torch.randint(0, N, (B, npoint,), device=xyz.device)
        # 使用随机索引对点云进行采样
        sampled_xyz = index_points(xyz, indices)
        return sampled_xyz, indices


    def query_weights(self, xyz):
        # find the nearest vertex

        knn_ret = ops.knn_points(xyz, self.smpl_verts.unsqueeze(0).repeat(xyz.shape[0],1,1))
        p_idx = knn_ret.idx

        pts_W = self.skinning_weights[p_idx, :]

        return pts_W


    def pixel_align(self, camera, input_xyz_points, feature_maps, full_proj_transform, trans_img2roi, roi_size, ho='hand'):
        batch_size, num_points_per_scene, _ = input_xyz_points.shape
        input_points = input_xyz_points.clone()
        input_points = input_points.reshape((-1, num_points_per_scene, 3))
    
        trans_img2roi = trans_img2roi

        cam_coord = torch.matmul(camera['R'], input_points.transpose(1, 2)).transpose(1, 2) + camera['T'].reshape(batch_size ,1, 3)

        xyz_2d = points3DToImg(cam_coord,camera['K'])[:, :, :2]

        ones = torch.ones((batch_size,num_points_per_scene, 1), dtype=torch.float32).to(input_points.device)
        uv_homogeneous = torch.cat([xyz_2d, ones], dim=-1)  # (b, N, 3)
        uv_2d_roi_unnorm = torch.matmul(trans_img2roi, uv_homogeneous.transpose(1, 2)).transpose(1, 2)[:,:,:2].unsqueeze(2) #b n 1 2

        uv_2d_roi = uv_2d_roi_unnorm / roi_size * 2 - 1

        sample_feat = torch.nn.functional.grid_sample(feature_maps, uv_2d_roi, align_corners=True)[:, :, :, 0].transpose(1, 2)
        sample_color = torch.nn.functional.grid_sample(camera['img_ROI'], uv_2d_roi, align_corners=True)[:, :, :,
                       0].transpose(1, 2)
        uv_2d_roi = uv_2d_roi.squeeze(2).reshape((batch_size, -1, 2))
        sample_feat = sample_feat.reshape((batch_size,uv_2d_roi.shape[1], -1))
        sample_color = sample_color.reshape((batch_size, uv_2d_roi.shape[1], -1))
       

        return sample_feat, sample_color

    def get_jtr(self, body):
        Jtrs = body['Jtr_a_pose']

        v_shaped = body['v_shaped']
        v_shaped = v_shaped.detach()

        center = torch.mean(v_shaped, dim=1).unsqueeze(1)
        minimal_shape_centered = v_shaped - center
        cano_max = minimal_shape_centered.max()
        cano_min = minimal_shape_centered.min()
        padding = (cano_max - cano_min) * 0.05

        # compute pose condition
        Jtrs = Jtrs - center
        Jtrs = (Jtrs - cano_min + padding) / (cano_max - cano_min) / 1.1
        Jtrs -= 0.5
        Jtrs *= 2.
        Jtrs = Jtrs.contiguous()
        return Jtrs