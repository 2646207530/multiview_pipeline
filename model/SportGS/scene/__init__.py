#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import time

import torch
from models import GaussianConverter
from scene.gaussian_model import GaussianModel
from dataset import load_dataset


class Scene:

    #gaussians : GaussianModel

    def __init__(self, cfg, gaussians_hand_group, gaussians_obj_group, save_dir : str, freeze_pose=False, multi_batch=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.cfg = cfg

        self.save_dir = save_dir
        self.gaussians_hand_group = gaussians_hand_group
        self.gaussians_obj_group = gaussians_obj_group

        self.train_dataset = load_dataset(cfg.dataset, split='train', multi_batch=multi_batch)
        self.metadata = self.train_dataset.metadata
        self.smpl_data = self.train_dataset.smpl_data
        self.metadata_obj = self.train_dataset.metadata_obj
       
        self.cameras_extent = self.metadata['right']['cameras_extent']
        print(f"freeze_pose: {freeze_pose}")
        self.converter = GaussianConverter(cfg, save_dir, self.metadata, self.metadata_obj, self.smpl_data, self.gaussians_hand_group.keys(), self.gaussians_obj_group.keys(), freeze_pose=freeze_pose).cuda()

    def train(self):
        self.converter.train()

    def eval(self):
        self.converter.eval()

    # from memory_profiler import profile
    # @profile
    def optimize(self, iteration):

        self.converter.optimize(iteration)



    def convert_gaussians(self, viewpoint_camera, iteration, compute_loss=True, prev_data=None):
       
        return self.converter(viewpoint_camera, iteration,compute_loss, prev_camera=prev_data)


    def get_skinning_loss(self, subject_id):
        loss_reg_r = getattr(self.converter,f"deformer_hand_{subject_id}_r").rigid.regularization()
        loss_reg_l = getattr(self.converter, f"deformer_hand_{subject_id}_l").rigid.regularization()
        loss_skinning = loss_reg_r.get('loss_skinning', torch.tensor(0.).cuda())+loss_reg_l.get('loss_skinning', torch.tensor(0.).cuda())
        return loss_skinning

   
    def save_checkpoint(self, iteration):
        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        torch.save((
                    self.converter.state_dict(),
                    self.converter.optimizer.state_dict(),
                    self.converter.scheduler.state_dict(),
                    iteration), self.save_dir + "/ckpt" + str(iteration) + ".pth")

    def load_checkpoint(self, path, strict=True):

        (converter_sd, converter_opt_sd, converter_scd_sd, first_iter) = torch.load(path)
       
        missing_keys, unexpected_keys = self.converter.load_state_dict(converter_sd, strict=strict)

        if missing_keys:
            assert ("missing keys:", missing_keys)
        if unexpected_keys:
            print("ignored unexpected keys:")
            print(unexpected_keys)
        if strict:
            self.converter.optimizer.load_state_dict(converter_opt_sd)
            self.converter.scheduler.load_state_dict(converter_scd_sd)


