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
import itertools
import numpy as np
import torch


def render(data,
           iteration,
           scene,
           pipe,
           bg_color : torch.Tensor,
           scaling_modifier = 1.0,
           override_color = None,
           compute_loss=True,
           return_opacity=False,
           pose_refine=False,
           white_bg=False,
           save=False,
           novel_data=None,
           prev_data=None,
           ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    loss_reg, updated_camera = \
    scene.convert_gaussians(data, iteration, compute_loss, prev_data=prev_data)

    return {
            "loss_reg": loss_reg,
            "updated_camera": updated_camera
            }

