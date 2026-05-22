
import os
import sys
sys.path.append(os.getcwd())


from typing import Dict, Tuple, List

import torch
import torch.nn as nn
from torch import Tensor

from .nflows.customs.flows import Flow, Glow


# =====================
class FlowPredictor(nn.Module):
    """Base of flow predictor for hand prediction"""
    flow : Flow

    def __init__(self) -> None:
        super().__init__()

    # ------------
    def get_inputs_log_prob(self, ctx:Tensor, **kwargs
                            ) -> Tuple[Tensor]:
        raise NotImplementedError('.')

    # ------------
    def generate_random_samples(self, ctx:Tensor, num_samples:int
                                ) -> Tuple[Tensor]:
        raise NotImplementedError('.')


# =====================
class GlowUVPredictor(FlowPredictor):
    def __init__(self,
                 context_features:int,
                 hidden_features:int,
                 num_layers: int=4,
                 num_blocks_per_layer: int=2,
                 num_joints: int=21,
                 noise_scale: float=0.025,
                 **kwargs) -> None:
        super().__init__()

        self.num_joints = num_joints
        self.z_dim = num_joints * 2
        self.c_dim = context_features
        self.noise_scale = noise_scale

        self.flow = Glow(features=self.z_dim,
                         hidden_features=hidden_features,
                         context_features=context_features,
                         num_layers=num_layers,
                         num_blocks_per_layer=num_blocks_per_layer,
                         **kwargs)


    # -----------------
    def get_inputs_log_prob(self, ctx:Tensor, UV:Tensor) -> Tuple[Tensor]:
        """
        In:
            ctx: [B, c_dim], input context
            UV: [B, N, J], 
        Out:
            log_p: [B, N]
            z: [B, N, z_dim]
        """

        # inputs check
        B, N, J = UV.shape[:3]
        assert J == self.num_joints, f'Expect {self.num_joints} joints, but get {J}'
        assert ctx.shape[1] == self.c_dim, f'Expect context features {self.c_dim}, but get {ctx.shape[1]}'

        ctx = ctx.reshape(B, 1, self.c_dim).repeat(1, N, 1) # [B, N, c_dim]
        # r6d = batch_rotMat2R6d_tensor(rmt).reshape(B, N, -1) # [B, N, J*6]

        if self.training: # add noise to relax overfit
            UV = UV + torch.randn_like(UV) * self.noise_scale

        log_p, z = self.flow.log_prob(UV.reshape(B*N, -1),
                                      ctx.reshape(B*N, -1))

        log_p = log_p.reshape(B, N) / self.z_dim # mean the prob to each compenent.
        z = z.reshape(B, N, self.z_dim)

        return log_p, z


    # ----------------
    def generate_random_samples(self, ctx:Tensor, num_samples:int) -> Tuple[Tensor]:
        """
        In:
            ctx: [B, c_dim]
            num_samples: int
        Out:
            rmt: [B, N, J, 3, 3], the first sample is always z0 sample
            r6d: [B, N, J, 6]
            log_p: [B, N], the log_p of generated samples
        """

        assert ctx.shape[1] == self.c_dim, f'Expect context features {self.c_dim}, but get {ctx.shape[1]}'

        B = ctx.shape[0]

        # randomly generate inputs in flow's distribution, and the first sample is always the z0 sample.
        zs :Tensor = self.flow._distribution.sample(num_samples-1, ctx).requires_grad_(False) # zs samples
        z0 :Tensor = self.flow._distribution.mean.reshape(1, 1, self.z_dim).repeat(B, 1, 1) # z0 sample, i.e. mode point.
        z = torch.cat([z0, zs], dim=1).to(device=ctx.device).requires_grad_(True)

        uv, log_p, _ = self.flow.sample_and_log_prob(num_samples, z, ctx) # [B, N, z_dim], [B, N]

        # rmt = batch_rotR6d2Mat_tensor(r6d.clone().reshape(-1, 6)).reshape(B, num_samples, -1, 3, 3)
        # r6d = r6d.reshape(B, num_samples, -1, 6)
        log_p = log_p.reshape(B, num_samples) / self.z_dim

        return uv, log_p



