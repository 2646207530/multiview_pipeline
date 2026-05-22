import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils.mano import MANO
import torchgeometry as tgm


def orthgonalProj(xy, scale, transl, img_size=256):
    scale = scale * img_size
    transl = transl * img_size / 2 + img_size / 2
    return xy * scale + transl


class ManoDecoder(nn.Module):
    def __init__(self, root_idx, bbox_3d, input_img_shape):
        super(ManoDecoder, self).__init__()
        self.mano_layer = MANO('right').layer
        self.joint_regressor = MANO().joint_regressor
        self.root_joint_idx = root_idx
        self.bbox_3d_size = bbox_3d
        self.input_img_shape = input_img_shape

    def rot6d_to_rotmat(self, x):
        x = x.reshape(-1, 3, 2)
        a1 = x[:, :, 0]
        a2 = x[:, :, 1]
        b1 = F.normalize(a1)
        b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
        b3 = torch.cross(b1, b2)
        return torch.stack((b1, b2, b3), dim=-1)

    def forward(self, pose, shape, cam):
        batch = pose.shape[0]
        # transform rot-6d to angle-axis
        pose = self.rot6d_to_rotmat(pose)
        # pose = kornia.geometry.conversions.rotation_matrix_to_angle_axis(pose).reshape(batch, -1)
        pose = torch.cat([pose, torch.zeros((pose.shape[0], 3, 1)).to(pose.device).float()], 2)
        pose_euler = tgm.rotation_matrix_to_angle_axis(pose).reshape(batch, -1)
        # get coordinates from MANO layer
        self.mano_layer = self.mano_layer.to(pose_euler.device)
        mano_mesh_cam, _ = self.mano_layer(pose_euler, shape)
        # mm to m
        mano_mesh_cam = mano_mesh_cam / 1000
        # get pose joints
        mano_joint_cam = torch.bmm(
            torch.from_numpy(self.joint_regressor).to(pose.device)[None, :, :].repeat(batch, 1, 1), mano_mesh_cam)
        coord_xyz = torch.cat((mano_mesh_cam, mano_joint_cam), dim=1)
        # root-relative
        coord_xyz = coord_xyz - mano_joint_cam[:, self.root_joint_idx, None]
        # project xy to uv
        coord_uv = orthgonalProj(coord_xyz[:, :, :2].clone(), cam[:, 0:1].unsqueeze(1), cam[:, 1:].unsqueeze(1))
        # normalization
        coord_xyz = coord_xyz / (self.bbox_3d_size / 2)
        coord_uv = coord_uv / (self.input_img_shape[0] // 2) - 1
        # coord_uvd = torch.cat((coord_uv, coord_xyz[:, :, 2:3]), dim=2)
        return coord_xyz, coord_uv, pose_euler, shape, cam