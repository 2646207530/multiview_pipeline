import torch.nn as nn
import torch
import torch.nn.functional as F
from pointNet.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation,PointNetSetAbstractionMsg


class PointNet2(nn.Module):
    def __init__(self, num_classes, in_channel=3, normal_channel=False):
        super(PointNet2, self).__init__()
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3+in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+3+in_channel, mlp=[128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, joint):
        pcl_feat = joint2pcloffset(joint, xyz, 0.8).permute(0, 2, 1)

        # Set Abstraction layers
        l0_xyz = xyz.permute(0, 2, 1)
        l0_points = torch.cat((l0_xyz, pcl_feat), dim=1)
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return [x]


class PointNet2_MSG(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2MSG, self).__init__()

        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=150, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        # Set Abstraction layers
        l0_points = xyz
        l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points],1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


class PointNet2_MSG_large(nn.Module):
    def __init__(self, joint_num):
        super(PointNet2_MSG_large, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(512, [0.05, 0.1], [16, 32], 3, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)

        out_dims = [joint_num*3, joint_num, joint_num]
        self.finals = nn.ModuleList()
        for out_dim in out_dims:
            self.finals.append(nn.Conv1d(in_channels=128, out_channels=out_dim, kernel_size=1, stride=1))

    def init_weights(self):
        for m in self.finals.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz):
        device = xyz.device
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        point_feature = F.relu(self.bn1(self.conv1(l0_points)))

        point_result = torch.Tensor().to(device)
        for layer in self.finals:
            temp = layer(point_feature)
            point_result = torch.cat((point_result, temp), dim=1)

        return [[xyz.permute(0, 2, 1), point_result.permute(0, 2, 1)]]


def joint2pcloffset(joint, pcl, kernel_size):
    """
    :param: joint BxJx3
    :param: pcl BxNx3
    """
    batch_size, joint_num, _ =joint.size()
    point_num = pcl.size(1)
    device = joint.device
    offset = joint.unsqueeze(2) - pcl.unsqueeze(1)
    dis = torch.sqrt(torch.sum(torch.pow(offset, 2), dim=-1))
    offset_norm = offset / dis.unsqueeze(-1)
    offset_norm = offset_norm.permute(0, 1, 3, 2).reshape(batch_size, joint_num*3, point_num)

    dis = (kernel_size - dis) / kernel_size
    mask = dis.ge(0).float()
    dis = dis * mask
    offset_norm = offset_norm * mask.view(batch_size,joint_num,1,point_num).repeat(1,1,3,1).reshape(batch_size,-1,point_num)
    return torch.cat((offset_norm, dis), dim=1).to(device).permute(0, 2, 1)


if __name__ == '__main__':
    model = PointNet2_MSG_large(32).cuda()
    point = torch.rand([32, 1024, 3]).cuda()
    print(model(point.permute(0,2,1)).size())
    # point = torch.rand([32, 1024, 3]).cuda()
    # joint = torch.rand([32, 21, 3]).cuda()
    # print(model(point, joint)[0].size())