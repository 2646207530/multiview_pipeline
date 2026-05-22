# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
Modified by
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/3/9 9:32 PM
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class Transform_Net(nn.Module):
    def __init__(self):
        super(Transform_Net, self).__init__()
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3 * 3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)  # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)  # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)  # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)  # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class DGCNN_partseg(nn.Module):
    def __init__(self, seg_num_all, k=40, emb_dims=1024, dropout=0.5):
        super(DGCNN_partseg, self).__init__()
        self.seg_num_all = seg_num_all
        self.k = k
        self.transform_net = Transform_Net()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1216, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

    def forward(self, x):
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)  # (batch_size, 3, 3)
        x = x.transpose(2, 1)  # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)  # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)  # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1088+64*3, num_points)

        x = self.conv8(x)  # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)  # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)  # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)

        return x


class DGCNN_semseg(nn.Module):
    def __init__(self, joint_num, k=40, emb_dims=1024):
        super(DGCNN_semseg, self).__init__()
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        out_dims = [joint_num*3, joint_num, joint_num]
        self.finals = nn.ModuleList()
        for out_dim in out_dims:
            self.finals.append(nn.Conv1d(in_channels=128, out_channels=out_dim, kernel_size=1, stride=1))

    def forward(self, x):
        device = x.device
        pcl = x.clone()
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, dim9=True)  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)  # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        point_feature = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

        point_result = torch.Tensor().to(device)
        for layer in self.finals:
            temp = layer(point_feature)
            point_result = torch.cat((point_result, temp), dim=1)

        return [[pcl.permute(0, 2, 1), point_result.permute(0, 2, 1)]]

def pcl_offset2joint(pcl_result, pcl, kernel_size, topk=30):
    """
    :param: pcl_result Bx(4*J)xN
    :param: pcl BxNx3
    """
    batch_size, joint_num, point_num = pcl_result.size()
    joint_num = int(joint_num / 4)

    coords = pcl.permute(0, 2, 1).view(batch_size, 1, 3, point_num)
    offset = pcl_result[:, :joint_num * 3, :].view(batch_size, joint_num, 3, point_num)
    heatmap = pcl_result[:, joint_num * 3:, :].view(batch_size, joint_num, 1, point_num)
    value, idx = torch.topk(heatmap,topk, dim=-1, largest=True)

    dist = kernel_size - heatmap * kernel_size
    joint = torch.sum(value * (offset * dist + coords).gather(-1, idx.repeat(1,1,3,1)), dim=-1) / (value.sum(-1) + 1e-8)

    return joint


def pcl_offset2joint_softmax(pcl_result, pcl, kernel_size, scale=30):
    """
    :param: pcl_result Bx(4*J)xN
    :param: pcl BxNx3
    """
    batch_size, joint_num, point_num = pcl_result.size()
    joint_num = int(joint_num / 4)

    coords = pcl.permute(0, 2, 1).view(batch_size, 1, 3, point_num)
    offset = pcl_result[:, :joint_num * 3, :].view(batch_size, joint_num, 3, point_num)
    heatmap = pcl_result[:, joint_num * 3:, :].view(batch_size, joint_num, 1, point_num)
    normal_heatmap = F.softmax(heatmap * scale, dim=-1)

    dist = kernel_size - heatmap * kernel_size
    joint = torch.sum((offset * dist + coords) * normal_heatmap, dim=-1)

    return joint


def pcl_weight_offset2joint_softmax(pcl_result, pcl, kernel_size, scale=30):
    """
    :param: pcl_result Bx(5*J)xN
    :param: pcl BxNx3
    """
    assert pcl.size(2) == 3
    batch_size, joint_num, point_num = pcl_result.size()
    joint_num = int(joint_num / 5)

    coords = pcl.permute(0, 2, 1).reshape(batch_size, 1, 3, point_num)
    offset = pcl_result[:, :joint_num * 3, :].view(batch_size, joint_num, 3, point_num)
    heatmap = pcl_result[:, joint_num * 3:joint_num * 4, :].view(batch_size, joint_num, 1, point_num)
    weight = pcl_result[:, joint_num * 4:, :].view(batch_size, joint_num, 1, point_num)
    normal_weight = F.softmax(weight * scale, dim=-1)

    dist = kernel_size - heatmap * kernel_size
    joint = torch.sum((offset * dist + coords) * normal_weight, dim=-1)/ torch.sum(normal_weight,dim=-1)

    return joint


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
    return torch.cat((offset_norm, dis), dim=1).to(device)

if __name__ == '__main__':
    model = DGCNN_semseg(21).cuda()
    point = torch.rand([32, 1024, 3]).cuda()
    print(model(point.permute(0, 2, 1)).size())
    # point = torch.rand([32, 1024, 3]).cuda()
    # joint = torch.rand([32, 21, 3]).cuda()
    # print(model(point, joint)[0].size())