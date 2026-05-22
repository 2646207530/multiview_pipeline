import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import torch
import cv2
from enum import Enum
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pyrender
import trimesh
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


color_hand_joints = [[1.0, 0.0, 0.0],
                     [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                     [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                     [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                     [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                     [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little
def get_param(dataset):
    if dataset == 'icvl' or dataset == 'nyu':
        return 240.99, 240.96, 160, 120
    elif dataset == 'msra':
        return 241.42, 241.42, 160, 120
    elif dataset == 'FHAD' or 'hands' in dataset:
        return 475.065948, 475.065857, 315.944855, 245.287079
    elif dataset == 'itop':
        return 285.71, 285.71, 160.0, 120.0


def get_joint_num(dataset):
    if dataset == 'nyu':
        return 14
    elif dataset == 'icvl':
        return 16
    elif dataset == 'FHAD' or 'hands' in dataset or 'msra' in dataset:
        return 21
    elif dataset == 'itop':
        return 15


def pixel2world(x, dataset):
    fx, fy, ux, uy = get_param(dataset)
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x


def world2pixel(x, dataset):
    fx,fy,ux,uy = get_param(dataset)
    x[:, :, 0] = x[:, :, 0] * fx/x[:, :, 2] + ux
    x[:, :, 1] = uy - x[:, :, 1] * fy / x[:, :, 2]
    return x


def jointImgTo3D(uvd, paras):
    fx, fy, fu, fv = paras
    ret = np.zeros_like(uvd, np.float32)
    if len(ret.shape) == 1:
        ret[0] = (uvd[0] - fu) * uvd[2] / fx
        ret[1] = (uvd[1] - fv) * uvd[2] / fy
        ret[2] = uvd[2]
    else:
        ret[:, 0] = (uvd[:,0] - fu) * uvd[:, 2] / fx
        ret[:, 1] = (uvd[:,1] - fv) * uvd[:, 2] / fy
        ret[:, 2] = uvd[:,2]
    return ret


def joint3DToImg(xyz, paras):
    fx, fy, fu, fv = paras
    ret = np.zeros_like(xyz, np.float32)
    if len(ret.shape) == 1:
        ret[0] = (xyz[0] * fx / xyz[2] + fu)
        ret[1] = (xyz[1] * fy / xyz[2] + fv)
        ret[2] = xyz[2]
    else:
        ret[:, 0] = (xyz[:, 0] * fx / xyz[:, 2] + fu)
        ret[:, 1] = (xyz[:, 1] * fy / xyz[:, 2] + fv)
        ret[:, 2] = xyz[:, 2]
    return ret


def get_sketch_setting(dataset):
    if dataset == 'FHAD' or 'hands' in dataset:
        # return [
        #         [0, 13], [13, 14], [14, 15], [15, 20],
        #         [0, 1], [1, 2], [2, 3], [3, 16],
        #         [0, 4], [4, 5], [5, 6], [6, 17],
        #         [0, 10], [10, 11], [11,  12], [12, 19],
        #         [0, 7], [7, 8], [8, 9], [9, 18]
        #         ]
        return [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
                [1, 6], [6, 7], [7, 8],
                [2, 9], [9, 10], [10, 11],
                [3, 12], [12, 13],[13, 14],
                [4, 15], [15, 16],[16, 17],
                [5, 18], [18, 19], [19, 20]]
    elif 'nyu' == dataset:
        return [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [9, 10], [1, 13],
                [3, 13], [5, 13], [7, 13], [10, 13], [11, 13], [12, 13]]
    elif 'nyu_all' == dataset:
        return [[0, 1], [1, 2], [2, 3],
                [4, 5], [5, 6], [6, 7],
                [8, 9], [9, 10], [10, 11],
                [12, 13], [13, 14], [14, 15],
                [16, 17], [17, 18], [18, 19],
                [3, 20], [7, 20], [11, 20], [15, 20], [19, 20],
                [20, 21], [20, 22]]
    elif dataset == 'icvl':
        return [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6],
                [0, 7], [7, 8], [8, 9], [0, 10], [10, 11], [11, 12],
                [0, 13], [13, 14], [14, 15]]
    elif dataset == 'msra':
        return [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
                [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16],
                [0, 17], [17, 18], [18, 19], [19, 20]]
    elif dataset == 'itop':
        return [[0, 1],
                [1, 2], [2, 4], [4, 6],
                [1, 3], [3, 5], [5, 7],
                [1, 8],
                [8, 9], [9, 11], [11, 13],
                [8, 10], [10, 12], [12, 14]]
    elif dataset == 'shrec' or 'DHG' in dataset:
        return [[0, 1],
                [0, 2], [2, 3], [3, 4], [4, 5],
                [0, 6], [6, 7], [7, 8], [8, 9],
                [0, 10], [10, 11], [11, 12], [12, 13],
                [0, 14], [14, 15], [15, 16], [16, 17],
                [0, 18], [18, 19], [19, 20], [20 ,21]]
    else:
        return [
                [0, 13], [13, 14], [14, 15], [15, 20],
                [0, 1], [1, 2], [2, 3], [3, 16],
                [0, 4], [4, 5], [5, 6], [6, 17],
                [0, 10], [10, 11], [11,  12], [12, 19],
                [0, 7], [7, 8], [8, 9], [9, 18]
                ]


def get_hierarchy_mapping(dataset):
    if 'mano' in dataset or 'hands' in dataset:
        return [[[0], [1, 2], [3, 16], [4, 5], [6,17], [10, 11], [12, 19], [7, 8],[9, 18], [13, 14],[15,20]],\
            [[0], [1, 2], [3, 4], [7, 8], [5, 6], [9, 10]], \
            [[0, 1, 2, 3, 4, 5]],
                ]
    elif 'nyu' == dataset:
        return [[[0, 1], [2,3], [4,5], [6,7], [8,9,10], [11,12,13]], ]
    elif 'nyu_all' == dataset:
        return [[[0, 1], [2, 3], [4,5], [6,7], [8,9], [10,11], [12,13], [14,15], [16,17],[18,19],[20]],\
                [[0,1], [2,3], [4,5], [6,7], [8,9], [10]],\
                [[0, 1, 2, 3, 4, 5]]]

def debug_mesh(verts, faces, batch_index, data_dir, img_type):
    batch_size = verts.size(0)
    verts = verts.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()
    for index in range(batch_size):
        path = data_dir + '/' + str(batch_index * batch_size + index) + '_' + img_type + '.obj'
        with open(path, 'w') as fp:
            for v in verts[index]:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

def get_hierarchy_sketch(dataset):
    if 'nyu' == dataset:
        return [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [9, 10], [1, 13],
                [3, 13], [5, 13], [7, 13], [10, 13], [11, 13], [12, 13]], \
               [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [1, 5], [2, 5], [3, 5], [4, 5]]
    elif 'nyu_all' == dataset:
        return [[0, 1], [1, 2], [2, 3],
                [4, 5], [5, 6], [6, 7],
                [8, 9], [9, 10], [10, 11],
                [12, 13], [13, 14], [14, 15],
                [16, 17], [17, 18], [18, 19],
                [3, 20], [7, 20], [11, 20], [15, 20], [19, 20],[20,21],[20,22]],\
               [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9],[1,10],[3,10],[5,10],[7,10],[9,10]],\
               [[0, 5], [1, 5], [2, 5], [3, 5], [4, 5]], \
               [[0, 0]]
    elif 'mano' == dataset or 'hands' in dataset:
        return [
                [0, 13], [13, 14], [14, 15], [15, 20],
                [0, 1], [1, 2], [2, 3], [3, 16],
                [0, 4], [4, 5], [5, 6], [6, 17],
                [0, 10], [10, 11], [11,  12], [12, 19],
                [0, 7], [7, 8], [8, 9], [9, 18]
                ],\
                [[0, 1], [0, 3], [0, 5], [0, 7], [0, 9], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], \
               [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]],\
                [[0, 0]]


class Color(Enum):
    RED = (0, 0, 255)
    GREEN = (75, 255, 66)
    BLUE = (255, 0, 0)
    YELLOW = (204, 153, 17) #(17, 240, 244)
    PURPLE = (255, 255, 0)
    CYAN = (255, 0, 255)
    BROWN = (204, 153, 17)


class Finger_color(Enum):
    THUMB = (0, 0, 255)
    INDEX = (75, 255, 66)
    MIDDLE = (255, 0, 0)
    RING = (17, 240, 244)
    LITTLE = (255, 255, 0)
    WRIST = (255, 0, 255)
    ROOT = (255, 0, 255)


def get_sketch_color(dataset):
    if dataset == 'FHAD' or 'hands' in dataset:
        # return (Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB,
        #         Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX,
        #        Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE,
        #         Finger_color.RING, Finger_color.RING, Finger_color.RING, Finger_color.RING,
        #        Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE)
        return [Finger_color.THUMB, Finger_color.INDEX, Finger_color.MIDDLE, Finger_color.RING, Finger_color.LITTLE,
                Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB,
                Finger_color.INDEX,  Finger_color.INDEX,  Finger_color.INDEX,
              Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE,
              Finger_color.RING, Finger_color.RING, Finger_color.RING,
              Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE,
              ]
    elif dataset == 'nyu':
        return (Finger_color.LITTLE,Finger_color.RING,Finger_color.MIDDLE,Finger_color.INDEX,Finger_color.THUMB,Finger_color.THUMB,
                Finger_color.LITTLE, Finger_color.RING, Finger_color.MIDDLE, Finger_color.INDEX, Finger_color.THUMB, Finger_color.THUMB,
                Finger_color.WRIST,Finger_color.WRIST)
    elif dataset == 'nyu_all':
        return (Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.LITTLE,
                Finger_color.RING,Finger_color.RING,Finger_color.RING,
                Finger_color.MIDDLE,Finger_color.MIDDLE,Finger_color.MIDDLE,
                Finger_color.INDEX,Finger_color.INDEX,Finger_color.INDEX,
                Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB,
                Finger_color.LITTLE, Finger_color.RING, Finger_color.MIDDLE, Finger_color.INDEX, Finger_color.THUMB, Finger_color.THUMB,
                Finger_color.WRIST,Finger_color.WRIST)
    elif dataset == 'icvl':
        return [Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB,Finger_color.INDEX,Finger_color.INDEX,Finger_color.INDEX,
                Finger_color.MIDDLE,Finger_color.MIDDLE,Finger_color.MIDDLE, Finger_color.RING,Finger_color.RING,Finger_color.RING,
                Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.LITTLE]
    elif dataset == 'msra':
        return [Finger_color.INDEX,Finger_color.INDEX,Finger_color.INDEX,Finger_color.INDEX,
                 Finger_color.MIDDLE,Finger_color.MIDDLE,Finger_color.MIDDLE,Finger_color.MIDDLE,
                 Finger_color.RING,Finger_color.RING,Finger_color.RING,Finger_color.RING,
                 Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.LITTLE,
                 Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB]
    elif dataset == 'itop':
        return [Color.RED,
                Color.GREEN, Color.GREEN, Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE,
                Color.CYAN,
                Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.PURPLE, Color.PURPLE, Color.PURPLE,
                ]
    elif dataset == 'shrec' or 'DHG' in dataset:
        return (Finger_color.ROOT,
            Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB,
         Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX,
         Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE,
         Finger_color.RING, Finger_color.RING, Finger_color.RING, Finger_color.RING,
         Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE,)
    elif dataset == 'smplerx':
        return (
            Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB,
         Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX,
         Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE,
         Finger_color.RING, Finger_color.RING, Finger_color.RING, Finger_color.RING,
         Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE,)
    else:
        return (Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB,
                Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX,
               Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE,
                Finger_color.RING, Finger_color.RING, Finger_color.RING, Finger_color.RING,
               Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE)


def get_joint_color(dataset):
    if dataset == 'FHAD'or 'hands' in dataset:
        # return [Finger_color.ROOT,
        #          Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX,
        #          Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE,
        #          Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE,
        #          Finger_color.RING, Finger_color.RING, Finger_color.RING,
        #          Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB,
        #          Finger_color.INDEX, Finger_color.MIDDLE, Finger_color.LITTLE, Finger_color.RING, Finger_color.THUMB,
        #         ]
        return [Finger_color.ROOT,
                Finger_color.THUMB, Finger_color.INDEX, Finger_color.MIDDLE, Finger_color.RING, Finger_color.LITTLE,
                Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB,
                Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX,
                Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE,
                Finger_color.RING, Finger_color.RING, Finger_color.RING,
                Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE]
    elif dataset == 'nyu':
        return [Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.RING,Finger_color.RING,Finger_color.MIDDLE,Finger_color.MIDDLE,
                Finger_color.INDEX, Finger_color.INDEX,Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB,
                Finger_color.WRIST,Finger_color.WRIST,Finger_color.WRIST]
    elif dataset == 'nyu_all':
        return [Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.LITTLE,
                Finger_color.RING,Finger_color.RING,Finger_color.RING,Finger_color.RING,
                Finger_color.MIDDLE,Finger_color.MIDDLE,Finger_color.MIDDLE,Finger_color.MIDDLE,
                Finger_color.INDEX, Finger_color.INDEX,Finger_color.INDEX, Finger_color.INDEX,
                Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB,
                Finger_color.WRIST,Finger_color.WRIST,Finger_color.WRIST]
    if dataset == 'icvl':
        return [Finger_color.ROOT,Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB,
                 Finger_color.INDEX,Finger_color.INDEX,Finger_color.INDEX,
                 Finger_color.MIDDLE,Finger_color.MIDDLE,Finger_color.MIDDLE,
                 Finger_color.RING,Finger_color.RING,Finger_color.RING,
                 Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.LITTLE]
    elif dataset == 'msra':
        return [Finger_color.WRIST,Finger_color.INDEX,Finger_color.INDEX,Finger_color.INDEX,Finger_color.INDEX,Finger_color.MIDDLE,
                Finger_color.MIDDLE,Finger_color.MIDDLE,Finger_color.MIDDLE,Finger_color.RING,Finger_color.RING,Finger_color.RING,Finger_color.RING,
                Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB]
    elif dataset == 'itop':
        return  [Color.RED,Color.BROWN,
                 Color.GREEN, Color.BLUE, Color.GREEN, Color.BLUE, Color.GREEN, Color.BLUE,
                 Color.CYAN,
                 Color.YELLOW,Color.PURPLE,Color.YELLOW,Color.PURPLE,Color.YELLOW,Color.PURPLE]
    elif dataset == 'shrec' or 'DHG' in dataset:
        return [Finger_color.ROOT, Finger_color.ROOT,
            Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB,
         Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX,
         Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE,
         Finger_color.RING, Finger_color.RING, Finger_color.RING, Finger_color.RING,
         Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE,]
    elif dataset == 'smplerx':
        return [Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB,
         Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX,
         Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE,
         Finger_color.RING, Finger_color.RING, Finger_color.RING, Finger_color.RING,
         Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE,]
    else:
        return [Finger_color.ROOT,
                 Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX,
                 Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE,
                 Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE,
                 Finger_color.RING, Finger_color.RING, Finger_color.RING,
                 Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB,
                 Finger_color.INDEX, Finger_color.MIDDLE, Finger_color.LITTLE, Finger_color.RING, Finger_color.THUMB,
                ]


def draw_point(dataset, img, pose):
    colors_joint = get_joint_color(dataset)
    idx = 0
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, colors_joint[0].value, -1)
        idx = idx + 1
    return img


def draw_pose(dataset, img, pose, scale=1):

    colors_joint = get_joint_color(dataset)
    idx = 0
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 2*scale, colors_joint[idx].value, -1)
        idx = idx + 1
        if idx >= len(colors_joint):
            break
    colors = get_sketch_color(dataset)
    idx = 0
    for index, (x, y) in enumerate(get_sketch_setting(dataset)):
        if x >= pose.shape[0] or y >= pose.shape[0]:
            break
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), colors[idx].value, 1*scale)
        idx = idx + 1
    return img

import torch.nn.functional as F
def debug_img_heatmap(img, heatmap2d, batch_index, data_dir, size, img_type='heatmap', save=False):
    cNorm = colors.Normalize(vmin=0, vmax=1.0)
    jet = plt.get_cmap('jet')
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    batch_size, head_num, height, width = heatmap2d.size()
    heatmap2d = heatmap2d.view(batch_size,head_num,-1)
    heatmap2d = (heatmap2d - heatmap2d.min(dim=-1, keepdim=True)[0])
    heatmap2d = heatmap2d / (heatmap2d.max(dim=-1, keepdim=True)[0] + 1e-8)
    heatmap2d = heatmap2d.view(batch_size, head_num, height, width)
    img = F.interpolate(img, (height, width))
    heatmap_list = []
    heatmap = heatmap2d.cpu().detach().numpy()
    img = (img.cpu().detach().numpy()+1)/2*255
    for index in range(heatmap2d.size(0)):
        for joint_index in range(heatmap2d.size(1)):
                img_dir = data_dir + '/' + img_type + '_' + str(batch_size * batch_index + index) + '_' + \
                          str(joint_index) + '.png'
                heatmap_draw = cv2.resize(heatmap[index, joint_index], (size, size))
                heatmap_color = 255 * scalarMap.to_rgba(1 - heatmap_draw)
                img_draw = cv2.cvtColor(img[index, 0], cv2.COLOR_GRAY2RGB)/2 + heatmap_color.reshape(size, size, 4)[:, :, 0:3]
                if save:
                    cv2.imwrite(img_dir, img_draw)
                heatmap_list.append(img_draw)
    return np.stack(heatmap_list, axis=0).squeeze()


def debug_2d_heatmap(heatmap2d, batch_index, data_dir, size, img_type='heatmap', save=False):
    cNorm = colors.Normalize(vmin=0, vmax=1.0)
    jet = plt.get_cmap('jet')
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    batch_size, head_num, height, width = heatmap2d.size()
    if batch_size==0:
        return 0
    # heatmap2d = heatmap2d.view(batch_size,head_num,-1)
    # heatmap2d = (heatmap2d - heatmap2d.min(dim=-1, keepdim=True)[0])
    # heatmap2d = heatmap2d / (heatmap2d.max(dim=-1, keepdim=True)[0] + 1e-8)
    # heatmap2d = heatmap2d.view(batch_size, head_num, height, width)

    # heatmap2d = F.interpolate(heatmap2d, size=[128, 128])
    heatmap_list = []
    heatmap = heatmap2d.cpu().detach().numpy()
    for index in range(heatmap2d.size(0)):
        for joint_index in range(heatmap2d.size(1)):
                img_dir = data_dir + '/' + img_type + '_' + str(batch_size * batch_index + index) + '_' + str(
                    joint_index) + '.png'
                heatmap_draw = cv2.resize(heatmap[index, joint_index], (size, size))
                heatmap_color = 255 * scalarMap.to_rgba(1 - heatmap_draw)
                if save:
                    cv2.imwrite(img_dir, heatmap_color.reshape(size, size, 4)[:, :, 0:3])
                heatmap_list.append(heatmap_color.reshape(size, size, 4)[:, :, 0:3])
                # ret, img_show = cv2.threshold(img_draw[index, 0] * 255.0, 245, 255, cv2.THRESH_BINARY)
                # img_show = cv2.cvtColor(img_show, cv2.COLOR_GRAY2RGB)
                # cv2.imwrite(img_dir, img_show/2 + heatmap_color.reshape(128, 128, 4)[:, :, 0:3])
    return np.stack(heatmap_list, axis=0).squeeze()


def debug_offset(data, batch_index, GFM_):
    img, pcl_sample, joint_world, joint_img, center, M, cube, pcl_normal, joint_normal, offset, coeff, max_bbx_len = data
    img_size = 32
    batch_size,joint_num,_ = joint_world.size()
    offset = GFM_.joint2offset(joint_img, img, feature_size=img_size)
    unit = offset[:, 0:joint_num*3, :, :].numpy()
    for index in range(batch_size):
        fig, ax = plt.subplots()
        unit_plam = unit[index, 0:3, :, :]
        x = np.arange(0,img_size,1)
        y = np.arange(0,img_size,1)

        X, Y = np.meshgrid(x, y)
        Y = img_size - 1 - Y
        ax.quiver(X, Y, unit_plam[0, ...], unit_plam[1, ...])
        ax.axis([0, img_size, 0, img_size])
        plt.savefig('./debug/offset_' + str(batch_index) + '_' + str(index) + '.png')


def debug_offset_heatmap(img, joint, batch_index, GFM_, kernel_size):
    img_size = 128
    batch_size,joint_num,_ = joint.size()
    offset = GFM_.joint2offset(joint, img, kernel_size, feature_size=img_size)
    heatmap = offset[:, joint_num*3:, :, :].numpy()
    cNorm = colors.Normalize(vmin=0, vmax=1.0)
    jet = plt.get_cmap('jet')
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    img_draw = img.numpy()
    for index in range(batch_size):
        for joint_index in range(joint_num):
            img_dir = './debug/' + str(batch_size * batch_index + index) + '_' + str(joint_index) + '.png'
            heatmap_color = 255 * scalarMap.to_rgba((kernel_size-heatmap[index, joint_index].reshape(128, 128)) / kernel_size)
            img_show = cv2.cvtColor(img_draw[index, 0] * 255.0/2.0, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(img_dir, img_show + heatmap_color.reshape(128, 128, 4)[:, :, 0:3])


def debug_2d_img(img, index, data_dir, name, batch_size):
    _, num, input_size, input_size = img.size()
    img_list = []
    for img_idx in range(img.size(0)):
        for channel_idx in range(img.size(1)):
            img_draw = (img.detach().cpu().numpy()[img_idx,channel_idx] + 1) / 2 * 255
            img_draw = cv2.cvtColor(img_draw, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(data_dir + '/' + str(batch_size * index + img_idx) + '_'+str(channel_idx)+"_" + name + '.png', img_draw)
            img_list.append(img_draw)
    return np.stack(img_list, axis=0)


def debug_2d_pose(img, joint_img, index, dataset, data_dir, name, batch_size, save=False):
    _, num, input_size, input_size = img.size()
    img_list = []
    for img_idx in range(joint_img.size(0)):
        joint_uvd = (joint_img.detach().cpu().numpy() + 1) / 2 * input_size
        img_draw = (img.detach().cpu().numpy() + 1) / 2 * 255
        img_show = draw_pose(dataset, cv2.cvtColor(img_draw[img_idx, 0], cv2.COLOR_GRAY2RGB),
                             joint_uvd[img_idx], input_size // 128)
        if save:
           cv2.imwrite(data_dir + '/' + str(batch_size * index + img_idx) + '_' + name + '.png', img_show)
        img_list.append(img_show)
    return np.stack(img_list, axis=0)


def debug_2d_pose_select(img, joint_img, index, dataset, data_dir, name, batch_size, select_id, save=False):
    _, num, input_size, input_size = img.size()
    img_list = []
    for img_index, img_id in enumerate(select_id):
        joint_uvd = (joint_img.detach().cpu().numpy() + 1) / 2 * input_size
        img_draw = (img.detach().cpu().numpy() + 1) / 2 * 255
        img_show = draw_pose(dataset, cv2.cvtColor(img_draw[img_index, 0], cv2.COLOR_GRAY2RGB),
                             joint_uvd[img_index], input_size // 128)
        if save:
           cv2.imwrite(data_dir + '/' + str(batch_size * index + img_id) + '_' + name + '.png', img_show)
        img_list.append(img_show)
    # return np.stack(img_list, axis=0)
    return 0

def draw_2d_pose(img, joint_img, dataset):
    num, input_size, input_size = img.size()
    joint_uvd = (joint_img.detach().cpu().numpy() + 1) / 2 * input_size
    img_draw = (img.detach().cpu().numpy() + 1) / 2 * 255
    img_show = draw_pose(dataset, cv2.cvtColor(img_draw[0], cv2.COLOR_GRAY2RGB), joint_uvd)
    return img_show


def draw_visible(dataset, img, pose, visible):
    idx = 0
    color = [Color.RED, Color.BLUE]
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, color[visible[idx]].value, -1)
        idx = idx + 1
    idx = 0
    for x, y in get_sketch_setting(dataset):
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), Color.BROWN.value, 1)
        idx = idx + 1
    return img


def debug_visible_joint(img, joint_img, visible, index, dataset, data_dir, name):
    batch_size,_,input_size,input_size = img.size()
    visible = visible.detach().cpu().numpy().astype(np.int)
    for img_idx in range(img.size(0)):
        joint_uvd = (joint_img.detach().cpu().numpy() + 1) / 2 * input_size
        img_draw = (img.detach().cpu().numpy() + 1) / 2 * 255
        img_show = draw_visible(dataset, cv2.cvtColor(img_draw[img_idx, 0], cv2.COLOR_GRAY2RGB), joint_uvd[img_idx], visible[img_idx])
        cv2.imwrite(data_dir + '/' + str(batch_size * index + img_idx) + '_' + name + '.png', img_show)


def draw_pcl(pcl, img_size, background_value=1):
    device = pcl.device
    batch_size = pcl.size(0)
    img_pcl = []
    for index in range(batch_size):
        img = torch.ones([img_size, img_size]).to(device) * background_value
        index_x = torch.clamp(torch.floor((pcl[index, :, 0] + 1) / 2 * img_size), 0, img_size - 1).long()
        index_y = torch.clamp(torch.floor((pcl[index, :, 1] + 1) / 2 * img_size), 0, img_size - 1).long()
        img[index_y, index_x] = -1
        img_pcl.append(img)
    return torch.stack(img_pcl, dim=0).unsqueeze(1)


def debug_pcl_pose(pcl, joint_xyz, index, dataset, data_dir, name):
    """
    :param pcl:
    :param joint_xyz:
    :param index:
    :param dataset:
    :param data_dir:
    :param name:
    :return:
    """
    batch_size = pcl.size(0)
    if batch_size == 0:
        return 0
    img = draw_pcl(pcl, 128)
    for img_idx in range(img.size(0)):
        joint_uvd = (joint_xyz.detach().cpu().numpy() + 1) / 2 * 128
        img_draw = (img.detach().cpu().numpy() + 1) / 2 * 255
        im_color = cv2.cvtColor(img_draw[img_idx, 0], cv2.COLOR_GRAY2RGB)
        img_show = draw_pose(dataset, im_color, joint_uvd[img_idx])
        cv2.imwrite(data_dir + '/' + str(batch_size * index + img_idx) + '-' + name + '.png', img_show)


def draw_muti_pic(batch_img_list, index, data_dir, name, text=None, save=True, max_col=7, batch_size=32):
    # batch_size = batch_img_list[0].shape[0]
    for batch_index in range(batch_size):
        img_list = []
        img_list_temp = []
        for img_index, imgs in enumerate(batch_img_list):
            img_list_temp.append(imgs[batch_index].squeeze())
            if (img_index + 1) % max_col == 0:
                img_list.append(np.hstack(img_list_temp))
                img_list_temp = []

        if img_index < max_col:
            imgs = np.hstack(img_list_temp)
        else:
            imgs = np.concatenate(img_list, axis=0)

        if text:
            cv2.putText(imgs, text[batch_index], (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200), 1)
        if save:
            cv2.imwrite(data_dir + '/' + name + '_' + str(batch_size * index + batch_index)  + '.png', imgs)
    return imgs
def draw_2d_skeleton(image, pose_uv):
    """
    :param image: H x W x 3
    :param pose_uv: 21 x 2
    wrist,
    thumb_mcp, thumb_pip, thumb_dip, thumb_tip
    index_mcp, index_pip, index_dip, index_tip,
    middle_mcp, middle_pip, middle_dip, middle_tip,
    ring_mcp, ring_pip, ring_dip, ring_tip,
    little_mcp, little_pip, little_dip, little_tip
    :return:
    """
    assert pose_uv.shape[0] == 21
    skeleton_overlay = image.copy()
    marker_sz = 2
    line_wd = 1
    root_ind = 0

    for joint_ind in range(pose_uv.shape[0]):
        joint = pose_uv[joint_ind, 0].astype('int32'), pose_uv[joint_ind, 1].astype('int32')
        cv2.circle(
            skeleton_overlay, joint,
            radius=marker_sz, color=color_hand_joints[joint_ind] * np.array(255), thickness=-1,
            lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            root_joint = pose_uv[root_ind, 0].astype('int32'), pose_uv[root_ind, 1].astype('int32')
            cv2.line(
                skeleton_overlay, root_joint, joint,
                color=color_hand_joints[joint_ind] * np.array(255), thickness=int(line_wd),
                lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)
        else:
            joint_2 = pose_uv[joint_ind - 1, 0].astype('int32'), pose_uv[joint_ind - 1, 1].astype('int32')
            cv2.line(
                skeleton_overlay, joint_2, joint,
                color=color_hand_joints[joint_ind] * np.array(255), thickness=int(line_wd),
                lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)


    return skeleton_overlay


def vis_keypoints_with_skeleton(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_keypoints(img, kps, alpha=1, radius=3, color=None):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    if color is None:
        colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        if color is None:
            cv2.circle(kp_mask, p, radius=radius, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(kp_mask, p, radius=radius, color=color, thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_mesh(img, mesh_vertex, alpha=0.5):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(mesh_vertex))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    mask = np.copy(img)

    # Draw the mesh
    for i in range(len(mesh_vertex)):
        p = mesh_vertex[i][0].astype(np.int32), mesh_vertex[i][1].astype(np.int32)
        cv2.circle(mask, p, radius=1, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, mask, alpha, 0)


def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, input_shape, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
        y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
        z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])

        if kpt_3d_vis[i1, 0] > 0 and kpt_3d_vis[i2, 0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1, 0] > 0:
            ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], c=colors[l], marker='o')
        if kpt_3d_vis[i2, 0] > 0:
            ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], c=colors[l], marker='o')

    x_r = np.array([0, input_shape[1]], dtype=np.float32)
    y_r = np.array([0, input_shape[0]], dtype=np.float32)
    z_r = np.array([0, 1], dtype=np.float32)

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()

    plt.show()
    cv2.waitKey(0)


def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0] + 1) + '/' + str(f[i][0] + 1) + ' ' + str(f[i][1] + 1) + '/' + str(
            f[i][1] + 1) + ' ' + str(f[i][2] + 1) + '/' + str(f[i][2] + 1) + '\n')
    obj_file.close()


def perspective_projection(vertices, cam_param):
    # vertices: [N, 3]
    # cam_param: [3]
    fx, fy = cam_param['focal']
    cx, cy = cam_param['princpt']
    vertices[:, 0] = vertices[:, 0] * fx / vertices[:, 2] + cx
    vertices[:, 1] = vertices[:, 1] * fy / vertices[:, 2] + cy
    return vertices


# def render_mesh(img, mesh, face, cam_param, mesh_as_vertices=False):
#     if mesh_as_vertices:
#         # to run on cluster where headless pyrender is not supported for A100/V100
#         vertices_2d = perspective_projection(mesh, cam_param)
#         img = vis_keypoints(img, vertices_2d, alpha=0.8, radius=2, color=(0, 0, 255))
#     else:
#         # mesh
#         mesh = trimesh.Trimesh(mesh, face)
#         # mesh.export('/home/cyc/pycharm/vGesture/lib/core/test_img/1129/test.obj')
#         rot = trimesh.transformations.rotation_matrix(
#             np.radians(180), [1, 0, 0])
#         mesh.apply_transform(rot)
#         material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE',
#                                                       baseColorFactor=(1.0, 1.0, 0.9, 1.0))
#         mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
#         scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
#         scene.bg_color = [0.0, 0.0, 0.0]  # 设置为黑色背景
#         scene.add(mesh, 'mesh')

#         focal, princpt = cam_param['focal'], cam_param['princpt']
#         camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
#         scene.add(camera)

#         # renderer
#         renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

#         # light
#         light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
#         light_pose = np.eye(4)
#         light_pose[:3, 3] = np.array([0, -1, 1])
#         scene.add(light, pose=light_pose)
#         light_pose[:3, 3] = np.array([0, 1, 1])
#         scene.add(light, pose=light_pose)
#         light_pose[:3, 3] = np.array([1, 1, 2])
#         scene.add(light, pose=light_pose)

#         # render
#         rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
#         rgb = rgb[:, :, :3].astype(np.float32)
#         only_rgb = rgb[:,:,:3].astype(np.uint8)
#         valid_mask = (depth > 0)[:, :, None]

#         # save to image
#         if len(img.shape) == 2:
#             img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#         img = rgb * valid_mask + img * (1 - valid_mask)
#     return img,only_rgb

def render_mesh(mesh, face, cam_param):
    import pyrender
    import trimesh
    import numpy as np

    # 创建 Trimesh 网格
    mesh = trimesh.Trimesh(vertices=mesh, faces=face)
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)

    # 平移网格到相机前
    mesh.apply_translation([-mesh.center_mass[0], -mesh.center_mass[1], 0])

    # 设置材质
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.5, roughnessFactor=0.5,
                                                  baseColorFactor=(1.0, 0.0, 0.0, 1.0))  # 红色网格

    # 创建场景
    scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5), bg_color=[0.1, 0.1, 0.1])
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    scene.add(mesh)

    # 设置相机
    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [0, 0, 1.0]  # 设置相机位置更近
    scene.add(camera, pose=camera_pose)

    # 添加光照
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    light_pose = np.eye(4)
    light_pose[:3, 3] = [0, 0, 2.0]  # 光源位置
    scene.add(light, pose=light_pose)

    # 渲染器
    renderer = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=800)
    rgb, _ = renderer.render(scene, flags=pyrender.RenderFlags.ALL_SOLID)

    # 释放渲染器资源
    renderer.delete()

    return rgb



def cam_equal_aspect_3d(ax, verts, flip_x=False, transpose=True):
    '''
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    :param ax:
    :param verts:
    :param flip_x:
    :return:
    '''
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    # min_lim, max_lim = np.min(centers - r), np.max(centers + r)
    if flip_x:
      ax.set_xlim(centers[0] + r, centers[0] - r)
      # ax.set_xlim(max_lim, min_lim)
    else:
      ax.set_xlim(centers[0] - r, centers[0] + r)
      # ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(centers[1] - r, centers[1] + r)
    ax.set_zlim(centers[2] + r, centers[2] - r)
    # ax.set_ylim(min_lim, max_lim)
    # ax.set_zlim(max_lim, min_lim)
    if transpose:
      ax.set_xlabel('X')
      ax.set_ylabel('Z')
      ax.set_zlabel('Y')
    else:
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')
    # ax.view_init(5, -5)
    ax.view_init(5, -85)


def draw_mesh(path, verts, faces, transpose=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # XYZ -> XZY
    if transpose:
        verts = verts[:, [0, 2, 1]]
    faces = faces.astype(int)
    mesh = Poly3DCollection(verts[faces], alpha=0.3)
    face_color = (141 / 255, 184 / 255, 226 / 255)
    edge_color = (50 / 255, 50 / 255, 50 / 255)
    mesh.set_facecolor(face_color)
    mesh.set_edgecolor(edge_color)

    ax.add_collection3d(mesh)

    cam_equal_aspect_3d(ax, verts, transpose=transpose)

    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def draw_mesh_without_axis(verts, faces, transpose=True):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import numpy as np

    # 创建绘图区域
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # XYZ -> XZY 转置
    if transpose:
        verts = verts[:, [0, 2, 1]]
    faces = faces.astype(int)

    # 创建网格
    mesh = Poly3DCollection(verts[faces], alpha=0.3)
    mesh.set_edgecolor((0, 0, 0))  # 设置边框为黑色
    mesh.set_facecolor((1, 1, 1, 0))  # 设置面为透明

    ax.add_collection3d(mesh)

    # 设置相机和视图比例
    cam_equal_aspect_3d(ax, verts, transpose=transpose)

    # 去除坐标系
    ax.axis('off')

    # 设置背景为透明
    fig.patch.set_alpha(0.0)

    # 将图像保存到内存
    t0=time.time()
    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    print('draw_mesh render time:',time.time()-t0)
    width, height = canvas.get_width_height()
    image = image.reshape((height, width, 3))

    # 关闭绘图
    plt.close(fig)

    return image





