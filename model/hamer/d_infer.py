import argparse
import glob
from tqdm import tqdm
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from rich import print
from line_profiler import LineProfiler, profile

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import open3d as o3d
import smplx
import trimesh

from hamer.models.mano_wrapper import MANO
from hamer.utils.mesh_renderer import MeshRenderer
# from onnx_utils import check_onnx_model
sys.path.append('../')
sys.path.append('../../')

from model.rootnet.Model_RGB import get_model

import onnxruntime as ort
import time
import torch
# from hamer.models.hamer import HAMER_INFER

from hamer.models import load_hamer
from hamer.utils.renderer import cam_crop_to_full, custom_cam_crop_to_full
from hamer.utils.geometry import perspective_projection

import cv2
import numpy as np
from skimage.filters import gaussian
from yolo.detector import Detector    
from config.yolo_config import yolo_opt
from hamer.datasets.utils import (convert_cvimg_to_tensor,
                    expand_to_aspect_ratio,
                    generate_image_patch_cv2)


lp = LineProfiler()

from config.hamer_config import hamer_opt

DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])

ckpt_path = "/home/pt/fbs/model/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
model_cfg = "/home/pt/fbs/model/hamer/_DATA/hamer_ckpts/checkpoints/model_config.yaml"
HAMER_ONNX_CKPT_PATH = "/home/pt/vGesture/software/hamer/_DATA/hamer_ckpts/onnx/hamer_inferpy.onnx"
HAMER_ONNX_INPUT_NAMES = ['img']
HAMER_ONNX_OUTPUT_NAMES = ['pred_cam', 'pred_cam_t', 'focal_length',
                           'pred_keypoints_3d', 'pred_vertices', 'pred_keypoints_2d',
                           'global_orient', 'hand_pose', 'betas', 'trans']

import torch
import torch.nn as nn

import torch
import torch.nn as nn
# =============================================================================
# 辅助函数：手动转换 轴角 -> 旋转矩阵 (避免 smplx 依赖)
# =============================================================================
def axis_angle_to_rotation_matrix_torch(rvec_tensor):
    """
    将轴角转换为旋转矩阵 (batch_rodrigues)
    Input:  Tensor [N, 3]
    Output: Tensor [N, 3, 3]
    """
    theta = torch.norm(rvec_tensor, dim=1, keepdim=True) + 1e-8
    r_hat = rvec_tensor / theta
    cos = torch.cos(theta)
    z_stick = torch.zeros(theta.shape[0], dtype=rvec_tensor.dtype, device=rvec_tensor.device)
    m = torch.stack([
        z_stick, -r_hat[:, 2], r_hat[:, 1],
        r_hat[:, 2], z_stick, -r_hat[:, 0],
        -r_hat[:, 1], r_hat[:, 0], z_stick], dim=1).reshape(-1, 3, 3)
    
    i_cube = torch.eye(3, dtype=rvec_tensor.dtype, device=rvec_tensor.device).unsqueeze(0).expand(rvec_tensor.shape[0], -1, -1)
    A = r_hat.unsqueeze(2) * r_hat.unsqueeze(1)
    
    return cos.unsqueeze(2) * i_cube + (1 - cos.unsqueeze(2)) * A + torch.sin(theta).unsqueeze(2) * m


class HAMER_ONNX_Wrapper(nn.Module):
    def __init__(self, hamer_model):
        super(HAMER_ONNX_Wrapper, self).__init__()
        self.hamer = hamer_model
        self.output_names = HAMER_ONNX_OUTPUT_NAMES
    
    @staticmethod
    def hamer_output_tuple_of_dict2tuple(output):
        # dict1: dict_keys(['pred_cam', 'pred_mano_params', 'pred_cam_t', 'focal_length', 'pred_keypoints_3d', 'pred_vertices', 'pred_keypoints_2d'])
        # dict2: dict_keys(['global_orient', 'hand_pose', 'betas', 'trans'])
        out_dict, params_dict = output
        return (
            out_dict['pred_cam'],
            out_dict['pred_cam_t'],
            out_dict['focal_length'],
            out_dict['pred_keypoints_3d'],
            out_dict['pred_vertices'],
            out_dict['pred_keypoints_2d'],
            params_dict['global_orient'],
            params_dict['hand_pose'],
            params_dict['betas'],
            params_dict['trans'],
        )

    def forward(self, img):
        # 将输入参数重新组装成batch字典，与estimate_from_rgb调用model(batch)一致
        batch = {'img': img}
        out, params = self.hamer(batch)
        return self.hamer_output_tuple_of_dict2tuple((out, params))


class hamer_inference():
    def __init__(self, cfg):
        ckpt_path = cfg.ckpt_path
        model_cfg = cfg.model_cfg
        use_onnx = cfg.use_onnx
        onnx_path = cfg.onnx_path
        
        self.use_onnx = use_onnx
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.use_onnx:
            print("Loading ONNX model...")
            self.ort_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            # In ONNX mode, we need to load the config separately for post-processing
            from .hamer.configs import get_config
            self.cfg = get_config(model_cfg)
            self.model = None
            # self.mesh_renderer = self.get_mesh_renderer()
        else:
            print("Loading PyTorch model...")
            print('ckpt_path:', ckpt_path)
            model, model_cfg_obj = load_hamer(ckpt_path)
            self.model = model.to(self.device)
            self.model.eval()
            self.cfg = model_cfg_obj
            self.ort_session = None
            # self.mesh_renderer = self.model.mesh_renderer

        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

    def get_mesh_renderer(self):
        mano_cfg = {k.lower(): v for k,v in dict(self.cfg.MANO).items()}
        self.mano = MANO(**mano_cfg)
        mesh_renderer = MeshRenderer(self.cfg, faces=self.mano.faces)
        return mesh_renderer

    def prepare_batch_bbox(self, img_0, bboxs):
        batch_imgs = []
        batch_centers = []
        batch_sizes = []
        batch_img_sizes = []
        batch_inv_trans = []
        batch_do_flips = []
        batch_trans = []
        """处理多个边界框数据"""
        for bbox in bboxs:
            # 解包类别和坐标
            if isinstance(bbox, list) and len(bbox) == 2:
                hand_cls = bbox[0]
                coords = bbox[1]

                # 验证坐标结构
                if not (isinstance(coords, list) and len(coords) == 4):
                    print(f"Invalid coordinates format: Expected [x1,y1,x2,y2], got {coords}")

                x1, y1, x2, y2 = coords
            else:
                hand_cls = 'left'
                print(f"Invalid bbox format: Expected [class, coords], got {bbox}")

            # 将类别映射为翻转标志
            do_flip = 0.0 if hand_cls == 'right' else 1.0

            # 计算中心点
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            # 计算边界框尺寸
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            # 设置缩放因子
            rescaling_factor = 2.5

            # 计算缩放后的尺寸，保持为数组格式
            scale = np.array([rescaling_factor * bbox_width / 200.0,
                              rescaling_factor * bbox_height / 200.0])

            # 计算最终的边界框尺寸
            BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
            if BBOX_SHAPE is not None:
                final_bbox_size = expand_to_aspect_ratio(scale * 200, target_aspect_ratio=BBOX_SHAPE).max()
            else:
                # 如果没有指定BBOX_SHAPE，使用原始逻辑
                final_bbox_size = max(bbox_width, bbox_height) * rescaling_factor

            # 图像块尺寸
            patch_width = patch_height = self.cfg.MODEL.IMAGE_SIZE  # 256

            # 处理输入图像
            cvimg = img_0.copy()

            # 防止锯齿处理
            t0 = time.time()
            # downsampling_factor = (final_bbox_size / patch_width) / 2.0
            # if downsampling_factor > 1.1:
            #     cvimg = gaussian(cvimg, sigma=(downsampling_factor - 1) / 2, channel_axis=2, preserve_range=True)
            t1 = time.time()
            # 生成图像块
            img_patch_cv, trans, inv_trans = generate_image_patch_cv2(
                cvimg,
                center_x, center_y,
                final_bbox_size, final_bbox_size,
                patch_width, patch_height,
                False, 1.0, 0,
                border_mode=cv2.BORDER_CONSTANT
            )
            t2 = time.time()
            
            # 将处理后的图像转换为张量
            img_patch_cv = img_patch_cv[:, :, ::-1]  # BGR to RGB
            if hand_cls != 'right':
                img_patch_cv = cv2.flip(img_patch_cv, 1)

            img_patch = convert_cvimg_to_tensor(img_patch_cv)

            # 应用标准化
            for n_c in range(3):  # 确保处理所有3个通道
                mean_val = float(self.mean[n_c]) if hasattr(self.mean, '__getitem__') else float(self.mean)
                std_val = float(self.std[n_c]) if hasattr(self.std, '__getitem__') else float(self.std)
                img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean_val) / std_val

            # 收集
            batch_imgs.append(torch.tensor(img_patch, dtype=torch.float32))  # (3,H,W)
            batch_centers.append(torch.tensor([center_x, center_y], dtype=torch.float32))  # (2,)
            batch_sizes.append(torch.tensor(final_bbox_size, dtype=torch.float32))  # (1,)
            batch_img_sizes.append(torch.tensor([cvimg.shape[1], cvimg.shape[0]], dtype=torch.float32))  # (2,)
            batch_inv_trans.append(torch.tensor(inv_trans, dtype=torch.float32))  # (3,3)
            batch_trans.append(torch.tensor(trans, dtype=torch.float32))  # (3,3)
            batch_do_flips.append(torch.tensor(do_flip, dtype=torch.float32))  # (1,)
        
        # 最后统一 stack 
        items = {
            'img': torch.stack(batch_imgs, dim=0).float(),  # (B,3,H,W)
            'box_center': torch.stack(batch_centers, dim=0),  # (B,2)
            'box_size': torch.stack(batch_sizes, dim=0),  # (B,1)
            'img_size': torch.stack(batch_img_sizes, dim=0),  # (B,2)
            'inv_trans': torch.stack(batch_inv_trans, dim=0),  # (B,3,3)
            'trans': torch.stack(batch_trans, dim=0),  # (B,3,3)
            'do_flip': torch.stack(batch_do_flips, dim=0),  # (B,1)
        }
        return items



    def prepare_item(self, img_0, bbox):
        """处理单个边界框数据"""
        # 打印调试信息
        # print(f"[DEBUG] Preparing item from bbox: {bbox}")
        
        # 解包类别和坐标
        if isinstance(bbox, list) and len(bbox) == 2:
            cls = bbox[0]
            coords = bbox[1]
            
            # 验证坐标结构
            if not (isinstance(coords, list) and len(coords) == 4):
                raise ValueError(f"Invalid coordinates format: Expected [x1,y1,x2,y2], got {coords}")
            
            x1, y1, x2, y2 = coords
        else:
            raise ValueError(f"Invalid bbox format: Expected [class, coords], got {bbox}")
        
        # 将类别映射为翻转标志
        do_flip = 0.0 if cls == 'right' else 1.0
        
        # 计算中心点
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        
        # 计算边界框尺寸
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # 设置缩放因子
        rescaling_factor = 2.5
        
        # 计算缩放后的尺寸，保持为数组格式
        scale = np.array([rescaling_factor * bbox_width / 200.0, 
                        rescaling_factor * bbox_height / 200.0])
        
        # 计算最终的边界框尺寸
        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
        if BBOX_SHAPE is not None:
            final_bbox_size = expand_to_aspect_ratio(scale * 200, target_aspect_ratio=BBOX_SHAPE).max()
        else:
            # 如果没有指定BBOX_SHAPE，使用原始逻辑
            final_bbox_size = max(bbox_width, bbox_height) * rescaling_factor
        
        # 图像块尺寸
        patch_width = patch_height = self.cfg.MODEL.IMAGE_SIZE #256
        
        # 处理输入图像
        cvimg = img_0.copy()
        
        # 防止锯齿处理
        downsampling_factor = (final_bbox_size / patch_width) / 2.0
        if downsampling_factor > 1.1:
            cvimg = gaussian(cvimg, sigma=(downsampling_factor-1)/2, channel_axis=2, preserve_range=True)
        
        # 生成图像块
        img_patch_cv, trans, inv_trans = generate_image_patch_cv2(
            cvimg,
            center_x, center_y,
            final_bbox_size, final_bbox_size,
            patch_width, patch_height,
            False, 1.0, 0,
            border_mode=cv2.BORDER_CONSTANT
        )
        
        # 将处理后的图像转换为张量
        img_patch_cv = img_patch_cv[:, :, ::-1]  # BGR to RGB
        if cls != 'right':
            img_patch_cv =cv2.flip(img_patch_cv,1)
            
        img_patch = convert_cvimg_to_tensor(img_patch_cv)
        
        # 应用标准化
        for n_c in range(3):  # 确保处理所有3个通道
            mean_val = float(self.mean[n_c]) if hasattr(self.mean, '__getitem__') else float(self.mean)
            std_val = float(self.std[n_c]) if hasattr(self.std, '__getitem__') else float(self.std)
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean_val) / std_val
        
        # 准备返回的项目
        item = {
            'img': torch.tensor(img_patch, dtype=torch.float32).unsqueeze(0),
            'box_center': torch.tensor([[center_x, center_y]], dtype=torch.float32),
            'box_size': torch.tensor([[final_bbox_size]], dtype=torch.float32),
            'img_size': torch.tensor([[cvimg.shape[1], cvimg.shape[0]]], dtype=torch.float32),
            'inv_trans': torch.tensor(inv_trans, dtype=torch.float32).unsqueeze(0),
            'do_flip': torch.tensor([do_flip], dtype=torch.float32),
            'trans': trans,
        }
        
        return item

    @torch.no_grad()
    def estimate_from_rgb(self, img_0, detections, k_real=None, depth_refine=None):
            """
            k_real: (3, 3) 的内参矩阵 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                    支持 numpy array 或 torch tensor
            """
            if not isinstance(detections, list) or len(detections) == 0:
                raise ValueError("Invalid detections format")
            
            batch = self.prepare_batch_bbox(img_0, detections)
            
            # 确保所有数据都是浮点类型并在正确设备上
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device).float()
            
            # === 推理部分 (不变) ===
            if self.use_onnx:
                input_img = batch['img'].cpu().numpy()
                onnx_outputs = self.ort_session.run(HAMER_ONNX_OUTPUT_NAMES, {'img': input_img})
                out_tensors = [torch.from_numpy(val).to(self.device) for val in onnx_outputs]
                out = dict(zip(HAMER_ONNX_OUTPUT_NAMES[:6], out_tensors[:6]))
                params = dict(zip(HAMER_ONNX_OUTPUT_NAMES[6:], out_tensors[6:]))
                out['pred_mano_params'] = params
            else:
                out, params = self.model(batch)
                
            # === 后处理准备 ===
            pred_cam = out['pred_cam']
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            do_flip = batch['do_flip'].float()
            trans = batch['trans']
            inv_trans = batch['inv_trans']
            
            # 处理 3D 关键点 (翻转处理)
            pred_keypoints_3d = out['pred_keypoints_3d'].float()
            pred_keypoints_3d[:, :, 0] = pred_keypoints_3d[:, :, 0] * do_flip.unsqueeze(1) 

            # =======================================================
            # === 核心修复：针对左手（flipped），反转 cam_bbox 的 x 偏移量 ===
            # =======================================================
            # pred_cam 形状为 [B, 3] -> (scale, tx, ty)
            # do_flip 形状为 [B, 1] 或 [B]，其中 1.0 表示左手(翻转过)，0.0 表示右手
            
            # 创建一个修正因子向量：右手为 1.0，左手为 -1.0
            # 公式：1 - 2 * 1 = -1 (左手); 1 - 2 * 0 = 1 (右手)
            flip_correction = 1.0 - 2.0 * do_flip.view(-1)
            
            # 复制一份 pred_cam 以免影响原数据（虽然直接改也可以）
            pred_cam_corrected = pred_cam.clone()
            
            # 修正 tx (索引为1)
            # 注意：只修正局部偏移的方向，不修正 scale 和 ty
            pred_cam_corrected[:, 1] = pred_cam_corrected[:, 1] * flip_correction
            
            # =======================================================

            # =======================================================
            # === 核心修改：根据 k_real 是否存在选择计算路径 ===
            # =======================================================
            if k_real is not None:
                # --- 路径 A: 使用真实内参 ---
                # print("使用真实内参")
                # 1. 解析 k_real (假设是 3x3 矩阵)
                if isinstance(k_real, np.ndarray):
                    k_real = torch.from_numpy(k_real).float().to(self.device)
                elif isinstance(k_real, torch.Tensor):
                    k_real = k_real.float().to(self.device)
                
                # 提取参数 (处理 batch 维度，如果 k_real 只有 (3,3) 则广播)
                if k_real.dim() == 2:
                    fx = k_real[0, 0]
                    fy = k_real[1, 1]
                    cx = k_real[0, 2]
                    cy = k_real[1, 2]
                else: # 假设是 (B, 3, 3)
                    fx = k_real[:, 0, 0]
                    fy = k_real[:, 1, 1]
                    cx = k_real[:, 0, 2]
                    cy = k_real[:, 1, 2]

                # 2. 调用自定义函数计算真实的 3D 平移
                print("计算真实相机平移...")
                pred_cam_t_full = custom_cam_crop_to_full(
                    pred_cam_corrected, box_center, box_size, img_size, fx, fy, cx, cy, depth_refine=depth_refine
                )
                
                # 3. 设置 scaled_focal_length (仅供渲染器使用，取 fx)
                scaled_focal_length = fx.unsqueeze(0) if fx.dim()==0 else fx

                # 4. 手动计算 2D 投影 (Manual Projection)
                # 为什么要手动？因为 perspective_projection 函数通常不支持偏移的主点 (cx, cy)
                # 公式: u = fx * (X/Z) + cx, v = fy * (Y/Z) + cy
                
                # 确保 pred_cam_t_full 形状正确
                if pred_cam_t_full.dim() == 3 and pred_cam_t_full.shape[1] == 1:
                    pred_cam_t_full = pred_cam_t_full.squeeze(1) # [B, 3]

                # 将相机平移加到关键点上 -> 变换到相机坐标系
                # pred_keypoints_3d: [B, N, 3], pred_cam_t_full: [B, 3] -> [B, 1, 3]
                kp_cam = pred_keypoints_3d + pred_cam_t_full.unsqueeze(1)
                
                # 透视除法
                depth = kp_cam[:, :, 2:3] + 1e-9 # Z
                x_norm = kp_cam[:, :, 0:1] / depth # X/Z
                y_norm = kp_cam[:, :, 1:2] / depth # Y/Z
                
                # 应用内参
                # 注意广播维度: fx, cx 可能需要 unsqueeze 来匹配 N (关键点数量)
                if torch.is_tensor(fx): fx_v = fx.view(-1, 1, 1)
                else: fx_v = fx
                if torch.is_tensor(fy): fy_v = fy.view(-1, 1, 1)
                else: fy_v = fy
                if torch.is_tensor(cx): cx_v = cx.view(-1, 1, 1)
                else: cx_v = cx
                if torch.is_tensor(cy): cy_v = cy.view(-1, 1, 1)
                else: cy_v = cy

                u_coords = x_norm * fx_v + cx_v
                v_coords = y_norm * fy_v + cy_v
                
                pred_keypoints_2d = torch.cat([u_coords, v_coords], dim=-1)

            else:
                # --- 路径 B: 原有默认逻辑 (估算焦距，假设中心主点) ---
                
                if img_size.dim() > 1:
                    img_size_max = img_size.max(dim=1)[0]
                else:
                    img_size_max = img_size.max()
                
                scaled_focal_length = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * img_size_max
                
                # 使用原版函数 (或上面的 custom 函数配合默认参数)
                # 这里为了保持一致性，我们用 custom 函数模拟原版行为
                # 默认主点假设在图像中心
                cx_def = img_size[:, 0] / 2.0
                cy_def = img_size[:, 1] / 2.0
                
                pred_cam_t_full = custom_cam_crop_to_full(
                    pred_cam_corrected, box_center, box_size, img_size, 
                    scaled_focal_length, scaled_focal_length, cx_def, cy_def
                )
                
                # 形状修复
                pred_cam_t_full = torch.tensor(pred_cam_t_full, dtype=torch.float32).to(self.device)
                if pred_cam_t_full.dim() == 3 and pred_cam_t_full.shape[1] == 1:
                    pred_cam_t_full = pred_cam_t_full.squeeze(1)

                # 焦距格式化
                if scaled_focal_length.dim() == 1:
                    focal_length_2d = torch.stack([scaled_focal_length, scaled_focal_length], dim=1)
                else:
                    focal_length_2d = scaled_focal_length
                    
                # 使用原来的 perspective_projection (因为它假设主点在中心，这与这里的逻辑一致)
                pred_keypoints_2d = perspective_projection(
                    pred_keypoints_3d,
                    translation=pred_cam_t_full,
                    focal_length=focal_length_2d
                )

            # =======================================================
            
            # 更新输出字典
            out['pred_keypoints_2d_full'] = pred_keypoints_2d
            out['pred_cam_t_full'] = pred_cam_t_full
            out['img'] = batch['img']
            out['focal_length'] = scaled_focal_length
            out['trans'] = trans
            out['do_flip'] = do_flip
            out['inv_trans'] = inv_trans
            
            return out, params
    
    @torch.no_grad()
    def get_image(self, dets, image, renderer):
        # 处理检测结果的正确方式
        if isinstance(dets, list) and len(dets) > 0:
            # 如果dets是嵌套列表，取第一个子列表
            if isinstance(dets[0], list) and len(dets[0]) > 0 and isinstance(dets[0][0], list):
                detection_list = dets[0]  # 提取实际的检测结果列表
            else:
                detection_list = dets  # 如果不是嵌套的，直接使用
            
            print(f"Processing {len(detection_list)} detections...")
            mesh_images =[]
            for i, bbox in enumerate(detection_list):  # 每个 bbox 是 ['right', [x1,y1,x2,y2]] 格式
                try:
                    print(f"\n--- Processing detection {i+1}/{len(detection_list)} ---")
                    print(f"Detection: {bbox}")
                    t0 = time.time()
                    # 直接传入单个 bbox
                    output, params = self.estimate_from_rgb(image, [bbox])
                    t1 = time.time()
                    # 处理输出...
                    mano_params = output['pred_mano_params']
                    global_orient = mano_params['global_orient']
                    hand_pose = mano_params['hand_pose']
                    betas = mano_params['betas']

                    pred_vertices = output['pred_vertices'][0]
                    pred_cam_t_full = output['pred_cam_t_full'][0]
                    
                    # 修复焦距处理 - 渲染器期望张量格式
                    focal_length_tensor = output['focal_length']
                    print(f"[DEBUG] focal_length_tensor shape: {focal_length_tensor.shape}")
                    print(f"[DEBUG] focal_length_tensor: {focal_length_tensor}")
                    
                    # 确保焦距是正确的张量格式供渲染器使用
                    if focal_length_tensor.dim() == 0:  # 0维张量（标量）
                        focal_length = focal_length_tensor.unsqueeze(0)  # 转换为1维张量
                    elif focal_length_tensor.dim() == 1:  # 已经是1维张量
                        focal_length = focal_length_tensor
                    else:  # 多维张量，展平并取第一个
                        focal_length = focal_length_tensor.flatten()[:1]  # 确保只有一个元素
                    
                    print(f"[DEBUG] Final focal_length for renderer shape: {focal_length.shape}")
                    print(f"[DEBUG] Final focal_length for renderer: {focal_length}")
                    t2 = time.time()
                    # 渲染网格
                    visulize_mesh = renderer.__call__(pred_vertices, pred_cam_t_full, image, focal_length, side_view=True)
                    t3 = time.time()
                    
                    print('推理时间：', t1 - t0)
                    print('mesh生成时间：', t3 - t2)
                    visulize_mesh *= 255
                    visulize_mesh = visulize_mesh.astype(np.uint8)
                    
                    # # 为每个检测结果保存不同的文件名
                    # bbox_info = f"{bbox[0]}_{int(bbox[1][0])}_{int(bbox[1][1])}"
                    # output_path = f'/home/pt/vGesture/software/hamer/test_hand_mesh_{bbox_info}.jpg'
                    # cv2.imwrite(output_path, visulize_mesh)
                    # print(f"Saved visualization for {bbox[0]} hand to {output_path}")
                    mesh_images.append(visulize_mesh)
                except Exception as e:
                    print(f"Error processing bbox {bbox}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
        else:
            print("No detections found or invalid detection format")
            mesh_images =[]

        return mesh_images
    
    def export_to_onnx(self, onnx_path: str, to_validate=True, to_simplify=True, shape_infer=True):
            """
            导出模型为 ONNX 格式
            """
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

            wrapper = HAMER_ONNX_Wrapper(self.model).to(self.device)
            wrapper.eval()

            # 临时禁用mesh_renderer，如有则移除以防图过大
            original_mesh_renderer = getattr(self.model, 'mesh_renderer', None)
            if original_mesh_renderer is not None:
                setattr(self.model, 'mesh_renderer', None)
            # 例如:
            dummy_img = torch.randn(1, 3, self.cfg.MODEL.IMAGE_SIZE, self.cfg.MODEL.IMAGE_SIZE, device=self.device)
            # dummy_box_center = torch.zeros(1, 2, dtype=torch.float32, device=self.device)  # 假设中心为0
            # dummy_box_size = torch.tensor([[200.0]], dtype=torch.float32, device=self.device) # 假设box_size
            # dummy_img_size = torch.tensor([[self.cfg.MODEL.IMAGE_SIZE, self.cfg.MODEL.IMAGE_SIZE]], dtype=torch.float32, device=self.device)
            # dummy_inv_trans = torch.zeros(1, 6, dtype=torch.float32, device=self.device)
            # dummy_do_flip = torch.zeros(1, dtype=torch.float32, device=self.device)

            # 导出 ONNX 模型
            torch.onnx.export(
                wrapper,
                (dummy_img),
                onnx_path,
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=["img"],
                output_names=HAMER_ONNX_OUTPUT_NAMES,
                # dynamic_axes={
                #     "img": {0: "batch_size", 2: "height", 3: "width"},
                #     "box_center": {0: "batch_size"},
                #     "box_size": {0: "batch_size"},
                #     "img_size": {0: "batch_size"},
                #     "inv_trans": {0: "batch_size"},
                #     "do_flip": {0: "batch_size"},
                #     "pred_keypoints_3d": {0: "batch_size"},
                #     "pred_keypoints_2d": {0: "batch_size"},
                #     "global_orient": {0: "batch_size"},
                #     "hand_pose": {0: "batch_size"},
                #     "betas": {0: "batch_size"},
                #     "pred_vertices": {0: "batch_size"},
                #     "pred_cam_t": {0: "batch_size"}
                # }
            )
            print(f"ONNX 模型已导出为 {onnx_path}")

            # 恢复 mesh_renderer
            if original_mesh_renderer is not None:
                setattr(self.model, 'mesh_renderer', original_mesh_renderer)

            if to_validate:
                check_onnx_model(onnx_path)

            # if to_simplify:
            #     print("Simplifying onnx model ...")
            #     onnx_model = onnx.load(onnx_path)
            #     # simplifying dynamic model
            #     simplified_model, is_success = simplify(onnx_model, overwrite_input_shapes={input_names[0]: [batch_size, 3, imheight, imwidth]})
            #     assert is_success, "Failed to simplify"
            #     onnx.save(simplified_model, onnx_path)
            #     check_onnx_model(onnx_path)

            # if shape_infer:
            #     print("Using shape inference ...")
            #     from onnx import shape_inference
            #     onnx_model = onnx.load(onnx_path)
            #     inferred_model = shape_inference.infer_shapes(onnx_model)
            #     onnx.save(inferred_model, onnx_path)
            #     check_onnx_model(onnx_path)

    @staticmethod
    def compare_pytorch_onnx(hamer, onnx_path):
        # 准备与导出时相同的虚拟输入
        dummy_img = torch.randn(1, 3, hamer.cfg.MODEL.IMAGE_SIZE, hamer.cfg.MODEL.IMAGE_SIZE, device=hamer.device)
        # dummy_box_center = torch.zeros(1, 2, dtype=torch.float32, device=hamer.device)
        # dummy_box_size = torch.tensor([[300.0]], dtype=torch.float32, device=hamer.device)
        # dummy_img_size = torch.tensor([[hamer.cfg.MODEL.IMAGE_SIZE, hamer.cfg.MODEL.IMAGE_SIZE]], dtype=torch.float32, device=hamer.device)
        # dummy_inv_trans = torch.zeros(1, 6, dtype=torch.float32, device=hamer.device)
        # dummy_do_flip = torch.zeros(1, dtype=torch.float32, device=hamer.device)

        # 创建包装器
        wrapper = HAMER_ONNX_Wrapper(hamer.model).to(hamer.device)
        wrapper.eval()

        # PyTorch 模型推理
        with torch.no_grad():
            pt_outputs = wrapper(dummy_img)

        # ONNX Runtime 推理
        ort_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])

         # 由于ONNX模型的输入名称为空，使用位置参数列表而不是命名字典
        ort_inputs = [
            dummy_img.cpu().numpy(),
            # dummy_box_center.cpu().numpy(),
            # dummy_box_size.cpu().numpy(),
            # dummy_img_size.cpu().numpy(),
            # dummy_inv_trans.cpu().numpy(),
            # dummy_do_flip.cpu().numpy()
        ]
        
        # 获取输入名称（如果有的话）get_image
        input_names = [input_info.name for input_info in ort_session.get_inputs()]
        if all(name for name in input_names):  # 如果所有输入都有名称
            ort_inputs_dict = {
                input_names[0]: ort_inputs[0],  # img
                # input_names[1]: ort_inputs[1],  # box_center
                # input_names[2]: ort_inputs[2],  # box_size
                # input_names[3]: ort_inputs[3],  # img_size
                # input_names[4]: ort_inputs[4],  # inv_trans
                # input_names[5]: ort_inputs[5]   # do_flip
            }
            ort_outs = ort_session.run(None, ort_inputs_dict)
        else:  # 使用位置参数
            # 创建以输入索引为键的字典
            ort_inputs_dict = {ort_session.get_inputs()[i].name if ort_session.get_inputs()[i].name else str(i): ort_inputs[i] for i in range(len(ort_inputs))}
            ort_outs = ort_session.run(None, ort_inputs_dict)


        # 将 ONNX 输出与 PyTorch 输出对比
        # pt_outputs 和 ort_outs 都是元组，对应相同顺序的输出
        for i, name in enumerate(HAMER_ONNX_OUTPUT_NAMES):
            if not isinstance(pt_outputs[i], torch.Tensor):
                print(f"跳过对 {name} 的差异比较")
            pt_tensor = pt_outputs[i].cpu().numpy()
            onnx_tensor = ort_outs[i]
            if np.allclose(pt_tensor, onnx_tensor, atol=1e-3):
                print(f"输出 {name} 匹配。")
            else:
                print(f"输出 {name} 不匹配！")
                # 如果需要可打印最大差异以调试
                diff = np.abs(pt_tensor - onnx_tensor)
                print(f"最大差异：{diff.max()}")


def image_fusion(ori, mesh_images):
    print('开始mesh与原图融合')
    print('ori，mesh_images形状：', ori.shape, mesh_images[0].shape)
    t0 = time.time()
    result = ori.copy()
    
    for overlay_img in mesh_images:
        mask = np.any(overlay_img[:, :, :3] > 0, axis=-1)
        overlay_rgb = overlay_img
        # 应用替换：非零区域用 overlay_img 替换，其他区域保持原图
        result = np.where(mask[:, :, np.newaxis], overlay_rgb, result)
    t1 = time.time()
    print(f'完成mesh与原图融合,用时{t1 -t0}')
    cv2.imwrite('/home/pt/vGesture/software/hamer/test_image/image_fusion.jpg', result)
    return result  # 可选：转为RGB去除Alpha通道


@profile
def main():
    hamer = hamer_inference(hamer_opt)
    renderer = hamer.get_mesh_renderer()
    
    img_path = '/home/pt/fbs/FoundationPoseROS2/FoundationPose/demo_data/ours/rgb/000000.jpg'
    image = cv2.imread(img_path)
    
    detector = Detector(yolo_opt)
    _, dets = detector.detect(image)
    print("Detections:", dets)
    
    # 处理检测结果的正确方式
    if isinstance(dets, list) and len(dets) > 0:
        # 如果dets是嵌套列表，取第一个子列表
        if isinstance(dets[0], list) and len(dets[0]) > 0 and isinstance(dets[0][0], list):
            detection_list = dets[0]  # 提取实际的检测结果列表
        else:
            detection_list = dets  # 如果不是嵌套的，直接使用
        
        print(f"Processing {len(detection_list)} detections...")
        t_start = time.time()
        mesh_images =[]
        for i, bbox in enumerate(detection_list):  # 每个 bbox 是 ['right', [x1,y1,x2,y2]] 格式
            try:
                print(f"\n--- Processing detection {i+1}/{len(detection_list)} ---")
                print(f"Detection: {bbox}")
                t0 = time.time()
                # 直接传入单个 bbox
                output, params = hamer.estimate_from_rgb(image, [bbox])
                t1 = time.time()
                # 处理输出...
                mano_params = output['pred_mano_params']
                global_orient = mano_params['global_orient']
                hand_pose = mano_params['hand_pose']
                betas = mano_params['betas']

                pred_vertices = output['pred_vertices'][0]
                pred_cam_t_full = output['pred_cam_t_full'][0]
                
                # 修复焦距处理 - 渲染器期望张量格式
                focal_length_tensor = output['focal_length']
                print(f"[DEBUG] focal_length_tensor shape: {focal_length_tensor.shape}")
                print(f"[DEBUG] focal_length_tensor: {focal_length_tensor}")
                
                # 确保焦距是正确的张量格式供渲染器使用
                if focal_length_tensor.dim() == 0:  # 0维张量（标量）
                    focal_length = focal_length_tensor.unsqueeze(0)  # 转换为1维张量
                elif focal_length_tensor.dim() == 1:  # 已经是1维张量
                    focal_length = focal_length_tensor
                else:  # 多维张量，展平并取第一个
                    focal_length = focal_length_tensor.flatten()[:1]  # 确保只有一个元素
                
                is_right = (bbox[0] == 'right')

                vertices_np = pred_vertices.detach().cpu().numpy()
                if not is_right:
                    # 如果是左手，模型输出的是翻转后的右手，我们需要把 X 轴取反还原回左手
                    vertices_np[:, 0] = -vertices_np[:, 0]

                hand_mesh = renderer.get_mesh(vertices_np)
                if is_right:
                    hand_mesh.export(f'/home/pt/fbs/test/handmesh_right.obj')
                else :
                    renderer.faces = renderer.faces[:, [0, 2, 1]]
                    hand_mesh.export(f'/home/pt/fbs/test/handmesh_left.obj')
                # print(f"[DEBUG] Final focal_length for renderer shape: {focal_length.shape}")
                # print(f"[DEBUG] Final focal_length for renderer: {focal_length}")
                # t2 = time.time()
                # # 渲染网格
                # visulize_mesh = renderer.__call__(pred_vertices, pred_cam_t_full, image, focal_length, side_view=True)
                # t3 = time.time()
                
                # print('推理时间：', t1 - t0)
                # print('mesh生成时间：', t3 - t2)
                # visulize_mesh *= 255
                # visulize_mesh = visulize_mesh.astype(np.uint8)
                
                # # # 为每个检测结果保存不同的文件名
                # # bbox_info = f"{bbox[0]}_{int(bbox[1][0])}_{int(bbox[1][1])}"
                # # output_path = f'/home/pt/vGesture/software/hamer/test_hand_mesh_{bbox_info}.jpg'
                # # cv2.imwrite(output_path, visulize_mesh)
                # # print(f"Saved visualization for {bbox[0]} hand to {output_path}")
                # mesh_images.append(visulize_mesh)

                # 创建点云对象
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(vertices_np)

                # 保存
                file_name = os.path.splitext(os.path.basename(img_path))[0]
                o3d.io.write_point_cloud(f'/home/pt/fbs/test/{file_name}_{str(bbox[0])}.ply', pcd)
            except Exception as e:
                print(f"Error processing bbox {bbox}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    else:
        print("No detections found or invalid detection format")
    # image_fusion(image, mesh_images)
    t_end = time.time()    
    print(f"Processing completed!总用时{t_end - t_start}")

    # hamer = hamer_inference(ckpt_path, model_cfg)

    # renderer = hamer.model.mesh_renderer
    # print('mode and render get')
    # #### TEST SINGLE IMAGE

    # img_path = '/home/pt/vGesture/software/hamer/example_data/test1.jpg'

    # image = cv2.imread(img_path)
    # detector = Detector(yolo_opt)
    # _, dets = detector.detect(image)
    # print("dets:", dets)
    # for bbox in dets:
    #     output, pramas = hamer.estimate_from_rgb(image, bbox)
    #     print("Output keys:", output.keys())
    #     for key, value in output.items():
    #         if isinstance(value, torch.Tensor):
    #             print(f"{key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
    #         else:
    #             print(f"{key}: {type(value)}")
    #     # 在hamer——infercenc中global_orient hand_pose betas被cat为pose
    #     # global_orient = output['global_orient'][0] # 手腕旋转矩阵 1×3×3 
    #     # hand_pose = output['hand_pose'][0] # 手部姿势旋转矩阵 15×3×3
    #     mano_params = output['pred_mano_params']
    #     global_orient = mano_params['global_orient']  # 手腕旋转矩阵
    #     print('global_orient shape:', global_orient.shape)
    #     hand_pose = mano_params['hand_pose']  # 手部局部旋转
    #     print('hand_pose shape:', hand_pose.shape)
    #     betas = mano_params['betas']  # 手部形状参数 10
    #     print('betas shape:', betas.shape)

    #     pred_vertices = output['pred_vertices'][0]
    #     pred_cam_t_full = output['pred_cam_t_full'][0]
    #     focal_length = output['focal_length'][0]
    #     visulize_mesh = renderer.__call__(pred_vertices, pred_cam_t_full, image, focal_length,side_view=True)
        
    #     visulize_mesh *= 255
    #     visulize_mesh = visulize_mesh.astype(np.uint8)
    #     cv2.imwrite('/home/cyc/pycharm/vGesture/software/hamer/test_hand_mesh.jpg', visulize_mesh)

    # # pass


def main_onnx():
    hamer = hamer_inference(ckpt_path, model_cfg, use_onnx=False)
    onnx_path = HAMER_ONNX_CKPT_PATH
    hamer.export_to_onnx(onnx_path)
    hamer.compare_pytorch_onnx(hamer, onnx_path)

def process_batch(input_folder, output_folder, k_real=None):
    # 0. 准备输出目录
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"输出目录已创建: {output_folder}")

    # 1. 模型初始化 (只执行一次)
    print("正在加载模型...")
    hamer = hamer_inference(hamer_opt)
    renderer = hamer.get_mesh_renderer()
    detector = Detector(yolo_opt)
    print("模型加载完成。")

    # 2. 获取图片列表
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
        image_paths.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    image_paths = sorted(list(set(image_paths)))
    print(f"共发现 {len(image_paths)} 张图片，开始处理...")

    # 3. 批量处理循环
    for img_path in tqdm(image_paths, desc="Batch Processing"):
        # 获取不带后缀的文件名 (例如: /path/0000.jpg -> 0000)
        file_name_with_ext = os.path.basename(img_path)
        file_name = os.path.splitext(file_name_with_ext)[0]
        
        try:
            image = cv2.imread(img_path)
            if image is None:
                continue

            # 检测手部
            _, dets = detector.detect(image)
            
            # 解析检测结果
            detection_list = []
            if isinstance(dets, list) and len(dets) > 0:
                if isinstance(dets[0], list) and len(dets[0]) > 0 and isinstance(dets[0][0], list):
                    detection_list = dets[0]
                else:
                    detection_list = dets
            
            if not detection_list:
                continue

            # 记录当前图片已保存的手，防止一张图里有两个右手导致文件名覆盖
            saved_counts = {'left': 0, 'right': 0}

            # 遍历检测到的手
            for bbox in detection_list:
                try:
                    # bbox 格式通常为 ['right', [x1, y1, x2, y2]]
                    hand_label = bbox[0] # 'right' or 'left'
                    # is_right = (hand_label == 'right')

                    # 推理
                    output, params = hamer.estimate_from_rgb(image, [bbox], k_real)
                    is_right = (output['do_flip'] == 0)
                    
                    
                    # --- 获取 MANO 参数 (Beta & Theta) ---
                    mano_params = output['pred_mano_params']
                    
                    # .detach().cpu().numpy() 将 Tensor 转为 numpy 数组以便保存
                    # betas: 形状参数 (1, 10)
                    betas_np = mano_params['betas'].detach().cpu().numpy()
                    # global_orient: 全局旋转 (1, 3, 3) 或者是轴角 (1, 3)，视具体模型配置而定
                    global_orient_np = mano_params['global_orient'].detach().cpu().numpy()
                    # hand_pose: 手指姿态 (1, 15, 3, 3) 或 (1, 45)
                    hand_pose_np = mano_params['hand_pose'].detach().cpu().numpy()
                    # 相机平移参数 (用于将手放到相机坐标系下)
                    cam_t_np = output['pred_cam_t_full'].detach().cpu().numpy()

                    # 获取顶点用于保存 obj
                    # pred_vertices = output['pred_vertices'][0]
                    # vertices_np = pred_vertices.detach().cpu().numpy()

                    # --- 左手特殊处理 ---
                    # 1. 顶点翻转：Hamer输出是右手系，如果是左手需翻转X轴还原
                    # if not is_right:
                    #     vertices_np[:, 0] = -vertices_np[:, 0]

                    # 2. 生成 Mesh 对象
                    # hand_mesh = renderer.get_mesh(vertices_np)

                    # 3. 面片法向翻转
                    # if not is_right:
                    #     hand_mesh.faces = hand_mesh.faces[:, [0, 2, 1]]

                    # --- 保存逻辑 ---
                    # 构建文件名后缀
                    suffix = f"_{hand_label}"
                    if saved_counts[hand_label] > 0:
                        suffix += f"_{saved_counts[hand_label] + 1}"
                    
                    # 1. 保存 OBJ (Mesh)
                    # obj_name = f"{file_name}{suffix}.obj"
                    # obj_path = os.path.join(output_folder, obj_name)
                    # hand_mesh.export(obj_path)
                    
                    # 2. 保存 MANO 参数 (NPZ) - 这里保存了 betas 和 pose (theta)
                    npz_name = f"{file_name}{suffix}.npz"
                    npz_path = os.path.join(output_folder, npz_name)
                    
                    np.savez(npz_path, 
                             betas=betas_np,             # 形状参数 Beta
                             global_orient=global_orient_np, # 姿态参数 Theta 的一部分 (根旋转)
                             hand_pose=hand_pose_np,     # 姿态参数 Theta 的一部分 (手指关节)
                             cam_t=cam_t_np,             # 相机位移
                             is_right=is_right           # 左右手标记
                    )
                    
                    # 计数增加
                    saved_counts[hand_label] += 1
                    
                except Exception as e_inner:
                    print(f"Error processing hand in {file_name}: {e_inner}")
                    import traceback
                    traceback.print_exc()
                    continue

        except Exception as e:
            print(f"Error processing file {img_path}: {e}")
            continue

    print("全部处理完成。")

# === 新增/修改的辅助函数 ===

def get_bbox_from_npy(npy_path, target_val=3):
    """
    读取 npy mask，计算指定值的最小外接矩形 (Bounding Box)
    Args:
        npy_path: .npy 文件的路径
        target_val: 目标像素值 (这里是 3)
    Returns:
        bbox: [x1, y1, x2, y2] 格式，如果没有找到目标值则返回 None
    """
    if not os.path.exists(npy_path):
        print(f"[Warning] Mask文件不存在: {npy_path}")
        return None

    # 加载 mask (H, W)
    mask = np.load(npy_path)

    # 找到所有值为 3 的像素索引
    # rows 对应 y (行), cols 对应 x (列)
    rows, cols = np.where(mask == target_val)

    # 如果没有找到值为 3 的像素
    if len(rows) == 0:
        return None

    # 计算最左、最右、最上、最下
    y1 = np.min(rows)
    y2 = np.max(rows)
    x1 = np.min(cols)
    x2 = np.max(cols)

    # 返回 [x1, y1, x2, y2]
    # 转换为 float 以兼容后续 Hamer 的处理逻辑
    return [float(x1), float(y1), float(x2), float(y2)]

def flip_axis_angle(rvec):
    """
    对轴角向量进行镜像翻转 (对应图像水平翻转)
    变换逻辑: [rx, ry, rz] -> [rx, -ry, -rz]
    """
    # return np.array([rvec[0], -rvec[1], -rvec[2]], dtype=rvec.dtype)
    return np.array([rvec[0], rvec[1], rvec[2]], dtype=rvec.dtype)

def matrix_to_axis_angle(rot_mats):
    """
    将旋转矩阵批量转换为轴角向量
    Input: (N, 3, 3) or (3, 3)
    Output: (N*3,) flattened
    """
    # 确保输入是列表或数组
    if isinstance(rot_mats, np.ndarray) and rot_mats.ndim == 2:
        rot_mats = [rot_mats]
    
    flattened_pose = []
    for mat in rot_mats:
        rvec, _ = cv2.Rodrigues(mat)
        flattened_pose.append(rvec.flatten())
    return np.concatenate(flattened_pose)

# === 核心处理函数的修改 ===
def process_batch_manopara_with_mask(input_folder, mask_folder, output_folder, intrinsics_path=None):
    # 0. 准备输出目录
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"输出目录已创建: {output_folder}")

    # --- 内参预处理逻辑 ---
    fixed_k = None
    intrinsics_dir = None
    
    if intrinsics_path:
        if os.path.isfile(intrinsics_path):
            print(f"[Info] 使用固定内参文件: {intrinsics_path}")
            fixed_k = load_intrinsics(intrinsics_path) # 需确保 load_intrinsics 已定义
        elif os.path.isdir(intrinsics_path):
            print(f"[Info] 使用内参文件夹: {intrinsics_path}")
            intrinsics_dir = intrinsics_path
        else:
            print(f"[Warning] 内参路径无效: {intrinsics_path}")

    # 1. 模型初始化 (只加载 Hamer，不需要 YOLO)
    print("正在加载 Hamer 模型...")
    hamer = hamer_inference(hamer_opt) # 需确保 hamer_inference 和 hamer_opt 可用
    print("模型加载完成。")

    # 2. 获取图片列表
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
        image_paths.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    image_paths = sorted(list(set(image_paths)))
    print(f"共发现 {len(image_paths)} 张图片，开始处理...")

    # 3. 批量处理循环
    for img_path in tqdm(image_paths, desc="Batch Processing (Mask)"):
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        image_results = {'left': None, 'right': None}

        # --- A. 获取 bbox ---
        npy_mask_path = os.path.join(mask_folder, f"{file_name}.npy")
        bbox_coords = get_bbox_from_npy(npy_mask_path, target_val=3)

        # 如果没有对应的 mask 或 mask 中没有值 3，跳过此图
        if bbox_coords is None:
            continue

        # 构造 detection_list: 只包含一个右手
        # 格式: [['right', [x1, y1, x2, y2]]]
        detection_list = [['right', bbox_coords]]

        # --- B. 获取当前帧内参 ---
        current_k_real = None
        if fixed_k is not None:
            current_k_real = fixed_k
        elif intrinsics_dir is not None:
            txt_path = os.path.join(intrinsics_dir, f"{file_name}.txt")
            if os.path.exists(txt_path):
                current_k_real = load_intrinsics(txt_path)

        try:
            image = cv2.imread(img_path)
            if image is None: continue

            # 遍历 detection_list (实际上现在只有一个)
            for bbox in detection_list:
                try:
                    hand_label = bbox[0] 
                    # 再次确认只处理右手 (双重保险)
                    if hand_label != 'right': continue
                    
                    is_right = True

                    # 推理
                    output, params = hamer.estimate_from_rgb(image, [bbox], current_k_real)
                    
                    # --- 提取参数 ---
                    mano_params = output['pred_mano_params']
                    
                    # Betas
                    betas_np = mano_params['betas'].detach().cpu().numpy().squeeze()
                    
                    # Hand Pose (转换为 Axis Angle)
                    hand_pose_mats = mano_params['hand_pose'].detach().cpu().numpy().squeeze()
                    hand_pose_aa = matrix_to_axis_angle(hand_pose_mats)
                    
                    # Global Orient (转换为 Axis Angle)
                    global_orient_mat = mano_params['global_orient'].detach().cpu().numpy().squeeze()
                    if global_orient_mat.ndim == 3: global_orient_mat = global_orient_mat[0]
                    global_orient_aa, _ = cv2.Rodrigues(global_orient_mat)
                    global_orient_aa = global_orient_aa.flatten()

                    # Camera Translation
                    cam_t_np = output['pred_cam_t_full'].detach().cpu().numpy().squeeze()
                    
                    # 拼接 Theta (Global + Hand Pose)
                    theta_np = np.concatenate((global_orient_aa, hand_pose_aa))

                    hand_data = {
                        'betas': betas_np,           
                        'theta': theta_np,           
                        'pose_hand': hand_pose_aa,   
                        'pose_global': global_orient_aa,
                        'cam_t': cam_t_np,
                        'is_right': is_right
                    }
                    image_results[hand_label] = hand_data
                    
                except Exception as e_inner:
                    print(f"Error processing hand in {file_name}: {e_inner}")
                    continue

            # 保存 .npy
            save_path = os.path.join(output_folder, f"{file_name}.npy")
            np.save(save_path, image_results)

        except Exception as e:
            print(f"Error processing file {img_path}: {e}")
            continue

    print("基于 Mask 的参数提取完成。")


def process_batch_manopara(input_folder, output_folder, k_real=None):
    # 0. 准备输出目录
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"输出目录已创建: {output_folder}")

    # 1. 模型初始化
    print("正在加载模型...")
    hamer = hamer_inference(hamer_opt)
    detector = Detector(yolo_opt)
    sar = get_model()
    print("模型加载完成。")

    # 2. 获取图片列表
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
        image_paths.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    image_paths = sorted(list(set(image_paths)))
    print(f"共发现 {len(image_paths)} 张图片，开始处理...")

    # 3. 批量处理循环
    for img_path in tqdm(image_paths, desc="Batch Processing"):
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        image_results = {'left': None, 'right': None}

        try:
            image = cv2.imread(img_path)
            if image is None: continue

            _, dets = detector.detect(image)
            
            detection_list = []
            if isinstance(dets, list) and len(dets) > 0:
                if isinstance(dets[0], list) and len(dets[0]) > 0 and isinstance(dets[0][0], list):
                    detection_list = dets[0]
                else:
                    detection_list = dets
            
            if not detection_list: continue

            saved_counts = {'left': 0, 'right': 0}

            for bbox in detection_list:
                try:
                    hand_label = bbox[0] 
                    is_right = (hand_label == 'right')

                    # 推理
                    depth_pred = sar.estimate_root_depth_custom(image, k_real, bbox[1])
                    output, params = hamer.estimate_from_rgb(image, [bbox], k_real, depth_refine=depth_pred)

                    print('depth_pred:', depth_pred)
                    
                    mano_params = output['pred_mano_params']
                    betas_np = mano_params['betas'].detach().cpu().numpy().squeeze()
                    
                    hand_pose_mats = mano_params['hand_pose'].detach().cpu().numpy().squeeze()
                    hand_pose_aa = matrix_to_axis_angle(hand_pose_mats)
                    
                    global_orient_mat = mano_params['global_orient'].detach().cpu().numpy().squeeze()
                    if global_orient_mat.ndim == 3: global_orient_mat = global_orient_mat[0]
                    global_orient_aa, _ = cv2.Rodrigues(global_orient_mat)
                    global_orient_aa = global_orient_aa.flatten()

                    cam_t_np = output['pred_cam_t_full'].detach().cpu().numpy().squeeze()
                    print('原始 cam_t_np:', cam_t_np)

                    # # 获取内参
                    # fx, fy = k_real[0, 0], k_real[1, 1]
                    # cx, cy = k_real[0, 2], k_real[1, 2]

                    # # 获取原始数据
                    # tz_old = cam_t_np[2]
                    # tz_new = float(depth_pred)

                    # # 计算手部在全图上的投影位置 (u, v)
                    # # 这一步是逆推：u = (tx * fx / tz) + cx
                    # # 注意：这里要确保 tx 包含了 custom_cam_crop_to_full 里的所有项
                    # u = (cam_t_np[0] * fx / tz_old) + cx
                    # v = (cam_t_np[1] * fy / tz_old) + cy

                    # # 根据新深度重新计算 Tx, Ty
                    # # 这样保证了 (u, v) 像素坐标绝对不变
                    # tx_new = (u - cx) * tz_new / fx
                    # ty_new = (v - cy) * tz_new / fy

                    # refined_cam_t = np.array([tx_new, ty_new, tz_new])
                    
                    # # === 重点：这里保存原始 RAW 数据，不进行任何左手翻转处理 ===
                    
                    
                    theta_np = np.concatenate((global_orient_aa, hand_pose_aa))

                    hand_data = {
                        'betas': betas_np,           
                        'theta': theta_np,           
                        'pose_hand': hand_pose_aa,   
                        'pose_global': global_orient_aa,
                        'cam_t': cam_t_np,
                        'is_right': is_right
                    }
                    image_results[hand_label] = hand_data
                    
                except Exception as e:
                    print(f"Error processing hand: {e}")
                    continue

            # 保存 .npy
            npy_path = os.path.join(output_folder, f"{file_name}.npy")
            np.save(npy_path, image_results)

        except Exception as e:
            print(f"Error processing file {img_path}: {e}")
            continue

    print("参数提取完成。")


def reconstruct_and_save_obj_with_wrapper(npy_folder, output_obj_folder, hamer_instance):
    """
    修复版 V2：
    1. 确保 betss 维度为 [1, 10]
    2. [关键修复] 确保 global_orient 和 hand_pose 都是 2维 Flattened 状态 ([1, 9] 和 [1, 135])
    3. 显式将模型移动到 GPU
    """
    device = hamer_instance.device
    
    if not os.path.exists(output_obj_folder):
        os.makedirs(output_obj_folder)

    # 1. 初始化 MANO 并移动到 GPU
    if not hasattr(hamer_instance, 'mano') or hamer_instance.mano is None:
        print("初始化 MANO 模型...")
        hamer_instance.get_mesh_renderer()
    
    mano_model = hamer_instance.mano
    mano_model.to(device)
    mano_model.eval()

    # 2. 读取文件
    npy_files = sorted(glob.glob(os.path.join(npy_folder, '*.npy')))
    print(f"开始重建，共 {len(npy_files)} 个文件...")

    for npy_path in tqdm(npy_files, desc="Reconstructing"):
        file_name = os.path.splitext(os.path.basename(npy_path))[0]
        
        try:
            data = np.load(npy_path, allow_pickle=True).item()
            scene_meshes = []

            for hand_type in ['right', 'left']:
                hand_data = data[hand_type]
                if hand_data is None: continue
                
                # --- A. 准备数据 & 维度检查 ---
                
                # 1. Betas: 确保是 [1, 10]
                betas_np = hand_data['betas']
                if betas_np.ndim == 1:
                    betas_np = betas_np[None, :]
                betas = torch.tensor(betas_np, dtype=torch.float32).to(device)
                
                # 2. Global Orient (AA): [1, 3]
                global_orient_aa_np = hand_data['pose_global']
                if global_orient_aa_np.ndim == 1:
                    global_orient_aa_np = global_orient_aa_np[None, :]
                global_orient_aa = torch.tensor(global_orient_aa_np, dtype=torch.float32).to(device)
                
                # 3. Hand Pose (AA): [15, 3]
                hand_pose_aa_np = hand_data['pose_hand'].reshape(-1, 3)
                hand_pose_aa = torch.tensor(hand_pose_aa_np, dtype=torch.float32).to(device)
                
                cam_t = hand_data['cam_t'] 
                is_right = hand_data['is_right']

                # --- B. 数据转换 (AA -> Flattened RotMat) ---
                
                # Global Orient: 1个关节 -> [1, 1, 3, 3]
                global_orient_mat = axis_angle_to_rotation_matrix_torch(global_orient_aa).view(1, 1, 3, 3)

                # Hand Pose: 15个关节 -> [1, 15, 3, 3]
                hand_pose_mat = axis_angle_to_rotation_matrix_torch(hand_pose_aa).view(1, 15, 3, 3)

                # --- C. 模型前向传播 ---
                output = mano_model(
                    betas=betas,
                    global_orient=global_orient_mat,
                    hand_pose=hand_pose_mat,
                    pose2rot=False # 明确告知我们传的是矩阵
                )
                
                vertices = output.vertices[0].detach().cpu().numpy() # [778, 3]
                
                # 获取 faces
                if hasattr(mano_model, 'faces'):
                    faces = mano_model.faces.astype(np.int32)
                else:
                    faces = mano_model.faces_tensor.cpu().numpy().astype(np.int32)

                # --- D. 几何还原 (左手处理) ---
                if is_right:
                    # 右手：直接加平移
                    vertices += cam_t
                    vertex_color = [100, 200, 100, 255] # 绿
                else:
                    # 左手：
                    # 1. 顶点 X 取反 (几何镜像)
                    vertices[:, 0] = -vertices[:, 0]
                    # 2. 面片法向翻转
                    faces = faces[:, [0, 2, 1]]
                    # 3. 平移修正 (Tx 取反)
                    real_cam_t = cam_t.copy()
                    real_cam_t[0] = real_cam_t[0]
                    vertices += real_cam_t
                    vertex_color = [200, 100, 100, 255] # 红

                # --- E. 创建 Mesh ---
                mesh = trimesh.Trimesh(vertices, faces, process=False)
                mesh.visual.vertex_colors = vertex_color
                scene_meshes.append(mesh)

            # 保存
            if len(scene_meshes) > 0:
                combined_mesh = trimesh.util.concatenate(scene_meshes)
                save_path = os.path.join(output_obj_folder, f"{file_name}.obj")
                combined_mesh.export(save_path)

        except Exception as e:
            print(f"Error reconstructing {file_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("完成。")


class TorchReprModifier:
    """from https://stackoverflow.com/questions/70704619/is-it-possible-to-show-variable-shapes-and-lengths-in-vscode-python-debugger-jus"""

    def __init__(self):
        self.original_torch_repr = torch.Tensor.__repr__

    def enable_custom_repr(self):
        # 定义自定义的PyTorch张量表示方法
        def custom_torch_repr(tensor):
            # return f'Tensor.shape:{tuple(tensor.shape)} {self.original_torch_repr(tensor)}'
            return f"{tuple(tensor.shape)} {tensor.device} {tensor.dtype} {self.original_torch_repr(tensor)}"

        torch.Tensor.__repr__ = custom_torch_repr


    def restore_original_repr(self):
        torch.Tensor.__repr__ = self.original_torch_repr


def load_intrinsics(txt_path):
    """
    读取 txt 文件并返回 3x3 的 numpy 矩阵 (float32)
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"文件未找到: {txt_path}")
    
    try:
        # np.loadtxt 会自动处理空格分隔和科学计数法 (e+02)
        k_real = np.loadtxt(txt_path, dtype=np.float32)
        
        # 确保形状是 3x3
        if k_real.shape != (3, 3):
            raise ValueError(f"矩阵形状错误，期望 (3, 3)，实际为 {k_real.shape}")
            
        return k_real
        
    except Exception as e:
        print(f"读取内参文件失败: {e}")
        return None

if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--vsdebug', action='store_true')
    # args = parser.parse_args()
    # if args.vsdebug:
    #     import debugpy
    #     debugpy.connect(("localhost", 5678))
    # TorchReprModifier().enable_custom_repr()
    
    # main()

    # main_onnx()
    # kernprof -lv infer.py
    # lp.print_stats()

    parser = argparse.ArgumentParser(description='Hamer Batch Processing')
    
    # 设置默认路径，你可以修改这里的 default 为你常用的路径
    default_input = '/home/pt/fbs/FoundationPoseROS2/FoundationPose/demo_data/ours/rgb'
    npy_folder = '/home/pt/fbs/test/manopara'
    default_output_obj = '/home/pt/fbs/test/manopara_obj'

    parser.add_argument('--input', type=str, default=default_input, help='Input folder containing images')
    parser.add_argument('--output', type=str, default=npy_folder, help='Output folder for results')
    
    args = parser.parse_args()
    
    txt_file = "/home/pt/fbs/FoundationPoseROS2/FoundationPose/demo_data/ours/cam_K.txt" # 替换为你的文件路径

    k_real = load_intrinsics(txt_file)

    # process_batch(default_input, npy_folder, k_real=k_real)


    process_batch_manopara(args.input, args.output, k_real=k_real)

    # process_batch_manopara_with_mask(args.input, '/home/pt/fbs/test/ding/ding_mask', args.output, intrinsics_path="/home/pt/fbs/test/ding/cam_K.txt")
    # 3. 初始化 Hamer 实例 (为了获取 MANO 模型)
    print("初始化 Hamer 用于 OBJ 重建...")
    hamer = hamer_inference(hamer_opt)
    renderer = hamer.get_mesh_renderer()
    
    # 2. 定义 MANO 模型路径 (请修改为你实际的路径)
    # 通常在 hamer/_DATA/data/mano/ 下
    path_to_mano_right = "/home/pt/fbs/MANO/MANO_RIGHT.pkl"
    # path_to_mano_left = "/home/pt/fbs/model/hamer/_DATA/data/mano/MANO_LEFT.pkl"
    
    # 检查路径是否存在
    if not os.path.exists(path_to_mano_right):
        print("错误：找不到 MANO .pkl 文件，请检查路径。")
    else:
        # 3. 使用 Hamer 内置的 MANO Wrapper 重建 Mesh
        reconstruct_and_save_obj_with_wrapper(
            npy_folder, 
            default_output_obj, 
            hamer
        )