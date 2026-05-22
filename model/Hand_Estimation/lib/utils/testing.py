import os
import pickle

import numpy as np
import torch
from manotorch.manolayer import ManoLayer

from lib.metrics.pck import Joint3DPCK, Vert3DPCK
from lib.utils.transform import (batch_cam_extr_transf, batch_cam_intr_projection, denormalize)
from lib.viztools.draw import save_a_image_with_mesh_joints
from lib.viztools.opendr_renderer import OpenDRRenderer

from .logger import logger


class IdleCallback():

    def __init__(self):
        pass

    def __call__(self, preds, inputs, step_idx, **kwargs):
        pass

    def on_finished(self):
        pass

    def reset(self):
        pass


class AUCCallback(IdleCallback):

    def __init__(self, exp_dir, val_min=0.0, val_max=0.02, steps=20):
        self.exp_dir = exp_dir
        self.val_min = val_min
        self.val_max = val_max
        self.steps = steps
        self.PCK_J = Joint3DPCK(EVAL_TYPE="joints_3d_rel", VAL_MIN=val_min, VAL_MAX=val_max, STEPS=steps)
        self.PCK_V = Vert3DPCK(EVAL_TYPE="verts_3d_rel", VAL_MIN=val_min, VAL_MAX=val_max, STEPS=steps)
        self.f_score, self.f_score_aligned = list(), list()
        self.f_threshs = [0.005, 0.015]
        self.n_eval_samples = 0

    def reset(self):
        self.PCK_J.reset()
        self.PCK_V.reset()

    def batch_align_w_scale(self, gt, pred):
        """批量对齐预测点云 (PyTorch实现)"""
        # 中心化
        t1 = gt.mean(dim=1, keepdim=True)
        t2 = pred.mean(dim=1, keepdim=True)
        gt_centered = gt - t1
        pred_centered = pred - t2
        
        # 缩放归一化
        scale_gt = torch.norm(gt_centered, dim=(1, 2), keepdim=True) + 1e-8
        scale_pred = torch.norm(pred_centered, dim=(1, 2), keepdim=True) + 1e-8
        gt_normalized = gt_centered / scale_gt
        pred_normalized = pred_centered / scale_pred
        
        # 批量Procrustes分析
        A = torch.bmm(gt_normalized.transpose(1, 2), pred_normalized)
        U, _, Vt = torch.linalg.svd(A)
        R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))
        S = torch.eye(3, device=gt.device).unsqueeze(0).repeat(R.size(0), 1, 1)
        S[:, 2, 2] = torch.det(R)
        R = torch.bmm(Vt.transpose(1, 2), torch.bmm(S, U.transpose(1, 2)))
        
        # 应用变换
        pred_aligned = torch.bmm(pred_normalized, R) * scale_gt + t1
        return pred_aligned

    def batch_calculate_fscore(self, gt, pred, pred_aligned, thresholds):
        """批量计算F-score (PyTorch实现)"""
        # 计算成对距离矩阵
        dist_matrix = torch.cdist(gt, pred)  # (B, V, V)
        
        # 计算最小距离
        min_gt_to_pred = dist_matrix.min(dim=2).values  # (B, V)
        min_pred_to_gt = dist_matrix.min(dim=1).values  # (B, V)
        
        # 对齐版本的距离计算
        dist_matrix_aligned = torch.cdist(gt, pred_aligned)
        min_gt_to_aligned = dist_matrix_aligned.min(dim=2).values
        min_pred_to_aligned = dist_matrix_aligned.min(dim=1).values
        
        # 初始化结果张量
        n_samples = gt.size(0)
        n_thresholds = len(thresholds)
        f_unaligned = torch.zeros((n_samples, n_thresholds), device=gt.device)
        f_aligned = torch.zeros((n_samples, n_thresholds), device=gt.device)
        
        # 逐阈值计算
        thresholds_tensor = torch.tensor(thresholds, device=gt.device).view(1, 1, -1)
        
        # 未对齐版本
        recall_unaligned = (min_pred_to_gt.unsqueeze(2) < thresholds_tensor).float().mean(dim=1)
        precision_unaligned = (min_gt_to_pred.unsqueeze(2) < thresholds_tensor).float().mean(dim=1)
        fscore_unaligned = 2 * recall_unaligned * precision_unaligned / (recall_unaligned + precision_unaligned + 1e-8)
        fscore_unaligned[recall_unaligned + precision_unaligned == 0] = 0
        
        # 对齐版本
        recall_aligned = (min_pred_to_aligned.unsqueeze(2) < thresholds_tensor).float().mean(dim=1)
        precision_aligned = (min_gt_to_aligned.unsqueeze(2) < thresholds_tensor).float().mean(dim=1)
        fscore_aligned = 2 * recall_aligned * precision_aligned / (recall_aligned + precision_aligned + 1e-8)
        fscore_aligned[recall_aligned + precision_aligned == 0] = 0
        
        return fscore_unaligned.cpu().numpy(), fscore_aligned.cpu().numpy()

    def __call__(self, preds, inputs, step_idx, **kwargs):
        img = inputs["image"]
        B, T, N, C, H, W = img.shape
        gt_T_c2m = inputs["target_cam_extr"].flatten(0, 1)
        gt_verts_3d = inputs['target_verts_3d'].flatten(0, 2) * 0.2 # (B*T*N, V, 3)
        gt_joints_3d = inputs['target_joints_3d'].flatten(0, 2) * 0.2
        pse_verts_3d = inputs['pse_verts_3d'].flatten(0, 2) * 0.2
        pse_joints_3d = inputs['pse_joints_3d'].flatten(0, 2) * 0.2
        gt_verts_3d = gt_verts_3d - gt_joints_3d[:, 9:10]
        pse_verts_3d = pse_verts_3d - pse_joints_3d[:, 9:10]
        # pred_verts = preds['master_verts_mvf']
        # pred_verts = pred_verts.unsqueeze(1).repeat(1, N, 1, 1)
        # pred_verts_in_cam = batch_cam_extr_transf(gt_T_c2m, pred_verts).flatten(0, 1)  # (B*T*N, V, 3)

        pred_verts_in_cam = pse_verts_3d
        
        # 批量对齐
        aligned_preds = self.batch_align_w_scale(gt_verts_3d, pred_verts_in_cam)
        
        # 批量计算F-score
        f_unaligned, f_aligned = self.batch_calculate_fscore(
            gt_verts_3d, 
            pred_verts_in_cam,
            aligned_preds,
            self.f_threshs
        )
        
        # 更新结果
        self.f_score.extend(f_unaligned.tolist())
        self.f_score_aligned.extend(f_aligned.tolist())
        self.n_eval_samples += B * T * N

        self.PCK_J.feed(preds, inputs)
        self.PCK_V.feed(preds, inputs)

    def on_finished(self):
        print('Total number of samples: %d' % self.n_eval_samples)
        print('F-scores')
        f_out = list()
        f_score, f_score_aligned = np.array(self.f_score).T, np.array(self.f_score_aligned).T
        for f, fa, t in zip(f_score, f_score_aligned, self.f_threshs):
            print('F@%.1fmm = %.3f' % (t * 1000, f.mean()), '\tF_aligned@%.1fmm = %.3f' % (t * 1000, fa.mean()))
            f_out.append('f_score_%d: %f' % (round(t * 1000), f.mean()))
            f_out.append('f_al_score_%d: %f' % (round(t * 1000), fa.mean()))

        logger.info(f"Dump AUC results to {self.exp_dir}")
        filepth_j = os.path.join(self.exp_dir, 'res_auc_j.pkl')
        auc_pth_j = os.path.join(self.exp_dir, 'auc_j.txt')
        filepth_v = os.path.join(self.exp_dir, 'res_auc_v.pkl')
        auc_pth_v = os.path.join(self.exp_dir, 'auc_v.txt')
        score_path = os.path.join(self.exp_dir, 'scores.txt')

        dict_J = self.PCK_J.get_measures()
        dict_V = self.PCK_V.get_measures()

        with open(filepth_j, 'wb') as f:
            pickle.dump(dict_J, f)
        with open(auc_pth_j, 'w') as ff:
            ff.write(str(dict_J["auc_all"]))

        with open(filepth_v, 'wb') as f:
            pickle.dump(dict_V, f)
        with open(auc_pth_v, 'w') as ff:
            ff.write(str(dict_V["auc_all"]))
        with open(score_path, 'w') as ff:
            for t in f_out:
                ff.write('%s\n' % t)

        logger.warning(f"auc_j: {dict_J['auc_all']}")
        logger.warning(f"auc_v: {dict_V['auc_all']}")
        self.reset()


class DrawingHandCallback(IdleCallback):

    def __init__(self, img_draw_dir):

        self.img_draw_dir = img_draw_dir
        os.makedirs(img_draw_dir, exist_ok=True)

        mano_layer = ManoLayer(mano_assets_root="assets/mano_v1_2")
        self.mano_faces = mano_layer.get_mano_closed_faces().numpy()
        self.renderer = OpenDRRenderer()

    def __call__(self, preds, inputs, step_idx, **kwargs):

        tensor_image = inputs["image"]  # (B, T, N, 3, H, W) 5 channels
        tensor_image = tensor_image.flatten(0, 1)  # (B*T, N, 3, H, W)
        batch_size = tensor_image.size(0)
        n_views = tensor_image.size(1)
        image = denormalize(tensor_image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False)
        image = image.permute(0, 1, 3, 4, 2)
        image = image.mul_(255.0).detach().cpu()  # (B, N, H, W, 3)
        image = image.numpy().astype(np.uint8)

        cam_param = inputs["target_cam_intr"].flatten(0, 1)  # (B*T, N, 4)
        mesh_xyz = preds["master_verts_mvf"].unsqueeze(1).repeat(1, n_views, 1, 1)
        pose_xyz = preds["master_joints_mvf"].unsqueeze(1).repeat(1, n_views, 1, 1)
        # mesh_xyz = inputs["master_verts_3d"].flatten(0, 1).unsqueeze(1).repeat(1, n_views, 1, 1)  # (B, N, 778, 3)
        # pose_xyz = inputs["master_joints_3d"].flatten(0, 1).unsqueeze(1).repeat(1, n_views, 1, 1)  # (B, N, 21, 3)


        # gt_T_c2m = torch.linalg.inv(inputs["target_cam_extr"])  # (B, N, 4, 4)
        gt_T_c2m = inputs["target_cam_extr"].flatten(0, 1)  # (B, N, 4, 4)
        mesh_xyz = batch_cam_extr_transf(gt_T_c2m, mesh_xyz)  # (B, N, 778, 3)
        pose_xyz = batch_cam_extr_transf(gt_T_c2m, pose_xyz)  # (B, N, 21, 3)
        pose_uv = batch_cam_intr_projection(cam_param, pose_xyz)  # (B, N, 21, 2)

        # tensor_image = inputs["image"]  # (B, N, 3, H, W) 5 channels
        # batch_size = tensor_image.size(0)
        # n_views = tensor_image.size(1)
        # image = denormalize(tensor_image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False)
        # image = image.permute(0, 1, 3, 4, 2)
        # image = image.mul_(255.0).detach().cpu()  # (B, N, H, W, 3)
        # image = image.numpy().astype(np.uint8)

        # cam_param = inputs["target_cam_intr"]
        # mesh_xyz = preds["pred_verts_3d"].unsqueeze(1).repeat(1, n_views, 1, 1)
        # pose_xyz = preds["pred_joints_3d"].unsqueeze(1).repeat(1, n_views, 1, 1)

        # gt_T_c2m = torch.linalg.inv(inputs["target_cam_extr"])  # (B, N, 4, 4)
        # mesh_xyz = batch_cam_extr_transf(gt_T_c2m, mesh_xyz)  # (B, N, 21, 3)
        # pose_xyz = batch_cam_extr_transf(gt_T_c2m, pose_xyz)  # (B, N, 778, 3)
        # pose_uv = batch_cam_intr_projection(cam_param, pose_xyz)  # (B, N, 21, 2)

        mesh_xyz = mesh_xyz.detach().cpu().numpy()
        pose_xyz = pose_xyz.detach().cpu().numpy()
        pose_uv = pose_uv.detach().cpu().numpy()
        cam_param = cam_param.detach().cpu().numpy()

        for i in range(batch_size):
            for j in range(n_views):
                file_name = os.path.join(self.img_draw_dir, f"step{step_idx}_frame{i}_view{j}.jpg")
                save_a_image_with_mesh_joints(image=image[i, j],
                                              cam_param=cam_param[i, j],
                                              mesh_xyz=mesh_xyz[i, j],
                                              pose_uv=pose_uv[i, j],
                                              pose_xyz=pose_xyz[i, j],
                                              face=self.mano_faces,
                                              with_mayavi_mesh=False,
                                              with_skeleton_3d=False,
                                              file_name=file_name,
                                              renderer=self.renderer)

    def on_finished(self):
        pass
