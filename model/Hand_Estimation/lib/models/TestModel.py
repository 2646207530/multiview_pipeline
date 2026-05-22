import math
import cv2

import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
# from manotorch.manolayer import ManoLayer

from ..metrics.basic_metric import LossMetric
from ..metrics.mean_epe import MeanEPE
from ..metrics.pa_eval import PAEval_new
from ..utils.builder import MODEL
from ..utils.logger import logger
from ..utils.misc import param_size
from ..utils.net_utils import init_weights, constant_init
from ..utils.recorder import Recorder
from ..utils.losses import MultiViewConsistencyLoss, joints_losses_uncertainty, \
    Keypoint2DLoss, Keypoint3DLoss, Mesh3DLoss, ParameterLoss, FnHeatmapLoss, RLELoss, pose_norm_loss
# from ..utils.mvc_loss import MultiViewConsistency
from ..utils.triangulation import batch_triangulate_dlt_torch, batch_triangulate_dlt_cfd_torch
from ..utils.transform import batch_cam_extr_transf, batch_cam_intr_projection, batch_persp_project, bchw_2_bhwc, \
    denormalize, _transform_coords
# from ..utils.poco_utils import POCOUtils
# from ..utils.train_utils import is_main_process
from ..viztools.draw import vis_confidence_error_scatter, draw_batch_joint_images, draw_batch_verts_images, plot_hand, \
    visualize_batch_3d_hand_mixed,visualize_batch_3d_hand
from lib.module_MVT.KPNet import KeypointPredictNet
from lib.module_MVT.CVINet_HM import CrossViewInteractNet
# from lib.module.MTNet import MultiFrameNet
from lib.models.model_abstraction import ModuleAbstract
# from lib.utils.mano import MANO
from .heads import build_head
from .heads.MVptEmb_head_MVT import mvf_head


# from ..viztools.draw import draw_mesh, visualize_batch_3d_hand_mixed
# from wilor.models import MANO
# from wilor.utils import SkeletonRenderer, MeshRenderer
# from wilor.utils.renderer import Renderer, cam_crop_to_full

@MODEL.register_module()
class TestMultiviewStereo(nn.Module, ModuleAbstract):
    def __init__(self, cfg):
        super(TestMultiviewStereo, self).__init__()
        self.name = type(self).__name__
        self.cfg = cfg
        self.cfg_loss = cfg.LOSS
        self.root_idx = cfg.LOSS.ROOT_IDX
        self.bbox_3d_size = cfg.LOSS.BBOX_3D_SIZE
        self.image_size = cfg.LOSS.IMAGE_SIZE
        self.data_preset_cfg = cfg.DATA_PRESET

        self.current_phase = 1

        self.kp_net = KeypointPredictNet(cfg.KP_HEAD, cfg.DATA_PRESET)
        self.cvi_net = CrossViewInteractNet(cfg.SV_HEAD)
        # self.mtf_net = MultiFrameNet(cfg.SV_HEAD)
        self.MVptEmb_head = mvf_head(cfg.MV_HEAD)

        # self.mvc_loss = MultiViewConsistencyLoss()
        self.heatmap_loss = FnHeatmapLoss()
        self.coord_loss = nn.L1Loss()
        # self.mesh_3d = Mesh3DLoss()
        # self.rle_loss = RLELoss()

        if self.cfg_loss.JOINTS_LOSS_TYPE == "l2":
            self.criterion_joints = torch.nn.MSELoss()
        else:
            self.criterion_joints = torch.nn.L1Loss()

        if self.cfg_loss.MESHES_LOSS_TYPE == "l2":
            self.criterion_meshes = torch.nn.MSELoss()
        else:
            self.criterion_meshes = torch.nn.L1Loss()
        self.criterion_regr = nn.MSELoss()

        self.loss_metric = LossMetric(cfg)

        self.train_log_interval = cfg.TRAIN.LOG_INTERVAL
        self.init_weights()

        logger.info(f"{self.name} has {param_size(self)}M parameters")
        logger.info(
            f"{self.name} loss type: joint {self.cfg_loss.JOINTS_LOSS_TYPE} verts {self.cfg_loss.MESHES_LOSS_TYPE}")

    def init_weights(self):
        init_weights(self, pretrained=self.cfg.PRETRAINED)

    def set_phase(self, phase):
        """切换训练阶段（无需重新加载模型）"""
        assert phase in [1, 2]
        self.current_phase = phase
        print(f"Start Phase {phase}")

    def setup(self, summary_writer, **kwargs):
        self.summary = summary_writer

    def _forward_impl(self, batch, mode, **kwargs):
        outputs = self.kp_net(batch)
        outputs.update(self.cvi_net(batch, outputs))
        outputs.update(self.MVptEmb_head(batch, outputs))

        return self._format_outputs(outputs)

    def _format_outputs(self, outputs):
        """统一格式化输出字典"""
        mano_params = outputs.get('pred_mano_params_mv', {})
        if mano_params is None:
            mano_params = {}
        preds = {
            'ref_joints_hm': outputs.get('ref_joints_master', None),                  ## 1x
            # 'ref_joints_in_cam': outputs.get('ref_joints_in_cam', None),           ## 1x
            # 'pred_jts': outputs.get('pred_jts', None),
            # 'sigma': outputs.get('sigma', None),
            # 'confidence': outputs.get('maxvals', None),
            # 'nf_loss': outputs.get('nf_loss', None),
            'heatmap_uv': outputs.get('pred_joints_uv', None),                     ## 1x
            'heatmap': outputs.get('pred_hmap', None),
            'confidence': outputs.get('confidence', None),
            'pred_uv': outputs.get('pred_uv', None),
            'anchor_uv': outputs.get('anchor_uv', None),                          ## /128 - 1
            'ref_joints_anchor': outputs.get('ref_joints', None),                        ## 5x
            'ambig_uv': outputs.get('ambig_uv', None),
            'ambig_ref_joints': outputs.get('ambig_ref_joints', None),
            'log_p_sampled': outputs.get('log_p_sampled', None),
            'log_p': outputs.get('log_p', None),
            'joints_preds_list': outputs.get('joints_preds_list', None),
            'master_joints_mvf': outputs.get('master_mano_keypoints_mv', None),         ## 1x
            'master_verts_mvf': outputs.get('master_mano_verts_mv', None),              ## 1x
            # 'coord_uv_master': outputs.get('coord_uv_mv', None),                   ## /128 - 1
            'pose_euler_mv': outputs.get('pred_mano_params_mv', None)['pose_euler'],
            'shape_mv': outputs.get('pred_mano_params_mv', None)['shape'],
            'cam_mv': outputs.get('pred_mano_params_mv', None)['cam'],
            'global_trans_mv': mano_params.get('global_trans', None),
            'global_scale_mv': mano_params.get('global_scale', None)
        }
        # # 检查每个输出是否包含 NaN 或 Inf
        # for key, value in preds.items():
        #     if value is not None:
        #         if torch.isnan(value).any():
        #             print(f"⚠️  WARNING: {key} contains NaN! Shape: {value.shape}")
        #             # 可选：打印具体值（如果很小）
        #             # print(f"  Value: {value[~torch.isnan(value)]}")
        #         if torch.isinf(value).any():
        #             print(f"⚠️  WARNING: {key} contains Inf! Shape: {value.shape}")
        #             # 可选：打印具体值
        #             # print(f"  Value: {value[torch.isinf(value)]}")

        return preds

    @staticmethod
    def loss_proj_to_multicam(pred_joints, T_c2m, K, gt_joints_2d, n_views, img_scale, conf_joints):
        pred_joints = pred_joints.unsqueeze(1).repeat(1, n_views, 1, 1)  # (B, N, 21, 3)
        pred_joints_in_cam = batch_cam_extr_transf(T_c2m, pred_joints)
        pred_joints_2d = batch_cam_intr_projection(K, pred_joints_in_cam).flatten(0, 1)  # (B*N, 21, 2)
        multicam_proj_offset = torch.clamp((conf_joints * pred_joints_2d - conf_joints * gt_joints_2d),
                                           min=-.5 * img_scale,
                                           max=.5 * img_scale) / img_scale
        loss_2d_joints = torch.sum(torch.pow(multicam_proj_offset, 2), dim=2)  # (B, N, 21, 2)
        loss_2d_joints = torch.mean(loss_2d_joints)
        return loss_2d_joints

    def get_proj_2d(self, pred_joints, T_c2m, K, n_views):
        pred_joints = pred_joints.unsqueeze(1).repeat(1, n_views, 1, 1)  # (B, N, 21, 3)
        pred_joints_in_cam = batch_cam_extr_transf(T_c2m, pred_joints)
        pred_joints_2d = batch_cam_intr_projection(K, pred_joints_in_cam)  # (B*N, 21, 2)

        return pred_joints_2d

    def compute_loss(self, preds, gt):
        loss_dict = {}
        total_loss = 0.0

        # ===== 阶段1基础损失 =====
        img = gt['image']
        pse_joints_uv = gt['pse_joints_2d'].flatten(0, 1)  # (BT, N, 21, 2)
        pgt_master_uv = gt['target_pseudo_uv'].flatten(0, 1)[:, 0]
        pgt_uv = gt['target_pseudo_uv'].flatten(0, 2)
        # pgt_uv = gt['target_joints_uvd'].flatten(0, 1)[:, :, :2]
        K = gt['target_cam_intr'].flatten(0, 1)  # (BT, N, 3, 3)
        T_c2m = gt['target_cam_extr'].flatten(0, 1)  # (BT, N, 4, 4)
        pse_vis = gt['cfd'].flatten(0, 1).unsqueeze(-1)
        # pse_vis = gt['target_joints_vis'].unsqueeze(-1)
        # rotation = T_c2m[:, :, :3, :3]
        B = img.size(0)
        T = img.size(1)
        N = img.size(2)
        H, W = img.size(-2), img.size(-1)

        confidence = pse_vis.flatten(0, 1) * preds['confidence'].detach()
        img_scale = math.sqrt(float(W ** 2 + H ** 2))

        # confidence_NS = confidence.unsqueeze(2).repeat(1, 1, 21, 1).reshape(batch_size*n_views, 21*21, 1)

        # 热图损失（始终计算）
        loss_heatmap = self.heatmap_loss(
            preds['heatmap'],
            gt['pse_joints_heatmap'].flatten(0, 2),
            gt['cfd'].flatten(0, 2).squeeze(-1),
            # gt['pse_joints_vis'].flatten(0, 2)
            # gt['target_joints_heatmap'].flatten(0, 1),
            # gt['target_joints_vis'].flatten(0, 1)
        ) * self.cfg.LOSS.HEATMAP
        total_loss += loss_heatmap
        loss_dict["heatmap"] = loss_heatmap

        # if self.current_phase == 1:
        # 热图2D关键点损失（始终计算）
        loss_heatmap_joints = torch.sum(
            torch.pow((pse_vis * preds['heatmap_uv'] - pse_vis * gt['pse_joints_2d'].flatten(0, 1))/ img_scale, 2),
            dim=3
        ).mean() * self.cfg.LOSS.HEATMAP_JOINTS_WEIGHT
        total_loss += loss_heatmap_joints
        loss_dict["heatmap_joints"] = loss_heatmap_joints

        loss_sv_2d = self.coord_loss(confidence * preds['anchor_uv'], confidence * pgt_uv) * self.cfg.LOSS.KEYPOINTS_2D
        total_loss += loss_sv_2d
        loss_dict['sv_2d'] = loss_sv_2d



        # Entropy loss
        loss_entropy = preds['log_p_sampled'].abs().mean() * self.cfg.LOSS.ENTRO
        total_loss += loss_entropy
        loss_dict['entropy'] = loss_entropy

        # NLL loss
        valid = pse_vis.flatten(0, 2).expand(-1, preds['log_p'].shape[1])
        loss_nll = (-(preds['log_p'] * valid).sum() / valid.sum() + 1e-8) * self.cfg.LOSS.NLL
        total_loss += loss_nll
        loss_dict['nll'] = loss_nll


        # 投影2D关键点损失
        loss_ref_2d_proj = 0
        for i in range(preds['joints_preds_list'].shape[0]):
            pred_proj_2d = self.get_proj_2d(preds['joints_preds_list'][i], T_c2m, K, N)
            loss_ref_2d_proj += self.coord_loss(
                confidence * (pred_proj_2d.flatten(0, 1) / 128 - 1),
                confidence * (gt['pse_joints_2d'].flatten(0, 2) / 128 - 1)
                # (pred_proj_2d.flatten(0, 1) / 128 - 1),
                # (gt['target_joints_2d'].flatten(0, 1) / 128 - 1)
            ) * self.cfg.LOSS.KEYPOINTS_2D
        total_loss += loss_ref_2d_proj
        loss_dict['ref_2d_proj_m2p'] = loss_ref_2d_proj

        pred_proj_master = self.get_proj_2d(preds['master_joints_mvf'], T_c2m, K, N)
        loss_mano_2d_proj = self.coord_loss(
            confidence * (pred_proj_master.flatten(0, 1) / 128 - 1),
            confidence * (gt['pse_joints_2d'].flatten(0, 2) / 128 - 1)
            # (pred_proj_2d.flatten(0, 1) / 128 - 1),
            # (gt['target_joints_2d'].flatten(0, 1) / 128 - 1)
        ) * self.cfg.LOSS.KEYPOINTS_2D
        total_loss += loss_mano_2d_proj
        loss_dict['mano_2d_proj_m2p'] = loss_mano_2d_proj

        loss_pose_mv = self.coord_loss(preds['pose_euler_mv'][:, 3:], torch.zeros_like(
            preds['pose_euler_mv'][:, 3:].to(img.device))) * self.cfg.LOSS.POSE_N
        # loss_pose_mv = pose_norm_loss(preds['pose_euler_mv'], 1, img.device) * self.cfg.LOSS.POSE_N
        total_loss += loss_pose_mv
        loss_dict['pose_mv'] = loss_pose_mv

        # shape normalize loss
        loss_shape_mv = self.coord_loss(preds['shape_mv'],
                                        torch.zeros_like(preds['shape_mv'].to(img.device))) * self.cfg.LOSS.SHAPE_N
        total_loss += loss_shape_mv
        loss_dict['shape_mv'] = loss_shape_mv


        loss_dict['loss'] = total_loss
        return total_loss, loss_dict


    def training_step(self, batch, step_idx, epoch, mode, **kwargs):
        img = batch["image"]  # (B, T, N, 3, H, W) 6 dimension
        B = img.size(0)
        T = img.size(1)
        N = img.size(2)

        preds = self._forward_impl(batch, mode, **kwargs)
        _, loss_dict = self.compute_loss(preds, batch)
        self.loss_metric.feed(loss_dict, B)
        # if self.current_phase == 1:


        if step_idx % self.train_log_interval == 0:
            if step_idx % (self.train_log_interval * 25) == 0:  # viz every 10 * interval batches
                # view_id = np.random.randint(n_views)
                K = batch['target_cam_intr'].flatten(0, 1)  # (BT, N, 3, 3)
                T_c2m = batch['target_cam_extr'].flatten(0, 1)
                pred_proj_2d = self.get_proj_2d(preds['joints_preds_list'][-1], T_c2m, K, N)

                pgt_J2d = batch['pse_joints_2d'].flatten(0, 1)
                pse_vis = batch['pse_joints_vis'].flatten(0, 2).unsqueeze(-1)
                pse_joints = batch_triangulate_dlt_cfd_torch(pgt_J2d, K, T_c2m, pse_vis.reshape(B * T, N, -1))




                for i in range(T):
                    anchor_3d = preds['ref_joints_anchor'].reshape(B, T, 21, 3)[:, i, ...]
                    pse_3d=pse_joints.reshape(B, T, 21, 3)[:, i, ...]
                    amb_3d = preds['ambig_ref_joints'].reshape(B, T, -1, 3)[:, i, ...]
                    mv_3d = preds['master_joints_mvf'].reshape(B, T, 21, 3)[:, i, ...]
                    #img_anchor_amb_array = visualize_batch_3d_hand_mixed(anchor_3d, amb_3d, 8)
                    #self.summary.add_image(f"point_cloud/anchor_amb_3d_train_seq_{i}", img_anchor_amb_array, step_idx,
                    #                       dataformats="NHWC")
                    #img_mv_amb_array = visualize_batch_3d_hand_mixed(mv_3d, amb_3d, 8)
                    #self.summary.add_image(f"point_cloud/mv_amb_3d_train_seq_{i}", img_mv_amb_array, step_idx,
                    #                      dataformats="NHWC")

                    #img_anchor_amb_array=visualize_batch_3d_hand(pse_3d,8)
                    #self.summary.add_image(f"point3d/mv_amb_3d_train_seq_{i}", img_anchor_amb_array, step_idx,
                    #                      dataformats="NHWC")

                    view_id = 0
                    pred_joints_3d = preds['master_joints_mvf'].reshape(B, T, 21, 3)[:, i, ...]
                    img_toshow = img[:, i, view_id, ...]  # (B, 3, H, W)
                    pred_proj = pred_proj_2d.reshape(B, T ,-1, 21, 2)[:, i, view_id,...]
                    extr_toshow = batch['target_cam_extr'][:, i, view_id, ...]
                    intr_toshow = batch['target_cam_intr'][:, i, view_id, ...]
                    pred_J3d_in_cam = (extr_toshow[:, :3, :3] @ pred_joints_3d.transpose(1, 2)).transpose(1, 2)
                    pred_J3d_in_cam = pred_J3d_in_cam + extr_toshow[:, :3, 3].unsqueeze(1)
                    pred_J2d = batch_persp_project(pred_J3d_in_cam, intr_toshow)

                    #pgt_J2d = batch['target_pseudo_uv'][:, i, view_id, :, :2]
                    pgt_J2d = batch['pse_joints_2d'][:, i, view_id, :, :2]
                    pred_J2d_hm = preds['heatmap_uv'].view(B, T, -1, 21, 2)[:, i, view_id, ...]

                    #img_array_pgt = draw_batch_joint_images(pred_J2d,(pgt_J2d + 1) * 128, img_toshow,
                    #                                        step_idx)
                    #self.summary.add_image(f"img_pgt/viz_joints_2d_val_seq_{i}", img_array_pgt, step_idx,
                    #                      dataformats="NHWC")

                    img_array_hm = draw_batch_joint_images(pred_J2d_hm, pgt_J2d, img_toshow, step_idx)
                    self.summary.add_image(f"img_hm/viz_joints_2d_val_seq_{i}", img_array_hm, step_idx,
                                           dataformats="NHWC")

                    img_array = draw_batch_joint_images(pred_proj, pgt_J2d , img_toshow, step_idx)
                    self.summary.add_image(f"img_master/vis_joints_2d_val_seq_{i}", img_array, step_idx,
                                           dataformats="NHWC")


        return preds, loss_dict

    def on_train_finished(self, recorder: Recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-train"
        recorder.record_loss(self.loss_metric, epoch_idx, comment=comment)

        self.loss_metric.reset()


    def validation_step(self, batch, step_idx, epoch, mode, **kwargs):
        img = batch["image"]  # (B, T, N, 3, H, W) 6 dimensions
        B = img.size(0)
        T = img.size(1)
        N = img.size(2)

        preds = self._forward_impl(batch, mode, **kwargs)

        if step_idx % (self.train_log_interval * 5) == 0:  # viz every 10 * interval batches

            for i in range(T):
                anchor_3d = preds['ref_joints_anchor'].reshape(B, T, 21, 3)[:, i, ...]
                amb_3d = preds['ambig_ref_joints'].reshape(B, T, -1, 3)[:, i, ...]
                mv_3d = preds['master_joints_mvf'].reshape(B, T, 21, 3)[:, i, ...]
                img_anchor_amb_array = visualize_batch_3d_hand_mixed(anchor_3d, amb_3d, 8)
                self.summary.add_image(f"point_cloud/anchor_amb_3d_val_seq_{i}", img_anchor_amb_array, step_idx,
                                       dataformats="NHWC")
                img_mv_amb_array = visualize_batch_3d_hand_mixed(mv_3d, amb_3d, 8)
                self.summary.add_image(f"point_cloud/mv_amb_3d_val_seq_{i}", img_mv_amb_array, step_idx,
                                       dataformats="NHWC")
        return preds

    def on_val_finished(self, recorder: Recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-val"
        self.loss_metric.reset()


    def select_2d(self, batch, mode):
        img = batch["image"]  # (B, N, 3, H, W) 5 dimensions
        B, N, C, H, W = img.shape
        img = img.flatten(0, 1)
        affine = batch['affine_inv'].flatten(0, 1)
        img_ori = batch['image_ori'].flatten(0, 1)
        joints_gt = batch["target_joints_uvd"].flatten(0, 1)[:, :, :2]
        joints_pgt = batch['target_pseudo_uv'].flatten(0, 1)
        master_joints_3d = batch["master_joints_3d"]
        frame_num = batch["frame_num"].flatten(0, 1)
        cam_num = batch["cam_num"].flatten(0, 1)
        file_num = batch["file_num"].flatten(0, 1)
        # joints_3d_gt = batch["target_joints_3d"]

        gt_T_c2m = batch["target_cam_extr"]  # (B, N, 4, 4)
        gt_K = batch["target_cam_intr"]  # (B, N, 3, 3)
        # gt_K = batch["cam_intr"]      # (B, N, 3, 3)
        # rotation = gt_T_c2m[:, :, :3, :3]
        # batch_size = img.size(0)
        # n_views = img.size(1)

        # tensor_image = img.detach().cpu()
        # image = bchw_2_bhwc(denormalize(tensor_image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False))
        # image = image.mul_(255.0).numpy().astype(np.uint8)  # (B, H, W, 3)
        image = img_ori.detach().cpu().numpy().astype(np.uint8)

        preds = self._forward_impl(batch, mode)
        pred_joints_3d = preds['master_joints_mvf']
        pred_proj_2d = self.get_proj_2d(pred_joints_3d, gt_T_c2m, gt_K, N).flatten(0, 1)
        # img_array = draw_batch_joint_images(pred_J2d, gt_J2d, img_toshow, step_idx)

        res = self.compute_error(preds, batch)
        error_joints_2d_mpse = res['error_joints_2d_mpse']

        idx = torch.where(error_joints_2d_mpse >= 40)[0].tolist()
        if len(idx) > 0:
            for i in idx:
                img_toshow = image[i]
                affine_toshow = affine[i].detach().cpu().numpy()
                joints_gt_toshow = joints_gt[i].detach().cpu().numpy()
                joints_pgt_toshow = joints_pgt[i].detach().cpu().numpy()
                pred_proj_toshow = pred_proj_2d[i].detach().cpu().numpy()
                joints_gt_ori = _transform_coords((joints_gt_toshow + 1) * 128, affine_toshow).astype(np.float32)
                joints_pgt_ori = _transform_coords((joints_pgt_toshow + 1) * 128, affine_toshow).astype(np.float32)
                pred_proj_ori = _transform_coords(pred_proj_toshow, affine_toshow).astype(np.float32)
                gt_img = plot_hand(img_toshow.copy(), joints_gt_ori)
                pgt_img = plot_hand(img_toshow.copy(), joints_pgt_ori)
                pred_img = plot_hand(img_toshow.copy(), pred_proj_ori)
                # gt_img = plot_hand(img_toshow.copy(), (joints_gt_toshow + 1) * 128)
                # pgt_img = plot_hand(img_toshow.copy(), (joints_pgt_toshow + 1) * 128)
                # pred_img = plot_hand(img_toshow.copy(), pred_proj_toshow)
                cv2.imwrite(f'./vis/2d/pred_vis_pgt/gt/gt_file{file_num[i]}_cam{cam_num[i]}_frame_{frame_num[i]}.jpg',
                            gt_img)
                cv2.imwrite(f'./vis/2d/pred_vis_pgt/pgt/pgt_file{file_num[i]}_cam{cam_num[i]}_frame_{frame_num[i]}.jpg',
                            pgt_img)
                cv2.imwrite(
                    f'./vis/2d/pred_vis_pgt/pred/pred_file{file_num[i]}_cam{cam_num[i]}_frame_{frame_num[i]}.jpg',
                    pred_img)

    def testing_step(self, batch, step_idx, mode, **kwargs):
        preds = self._forward_impl(batch, mode)

        if "callback" in kwargs:
            callback = kwargs.pop("callback")
            if callable(callback):
                callback(preds, batch, step_idx, **kwargs)

        return preds

    def on_test_finished(self, recorder: Recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-test"
        self.loss_metric.reset()

    def format_metric(self, mode, epoch_idx):
        if mode == "train":
            if self.current_phase == 1:
                return (f"L: {self.loss_metric.get_loss('loss'):.4f} | "
                        f"L_ENTRO: {self.loss_metric.get_loss('entropy'):.4f} | "
                        f"L_NLL: {self.loss_metric.get_loss('nll'):.4f} | "
                        f"L_HM: {self.loss_metric.get_loss('heatmap'):.4f} | "
                        f"L_J2D_HM: {self.loss_metric.get_loss('heatmap_joints'):.4f} | "
                        f"L_SV_2D: {self.loss_metric.get_loss('sv_2d'):.4f} | "
                        f"L_R2D_PJ_M: {self.loss_metric.get_loss('ref_2d_proj_m2p'):.4f} | "
                        f"L_M2D_PJ_M: {self.loss_metric.get_loss('mano_2d_proj_m2p'):.4f} | "
                        # f"L_MA: {self.loss_metric.get_loss('master_abs'):.4f} | "
                        f"L_POSE_MV: {self.loss_metric.get_loss('pose_mv'):.4f} | "
                        f"L_SHAPE_MV: {self.loss_metric.get_loss('shape_mv'):.4f}")
            else:
                return (f"L: {self.loss_metric.get_loss('loss'):.4f} | "
                        # f"L_UNC: {self.loss_metric.get_loss('uncert_loss'):.4f} | "
                        # f"L_NF: {self.loss_metric.get_loss('loss_nf'):.4f} | "
                        # f"L_MANO_SV: {self.loss_metric.get_loss('mano_params_sv'):.4f} | "
                        # f"L_MANO_MV: {self.loss_metric.get_loss('mano_params_mvf'):.4f} | "
                        # f"L_HM: {self.loss_metric.get_loss('heatmap'):.4f} | "
                        # f"L_J2D_HM: {self.loss_metric.get_loss('heatmap_joints'):.4f} | "
                        f"L_SV_2D: {self.loss_metric.get_loss('sv_2d'):.4f} | "
                        # f"L_MAS_2D: {self.loss_metric.get_loss('master_2d'):.4f} | "
                        f"L_RLE_2D: {self.loss_metric.get_loss('rle_2d'):.4f} | "
                        f"L_J2D_PJ_M: {self.loss_metric.get_loss('kpts_2d_proj_m2p'):.4f} | "
                        # f"L_MAS_PJ: {self.loss_metric.get_loss('kpts_master_proj'):.4f} | "
                        # f"L_MVC: {self.loss_metric.get_loss('mvc'):.4f} | "
                        # f"L_TRI: {self.loss_metric.get_loss('triangulate'):.4f} | "
                        # f"L_J2D_SV: {self.loss_metric.get_loss('kpts_2d_sv'):.4f} | "
                        # f"L_J2D_MVF: {self.loss_metric.get_loss('kpts_2d_mvf'):.4f} | "
                        # f"L_SV_3D: {self.loss_metric.get_loss('sv_3d'):.4f} | "
                        f"L_MA_3D: {self.loss_metric.get_loss('master_recon'):.4f} | "
                        f"L_MA: {self.loss_metric.get_loss('master_abs'):.4f} | "
                        f"L_VM: {self.loss_metric.get_loss('verts_3d_mv'):.4f} | "
                        f"L_JM: {self.loss_metric.get_loss('joints_3d_mv'):.4f} | "
                        f"L_POSE_SV: {self.loss_metric.get_loss('pose_sv'):.4f} | "
                        f"L_SHAPE_SV: {self.loss_metric.get_loss('shape_sv'):.4f} | "
                        f"L_POSE_MV: {self.loss_metric.get_loss('pose_mv'):.4f} | "
                        f"L_SHAPE_MV: {self.loss_metric.get_loss('shape_mv'):.4f}")
                # f"L_MAL: {self.loss_metric.get_loss('master_align'):.4f}")
                # f"L_J2D_PJ_S: {self.loss_metric.get_loss('kpts_2d_proj_sv2m'):.4f} | ")
        else:
            # metric_toshow = [self.MPJPE_2D_PSEUDO, self.MPJPE_2D_KP, self.MPJPE_2D_SV, self.PA_3D_PGT, self.PA_3D_SV, self.PA_3D_MP, self.PA_3D_MR]
            metric_toshow = [self.MPJPE_2D_PSEUDO, self.MPJPE_2D_HM, self.MPJPE_2D_AC,
                             self.MPJPE_2D_MV, self.MPJPE_3D_HM, self.MPJPE_3D_AC, self.MPJPE_3D_MV, self.MPJPE_3D_MVN]

        return " | ".join([str(me) for me in metric_toshow])

    def forward(self, inputs, step_idx=0, epoch=0, mode="test", **kwargs):
        if mode == "train":
            return self.training_step(inputs, step_idx, epoch, mode="train", **kwargs)
        elif mode == "val":
            return self.validation_step(inputs, step_idx, epoch, mode="val", **kwargs)
        elif mode == "test":
            return self.testing_step(inputs, step_idx, mode="test", **kwargs)
        else:
            raise ValueError(f"Only 'train' and 'val' are supported in forward method, got {mode}.")

