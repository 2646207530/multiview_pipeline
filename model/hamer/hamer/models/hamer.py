import torch
import pytorch_lightning as pl
from typing import Any, Dict, Mapping, Tuple
import numpy as np
from yacs.config import CfgNode
import cv2
from ..utils import SkeletonRenderer, MeshRenderer
from ..utils.geometry import aa_to_rotmat, perspective_projection

from .backbones import create_backbone
from .heads import build_mano_head
from .discriminator import Discriminator
from .losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss
from . import MANO

from .backbones.selective_vit_adapter import apply_patch

    
class HAMER(pl.LightningModule):

    def __init__(self, cfg: CfgNode, init_renderer: bool = False):
        """
        Setup HAMER model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()
        print('hamer')
        # Save hyperparameters
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])

        self.cfg = cfg
        # Create backbone feature extractor
        self.backbone = create_backbone(cfg)
        # if cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS', None):
        #     # self.backbone.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location='cpu')['state_dict'])
        #     self.backbone.load_state_dict(torch.load('/home/cyc/pycharm/hamer/_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth', map_location='cpu')['state_dict'])
        # Create MANO head
        self.mano_head = build_mano_head(cfg)

        # Create discriminator
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            self.discriminator = Discriminator()

        # Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.mano_parameter_loss = ParameterLoss()

        # Instantiate MANO model
        mano_cfg = {k.lower(): v for k,v in dict(cfg.MANO).items()}
        self.mano = MANO(**mano_cfg)
        # 默认值
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)
        self.to(self.device)
        self.eval()
        # Buffer that shows whetheer we need to initialize ActNorm layers
        self.register_buffer('initialized', torch.tensor(False))
        # Setup renderer for visualization
        if init_renderer:
            print('ske and mesh render')
            self.renderer = SkeletonRenderer(self.cfg)
            
            self.mesh_renderer = MeshRenderer(self.cfg, faces=self.mano.faces)
            
        else:
            self.renderer = None
            self.mesh_renderer = None

        # Disable automatic optimization since we use adversarial training
        self.automatic_optimization = False
        
    def create_render(self):
        return MeshRenderer(self.cfg, faces=self.mano.faces)

    def get_parameters(self):
        all_params = list(self.mano_head.parameters())
        all_params += list(self.backbone.parameters())
        return all_params

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """
        Setup model and distriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        param_groups = [{'params': filter(lambda p: p.requires_grad, self.get_parameters()), 'lr': self.cfg.TRAIN.LR}]

        optimizer = torch.optim.AdamW(params=param_groups,
                                        # lr=self.cfg.TRAIN.LR,
                                        weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        optimizer_disc = torch.optim.AdamW(params=self.discriminator.parameters(),
                                            lr=self.cfg.TRAIN.LR,
                                            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)

        return optimizer, optimizer_disc

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """

        # Use RGB image as input
        x = batch['img']
        # img_np = x.squeeze(0).permute(1,2,0).cpu().numpy()
        # img_np = (img_np * self.std + self.mean)
        # img_np = img_np.astype(np.uint8)
        # cv2.imwrite('/home/cyc/pycharm/vGesture/lib/core/hamer_patch_foeward.png',img_np)
        batch_size = x.shape[0]

        # Compute conditioning features using the backbone
        # if using ViT backbone, we need to use a different aspect ratio
        conditioning_feats = self.backbone(x[:,:,:,32:-32])

        pred_mano_params, pred_cam, _ = self.mano_head(conditioning_feats)
        
        # Store useful regression outputs to the output dict
        output = {}
        output['pred_cam'] = pred_cam
        output['pred_mano_params'] = {k: v.clone() for k,v in pred_mano_params.items()}
        
        # Compute camera translation
        device = pred_mano_params['hand_pose'].device
        dtype = pred_mano_params['hand_pose'].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2*focal_length[:, 0]/(self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] +1e-9)],dim=-1)
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length

        # Compute model vertices, joints and the projected joints
        pred_mano_params['global_orient'] = pred_mano_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_mano_params['hand_pose'] = pred_mano_params['hand_pose'].reshape(batch_size, -1, 3, 3)
        pred_mano_params['betas'] = pred_mano_params['betas'].reshape(batch_size, -1)
        # output['beta'] = pred_mano_params['betas']
        mano_output = self.mano(**{k: v for k,v in pred_mano_params.items()}, pose2rot=False)
        pred_keypoints_3d = mano_output.joints
        pred_vertices = mano_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)
        # print('self.cfg.MODEL.IMAGE_SIZE:',self.cfg.MODEL.IMAGE_SIZE)#256
        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)
        pred_mano_params['trans'] = pred_cam_t
        return output, pred_mano_params

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """

        pred_mano_params = output['pred_mano_params']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']


        batch_size = pred_mano_params['hand_pose'].shape[0]
        device = pred_mano_params['hand_pose'].device
        dtype = pred_mano_params['hand_pose'].dtype

        # Get annotations
        gt_keypoints_2d = batch['keypoints_2d']
        gt_keypoints_3d = batch['keypoints_3d']
        gt_mano_params = batch['mano_params']
        has_mano_params = batch['has_mano_params']
        is_axis_angle = batch['mano_params_is_axis_angle']

        # Compute 3D keypoint loss
        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=0)

        # Compute loss on MANO parameters
        loss_mano_params = {}
        for k, pred in pred_mano_params.items():
            gt = gt_mano_params[k].view(batch_size, -1)
            if is_axis_angle[k].all():
                gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
            has_gt = has_mano_params[k]
            loss_mano_params[k] = self.mano_parameter_loss(pred.reshape(batch_size, -1), gt.reshape(batch_size, -1), has_gt)

        loss = self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D'] * loss_keypoints_3d+\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D'] * loss_keypoints_2d+\
               sum([loss_mano_params[k] * self.cfg.LOSS_WEIGHTS[k.upper()] for k in loss_mano_params])

        losses = dict(loss=loss.detach(),
                      loss_keypoints_2d=loss_keypoints_2d.detach(),
                      loss_keypoints_3d=loss_keypoints_3d.detach())

        for k, v in loss_mano_params.items():
            losses['loss_' + k] = v.detach()

        output['losses'] = losses

        return loss

    # Tensoroboard logging should run from first rank only
    @pl.utilities.rank_zero.rank_zero_only
    def tensorboard_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True, write_to_summary_writer: bool = True) -> None:
        """
        Log results to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """

        mode = 'train' if train else 'val'
        # batch_size = batch['keypoints_2d'].shape[0]
        batch_size = batch['joint_img'].shape[0]
        images = batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        #images = 255*images.permute(0, 2, 3, 1).cpu().numpy()

        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, -1, 3)
        pred_vertices = output['pred_vertices'].detach().reshape(batch_size, -1, 3)
        focal_length = output['focal_length'].detach().reshape(batch_size, 2)
        # gt_keypoints_3d = batch['keypoints_3d']
        # gt_keypoints_2d = batch['keypoints_2d']
        gt_keypoints_2d = batch['joint_img'][:, 2]
        gt_keypoints_2d = torch.cat([gt_keypoints_2d, torch.ones_like(gt_keypoints_2d[...,:1])], dim=-1)
        pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, 3)
        pred_keypoints_2d = output['pred_keypoints_2d'].detach().reshape(batch_size, -1, 2)

        if write_to_summary_writer:
            summary_writer = self.logger.experiment
            for loss_name, val in output['losses'].items():
                summary_writer.add_scalar(mode +'/' + loss_name, val.detach().item(), step_count)
        num_images = min(batch_size, self.cfg.EXTRA.NUM_LOG_IMAGES)

        # gt_keypoints_3d = batch['keypoints_3d']
        # pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, -1, 3)

        # We render the skeletons instead of the full mesh because rendering a lot of meshes will make the training slow.
        #predictions = self.renderer(pred_keypoints_3d[:num_images],
        #                            gt_keypoints_3d[:num_images],
        #                            2 * gt_keypoints_2d[:num_images],
        #                            images=images[:num_images],
        #                            camera_translation=pred_cam_t[:num_images])
        predictions = self.mesh_renderer.visualize_tensorboard(pred_vertices[:num_images].cpu().numpy(),
                                                               pred_cam_t[:num_images].cpu().numpy(),
                                                               images[:num_images].cpu().numpy(),
                                                               pred_keypoints_2d[:num_images].cpu().numpy(),
                                                               gt_keypoints_2d[:num_images].cpu().numpy(),
                                                               focal_length=focal_length[:num_images].cpu().numpy())
        if write_to_summary_writer:
            summary_writer.add_image('%s/predictions' % mode, predictions, step_count)

        return predictions

    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)
    # def forward(self, img: torch.Tensor, bbox: torch.Tensor) -> Tuple[
    #     torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     前向推理，并在内部进行预处理:
    #     Args:
    #         img (torch.Tensor): 输入图像, shape [B, 3, H, W]
    #         bbox (torch.Tensor): Bounding box张量, shape [B, 5](cls, conf, x1, y1, x2, y2).
    #     Returns:
    #         pred_keypoints_3d (torch.Tensor): [B, 21, 3]
    #         pred_keypoints_2d (torch.Tensor): [B, 21, 2]
    #         global_orient (torch.Tensor): [B, 1, 3, 3]
    #         hand_pose (torch.Tensor): [B, 15, 3, 3]
    #         betas (torch.Tensor): [B, 10]
    #         pred_vertices (torch.Tensor): [B, 778, 3]
    #         pred_cam_t (torch.Tensor): [B, 3]
    #     """

    #     # 假设 batch size = 1 或者你有batched输入，需要对每个bbox处理对应的img
    #     # 这里简化处理假设batch=1，如需batch支持，请根据bbox数量循环或并行处理
    #     # bbox: [B, 6]
    #     cls, conf, x1, y1, x2, y2 = bbox[0]  # 假设batch=1
    #     xyxy = [[], [x1.item(), y1.item(), x2.item(), y2.item()]]
    #     do_flip = cls.item()

    #     if isinstance(xyxy[1], list):
    #         xyxy = xyxy[1]
    #         center_x = xyxy[0] + (xyxy[2] - xyxy[0]) / 2.
    #         center_y = xyxy[1] + (xyxy[3] - xyxy[1]) / 2.
    #     else:
    #         raise ValueError(f"Unexpected structure in xyxy: {xyxy}")

    #     xyxy[2:4] = [max(xyxy[2], xyxy[3])] * 2
    #     xyxy[2:4] = [float(x) for x in xyxy[2:4]]
    #     scale = [1.5 * value / 200.0 for value in xyxy[2:4]]
    #     scale = np.array(scale)
    #     BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
    #     bbox_size = np.array(expand_to_aspect_ratio(scale * 200, target_aspect_ratio=BBOX_SHAPE)).max()

    #     patch_width = patch_height = self.cfg.MODEL.IMAGE_SIZE

    #     # 将img从tensor转为numpy进行处理 (假设img是[B,3,H,W], 需转为H,W,C)
    #     # 如果 img 是 [B,3,H,W]，先转回CPU numpy再操作
    #     img_np = img[0].permute(1,2,0).cpu().numpy() # [H,W,C]
    #     img_height, img_width, img_channels = img_np.shape

    #     downsampling_factor = ((bbox_size*1.0) / patch_width)
    #     downsampling_factor = downsampling_factor / 2.0
    #     if downsampling_factor > 1.1:
    #         img_np = gaussian(img_np, sigma=(downsampling_factor-1)/2, channel_axis=2, preserve_range=True)

    #     img_patch_cv, trans, inv_trans = generate_image_patch_cv2(
    #         img_np,
    #         center_x, center_y,
    #         bbox_size, bbox_size,
    #         patch_width, patch_height,
    #         False, 1.0, 0,
    #         border_mode=cv2.BORDER_CONSTANT
    #     )
    #     img_patch_cv = img_patch_cv[:, :, ::-1]
    #     img_patch = convert_cvimg_to_tensor(img_patch_cv) # [C,H,W]

    #     for n_c in range(min(img_channels, 3)):
    #         img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

    #     # 构造item字典
    #     item = {
    #         'img': torch.tensor(img_patch, dtype=torch.float32)[None],  # [1,3,256,256]
    #         'box_center': torch.tensor([center_x, center_y], dtype=torch.float32)[None],
    #         'box_size': torch.tensor(bbox_size, dtype=torch.float32)[None],
    #         'img_size': torch.tensor([img_width, img_height], dtype=torch.float32)[None],
    #         'inv_trans': torch.tensor(inv_trans, dtype=torch.float32)[None],
    #         'do_flip': torch.tensor([do_flip], dtype=torch.float32)
    #     }

    #     # 移动到 self.device
    #     batch = recursive_to(item, self.device)
    #     batch['img'] = batch['img'].float()

    #     with torch.no_grad():
    #         # 如果当前类本身就是主模型类，请将下行替换为self.forward_step(batch, train=False)
    #         output, mano_params = self.forward_step(batch, train=False)

    #     pred_keypoints_3d = output['pred_keypoints_3d']    # [B,21,3]
    #     pred_keypoints_2d = output['pred_keypoints_2d']
    #     global_orient = mano_params['global_orient']       # [B,1,3,3]
    #     hand_pose = mano_params['hand_pose']               # [B,15,3,3]
    #     betas = mano_params['betas']                       # [B,10]
    #     pred_vertices = output['pred_vertices']            # [B,778,3]
    #     pred_cam_t = output['pred_cam_t']                  # [B,3]

    #     return pred_keypoints_3d, pred_keypoints_2d, global_orient, hand_pose, betas, pred_vertices, pred_cam_t


    def training_step_discriminator(self, batch: Dict,
                                    hand_pose: torch.Tensor,
                                    betas: torch.Tensor,
                                    optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """
        Run a discriminator training step
        Args:
            batch (Dict): Dictionary containing mocap batch data
            hand_pose (torch.Tensor): Regressed hand pose from current step
            betas (torch.Tensor): Regressed betas from current step
            optimizer (torch.optim.Optimizer): Discriminator optimizer
        Returns:
            torch.Tensor: Discriminator loss
        """
        batch_size = hand_pose.shape[0]
        gt_hand_pose = batch['hand_pose']
        gt_betas = batch['betas']
        gt_rotmat = aa_to_rotmat(gt_hand_pose.view(-1,3)).view(batch_size, -1, 3, 3)
        disc_fake_out = self.discriminator(hand_pose.detach(), betas.detach())
        loss_fake = ((disc_fake_out - 0.0) ** 2).sum() / batch_size
        disc_real_out = self.discriminator(gt_rotmat, gt_betas)
        loss_real = ((disc_real_out - 1.0) ** 2).sum() / batch_size
        loss_disc = loss_fake + loss_real
        loss = self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_disc
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss_disc.detach()

    def training_step(self, joint_batch: Dict, batch_idx: int) -> Dict:
        """
        Run a full training step
        Args:
            joint_batch (Dict): Dictionary containing image and mocap batch data
            batch_idx (int): Unused.
            batch_idx (torch.Tensor): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch = joint_batch['img']
        mocap_batch = joint_batch['mocap']
        optimizer = self.optimizers(use_pl_optimizer=True)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            optimizer, optimizer_disc = optimizer

        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=True)
        pred_mano_params = output['pred_mano_params']
        if self.cfg.get('UPDATE_GT_SPIN', False):
            self.update_batch_gt_spin(batch, output)
        loss = self.compute_loss(batch, output, train=True)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            disc_out = self.discriminator(pred_mano_params['hand_pose'].reshape(batch_size, -1), pred_mano_params['betas'].reshape(batch_size, -1))
            loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size
            loss = loss + self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv

        # Error if Nan
        if torch.isnan(loss):
            raise ValueError('Loss is NaN')

        optimizer.zero_grad()
        self.manual_backward(loss)
        # Clip gradient
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.cfg.TRAIN.GRAD_CLIP_VAL, error_if_nonfinite=True)
            self.log('train/grad_norm', gn, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        optimizer.step()
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            loss_disc = self.training_step_discriminator(mocap_batch, pred_mano_params['hand_pose'].reshape(batch_size, -1), pred_mano_params['betas'].reshape(batch_size, -1), optimizer_disc)
            output['losses']['loss_gen'] = loss_adv
            output['losses']['loss_disc'] = loss_disc

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            self.tensorboard_logging(batch, output, self.global_step, train=True)

        self.log('train/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False)

        return output

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:
        """
        Run a validation step and log to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            batch_idx (int): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        # batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)
        output['loss'] = loss
        self.tensorboard_logging(batch, output, self.global_step, train=False)

        return output


class HAMER_INFER(torch.nn.Module):
    def __init__(self, cfg: CfgNode, init_renderer: bool = True, token_merge=False):
        """
        Setup HAMER model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()
        print('hamer infer')
        self.cfg = cfg
        # Create backbone feature extractor
        self.backbone = create_backbone(cfg)

        if token_merge:
            apply_patch(self.backbone)
            self.backbone.r = (8, -1)

        # Create MANO head
        self.mano_head = build_mano_head(cfg)

        # Instantiate MANO model
        mano_cfg = {k.lower(): v for k,v in dict(cfg.MANO).items()}
        self.mano = MANO(**mano_cfg)

        # Buffer that shows whetheer we need to initialize ActNorm layers
        self.register_buffer('initialized', torch.tensor(False))

    @torch.no_grad()
    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """

        # Use RGB image as input
        x = batch['img']
        batch_size = x.shape[0]

        # Compute conditioning features using the backbone
        # if using ViT backbone, we need to use a different aspect ratio
        conditioning_feats = self.backbone(x[:,:,:,32:-32])

        pred_mano_params, pred_cam, _ = self.mano_head(conditioning_feats)

        # Store useful regression outputs to the output dict
        output = {}
        output['pred_cam'] = pred_cam
        output['pred_mano_params'] = {k: v.clone() for k,v in pred_mano_params.items()}

        # Compute camera translation
        device = pred_mano_params['hand_pose'].device
        dtype = pred_mano_params['hand_pose'].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2*focal_length[:, 0]/(self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] +1e-9)],dim=-1)
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length

        # Compute model vertices, joints and the projected joints
        pred_mano_params['global_orient'] = pred_mano_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_mano_params['hand_pose'] = pred_mano_params['hand_pose'].reshape(batch_size, -1, 3, 3)
        pred_mano_params['betas'] = pred_mano_params['betas'].reshape(batch_size, -1)
        #拼接了orient和handpose
        pose = torch.cat([pred_mano_params['global_orient'], pred_mano_params['hand_pose']], dim=1)

        output['pose'] = pose
        output['beta'] = pred_mano_params['betas']

        mano_output = self.mano(**{k: v for k,v in pred_mano_params.items()}, pose2rot=False)
        pred_keypoints_3d = mano_output.joints
        pred_vertices = mano_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)

        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)
        return output
    
    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)