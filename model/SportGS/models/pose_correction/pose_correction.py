import os

from common.rot import matrix_to_axis_angle, axis_angle_to_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.spatial.transform import Rotation
from utils.general_utils import get_jtr
from right_hand_model import MANO
import models
from .lbs import lbs
# from models.network_utils import get_mlp

def get_transforms_02v(Jtr):
    device = Jtr.device

    from scipy.spatial.transform import Rotation as R
    rot45p = torch.tensor(R.from_euler('z', 45, degrees=True).as_matrix(), dtype=torch.float32, device=device)
    rot45n = torch.tensor(R.from_euler('z', -45, degrees=True).as_matrix(), dtype=torch.float32, device=device)
    # Specify the bone transformations that transform a SMPL A-pose mesh
    # to a star-shaped A-pose (i.e. Vitruvian A-pose)
    bone_transforms_02v = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 4, 4).repeat(24, 1, 1)

    # First chain: L-hip (1), L-knee (4), L-ankle (7), L-foot (10)
    R_02v_l = []
    t_02v_l = []
    chain = [1, 4, 7, 10]
    rot = rot45p
    for i, j_idx in enumerate(chain):
        R_02v_l.append(rot)
        t = Jtr[j_idx]
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent]
            t = torch.matmul(rot, t - t_p)
            t = t + t_02v_l[i-1]

        t_02v_l.append(t)

    R_02v_l = torch.stack(R_02v_l, dim=0)
    t_02v_l = torch.stack(t_02v_l, dim=0)
    t_02v_l = t_02v_l - torch.matmul(Jtr[chain], rot.transpose(0, 1))

    R_02v_l = F.pad(R_02v_l, (0, 0, 0, 1))  # 4 x 4 x 3
    t_02v_l = F.pad(t_02v_l, (0, 1), value=1.0)   # 4 x 4

    bone_transforms_02v[chain] = torch.cat([R_02v_l, t_02v_l.unsqueeze(-1)], dim=-1)

    # Second chain: R-hip (2), R-knee (5), R-ankle (8), R-foot (11)
    R_02v_r = []
    t_02v_r = []
    chain = [2, 5, 8, 11]
    rot = rot45n
    for i, j_idx in enumerate(chain):
        # bone_transforms_02v[j_idx, :3, :3] = rot
        R_02v_r.append(rot)
        t = Jtr[j_idx]
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent]
            t = torch.matmul(rot, t - t_p)
            t = t + t_02v_r[i-1]

        t_02v_r.append(t)

    # bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)
    R_02v_r = torch.stack(R_02v_r, dim=0)
    t_02v_r = torch.stack(t_02v_r, dim=0)
    t_02v_r = t_02v_r - torch.matmul(Jtr[chain], rot.transpose(0, 1))

    R_02v_r = F.pad(R_02v_r, (0, 0, 0, 1))  # 4 x 3
    t_02v_r = F.pad(t_02v_r, (0, 1), value=1.0)   # 4 x 4

    bone_transforms_02v[chain] = torch.cat([R_02v_r, t_02v_r.unsqueeze(-1)], dim=-1)

    return bone_transforms_02v

class NoPoseCorrection(nn.Module):
    def __init__(self, config, metadata=None):
        super(NoPoseCorrection, self).__init__()

    def forward(self, camera, iteration):
        return camera, {}

    def regularization(self, out):
        return {}

class PoseCorrection(nn.Module):
    def __init__(self, config, metadata=None, metadata_obj=None, hand_side='right'):
        super(PoseCorrection, self).__init__()

        self.config = config
        self.metadata = metadata

        self.frame_dict = metadata['frame_dict']

        v_template = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/v_templates.npz')[hand_side+'Hand']
        lbs_weights = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/skinning_weights_all.npz')[hand_side+'Hand']
        posedirs = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/posedirs_all.npz')[hand_side+'Hand']
        posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1]).T
        shapedirs = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/shapedirs_all.npz')[hand_side+'Hand']
        J_regressor = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/J_regressors.npz')[hand_side+'Hand']
        kintree_table = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/kintree_table.npy')

        self.register_buffer('v_template', torch.tensor(v_template, dtype=torch.float32).unsqueeze(0))
        self.register_buffer('posedirs', torch.tensor(posedirs, dtype=torch.float32))
        self.register_buffer('shapedirs', torch.tensor(shapedirs, dtype=torch.float32))
        self.register_buffer('J_regressor', torch.tensor(J_regressor, dtype=torch.float32))
        self.register_buffer('lbs_weights', torch.tensor(lbs_weights, dtype=torch.float32))
        self.register_buffer('kintree_table', torch.tensor(kintree_table, dtype=torch.int32))

    def forward_smpl(self, betas, root_orient, pose_body, pose_hand, trans):
        full_pose = torch.cat([root_orient, pose_body, pose_hand], dim=-1)
        verts_posed, Jtrs_posed, Jtrs, bone_transforms, _, v_posed, v_shaped, rot_mats = lbs(betas=betas,
                                                                                             pose=full_pose,
                                                                                             v_template=self.v_template.clone(),
                                                                                             clothed_v_template=None,
                                                                                             shapedirs=self.shapedirs.clone(),
                                                                                             posedirs=self.posedirs.clone(),
                                                                                             J_regressor=self.J_regressor.clone(),
                                                                                             parents=self.kintree_table[
                                                                                                 0].long(),
                                                                                             lbs_weights=self.lbs_weights.clone(),
                                                                                             dtype=torch.float32)

        rots = torch.cat([torch.eye(3).reshape(1, 1, 3, 3).to(rot_mats.device), rot_mats[:, 1:]], dim=1)
        rots = rots.reshape(1, -1, 9).contiguous()

        # print(bone_transforms)
        bone_transforms_02v = get_transforms_02v(Jtrs.squeeze(0))
        bone_transforms = torch.matmul(bone_transforms.squeeze(0), torch.inverse(bone_transforms_02v))
        # print(bone_transforms)


        bone_transforms[:, :3, 3] = bone_transforms[:, :3, 3] + trans

        v_shaped = v_shaped.detach()
        center = torch.mean(v_shaped, dim=1)
        minimal_shape_centered = v_shaped - center
        cano_max = minimal_shape_centered.max()
        cano_min = minimal_shape_centered.min()
        padding = (cano_max - cano_min) * 0.05

        # compute pose condition
        Jtrs = Jtrs - center
        Jtrs = (Jtrs - cano_min + padding) / (cano_max - cano_min) / 1.1
        Jtrs -= 0.5
        Jtrs *= 2.
        Jtrs = Jtrs.contiguous()

        verts_posed = verts_posed + trans[None]

        return rots, Jtrs, bone_transforms, verts_posed, v_posed, Jtrs_posed

    def forward(self, camera, iteration):
        frame = camera.frame_id
        if frame not in self.frame_dict:
            return camera, {}
        return self.pose_correct(camera, iteration)

    def regularization(self, out):
        raise NotImplementedError

    def pose_correct(self, camera, iteration):
        raise NotImplementedError

class DirectPoseOptimization(nn.Module):
    def __init__(self, config, metadata, freeze_pose):
        super(DirectPoseOptimization, self).__init__()
        print(metadata.keys())
        self.frame_dict = metadata['frame']
        _MANO_DIR = '/data2/fubingshuai/golf/golf-hand-object/MANO'
        # ⚠️ 故意保留 flat_hand_mean=False (与 train_contact/finetune 里的 flat=True 不一致).
        # 这看似是 bug, 但优化流程依赖这个"差量"约定才稳定:
        # - 这里 body_model_r/l 在 forward (L241,257) 算 bone_transforms / Jtrs,
        #   用 flat=False 时内部会 +hand_mean, 等价于把 pose 当成差量解释.
        # - render_mesh 用的 body_model (train_contact:61-62) 是 flat=True,
        #   把 pose 当成绝对角解释.
        # 两边对同一份 pose 看法不同, 优化时是经验调出来的; 改"对齐"反而把
        # opt_contact / force_closure 搞烂. 保留原行为.
        self.body_model_r = MANO(model_path=_MANO_DIR, flat_hand_mean=False)  # .cuda()
        self.body_model_l = MANO(model_path=_MANO_DIR,
                            is_rhand=False, flat_hand_mean=False)
        self.lbs_weights_r = np.load('./hand_models/misc/skinning_weights_all.npz')['rightHand']
        self.faces = {'right': self.body_model_r.faces, 'left':self.body_model_l.faces}

        self.cfg = config

        rot_l = metadata['rot_l']
        pose_l = metadata['pose_l']
        trans_l = metadata['trans_l']

        rot_r = metadata['rot_r']
        pose_r = metadata['pose_r']
        trans_r = metadata['trans_r']
        
        frame_dict = metadata['frame_dict']

        obj_trans = metadata['obj_trans']
        obj_rots = metadata['obj_rots']

        betas_l = metadata['beta_l']
        betas_r = metadata['beta_r']
        
        self.register_parameter('betas_l', nn.Parameter(torch.tensor(betas_l, dtype=torch.float32)))
        self.register_parameter('betas_r', nn.Parameter(torch.tensor(betas_r, dtype=torch.float32)))

        self.frame_dict = frame_dict

        # use nn.Embedding
        rot_l = np.array(rot_l)
        pose_l = np.array(pose_l)
        trans_l = np.array(trans_l)
        self.rot_l = nn.Embedding.from_pretrained(torch.from_numpy(rot_l).float(), freeze=False)
        self.pose_l = nn.Embedding.from_pretrained(torch.from_numpy(pose_l).float(), freeze=False)
        self.trans_l = nn.Embedding.from_pretrained(torch.from_numpy(trans_l).float(), freeze=freeze_pose)

        rot_r = np.array(rot_r)
        pose_r = np.array(pose_r)
        trans_r = np.array(trans_r)
        self.rot_r = nn.Embedding.from_pretrained(torch.from_numpy(rot_r).float(), freeze=False)
        self.pose_r = nn.Embedding.from_pretrained(torch.from_numpy(pose_r).float(), freeze=False)
        self.trans_r = nn.Embedding.from_pretrained(torch.from_numpy(trans_r).float(), freeze=freeze_pose)

        self.obj_trans = nn.Embedding.from_pretrained(torch.from_numpy(np.array(obj_trans)).float(), freeze=freeze_pose)
        self.obj_rots = nn.Embedding.from_pretrained(torch.from_numpy(np.array(obj_rots)).float(), freeze=freeze_pose)

        self.pose_r_ori = nn.Embedding.from_pretrained(torch.from_numpy(pose_r).float(), freeze=True)
        self.pose_l_ori = nn.Embedding.from_pretrained(torch.from_numpy(pose_l).float(), freeze=True)
        self.trans_r_ori = nn.Embedding.from_pretrained(torch.from_numpy(trans_r).float(), freeze=True)
        self.trans_l_ori = nn.Embedding.from_pretrained(torch.from_numpy(trans_l).float(), freeze=True)


    def forward(self, camera, iteration):
        if iteration < self.cfg.get('delay', 0):
            return camera, {}

        frame = camera.frame_id

        # use nn.Embedding
        idx = torch.Tensor([self.frame_dict[frame]]).long().to(self.betas_l.device)
        rot_l = self.rot_l(idx)
        pose_l = self.pose_l(idx)
        trans_l = self.trans_l(idx)
        betas_l = self.betas_l[idx]
        
        body_l = self.body_model_l(global_orient=rot_l, hand_pose=pose_l,betas=betas_l, transl=trans_l)

        bone_transforms_l = body_l['bone_transforms']

        # rot_mats = body_l['rot_mats']
        # rots = torch.cat([torch.eye(3).reshape(1, 1, 3, 3).repeat(rot_mats.shape[0], 1, 1, 1).to(rot_mats.device),
        #                   rot_mats[:, 1:]], dim=1)
        # rots_l = rots.reshape(rot_mats.shape[0], -1, 9).contiguous()
        bone_transforms_l[:, :, :3, 3] = bone_transforms_l[:, :, :3, 3] + trans_l.unsqueeze(1)
        Jtrs_l = get_jtr(body_l)


        rot_r = self.rot_r(idx)
        pose_r = self.pose_r(idx)
        trans_r = self.trans_r(idx)
        betas_r = self.betas_r[idx]
        body_r = self.body_model_r(global_orient=rot_r, 
                                   hand_pose=pose_r, betas=betas_r, transl=trans_r)

        bone_transforms_r = body_r['bone_transforms']

        # rot_mats = body_r['rot_mats']
        # rots = torch.cat([torch.eye(3).reshape(1, 1, 3, 3).repeat(rot_mats.shape[0], 1, 1, 1).to(rot_mats.device),
        #                   rot_mats[:, 1:]], dim=1)
        # rots_r = rots.reshape(rot_mats.shape[0], -1, 9).contiguous()
        bone_transforms_r[:, :, :3, 3] = bone_transforms_r[:, :, :3, 3] + trans_r.unsqueeze(1)
        Jtrs_r = get_jtr(body_r)

        updated_camera = camera.copy()

        obj_rots = self.obj_rots(idx)
        obj_trans = self.obj_trans(idx)
        trans_r = self.trans_r(idx)
        rot_r = self.rot_r(idx)

        # =========================
        # Temporal Smooth Regularization (2nd order)
        # =========================

        num_frames = self.obj_trans.num_embeddings

        idx_prev = torch.clamp(idx - 1, min=0)
        idx_next = torch.clamp(idx + 1, max=num_frames - 1)

        # ---- translation smooth ----
        trans_prev = self.obj_trans(idx_prev)
        trans_next = self.obj_trans(idx_next)

        loss_trans_smooth = ((trans_next - 2 * obj_trans + trans_prev) ** 2).mean()

        trans_r_prev = self.trans_r(idx_prev)
        trans_r_next = self.trans_r(idx_next)

        loss_trans_smooth_hand = ((trans_r_next - 2 * trans_r + trans_r_prev) ** 2).mean()


        # ---- rotation smooth (SO3 relative) ----
        rot_prev = self.obj_rots(idx_prev)
        rot_next = self.obj_rots(idx_next)

        R_prev = axis_angle_to_matrix(rot_prev.view(1, 3))
        R_curr = axis_angle_to_matrix(obj_rots.view(1, 3))
        R_next = axis_angle_to_matrix(rot_next.view(1, 3))

        # second order in Lie algebra
        R_rel_prev = torch.matmul(R_prev.transpose(-1, -2), R_curr)
        R_rel_next = torch.matmul(R_curr.transpose(-1, -2), R_next)

        w_prev = matrix_to_axis_angle(R_rel_prev)
        w_next = matrix_to_axis_angle(R_rel_next)

        loss_rot_smooth = ((w_next - w_prev) ** 2).mean()

        # hand
        rot_r_prev = self.rot_r(idx_prev)
        rot_r_next = self.rot_r(idx_next)

        R_prev = axis_angle_to_matrix(rot_r_prev.view(1, 3))
        R_curr = axis_angle_to_matrix(rot_r.view(1, 3))
        R_next = axis_angle_to_matrix(rot_r_next.view(1, 3))

        # second order in Lie algebra
        R_rel_prev = torch.matmul(R_prev.transpose(-1, -2), R_curr)
        R_rel_next = torch.matmul(R_curr.transpose(-1, -2), R_next)

        w_prev = matrix_to_axis_angle(R_rel_prev)
        w_next = matrix_to_axis_angle(R_rel_next)

        loss_rot_smooth_hand = ((w_next - w_prev) ** 2).mean()

        loss_obj_smooth = (
            0.5 * (loss_trans_smooth+loss_trans_smooth_hand) +
            0.5 * (loss_rot_smooth+loss_rot_smooth_hand)
        )
            

        theta_l = torch.cat([rot_l, pose_l],dim=-1)
        theta_r = torch.cat([rot_r, pose_r],dim=-1)
        

        # rots_diff = ((matrix_to_axis_angle(camera.rots_l.view(16,3,3)).view(48) - theta_l.view(48))**2).mean()+\
        #             ((matrix_to_axis_angle(camera.rots_r.view(16,3,3)).view(48) - theta_r.view(48))**2).mean()+\
        #             ((matrix_to_axis_angle(camera.obj_rots).view(3) - obj_rots.view(3))**2).mean()
        rots_diff = torch.norm(pose_r - self.pose_r_ori(idx), dim=-1).mean()+torch.norm(pose_l - self.pose_l_ori(idx), dim=-1).mean()
        #torch.norm(trans_r - self.trans_r_ori(idx), dim=-1).mean()+torch.norm(trans_l - self.trans_l_ori(idx), dim=-1).mean()
        
        updated_camera.update(
            rots_l= axis_angle_to_matrix(theta_l.view(16,3)).view(16,9),
            Jtrs_l= Jtrs_l.squeeze(0),
            bone_transforms_l= bone_transforms_l.squeeze(0),
            hand_param_l= torch.cat([rot_l, pose_l, betas_l, trans_l], dim=-1),
            rots_r= axis_angle_to_matrix(theta_r.view(16,3)).view(16,9),
            Jtrs_r= Jtrs_r.squeeze(0),
            bone_transforms_r= bone_transforms_r.squeeze(0),
            hand_param_r= torch.cat([rot_r, pose_r, betas_r, trans_r], dim=-1),

            obj_rots= axis_angle_to_matrix(obj_rots.view(1,3)).view(3,3),
            obj_trans= obj_trans.view(3),
            
        )

        loss_pose = rots_diff.mean()

        return updated_camera, {
            'pose': loss_pose,
            'smooth': loss_obj_smooth
        }

    def regularization(self, out):
        loss = (out['rots_diff'] ** 2).mean()
        return {'pose_reg': loss}

    def export(self, file_name, data_path=None, output_path=None):
        data_p = data_path if data_path is not None else "/home/pt/fbs/data_fixed.npy"

        data = np.load(data_p, allow_pickle=True).item()
        seq_name = next(iter(data["data_dict"].keys()))
        data_params = data["data_dict"][seq_name]['params']
        num_frames = self.obj_trans.num_embeddings
        for vidx in range(num_frames):
            # use nn.Embedding
            idx = torch.Tensor([self.frame_dict[vidx]]).long().to(self.betas_l.device)

            rot_r = self.rot_r(idx)
            pose_r = self.pose_r(idx)
            trans_r = self.trans_r(idx)
            rot_l = self.rot_l(idx)
            pose_l = self.pose_l(idx)
            trans_l = self.trans_l(idx)
            data_params['right hand']["pose_r"][vidx] = pose_r.detach().cpu().numpy()
            data_params['right hand']["trans_r"][vidx] = trans_r.detach().cpu().numpy()
            data_params['right hand']["rot_r"][vidx] = rot_r.detach().cpu().numpy()
            data_params['left hand']["pose_l"][vidx] = pose_l.detach().cpu().numpy()
            data_params['left hand']["trans_l"][vidx] = trans_l.detach().cpu().numpy()
            data_params['left hand']["rot_l"][vidx] = rot_l.detach().cpu().numpy()

            data_params['object']["obj_rot"][vidx] = self.obj_rots(idx).detach().cpu().numpy()
            data_params['object']["obj_trans"][vidx] = self.obj_trans(idx).detach().cpu().numpy()
        # 重要：将修改后的字典重新封装为 numpy 对象数组再保存
        data_to_save = np.array(data, dtype=object)

        # 可选：保存到原路径覆盖，或新路径
        out_path = output_path if output_path is not None else "/home/pt/fbs/test/data_{}".format(file_name)
        
        np.save(out_path, data_to_save, allow_pickle=True)

        print(f"数据已成功导出到: {out_path}")
        


def get_pose_correction(cfg, smpl_data, freeze_pose):
    return DirectPoseOptimization(cfg, smpl_data, freeze_pose)