import os
import random
import re
import sys
import glob
import cv2
import common.transforms as tf
from common import data_utils
from common.data_utils import read_img
from common.mesh import Mesh
from common.object_tensors import ObjectTensors
from right_hand_model import MANO
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from utils.dataset_utils import get_02v_bone_transforms, fetchPly, storePly, AABB, get_valid, pad_jts2d, \
    transform_2d_for_speedup, apply_w2c_pose_numpy, update_K_after_bbox_crop_resize
from scene.cameras import Camera
from utils.camera_utils import freeview_camera

import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
import trimesh


class GolfDataset(Dataset):
    def __init__(self, cfg, split='train', test_split='SDF', multi_batch=False):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self._load_data(cfg, split)
        self.root_dir = self.imgpath or cfg.root_dir
        self.multi_batch = multi_batch

        self.white_bg = cfg.white_background
        self.H, self.W = 1080, 1440
        

        self.skinning_weights = np.load('./hand_models/misc/skinning_weights_all.npz')
        self.posedirs = np.load('./hand_models/misc/posedirs_all.npz')
        self.J_regressor = np.load('./hand_models/misc/J_regressors.npz')

        self.cam_params = {}
        self.world2cam = np.eye(4)

        _MANO_DIR = '/data2/fubingshuai/golf/golf-hand-object/MANO'
        self.body_model_r = MANO(model_path=_MANO_DIR, flat_hand_mean=True)  # .cuda()
        self.body_model_l = MANO(model_path=_MANO_DIR,
                            is_rhand=False, flat_hand_mean=True)
        self.faces = {'right': self.body_model_r.faces, 'left':self.body_model_l.faces}
        #self.color_scales = np.load("../lib/color_scales_25k.npy")

        self.metadata = {}
        self.get_metadata('right')
        self.get_metadata('left')
        self.smpl_data = self.get_smpl_data()
        self.get_obj_data()

        

    def _load_data(self, cfg, split):
        self.data = {}
        self.imgnames = {}
        
        data_p = cfg.pose_path
       

        data = np.load(data_p, allow_pickle=True).item()

        seq_name = cfg.get('seq_name', None)
        if seq_name is None:
            seq_name = next(iter(data["data_dict"].keys()))
        self.data = data["data_dict"][str(seq_name)]
        imgnames = data["imgnames"]
        print(len(imgnames))
    
        self.imgpath = data.get("imgpath")

        frames = []
        self.imgnames = []
        for imgname in imgnames:
            # 解析路径
            _m = re.search(r'(\d+)', imgname)
            vidx = int(_m.group(1)) if _m else 0
            
            frames.append(vidx)
            self.imgnames.append(imgname)

        self.frame_dict = {
            frame: i for i, frame in enumerate(frames)
        }


    def get_smpl_data(self):
        from collections import defaultdict
        smpl_data = defaultdict(list)
        
        for idx, imgname in enumerate(self.imgnames):
        
            _m = re.search(r'(\d+)', imgname)
            vidx = int(_m.group(1)) if _m else 0
            data_params = self.data["params"]
            
            pose_r = data_params['right hand']["pose_r"][vidx].copy()
            trans_r = data_params['right hand']["trans_r"][vidx].copy()
            betas_r = data_params['right hand']["shape_r"][vidx].copy()
            rot_r = data_params['right hand']["rot_r"][vidx].copy()

            pose_l = data_params['left hand']["pose_l"][vidx].copy()
            trans_l = data_params['left hand']["trans_l"][vidx].copy()
            betas_l = data_params['left hand']["shape_l"][vidx].copy()
            rot_l = data_params['left hand']["rot_l"][vidx].copy()

            obj_rot = data_params['object']["obj_rot"][vidx].copy()
            obj_trans = data_params['object']["obj_trans"][vidx].copy()

            world2cam = self.world2cam
            obj_rot, obj_trans = apply_w2c_pose_numpy(obj_rot, obj_trans , world2cam)


            obj_rot = Rotation.from_rotvec(obj_rot).as_matrix()       # 3×3
            # -------- add 20~40 degree rotation noise --------
            noise=False
            if noise:
                axis = np.random.randn(3)
                axis = axis / np.linalg.norm(axis)
                angle = np.deg2rad(np.random.uniform(20.0, 40.0))
                noise_rot = Rotation.from_rotvec(axis * angle).as_matrix()

                # 物体坐标系下扰动（常用）
                obj_rot = noise_rot @ obj_rot

                # -------- add 30~50 mm xy translation noise --------
                noise_xy = np.random.uniform(0.03, 0.05, size=2)          # meter
                noise_xy *= np.random.choice([-1, 1], size=2)

                obj_trans = obj_trans.copy()
                obj_trans[0] += noise_xy[0]
                obj_trans[1] += noise_xy[1]
            
            obj_rot = Rotation.from_matrix(obj_rot).as_rotvec()


            rot_r = Rotation.from_rotvec(rot_r).as_matrix()       # 3×3
            rot_r = world2cam[:3, :3] @ rot_r   # 3×3
            rot_r = Rotation.from_matrix(rot_r).as_rotvec()
            rot_l = Rotation.from_rotvec(rot_l).as_matrix()       # 3×3
            rot_l = world2cam[:3, :3] @ rot_l   # 3×3
            rot_l = Rotation.from_matrix(rot_l).as_rotvec()

            trans_l = world2cam[:3, :3] @ trans_l + world2cam[:3, 3]
            trans_r = world2cam[:3, :3] @ trans_r + world2cam[:3, 3]
            
            #smpl_data['frame'].append(vidx)
            smpl_data['pose_r'].append(pose_r.astype(np.float32))
            smpl_data['beta_r'].append(betas_r.astype(np.float32))
            smpl_data['trans_r'].append(trans_r.astype(np.float32))
            smpl_data['rot_r'].append(rot_r.astype(np.float32))

            smpl_data['pose_l'].append(pose_l.astype(np.float32))
            smpl_data['beta_l'].append(betas_l.astype(np.float32))
            smpl_data['trans_l'].append(trans_l.astype(np.float32))
            smpl_data['rot_l'].append(rot_l.astype(np.float32))


            smpl_data['obj_trans'].append(obj_trans.astype(np.float32))
            smpl_data['obj_rots'].append(obj_rot.astype(np.float32))
        smpl_data['frame_dict'] = self.frame_dict
        return smpl_data

    def get_metadata(self, hand_side='right'):

        cano_data = self.get_cano_mano_verts(hand_side)

        self.metadata[hand_side]={
            'faces': self.faces[hand_side],
            'posedirs': self.posedirs[hand_side+'Hand'],
            'J_regressor': self.J_regressor[hand_side+'Hand'],
            'cameras_extent': 1.0395,
            'frame_dict': self.frame_dict,
        }
        self.metadata[hand_side].update(cano_data)


    def get_obj_data(self):
        self.metadata_obj = {}
        for obj_name in self.cfg._YCB_CLASSES:

            path = self.cfg.get('obj_mesh_path', None)
            if path is None:
                raise ValueError(
                    "dataset.obj_mesh_path is required for golf (path to club mesh in meters)."
                )

            # 加载 mesh
            mesh = trimesh.load_mesh(path, process=False)
            _orig_faces = int(mesh.faces.shape[0])
            _max_faces = int(self.cfg.get('obj_max_faces', 30000))
            if mesh.faces.shape[0] > _max_faces:
                print(f"[GolfDataset] decimating obj mesh {mesh.faces.shape[0]} -> {_max_faces} faces")
                mesh = mesh.simplify_quadric_decimation(face_count=_max_faces)
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)

            # 把 SportGS 实际使用的几何导出到输入 STL 同目录, 方便人眼 / 可视化核对.
            # 文件名规则: <input_stem>_used.stl
            try:
                import os as _os
                _base, _ext = _os.path.splitext(path)
                used_path = f"{_base}_used{_ext if _ext else '.stl'}"
                mesh.export(used_path)
                print(f"[GolfDataset] 实际使用的 obj mesh 已导出: {used_path}  "
                      f"(orig_faces={_orig_faces}, used_verts={len(verts)}, "
                      f"used_faces={len(faces)})")
            except Exception as _e:
                print(f"[GolfDataset] 导出 used mesh 失败 (不影响优化): {_e}")

            sdf_grid, sdf_center, sdf_extent = None, None, None

            #sdf_grid, sdf_center, sdf_extent = self._build_object_sdf_grid(mesh)
           
            obj_triangles= mesh.vertices[mesh.faces].astype(np.float32)

            # 构建角点 AABB
            aabb_min = verts.min(axis=0)
            aabb_max = verts.max(axis=0)


            obj_aabb = AABB(aabb_min, aabb_max)

            # 8个角点（AABB）作为 obj3DCorners
            obj3DCorners = np.array([
                [aabb_min[0], aabb_min[1], aabb_min[2]],
                [aabb_min[0], aabb_min[1], aabb_max[2]],
                [aabb_min[0], aabb_max[1], aabb_min[2]],
                [aabb_min[0], aabb_max[1], aabb_max[2]],
                [aabb_max[0], aabb_min[1], aabb_min[2]],
                [aabb_max[0], aabb_min[1], aabb_max[2]],
                [aabb_max[0], aabb_max[1], aabb_min[2]],
                [aabb_max[0], aabb_max[1], aabb_max[2]],
            ], dtype=np.float32)

            self.metadata_obj[obj_name] = {
                'obj3DCorners': obj3DCorners,  # 8×3
                'obj_aabb': obj_aabb,
                'obj_triangles': obj_triangles,
                'obj_sdf_grid': sdf_grid,
                'obj_sdf_center': sdf_center,
                'obj_sdf_extent': sdf_extent,
                'faces': faces,
                'obj_points': verts,
                'frame_dict': self.frame_dict
                #'obj_mesh': mesh
            }

    def _build_object_sdf_grid(self, mesh, resolution=64, pad_ratio=0.1, chunk_size=32768):
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        aabb_min = verts.min(axis=0)
        aabb_max = verts.max(axis=0)

        center = (aabb_min + aabb_max) * 0.5
        extent = np.max(aabb_max - aabb_min) * 0.5 * (1.0 + pad_ratio)
        extent = np.float32(max(extent, 1e-6))

        grid_coords = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
        xx, yy, zz = np.meshgrid(grid_coords, grid_coords, grid_coords, indexing='ij')
        grid_norm = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        points_obj = grid_norm * extent + center[None, :]

        query = trimesh.proximity.ProximityQuery(mesh)
        signed = np.zeros(points_obj.shape[0], dtype=np.float32)
        for s in range(0, points_obj.shape[0], chunk_size):
            e = min(s + chunk_size, points_obj.shape[0])
            pts = points_obj[s:e]
            closest, _, _ = query.on_surface(pts)
            unsigned = np.linalg.norm(closest - pts, axis=1).astype(np.float32)
            inside = mesh.contains(pts)
            # outside positive, inside negative
            signed[s:e] = np.where(inside, -unsigned, unsigned)

        sdf_grid = signed.reshape(resolution, resolution, resolution)
        return sdf_grid.astype(np.float32), center.astype(np.float32), extent


    def get_cano_mano_verts(self, hand_side='right'):
        # compute scale from Mano
        if hand_side=='right':
            body = self.body_model_r()
        else:
            body = self.body_model_l()
        # 3D models and points
        minimal_shape = body['v'][0].detach().numpy()
        #print('minimal_shape',minimal_shape[0])
        # Break symmetry if given in float16:
        if minimal_shape.dtype == np.float16:
            minimal_shape = minimal_shape.astype(np.float32)
            minimal_shape += 1e-4 * np.random.randn(*minimal_shape.shape)
        else:
            minimal_shape = minimal_shape.astype(np.float32)

        # Minimally clothed shape
        J_regressor = self.J_regressor[hand_side+'Hand']
        Jtr = np.dot(J_regressor, minimal_shape)

        skinning_weights = self.skinning_weights[hand_side+'Hand']

        # bone_transforms = model_dict['bone_transforms']
        bone_transforms = np.repeat(np.eye(4)[np.newaxis, ...], 16, axis=0)
        T = np.matmul(skinning_weights, bone_transforms.reshape([-1, 16])).reshape([-1, 4, 4])
        vertices = np.matmul(T[:, :3, :3], minimal_shape[..., np.newaxis]).squeeze(-1) + T[:, :3, -1]

        coord_max = np.max(vertices, axis=0)
        coord_min = np.min(vertices, axis=0)
        padding_ratio = self.cfg.padding
        padding_ratio = np.array(padding_ratio, dtype=np.float64)
        padding = (coord_max - coord_min) * padding_ratio
        coord_max += padding
        coord_min -= padding

        cano_mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=self.faces[hand_side])

        return {
            'smpl_verts': vertices.astype(np.float32),
            'minimal_shape': minimal_shape,
            'Jtr': Jtr,
            'skinning_weights': skinning_weights,
            'bone_transforms': bone_transforms,
            'cano_mesh': cano_mesh,
            'faces': self.faces[hand_side],
            'coord_max': coord_max,
            'coord_min': coord_min,
            'aabb': AABB(coord_min, coord_max),
        }


    def process_bbox(self, bbox, img_width, img_height, expansion_factor=1.25):
        # sanitize bboxes
        x, y, w, h = bbox
        x1 = np.max((0, x))
        y1 = np.max((0, y))
        x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
        if w * h > 0 and x2 >= x1 and y2 >= y1:
            bbox = np.array([x1, y1, x2 - x1, y2 - y1])
        else:
            return None

        # aspect ratio preserving bbox
        w = bbox[2]
        h = bbox[3]
        c_x = bbox[0] + w / 2.
        c_y = bbox[1] + h / 2.
        aspect_ratio = 1
        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        bbox[2] = w * expansion_factor
        bbox[3] = h * expansion_factor
        bbox[0] = c_x - bbox[2] / 2.
        bbox[1] = c_y - bbox[3] / 2.

        return bbox

    def generate_patch_image(self, cvimg, bbox, input_shape, do_flip=False, scale=1, rot=0):
        """
        @description: Modified from https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/blob/master/data/dataset.py.
                      generate the patch image from the bounding box and other parameters.
        ---------
        @param: input image, bbox(x1, y1, h, w), dest image shape, do_flip, scale factor, rotation degrees.
        -------
        @Returns: processed image, affine_transform matrix to get the processed image.
        -------
        """

        img = cvimg.copy()
        img_height, img_width, _ = img.shape

        bb_c_x = float(bbox[0])
        bb_c_y = float(bbox[1])
        bb_width = float(bbox[2])
        bb_height = float(bbox[3])

        if do_flip:
            img = img[:, ::-1, :]
            bb_c_x = img_width - bb_c_x - 1

        trans = self.gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, input_shape[1], input_shape[0], scale,
                                             rot, inv=False)
        img_patch = cv2.warpAffine(img, trans, (int(input_shape[1]), int(input_shape[0])), flags=cv2.INTER_LINEAR)
        new_trans = np.zeros((3, 3), dtype=np.float32)
        new_trans[:2, :] = trans
        new_trans[2, 2] = 1

        return img_patch, new_trans

    def gen_trans_from_patch_cv(self, c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
        """
        @description: Modified from https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/blob/master/data/dataset.py.
                      get affine transform matrix
        ---------
        @param: image center, original image size, desired image size, scale factor, rotation degree, whether to get inverse transformation.
        -------
        @Returns: affine transformation matrix
        -------
        """

        def rotate_2d(pt_2d, rot_rad):
            x = pt_2d[0]
            y = pt_2d[1]
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            xx = x * cs - y * sn
            yy = x * sn + y * cs
            return np.array([xx, yy], dtype=np.float32)

        # augment size with scale
        src_w = src_width * scale
        src_h = src_height * scale
        src_center = np.array([c_x, c_y], dtype=np.float32)

        # augment rotation
        rot_rad = np.pi * rot / 180
        src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
        src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

        dst_w = dst_width
        dst_h = dst_height
        dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
        dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
        dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = src_center
        src[1, :] = src_center + src_downdir
        src[2, :] = src_center + src_rightdir

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = dst_center
        dst[1, :] = dst_center + dst_downdir
        dst[2, :] = dst_center + dst_rightdir

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def __len__(self):
        return len(self.imgnames)

    def __getitem__(self, idx):
        imgname = self.imgnames[idx]

        _m = re.search(r'(\d+)', imgname)
        vidx = int(_m.group(1)) if _m else 0
        
        seq_data = self.data
        data_params = seq_data["params"]

        cam_K = data_params['camera'].get('k_use', None)
        if cam_K is None:
            _K = data_params['camera']['K']
            cam_K = np.asarray(_K[0] if isinstance(_K, (list, tuple)) else _K)
        intrx = np.asarray(cam_K).copy()

        pose_r = data_params['right hand']["pose_r"][vidx].copy()
        trans_r = data_params['right hand']["trans_r"][vidx].copy()
        betas_r = data_params['right hand']["shape_r"][vidx].copy()
        rot_r = data_params['right hand']["rot_r"][vidx].copy()

        pose_l = data_params['left hand']["pose_l"][vidx].copy()
        trans_l = data_params['left hand']["trans_l"][vidx].copy()
        betas_l = data_params['left hand']["shape_l"][vidx].copy()
        rot_l = data_params['left hand']["rot_l"][vidx].copy()

        #image_size = {"width": self.W, "height": self.H}

        # scale and center in the original image space
        # bbox = data_bbox[vidx, view_idx]  # original bbox
        # intrx = update_K_after_bbox_crop_resize(intrx, bbox, cap_dim=1000, crop_ratio=1.5)

        img_name = os.path.join(self.root_dir, imgname)

        if '/images/' in img_name:
            seg_path = img_name.replace('/images/', '/masks/').replace('jpg', 'npy')
        else:
            base, _ = os.path.splitext(imgname)
            seg_path = os.path.join(self.root_dir + '_masks', base + '.npy')
        
        image, img_status = read_img(img_name, (self.H, self.W, 3))
        mask = np.load(seg_path)
        obj_mask = (mask == 1)
        mask_r = (mask == 2)
        mask_l = (mask == 3)
        

        R = np.eye(3).astype(np.float32)
        T = np.zeros(3).astype(np.float32)



        color_jitting = False
        color_factor = 0.3
        if color_jitting:
            if self.split == 'train':
                c_up = 1.0 + color_factor
                c_low = 1.0 - color_factor
                color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]
            else:
                # pre defined color_jitting for testset
                color_scale = self.color_scales[idx].astype(np.float32)
            for i in range(3):
               image[:, :, i] = np.clip(image[:, :, i] * color_scale[i], 0, 255)


        #image = cv2.resize(image, (self.w, self.h))

        obj_image = image.copy()
        full_image = image.copy()
        hand_image = image.copy()

        scale, rot, do_flip, color_scale = 1, 0, False, [1.0, 1.0, 1.0]
        roi_size = self.cfg.get('roi_size', 224)

        img_ROI, trans_img2roi = self.generate_patch_image(full_image,[0,0,self.W,self.H], [roi_size, roi_size], do_flip, scale, rot)

        hand_mask = mask_r | mask_l
        full_mask = hand_mask | obj_mask
        
        img_ROI = img_ROI / 255.
        img_ROI = torch.from_numpy(img_ROI).permute(2, 0, 1).float()

        hand_image[hand_mask == 0] = 255. if self.white_bg else 0.
        hand_image = hand_image / 255.
        hand_image = torch.from_numpy(hand_image).permute(2, 0, 1).float()

        obj_image[obj_mask == 0] = 255. if self.white_bg else 0.
        obj_image = obj_image / 255.
        obj_image = torch.from_numpy(obj_image).permute(2, 0, 1).float()

        full_image_ori = full_image.copy()
        full_image_ori = full_image_ori/255.
        full_image_ori = torch.from_numpy(full_image_ori).permute(2, 0, 1).float()

        full_image[full_mask == 0] = 255. if self.white_bg else 0.
        full_image = full_image / 255.
        full_image = torch.from_numpy(full_image).permute(2, 0, 1).float()

        full_mask = torch.from_numpy(full_mask.astype(np.float32)).unsqueeze(0).float()
        hand_mask = torch.from_numpy(hand_mask.astype(np.float32)).unsqueeze(0).float()
        obj_mask = torch.from_numpy(obj_mask.astype(np.float32)).unsqueeze(0).float()
        
        mask_r = torch.from_numpy(mask_r.astype(np.float32)).float()
        mask_l = torch.from_numpy(mask_l.astype(np.float32)).float()


        focal_length_x = intrx[0, 0]
        focal_length_y = intrx[1, 1]
        FovY = focal2fov(focal_length_y, self.H)
        FovX = focal2fov(focal_length_x, self.W)

       
        obj_rot = data_params['object']["obj_rot"][vidx].copy()
        obj_trans = data_params['object']["obj_trans"][vidx].copy()
        
        world2cam = self.world2cam
        obj_rot, obj_trans = apply_w2c_pose_numpy(obj_rot, obj_trans, world2cam)
        
            #rot_r, trans_r = apply_w2c_pose_numpy(rot_r, trans_r, world2cam)
            #rot_l, trans_l = apply_w2c_pose_numpy(rot_l, trans_l, world2cam)
        # if self.split == 'train':
        #     print('train:', obj_trans)

        obj_rot, _ = cv2.Rodrigues(obj_rot)
        

        body_r = self.body_model_r(betas=torch.from_numpy(betas_r).float().reshape(-1, 10))
        minimal_shape_r = body_r['v'][0].detach().cpu().numpy()
        body_l = self.body_model_l(betas=torch.from_numpy(betas_l).float().reshape(-1, 10))
        minimal_shape_l = body_l['v'][0].detach().cpu().numpy()

        body_r_world = self.body_model_r(global_orient=torch.from_numpy(rot_r).float().reshape(-1, 3),
                                   hand_pose=torch.from_numpy(pose_r).float().reshape(-1, 45),
                                   betas=torch.from_numpy(betas_r).float().reshape(-1, 10),
                                   #trans_l=torch.from_numpy(trans_r).float().reshape(-1, 3),
                                   )

        body_l_world = self.body_model_l(global_orient=torch.from_numpy(rot_l).float().reshape(-1, 3),
                                   hand_pose=torch.from_numpy(pose_l).float().reshape(-1, 45),
                                   betas=torch.from_numpy(betas_l).float().reshape(-1, 10),
                                   #trans_l=torch.from_numpy(trans_l).float().reshape(-1, 3),
                                   )

        bone_transforms_r, Jtr_r, Jtr_norm_r, pose_rot_r = compute_posed_mano_hand(body_r_world, minimal_shape_r, rot_r, pose_r, trans_r, world2cam)
        bone_transforms_l, Jtr_l, Jtr_norm_l, pose_rot_l = compute_posed_mano_hand(body_l_world, minimal_shape_l, rot_l, pose_l, trans_l, world2cam)
        hand_param_r= torch.tensor(0.)
        hand_param_l= torch.tensor(0.)

        camera = Camera(
            offaxis=True,
            frame_id=int(vidx),
            cam_id=0,
            subject_id='s01',
            obj_id=self.cfg._YCB_CLASSES[0],
            K=intrx, R=R, T=np.squeeze(T),
            #bbox=torch.from_numpy(bbox),
            FoVx=FovX,
            FoVy=FovY,
            image=hand_image,
            mask=hand_mask,
            obj_image=obj_image,
            obj_mask=obj_mask,
            full_image=full_image,
            full_image_ori=full_image_ori,
            full_mask=full_mask,
            img_ROI=img_ROI,
            trans_img2roi=trans_img2roi,
            image_name=imgname.replace('cropped_images/', ''),
            data_device=self.cfg.data_device,
            # human params
            rots_r=torch.from_numpy(pose_rot_r).float(),
            Jtrs_r=torch.from_numpy(Jtr_norm_r).float(),
            bone_transforms_r=torch.from_numpy(bone_transforms_r),
            hand_param_r=hand_param_r,
            hand_param_l=hand_param_l,
            rots_l=torch.from_numpy(pose_rot_l).float(),
            Jtrs_l=torch.from_numpy(Jtr_norm_l).float(),
            bone_transforms_l=torch.from_numpy(bone_transforms_l),
            # obj params
            obj_rots=torch.from_numpy(obj_rot).float().view(3,3),
            obj_trans=torch.from_numpy(obj_trans).float().view(3),
            Jtrs_l_3d=torch.from_numpy(Jtr_l).float(),
            Jtrs_r_3d=torch.from_numpy(Jtr_r).float(),
        )
        return camera



    def readPointCloud(self, sub_id, mano_side='right'):

        ply_path = os.path.join(self.root_dir,'canonical', 'cano_mano_{}_{}.ply'.format(sub_id, mano_side))
        os.makedirs(os.path.join(self.root_dir,'canonical'), exist_ok=True)
        try:
            pcd = fetchPly(ply_path)
        except:
            verts = self.metadata[mano_side]['smpl_verts']
            faces = self.faces[mano_side]
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            n_points = 5000

            xyz = mesh.sample(n_points)
            rgb = np.ones_like(xyz) * 255
            storePly(ply_path, xyz, rgb)

            pcd = fetchPly(ply_path)

        return pcd

    def randomPointCloud(self, obj_id):
        ply_path = os.path.join(self.root_dir,'canonical','random_pc_obj_{}.ply'.format(obj_id))
        os.makedirs(os.path.join(self.root_dir,'canonical'), exist_ok=True)
        n_points = 5000

        objcoords = self.metadata_obj[obj_id]['obj3DCorners']

        coord_min = objcoords.min(axis=0)
        coord_max = objcoords.max(axis=0)
        xyz = np.random.uniform(coord_min, coord_max, (n_points, 3))
        rgb = np.ones_like(xyz) * 255
        storePly(ply_path, xyz, rgb)

        pcd = fetchPly(ply_path)

        return pcd


def compute_posed_mano_hand(body, minimal_shape, rot, pose, trans, w2c):
    # compute posed mano hand


    Jtr = body['Jtr'][0].detach().numpy()
    bone_transforms = body['bone_transforms'][0].detach().numpy()

    Jtr = (w2c[:3, :3] @ Jtr.T).T + w2c[:3, 3]

    # canonical SMPL vertices without pose correction, to normalize joints
    center = np.mean(minimal_shape, axis=0)
    minimal_shape_centered = minimal_shape - center
    cano_max = minimal_shape_centered.max()
    cano_min = minimal_shape_centered.min()
    padding = (cano_max - cano_min) * 0.05

    # compute pose condition
    Jtr_norm = Jtr - center
    Jtr_norm = (Jtr_norm - cano_min + padding) / (cano_max - cano_min) / 1.1
    Jtr_norm -= 0.5
    Jtr_norm *= 2.

    bone_transforms1 = np.repeat(np.eye(4)[np.newaxis, ...], 16, axis=0)
    bone_transforms = bone_transforms @ np.linalg.inv(bone_transforms1)
    bone_transforms = bone_transforms.astype(np.float32)
    bone_transforms[:, :3, 3] += trans

    pose6d = np.concatenate([rot, pose], axis=-1)
    pose6d = Rotation.from_rotvec(pose6d.reshape([-1, 3]))
    pose_mat_full = pose6d.as_matrix()
    pose_mat = pose_mat_full[1:, ...].copy()
    pose_rot = np.concatenate([np.expand_dims(np.eye(3), axis=0), pose_mat], axis=0).reshape(
        [-1, 9])

    bone_transforms = w2c @ bone_transforms

    return bone_transforms.astype(np.float32), Jtr, Jtr_norm, pose_rot


