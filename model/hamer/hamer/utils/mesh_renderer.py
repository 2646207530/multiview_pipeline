import os
import time
from line_profiler import profile
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh
import cv2
import torch.nn.functional as F

from .render_openpose import render_openpose

def create_raymond_lights():
    import pyrender
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes

# class MeshRenderer:

#     def __init__(self, cfg, faces=None):
#         self.cfg = cfg
#         self.focal_length = cfg.EXTRA.FOCAL_LENGTH
#         self.img_res = cfg.MODEL.IMAGE_SIZE
#         self.renderer = pyrender.OffscreenRenderer(viewport_width=self.img_res,
#                                        viewport_height=self.img_res,
#                                        point_size=1.0) # FIXME: 这里的渲染器没有在__call__被使用
        
#         self.camera_center = [self.img_res // 2, self.img_res // 2]
#         self.faces = faces

    # def visualize(self, vertices, camera_translation, images, focal_length=None, nrow=3, padding=2):
    #     images_np = np.transpose(images, (0,2,3,1))
    #     rend_imgs = []
    #     for i in range(vertices.shape[0]):
    #         fl = self.focal_length
    #         rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=False), (2,0,1))).float()
    #         rend_img_side = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=True), (2,0,1))).float()
    #         rend_imgs.append(torch.from_numpy(images[i]))
    #         rend_imgs.append(rend_img)
    #         rend_imgs.append(rend_img_side)
    #     rend_imgs = make_grid(rend_imgs, nrow=nrow, padding=padding)
    #     return rend_imgs

    # def visualize_tensorboard(self, vertices, camera_translation, images, pred_keypoints, gt_keypoints, focal_length=None, nrow=5, padding=2):
    #     images_np = np.transpose(images, (0,2,3,1))
    #     rend_imgs = []
    #     pred_keypoints = np.concatenate((pred_keypoints, np.ones_like(pred_keypoints)[:, :, [0]]), axis=-1)
    #     pred_keypoints = self.img_res * (pred_keypoints + 0.5)
    #     gt_keypoints[:, :, :-1] = self.img_res * (gt_keypoints[:, :, :-1] + 0.5)
    #     #keypoint_matches = [(1, 12), (2, 8), (3, 7), (4, 6), (5, 9), (6, 10), (7, 11), (8, 14), (9, 2), (10, 1), (11, 0), (12, 3), (13, 4), (14, 5)]
    #     for i in range(vertices.shape[0]):
    #         fl = self.focal_length
    #         rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=False), (2,0,1))).float()
    #         rend_img_side = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=True), (2,0,1))).float()
    #         hand_keypoints = pred_keypoints[i, :21]
    #         #extra_keypoints = pred_keypoints[i, -19:]
    #         #for pair in keypoint_matches:
    #         #    hand_keypoints[pair[0], :] = extra_keypoints[pair[1], :]
    #         pred_keypoints_img = render_openpose(255 * images_np[i].copy(), hand_keypoints) / 255
    #         hand_keypoints = gt_keypoints[i, :21]
    #         #extra_keypoints = gt_keypoints[i, -19:]
    #         #for pair in keypoint_matches:
    #         #    if extra_keypoints[pair[1], -1] > 0 and hand_keypoints[pair[0], -1] == 0:
    #         #        hand_keypoints[pair[0], :] = extra_keypoints[pair[1], :]
    #         gt_keypoints_img = render_openpose(255*images_np[i].copy(), hand_keypoints) / 255
    #         rend_imgs.append(torch.from_numpy(images[i]))
    #         rend_imgs.append(rend_img)
    #         rend_imgs.append(rend_img_side)
    #         rend_imgs.append(torch.from_numpy(pred_keypoints_img).permute(2,0,1))
    #         rend_imgs.append(torch.from_numpy(gt_keypoints_img).permute(2,0,1))
    #     rend_imgs = make_grid(rend_imgs, nrow=nrow, padding=padding)
    #     return rend_imgs

#     @profile
#     def __call__(self, vertices, camera_translation, image, focal_length=5000, text=None, resize=None, side_view=False, baseColorFactor=(1.0, 1.0, 0.9, 1.0), rot_angle=90):
#         renderer = pyrender.OffscreenRenderer(viewport_width=image.shape[1],
#                                               viewport_height=image.shape[0],
#                                               point_size=1.0)
#         material = pyrender.MetallicRoughnessMaterial(
#             metallicFactor=0.0,
#             alphaMode='OPAQUE',
#             baseColorFactor=baseColorFactor)

#         camera_translation[0] *= -1.
#         mesh = trimesh.Trimesh(vertices.detach().cpu().numpy(), 
#                        self.faces.detach().cpu().numpy() if isinstance(self.faces, torch.Tensor) else self.faces)
#         # mesh: 一个 trimesh.Trimesh 对象
#         # mesh.export('/home/pt/vGesture/software/hamer/test_hand_mesh.obj')  # 导出为 obj 文件 # FIXME: debug用的吗

#         # mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
#         # if side_view:
#         #     rot = trimesh.transformations.rotation_matrix(
#         #         np.radians(rot_angle), [0, 1, 0])
#         #     mesh.apply_transform(rot)
#         rot = trimesh.transformations.rotation_matrix(
#             np.radians(180), [1, 0, 0])
#         mesh.apply_transform(rot)
#         mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

#         scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
#                                ambient_light=(0.3, 0.3, 0.3))
#         scene.add(mesh, 'mesh')

#         camera_pose = np.eye(4)
#         camera_pose[:3, 3] = camera_translation.cpu().numpy()
#         camera_center = [image.shape[1] / 2., image.shape[0] / 2.]
#         # 提取标量值
#         # print('focal_length:', focal_length)
#         fx = focal_length[0].item()  # 提取第一个值
#         fy = focal_length[0].item()  # 提取第二个值
#         camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy,
#                                            cx=camera_center[0], cy=camera_center[1])
#         # print('fx, fy, cx, cy', fx, fy, camera_center[0], camera_center[1])
#         scene.add(camera, pose=camera_pose)


#         light_nodes = create_raymond_lights()
#         for node in light_nodes:
#             scene.add_node(node)

#         color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
#         color = color.astype(np.float32) / 255.0
#         valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
#         if not side_view:
#             output_img = (color[:, :, :3] * valid_mask +
#                       (1 - valid_mask) * image)
#         else:
#             output_img = color[:, :, :3]
#         if resize is not None:
#             output_img = cv2.resize(output_img, resize)

#         output_img = output_img.astype(np.float32)
#         # renderer.delete()
#         return output_img
    

class MeshRenderer:
    def __init__(self, cfg, faces=None):
        self.cfg = cfg
        self.focal_length = cfg.EXTRA.FOCAL_LENGTH
        self.img_res = cfg.MODEL.IMAGE_SIZE
        
        # 预先创建渲染器，避免每次调用都重新创建
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.img_res,
            viewport_height=self.img_res,
            point_size=1.0
        )
        
        # 预先创建常用对象
        self.camera_center = [self.img_res // 2, self.img_res // 2]
        self.faces = faces
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                   ambient_light=(0.3, 0.3, 0.3))
        self.light_nodes = create_raymond_lights()
        
        # 缓存常用的材质
        self.default_material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0)
        )

    def visualize(self, vertices, camera_translation, images, focal_length=None, nrow=3, padding=2):
        images_np = np.transpose(images, (0,2,3,1))
        rend_imgs = []
        
        # 预先计算所有需要的渲染
        for i in range(vertices.shape[0]):
            fl = self.focal_length
            rend_img = self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=False)
            rend_img_side = self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=True)
            
            # 直接在GPU上进行转置操作
            rend_img = torch.from_numpy(rend_img.transpose(2,0,1)).float()
            rend_img_side = torch.from_numpy(rend_img_side.transpose(2,0,1)).float()
            
            rend_imgs.append(torch.from_numpy(images[i]))
            rend_imgs.append(rend_img)
            rend_imgs.append(rend_img_side)
            
        rend_imgs = make_grid(rend_imgs, nrow=nrow, padding=padding)
        return rend_imgs

    def visualize_tensorboard(self, vertices, camera_translation, images, pred_keypoints, gt_keypoints, focal_length=None, nrow=5, padding=2):
        images_np = np.transpose(images, (0,2,3,1))
        rend_imgs = []
        
        # 预处理关键点数据
        pred_keypoints = np.concatenate((pred_keypoints, np.ones_like(pred_keypoints)[:, :, [0]]), axis=-1)
        pred_keypoints = self.img_res * (pred_keypoints + 0.5)
        gt_keypoints[:, :, :-1] = self.img_res * (gt_keypoints[:, :, :-1] + 0.5)

        for i in range(vertices.shape[0]):
            fl = self.focal_length
            rend_img = self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=False)
            rend_img_side = self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=True)
            
            # 直接在GPU上进行转置操作
            rend_img = torch.from_numpy(rend_img.transpose(2,0,1)).float()
            rend_img_side = torch.from_numpy(rend_img_side.transpose(2,0,1)).float()
            
            # 处理关键点图像
            hand_keypoints = pred_keypoints[i, :21]
            pred_keypoints_img = render_openpose(255 * images_np[i].copy(), hand_keypoints) / 255
            hand_keypoints = gt_keypoints[i, :21]
            gt_keypoints_img = render_openpose(255*images_np[i].copy(), hand_keypoints) / 255
            
            rend_imgs.append(torch.from_numpy(images[i]))
            rend_imgs.append(rend_img)
            rend_imgs.append(rend_img_side)
            rend_imgs.append(torch.from_numpy(pred_keypoints_img).permute(2,0,1))
            rend_imgs.append(torch.from_numpy(gt_keypoints_img).permute(2,0,1))
            
        rend_imgs = make_grid(rend_imgs, nrow=nrow, padding=padding)
        return rend_imgs

    def __call__(self, vertices, camera_translation, image, focal_length=5000, text=None, resize=None, side_view=False, 
                    baseColorFactor=(1.0, 1.0, 0.9, 1.0), rot_angle=90, trans=None, do_flip=None, inv_trans=None):
            
            # 1. 基础设置
            renderer = self.renderer
            renderer.viewport_width = image.shape[1]
            renderer.viewport_height = image.shape[0]
            
            # 2. 材质
            material = self.default_material if baseColorFactor == (1.0, 1.0, 0.9, 1.0) else pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode='OPAQUE',
                baseColorFactor=baseColorFactor)

            # 3. 相机位姿处理
            # 【重要修复】不要随意翻转 X 轴，除非你非常确定坐标系差异。
            # 通常 HaMeR 的 pred_cam_t_full 已经是正确的方向。
            # camera_translation[0] *= -1.  <-- 删掉或注释这行
            
            # 4. 创建 Mesh
            vertices_np = vertices.detach().cpu().numpy()
            faces_np = self.faces.detach().cpu().numpy() if isinstance(self.faces, torch.Tensor) else self.faces
            mesh = trimesh.Trimesh(vertices_np, faces_np)

            # 应用旋转 (OpneCV +Y down -> OpenGL +Y up)
            rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot)
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

            # 5. 构建场景
            self.scene.clear()
            self.scene.add(mesh, 'mesh')

            # 6. 设置相机
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = camera_translation.cpu().numpy()
            
            camera_center = [image.shape[1] / 2., image.shape[0] / 2.]
            
            # 处理焦距
            if isinstance(focal_length, torch.Tensor):
                fx = focal_length.flatten()[0].item() # 更加鲁棒的写法
                fy = fx
            else:
                fx = fy = focal_length
            
            camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=camera_center[0], cy=camera_center[1])
            self.scene.add(camera, pose=camera_pose)

            # 7. 【重要保底】显式添加跟随相机的光照，防止全黑
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
            self.scene.add(light, pose=camera_pose)

            # 8. 渲染
            color, rend_depth = renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)
            
            # 9. 【关键修复】正确处理左手翻转和像素类型
            # 注意：这里我们保持 float32 [0,1] 返回，但在 main 函数里必须转回 uint8
            color = color.astype(np.float32) / 255.0
            
            # 如果是左手且提供了变换矩阵，处理翻转逻辑
            # 注意：do_flip 是 tensor，要取值
            is_flipped = (do_flip is not None) and (float(do_flip) == 1.0)
            
            if is_flipped and trans is not None and inv_trans is not None:
                # 这里的逻辑比较绕：如果渲染是在全图做的，其实不需要 warp。
                # 但如果你的逻辑是先渲染再翻转像素，请确保逻辑正确。
                # 假设你是想翻转图像内容（因为左手预测是基于翻转图的右手）：
                
                # 方法 A: 简单的图像水平翻转 (如果渲染也是镜像的)
                # color = cv2.flip(color, 1) 
                
                # 方法 B: 你的原始逻辑 (似乎是把crop区域扣出来翻转再放回去？)
                # 这段逻辑风险很大，建议先注释，除非确定需要像素级warp
                pass 

            # 返回 [0.0 - 1.0] 的 float 图像
            return color
    
    def any_hand(self, vertices, camera_pose, M):
        viewport_width = 1920
        viewport_height = 1080
        
        renderer = self.renderer
        renderer.viewport_width = 1920
        renderer.viewport_height = 1080
        
        # 创建简单的测试材质
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.8, 0.8, 0.8, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0
        )
        
        # 创建网格
        vertices_np = vertices.reshape(-1, 3)
        faces_np = self.faces.detach().cpu().numpy() if isinstance(self.faces, torch.Tensor) else self.faces
        
        print(f"Vertices shape: {vertices_np.shape}")
        print(f"Faces shape: {faces_np.shape}")
        print(f"Vertices range: [{vertices_np.min()}, {vertices_np.max()}]")
        
        mesh = trimesh.Trimesh(vertices_np, faces_np)
        
        # 应用变换
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        
        # 计算边界框和中心
        bbox = mesh.bounds
        center = mesh.centroid
        print(f"Bounding box: {bbox}")
        print(f"Center: {center}")
        
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        
        # 创建场景
        self.scene.clear()
        self.scene.add(mesh)
        
        # 设置相机
        fx = M[0, 0]
        fy = M[1, 1]
        cx = M[0, 2]
        cy = M[1, 2]
        
        print(f"Camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        
        camera = pyrender.IntrinsicsCamera(
            fx=fx, fy=fy, cx=cx, cy=cy,
            znear=0.1, zfar=2000.0
        )
        
        # 如果相机姿态有问题，使用默认视角
        if camera_pose is None or np.allclose(camera_pose, np.eye(4)):
            print("Using default camera pose")
            # 将相机放在模型前方
            camera_pose = np.eye(4)
            camera_pose[2, 3] = center[2] + 500  # 在Z方向偏移
        
        print(f"Camera pose:\n{camera_pose}")
        
        self.scene.add(camera, pose=camera_pose)
        
        # 添加光照
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        self.scene.add(light, pose=camera_pose)
        
        # 渲染

        color, depth = renderer.render(self.scene)
        print(f"Render result - color range: [{color.min()}, {color.max()}]")
        print(f"Render result - depth range: [{depth.min()}, {depth.max()}]")
        
        if color.max() == 0:
            print("Warning: Rendered image is completely black!")
            
        color = color.astype(np.float32) / 255.0
        return color

    def get_mesh(self, vertices):
        """
        根据传入的顶点返回 trimesh 对象
        Args:
            vertices: (N, 3) 或 (1, N, 3) 的 tensor 或 numpy 数组
        Returns:
            trimesh.Trimesh 对象
        """
        # 1. 处理顶点数据转为 Numpy
        if isinstance(vertices, torch.Tensor):
            vertices_np = vertices.detach().cpu().numpy()
        else:
            vertices_np = vertices

        # 2. 处理 Batch 维度: 如果是 (1, 778, 3) 变成 (778, 3)
        if vertices_np.ndim == 3:
            vertices_np = vertices_np[0]

        # 3. 处理面数据转为 Numpy
        faces_np = self.faces
        if isinstance(faces_np, torch.Tensor):
            faces_np = faces_np.detach().cpu().numpy()

        # 4. 创建并返回 Trimesh 对象
        mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)
        
        # 可选：如果需要和渲染时的坐标系一致（你的代码里有个绕X轴翻转180度），取消下面注释
        # rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        # mesh.apply_transform(rot)

        return mesh