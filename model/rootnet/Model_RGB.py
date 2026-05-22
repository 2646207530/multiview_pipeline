import numpy as np
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as standard
import time
import sys
sys.path.append('../../')
sys.path.append('../../lib/')
sys.path.append('../../software/')
from .convnext import convnext_base
from .vis_tool import draw_pose, draw_2d_skeleton
from .mano import MANO
from .preprocessing import generate_patch_image, uvd2xyz, process_bbox

# from config import self.cfg


class NormalVectorLoss(nn.Module):
    def __init__(self, face):
        super(NormalVectorLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt):
        face = torch.LongTensor(self.face).cuda()

        v1_out = coord_out[:, face[:, 1], :] - coord_out[:, face[:, 0], :]
        v1_out = F.normalize(v1_out, p=2, dim=2)  # L2 normalize to make unit vector
        v2_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 0], :]
        v2_out = F.normalize(v2_out, p=2, dim=2)  # L2 normalize to make unit vector
        v3_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 1], :]
        v3_out = F.normalize(v3_out, p=2, dim=2)  # L2 nroamlize to make unit vector

        v1_gt = coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 0], :]
        v1_gt = F.normalize(v1_gt, p=2, dim=2)  # L2 normalize to make unit vector
        v2_gt = coord_gt[:, face[:, 2], :] - coord_gt[:, face[:, 0], :]
        v2_gt = F.normalize(v2_gt, p=2, dim=2)  # L2 normalize to make unit vector
        normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
        normal_gt = F.normalize(normal_gt, p=2, dim=2)  # L2 normalize to make unit vector


        cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True))
        cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True))
        cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True))
        loss = torch.cat((cos1, cos2, cos3), 1)
        return loss

class EdgeLengthLoss(nn.Module):
    def __init__(self, face):
        super(EdgeLengthLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt):
        face = torch.LongTensor(self.face).cuda()

        d1_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 1], :] - coord_out[:, face[:, 2], :]) ** 2, 2, keepdim=True))

        d1_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))

        diff1 = torch.abs(d1_out - d1_gt)
        diff2 = torch.abs(d2_out - d2_gt)
        diff3 = torch.abs(d3_out - d3_gt)
        loss = torch.cat((diff1, diff2, diff3), 1)
        return loss


class SoftHeatmap(nn.Module):
    def __init__(self, size, kp_num):
        super(SoftHeatmap, self).__init__()
        self.size = size
        self.beta = nn.Conv2d(kp_num, kp_num, 1, 1, 0, groups=kp_num, bias=False)
        self.wx = torch.arange(0.0, 1.0 * self.size, 1).view([1, self.size]).repeat([self.size, 1])
        self.wy = torch.arange(0.0, 1.0 * self.size, 1).view([self.size, 1]).repeat([1, self.size])
        self.wx = nn.Parameter(self.wx, requires_grad=False)
        self.wy = nn.Parameter(self.wy, requires_grad=False)

    def forward(self, x):
        s = list(x.size())
        scoremap = self.beta(x)
        scoremap = scoremap.view([s[0], s[1], s[2] * s[3]])
        scoremap = F.softmax(scoremap, dim=2)
        scoremap = scoremap.view([s[0], s[1], s[2], s[3]])
        scoremap_x = scoremap.mul(self.wx)
        scoremap_x = scoremap_x.view([s[0], s[1], s[2] * s[3]])
        soft_argmax_x = torch.sum(scoremap_x, dim=2)
        scoremap_y = scoremap.mul(self.wy)
        scoremap_y = scoremap_y.view([s[0], s[1], s[2] * s[3]])
        soft_argmax_y = torch.sum(scoremap_y, dim=2)
        keypoint_uv = torch.stack([soft_argmax_x, soft_argmax_y], dim=2)
        return keypoint_uv, scoremap

class GraphConv(nn.Module):
    def __init__(self, num_joint, in_features, out_features):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.adj = nn.Parameter(torch.eye(num_joint).float(), requires_grad=True)

    def laplacian(self, A_hat):
        D_hat = torch.sum(A_hat, 1, keepdim=True) + 1e-5
        L = 1 / D_hat * A_hat
        return L

    def forward(self, x):
        batch = x.size(0)
        A_hat = self.laplacian(self.adj)
        A_hat = A_hat.unsqueeze(0).repeat(batch, 1, 1)
        out = self.fc(torch.matmul(A_hat, x))
        return out

class SAIGB(nn.Module):
    def __init__(self, backbone_channels, num_FMs, feature_size, num_vert, template):
        super(SAIGB, self).__init__()
        self.template = nn.Parameter(torch.Tensor(template), requires_grad=False)  # self.mano.template
        self.backbone_channels = backbone_channels
        self.feature_size = feature_size
        self.num_vert = num_vert
        self.num_FMs = num_FMs
        self.group = nn.Sequential(
            nn.Conv2d(self.backbone_channels, self.num_FMs * self.num_vert, 1),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        feature = self.group(x).view(-1, self.num_vert, self.feature_size * self.num_FMs)
        template = self.template.repeat(x.shape[0], 1, 1)
        init_graph = torch.cat((feature, template), dim=2)
        return init_graph

class GBBMR(nn.Module):
    def __init__(self, in_dim, num_vert, num_joint, heatmap_size):
        super(GBBMR, self).__init__()
        self.in_dim = in_dim
        self.num_vert = num_vert
        self.num_joint = num_joint
        self.num_total = num_vert + num_joint
        self.heatmap_size = heatmap_size
        self.soft_heatmap = SoftHeatmap(self.heatmap_size, self.num_total)
        self.reg_xy = nn.Sequential(
            GraphConv(self.num_vert, self.in_dim, self.heatmap_size ** 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            GraphConv(self.num_vert, self.heatmap_size ** 2, self.heatmap_size ** 2),
        )
        self.reg_z = nn.Sequential(
            GraphConv(self.num_vert, self.in_dim, self.heatmap_size ** 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            GraphConv(self.num_vert, self.heatmap_size ** 2, self.heatmap_size ** 2),
        )
        self.mesh2pose_hm = nn.Linear(self.num_vert, self.num_joint)
        self.mesh2pose_dm = nn.Linear(self.num_vert, self.num_joint)

    def forward(self, x):
        init_graph = x
        heatmap_xy_mesh = self.reg_xy(init_graph).view(-1, self.num_vert, self.heatmap_size, self.heatmap_size)
        heatmap_z_mesh = self.reg_z(init_graph).view(-1, self.num_vert, self.heatmap_size, self.heatmap_size)
        heatmap_xy_joint = self.mesh2pose_hm(heatmap_xy_mesh.transpose(1, 3)).transpose(1, 3)
        heatmap_z_joint = self.mesh2pose_dm(heatmap_z_mesh.transpose(1, 3)).transpose(1, 3)
        heatmap_xy = torch.cat((heatmap_xy_mesh, heatmap_xy_joint), dim=1)
        heatmap_z = torch.cat((heatmap_z_mesh, heatmap_z_joint), dim=1)
        coord_xy, latent_heatmaps = self.soft_heatmap(heatmap_xy)
        depth_maps = latent_heatmaps * heatmap_z
        coord_z = torch.sum(
            depth_maps.view(-1, self.num_total, depth_maps.shape[2] * depth_maps.shape[3]), dim=2, keepdim=True)
        joint_coord = torch.cat((coord_xy, coord_z), 2)
        joint_coord[:, :, :2] = joint_coord[:, :, :2] / (self.heatmap_size // 2) - 1
        return joint_coord, latent_heatmaps, depth_maps


class SARresnet34(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet34(pretrained=False)
        self.extract_mid = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu,
                                         backbone.maxpool, backbone.layer1, backbone.layer2)
        extract_high = []

        extract_high.append(nn.Sequential(backbone.layer3, backbone.layer4))
        self.extract_high = nn.ModuleList(extract_high)

    def forward(self, x):
        feat_mid = self.extract_mid(x)

        feat = feat_mid
        feat_high = self.extract_high[0](feat)
        
        return feat_high

class SARhead(nn.Module):
    def __init__(self, 
                 in_channels=1024, 
                 num_FMs=8, 
                 num_joints=21, 
                 num_verts=778, 
                 feature_size=64, 
                 heatmap_size=32, **kwargs):
        super().__init__()
        mano = MANO()
        self.feature_size = feature_size
        self.saigb = SAIGB(in_channels, num_FMs, self.feature_size, num_verts, mano.template)
        self.gbbmr = GBBMR(num_FMs*self.feature_size+3, num_verts, num_joints, heatmap_size)

    def forward(self, x, target=None):
        '''x: (B,C,Hp,Wp)'''
        B = x.shape[0]
        features = x

        init_graph = self.saigb(features)
        coord, lhm, dm = self.gbbmr(init_graph)

        return coord
    

class SAR(nn.Module):
    def __init__(self, backbone='convnext', **kwargs):
        super().__init__()
        if backbone == 'convnext':
            self.backbone = convnext_base(pretrained=False, in_22k=True, num_classes=21841)
        elif backbone == 'resnet34':
            self.backbone = SARresnet34()
        else:
            raise NotImplementedError()
        self.head = SARhead(**kwargs)

    def forward(self, x):
        features = self.backbone(x)
        results = self.head(features)
        return results


class ResRootNet(nn.Module):
    def __init__(self, inplanes=512, outplanes=256):
        self.inplanes = inplanes
        self.outplanes = outplanes

        super().__init__()
        # self.deconv_layers = self._make_deconv_layer(3)
        # self.xy_layer = nn.Conv2d(
        #     in_channels=self.outplanes,
        #     out_channels=1,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0
        # )
        self.depth_layer = nn.Conv2d(
            in_channels=self.inplanes,
            out_channels=1, 
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward_coord(self, x, k_value):
        #  # x,y
        # xy = self.deconv_layers(x)
        # xy = self.xy_layer(xy)
        # xy = xy.view(-1,1,cfg.output_shape[0]*cfg.output_shape[1])
        # xy = F.softmax(xy,2)
        # xy = xy.view(-1,1,cfg.output_shape[0],cfg.output_shape[1])

        # hm_x = xy.sum(dim=(2))
        # hm_y = xy.sum(dim=(3))

        # coord_x = hm_x * torch.arange(cfg.output_shape[1]).float().cuda()
        # coord_y = hm_y * torch.arange(cfg.output_shape[0]).float().cuda()
        
        # coord_x = coord_x.sum(dim=2)
        # coord_y = coord_y.sum(dim=2)

        # z
        img_feat = torch.mean(x.view(x.size(0), x.size(1), x.size(2)*x.size(3)), dim=2) # global average pooling
        img_feat = torch.unsqueeze(img_feat,2); img_feat = torch.unsqueeze(img_feat,3)
        gamma = self.depth_layer(img_feat)
        gamma = gamma.view(-1,1)
        depth = gamma * k_value.view(-1,1)

        # coord = torch.cat((coord_x, coord_y, depth), dim=1)
        return depth

    def forward(self, x, k_value, target=None):
        coord = self.forward_coord(x, k_value)

        if target is None:
            return coord
        else:
            target_coord = target['root_img']
            
            ## coordrinate loss
            loss_coord = F.smooth_l1_loss(coord, target_coord[:,None])
            # loss_coord = torch.abs(coord - target_coord)
            # loss_coord = (loss_coord[:,0] + loss_coord[:,1] + loss_coord[:,2])/3.
            return loss_coord


# class EstimateRGB(object):
class EstimateRGB(nn.Module):
    def __init__(self, cfg):
        super(EstimateRGB, self).__init__()
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = standard.Compose([standard.ToTensor(), standard.Normalize(*mean_std)])
        self.mode='estimate'
        self.cfg = cfg
        # self.cam_paras = cam_paras
        # mano
        self.mano = MANO()
        self.face = self.mano.face
        self._make_model()

    def _make_model(self):
        model = SAR(self.cfg.backbone, in_channels=self.cfg.in_channels).to(self.cfg.device)
        checkpoint = torch.load(self.cfg.checkpoint)
        if 'net' in checkpoint:
            model.load_state_dict(checkpoint['net'])
        else:
            model.load_state_dict(checkpoint['network'], strict=False)
        model.eval()
        self.model = model
        
        if 'rootnet' in checkpoint:
            rootnet = ResRootNet(inplanes=self.cfg.in_channels).to(self.cfg.device)
            rootnet.load_state_dict(checkpoint['rootnet'])
            rootnet.eval()
            self.rootnet = rootnet
            print('钩子函数在进行')
            self.cnn_feats = []
            def layer_hook(module, in_, out_):
                self.cnn_feats.append(out_.detach())
            self.model.backbone.register_forward_hook(layer_hook)
        else:
            self.rootnet = None
            print('钩子函数不在进行')
            
    def export_onnx(self):
        import onnx
        import onnxruntime as ort
        print('export begin')
        model = SAR(self.cfg.backbone, in_channels=self.cfg.in_channels).to(self.cfg.device)
        checkpoint = torch.load(self.cfg.checkpoint)
        if 'net' in checkpoint:
            model.load_state_dict(checkpoint['net'])
        else:
            model.load_state_dict(checkpoint['network'], strict=False)
        model.eval()
        
        dummy_input = torch.randn(1, 3, 256, 256).to(self.cfg.device)
        model_path = "/home/cyc/pycharm/vGesture/lib/core/sar_output/sar_model.onnx"
        # 导出模型为 ONNX 格式
        torch.onnx.export(
            model,
            dummy_input,
            model_path,
            verbose=False,
            opset_version=12,  # 可以根据需要调整
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},  # 使批量大小可变
                'output': {0: 'batch_size'}
            }
        )
        onnx_model = onnx.load(model_path)

        # 检查模型
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid")
        # 创建 ONNX Runtime 会话
        ort_session = ort.InferenceSession(model_path)

        # 准备虚拟输入
        dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)

        # 运行推理
        outputs = ort_session.run(None, {'input': dummy_input})
        torch_dummy_input = torch.from_numpy(dummy_input).to(self.cfg.device)
        sar_outputs = self.model(torch_dummy_input)
        print("ONNX Runtime 输出：", outputs)
        print("SAR Runtime 输出：", sar_outputs)
        return outputs
    
    def sar_trt(self,img):
        import numpy as np
        import tensorrt as trt
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        # 加载 TensorRT 引擎
        trt_path = '/home/cyc/pycharm/vGesture/lib/core/sar_output/sar_model.trt'
        with open(trt_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        # 创建执行上下文
        context = engine.create_execution_context()
        # 准备输入数据
        # 检查并准备输入数据
        # 确保输入 img 的大小是 [1, 3, 256, 256]
        assert img.shape == (1, 3, 256, 256), f"Expected input size [1, 3, 256, 256], but got {img.shape}"
        torch_dummy_input = img.to(self.cfg.device)
        # 创建输出张量（SAR模型的输出形状可以是变化的，这里假设为[1, 799, 3]）
        output_shape = (1, 799, 3)  # 确保形状与实际的推理输出一致
        output_tensor = torch.empty(output_shape).to(self.cfg.device)  # 分配输出张量
        
        # 将输入和输出的内存指针绑定到 TensorRT
        bindings = [int(torch_dummy_input.data_ptr()), int(output_tensor.data_ptr())]
        
        t0 = time.time()
        # 执行推理
        context.execute_v2(bindings)
        print('trt inference time:',time.time()-t0)
        
        # 推理后的输出已经存储在 output_tensor 中
        trt_output_data = output_tensor # 将输出从 GPU 移动到 CPU 并转换为 NumPy 数组
        print("TRT inference result:", trt_output_data)
        # 通过 PyTorch 模型获取输出进行对比
        t1 = time.time()
        sar_output = self.model(torch_dummy_input)
        print('SAR inference time:',time.time()-t1)
        print('SAR inference result:', sar_output)

        return trt_output_data

    def post_processing(self, outs, meta_info, img_width, do_flip=False):
        # print('sar out shape:',type(outs))
        for key, value in outs.items():
            print(f"Key: {key}, Value: {value}")

        crop_img = meta_info['crop_img']
        coords_uvd = outs['coords']
        batch = coords_uvd.shape[0]
        eval_result = {'pose_uvd': list(), 'mesh_uvd': list(), 'pose_xyz': list(), 'mesh_xyz': list()}
        for i in range(batch):
            coord_uvd_crop, root_depth,  bb2img_trans, img2bb_trans, K = \
                coords_uvd[i], meta_info['root_depth'][i], meta_info['bb2img_trans'][i], meta_info['img2bb_trans'][i], meta_info['K'][i]
            coord_uvd_crop[:, 2] = coord_uvd_crop[:, 2] * self.cfg.depth_box + root_depth
            # coord_uvd_crop[:, :2] = (coord_uvd_crop[:, :2] + 1) * (self.cfg.input_img_shape[0] // 2)
            coord_uvd_crop[:, :2] = (coord_uvd_crop[:, :2] + 0.5) * self.cfg.input_img_shape[0]
            # back to original image
            coord_uvd_full = coord_uvd_crop.copy()
            uv1 = np.concatenate((coord_uvd_full[:, :2], np.ones_like(coord_uvd_full[:, :1])), 1)
            coord_uvd_full[:, :2] = np.dot(bb2img_trans, uv1.transpose(1, 0)).transpose(1, 0)[:, :2]
            if do_flip:
                coord_uvd_full[:, 0] = img_width - coord_uvd_full[:,0] - 1
            eval_result['pose_uvd'].append(coord_uvd_full[self.cfg.num_vert:])
            eval_result['mesh_uvd'].append(coord_uvd_full[:self.cfg.num_vert])

            coord_xyz = uvd2xyz(coord_uvd_full, K)
            pose_xyz = coord_xyz[self.cfg.num_vert:]
            center_xyz = np.mean(pose_xyz, axis=0, keepdims=True)
            mesh_xyz = coord_xyz[:self.cfg.num_vert]
            eval_result['pose_xyz'].append(pose_xyz)
            eval_result['mesh_xyz'].append(mesh_xyz)

            #vis
            crop_img = crop_img[0]
            print('sar 2d pose:',coord_uvd_crop[self.cfg.num_vert:, :2])
            pose_img = draw_2d_skeleton(crop_img, coord_uvd_crop[self.cfg.num_vert:, :2])
            pose_img_full = draw_2d_skeleton(meta_info['ori_img'], coord_uvd_full[self.cfg.num_vert:, :2])
            M = torch.cat((torch.tensor(meta_info['img2bb_trans'][0]), torch.tensor([[0, 0, 1]])), dim=0).unsqueeze(0)
            # print(M)

            meta_info_output = {
                'crop_img_rgb': crop_img,
                'crop_img_d': None,
                'pose_img_rgb': pose_img,
                'pose_img_d': None,
                'joint_xyz_world': pose_xyz,
                'cam_para': self.cfg.cam_para,
                'center': torch.from_numpy(center_xyz),
                'cube': self.cfg.depth_box * 1000,
                'M': M,
                'img2bb_trans': meta_info['img2bb_trans'][0]
            }

        return eval_result, meta_info_output
    
    def convert2origin_pixel(self, uvd, inv_trans):
        '''
        uvd: B,J,3 joints uvd coord
        trans: B,2,3 img2bbox trans
        '''
        uv = (uvd[:,:,:2] + 0.5) * self.cfg.input_img_shape[1]

        uv1 = torch.cat((uv[:, :, :2], torch.ones_like(uvd[:, :, :1])), dim=2)
        uv = (inv_trans @ uv1.transpose(-1, -2)).transpose(-1, -2)

        return uv

    def calculate_k(self, bbox, fx, fy):
        area = bbox[-1] * bbox[-2]
        real_area = torch.tensor(self.cfg.bbox_real[0] * self.cfg.bbox_real[1])

        return torch.sqrt(real_area*fx*fy/(area)).unsqueeze(0).to(self.cfg.device)

    def run(self,input):
        input = input[0]

        img_rgb, bbox, hand_type = input['rgb'], input['rgb_bbox'], input['hand_type']
        do_flip = hand_type == "left"
        
        if 'depth' in input:
            depth_img = input['depth']
        else:
            depth_img = None

        x1, y1, x2, y2 = bbox

        w = x2 - x1
        h = y2 - y1 
        bbox_ = [x1, y1, w, h]

        [fx, fy, fu, fv] = self.cfg.cam_para
        K = np.array([[fx,0,fu],[0,fy,fv],[0,0,1]])

        height, width = img_rgb.shape[:-1]
        bbox_ = process_bbox(bbox_, width, height, self.cfg.input_img_shape, 1.5)

        img, img2bb_trans, bb2img_trans = \
            generate_patch_image(cvimg=img_rgb,
                                 bbox=bbox_, scale=1.0, rot=0.0,
                                 do_flip=do_flip,
                                 out_shape=self.cfg.input_img_shape)
        imgcrop = img.astype(np.uint8).copy()

        img = img[:,:,::-1].astype(np.uint8).copy()# BGR -> RGB
        img = self.transform(img).unsqueeze(0).to(self.cfg.device)

        pred_root = [0.]
        with torch.no_grad():
            self.cnn_feats = []
            outs = self.model(img)
            # outs = self.sar_trt(img)
            if depth_img is not None:
                depth_img = torch.from_numpy(depth_img.astype(np.float32) / 1000.)[None, None]
                root_uvd = self.convert2origin_pixel(outs[:,778:779].cpu(), torch.from_numpy(bb2img_trans))

                root_2D = root_uvd[:,:,:2].clone()
                root_2D[:,:,0] = root_2D[:,:,0] / (width// 2) - 1
                root_2D[:,:,1] = root_2D[:,:,1] / (height // 2) - 1

                pred_root = torch.nn.functional.grid_sample(depth_img, root_2D[:,None])[:,0,0]

            elif K is not None and self.rootnet is not None:
                feats = self.cnn_feats[-1]
                k_value = self.calculate_k(bbox_, fx, fy)
                pred_root = self.rootnet(feats, k_value)

        meta_info_vis = {
            'ori_img': input['rgb'].copy(),
            'crop_img': [imgcrop],
            'img2bb_trans':[np.float32(img2bb_trans)],
            'bb2img_trans': [np.float32(bb2img_trans)],
            'root_depth': np.float32(pred_root),
            'K':[K],
            'scale':[1.]
        }

        outs = {'coords': outs.cpu().numpy()}
        meta_info_vis = {k: v for k, v in meta_info_vis.items()}
        output, meta_info_output = self.post_processing(outs, meta_info_vis, width, do_flip)

        # meta_info_output = {k: v[0] for k,v in meta_info_output.items()}
        output = {k: v[0] for k,v in output.items()}

        return meta_info_output, output

    def estimate_root_depth_custom(self, img, K, bbox):
        """
        独立估算根节点深度的方法
        Args:
            img: 原始图像 (numpy array, HxWx3, BGR format usually from cv2)
            K: 相机内参矩阵 (3x3 numpy array or list)
            bbox: 手部边界框 [x1, y1, x2, y2]

        Returns:
            root_depth: 预测的根节点绝对深度 (float, 单位通常是米或毫米，取决于训练配置)
        """
        # 1. 检查是否加载了 RootNet
        if self.rootnet is None:
            raise RuntimeError("RootNet is not loaded in the checkpoint!")

        # 2. 解析 BBox 并处理 (Padding 等)
        # 假设输入 bbox 是 [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        bbox_xywh = [x1, y1, w, h] # 转换为 [x, y, w, h]

        height, width = img.shape[:2]
        # 使用原代码中的 process_bbox 进行处理 (扩展宽高比等)
        bbox_processed = process_bbox(bbox_xywh, width, height, self.cfg.input_img_shape, 1.5)

        # 3. 图像裁剪与预处理
        # 注意：这里假设输入 img 是 BGR (Opencv 读取)，需要转 RGB
        # generate_patch_image 会返回裁剪后的 patch, 以及变换矩阵
        patch, _, _ = generate_patch_image(
            cvimg=img,
            bbox=bbox_processed,
            scale=1.0,
            rot=0.0,
            do_flip=False,
            out_shape=self.cfg.input_img_shape
        )
        
        # 转换颜色空间 BGR -> RGB，归一化，转 Tensor
        patch = patch[:, :, ::-1].astype(np.uint8).copy() 
        patch_tensor = self.transform(patch).unsqueeze(0).to(self.cfg.device)

        # 4. 运行 Backbone 提取特征
        # RootNet 依赖于 Backbone 的中间特征，这些特征通过 hook (self.cnn_feats) 获取
        self.cnn_feats = [] # 清空之前的特征
        with torch.no_grad():
            # 只需要运行 backbone 即可触发 hook，不需要运行整个 head
            _ = self.model.backbone(patch_tensor)
            
            # 获取最后一层特征，根据原代码 run() 函数逻辑: feats = self.cnn_feats[-1]
            feats = self.cnn_feats[-1]

        # 5. 计算 k_value
        # K 矩阵通常是 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        if isinstance(K, np.ndarray):
            fx = K[0, 0]
            fy = K[1, 1]
        else:
            fx, fy = K[0][0], K[1][1]
            
        k_value = self.calculate_k(bbox_processed, fx, fy)

        # 6. 预测深度
        with torch.no_grad():
            pred_depth = self.rootnet(feats, k_value)

        # 返回标量深度值
        return pred_depth.item()

def get_model():
    from .sar_config_stage_1 import rgb_opt
    return EstimateRGB(rgb_opt)

if __name__ == '__main__':
    import torch
    input = {'img':torch.rand(1, 3, 256, 256)}
    net = SAR()
    output = net(input)
    print(output)



