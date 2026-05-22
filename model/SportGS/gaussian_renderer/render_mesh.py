
import os
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    BlendParams,
)
from pytorch3d.structures.meshes import Meshes
import numpy as np
import torch
import torch.nn.functional as F
from utils.general_utils import nearest_points
from utils.loss_utils import FCLoss, compute_contact_loss, compute_contact_sdf_grid_loss, mano_self_collision_loss
from utils.contact_config import get_attract_tips, get_hand_hand_contacts
from pytorch3d.ops import knn_points



def render_mesh(data, 
           iteration,
           scene,
           body_model_r, 
           body_model_l,
           mano_finger_labels,
           vis=False,
           finetune=False
           ):
    """
    Render the mesh. 
    """
    # get hand-object posed meshes
    obj_points = torch.from_numpy(scene.metadata_obj[data.obj_id]['obj_points']).float().cuda()
    obj_faces = torch.from_numpy(scene.metadata_obj[data.obj_id]['faces']).float().cuda()

    n_pts = obj_points.shape[0]
   
    rot = data.obj_rots.to(obj_points.device)
    trans = data.obj_trans.to(obj_points.device)

    # print(rot.requires_grad) # true
    # print(trans.requires_grad) # true
    obj_points_posed = (rot @ obj_points.T).T + trans
   
    # T_fwd = torch.eye(4).to(rot.device)
    # T_fwd[:3, :3] = rot.float().to(rot.device)
    # T_fwd[:3, 3] = trans
    # T_fwd = T_fwd.repeat(n_pts, 1, 1)

    # print(T_fwd.requires_grad) # true

    # homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=obj_points.device)
    # x_hat_homo = torch.cat([obj_points, homo_coord], dim=-1).view(n_pts, 4, 1)
    # obj_points_posed = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]

    hand_param_l = data.hand_param_l.float().reshape(-1, 3+45+10+3)
    hand_param_r = data.hand_param_r.float().reshape(-1, 3+45+10+3)
    rot_l, pose_l, betas_l, trans_l = hand_param_l[:,:3],hand_param_l[:,3:45+3],hand_param_l[:,45+3:45+3+10],hand_param_l[:,45+3+10:]
    rot_r, pose_r, betas_r, trans_r = hand_param_r[:,:3],hand_param_r[:,3:45+3],hand_param_r[:,45+3:45+3+10],hand_param_r[:,45+3+10:]
    
    body_l = body_model_l(global_orient=rot_l.float().reshape(-1, 3),
                                   hand_pose=pose_l.float().reshape(-1, 45),
                                   betas=betas_l.float().reshape(-1, 10),
                                   #transl=trans_l.float().reshape(-1, 3),
                                   )
    body_r = body_model_r(global_orient=rot_r.float().reshape(-1, 3),
                                   hand_pose=pose_r.float().reshape(-1, 45),
                                   betas=betas_r.float().reshape(-1, 10),
                                   transl=trans_r.float().reshape(-1, 3),
                                   )
    
    def right_hand_to_left(body_l_can, trans_l, body_model_l):
        body_l_can['v'][0][:, 0] = body_l_can['v'][0][:, 0] * -1
        points_l = body_l_can['v'][0]+trans_l.float().reshape( 3)
        faces_l = body_model_l.faces[:, [0, 2, 1]]
        return points_l, faces_l
    points_l, faces_l = right_hand_to_left(body_l, trans_l, body_model_l)
   
    faces = torch.cat([torch.from_numpy(body_model_r.faces.astype(np.int64)).to(obj_points.device), 
                       torch.from_numpy(faces_l.astype(np.int64)).to(obj_points.device), obj_faces], dim=0)
    points = torch.cat([body_r['v'][0], 
                        points_l, 
                        obj_points_posed
                        ], dim=0)
    # 计算每个模型的点云数量
    num_body_r_points = body_r['v_shaped'][0].shape[0]
    num_body_l_points = body_l['v_a_pose'][0].shape[0]
    num_obj_points = obj_points_posed.shape[0]

    # 偏移量：左手和右手的面片索引需要加上相应的点云数量
    # 右手的面片索引需要加上左手的点云数量，再加上右手的点云数量（即每个面片的点云索引偏移）
    body_model_r_faces_offset = 0
    body_model_l_faces_offset = num_body_r_points
    obj_faces_offset = num_body_r_points + num_body_l_points

    # 更新面片索引
    updated_r_faces = body_model_r.faces + body_model_r_faces_offset
    updated_l_faces = faces_l + body_model_l_faces_offset
    updated_obj_faces = obj_faces + obj_faces_offset

    # 拼接面片数据
    faces = torch.cat([
        torch.from_numpy(updated_r_faces.astype(np.int64)).to(obj_points.device),
        torch.from_numpy(updated_l_faces.astype(np.int64)).to(obj_points.device),
        updated_obj_faces.to(obj_points.device)
    ], dim=0)

    # ===============================
    # Right Hand - Object Interaction Loss
    # ===============================

    right_hand = body_r['v']          # (B,Nr, 3)
    left_hand = points_l.unsqueeze(0)
    obj_verts  = obj_points_posed.unsqueeze(0)        # (B,No, 3)
    
        
    obj_triangles = []
    
    obj_triangles = [torch.from_numpy(scene.metadata_obj[data.obj_id]['obj_triangles'])]
    obj_triangles = torch.stack(obj_triangles, dim=0).cuda()

    obj_triangles = torch.matmul(rot.unsqueeze(0), obj_triangles.view(1, -1, 3).transpose(1, 2)).transpose(1, 2) \
                    + trans.unsqueeze(0).unsqueeze(0)
    obj_triangles = obj_triangles.view(1, -1, 3, 3)

    collision_mode = 'dist' if finetune else 'dist_sq'
    loss_attract, loss_collision, contact_info, _ = compute_contact_loss(
        right_hand,
        obj_verts,
        obj_triangles,
        contact_thresh=0.01,
        contact_mode="dist_tanh",      # 可以保留
        collision_thresh=0.005,
        collision_mode=collision_mode,
        contact_zones="zones",
    )
    penetr_verts = contact_info['penetr_verts']
    loss_attract_l, loss_collision_l, _, _ = compute_contact_loss(
        left_hand,
        obj_verts,
        obj_triangles,
        contact_thresh=0.01,
        contact_mode="dist_tanh",      # 可以保留
        collision_thresh=0.005,  
        collision_mode=collision_mode,   
        contact_zones="zones",
    )

    right_faces = torch.from_numpy(body_model_r.faces.astype(np.int64)).to(obj_points.device)
    right_hand_triangles = right_hand[:, right_faces].view(right_hand.shape[0], -1, 3, 3)

    _, loss_collision_h2h, _, _ = compute_contact_loss(
        left_hand,
        right_hand,
        right_hand_triangles,
        contact_thresh=0.01,
        contact_mode="dist_tanh",
        collision_thresh=0.005,
        collision_mode="dist",     
        contact_zones="zones",
    )

    # self_pen

    
    lambda_collision = 10.0 
    if finetune:
        lambda_collision *= 10

    lambda_attract   = 1.0
    if finetune:
        lambda_attract *= 5.0 
    loss_contact = lambda_collision * (loss_collision+loss_collision_l+loss_collision_h2h) + lambda_attract* (loss_attract+loss_attract_l)
    # loss_contact = lambda_collision * loss_collision + lambda_attract* loss_attract
    
    # 确保 loss_contact 是标量（0维 tensor）
    if len(loss_contact.shape) != 0:
        loss_contact = loss_contact.mean() if loss_contact.numel() > 0 else torch.tensor(0.0, device=loss_contact.device)

    # ---- 预定义 hand-hand 接触: 来自 config/baseball_golf.json -> hand_hand_contacts ----
    # 每条 (right_mano_vid, left_mano_vid, weight) 都是一对应当贴近的顶点; 把
    # 加权平方距离加到 loss_contact 里, 让两手对应顶点被拉到一起。
    # ⚠️ 仅在二阶段 (finetune=True, 力闭合) 生效, 引导手指最终位置;
    #     一阶段 Contact 优化阶段跳过, 避免手指被预定义点过度约束、阻碍 mask 对齐.
    hh_pairs = get_hand_hand_contacts() if finetune else []
    if hh_pairs:
        device = right_hand.device
        r_vids = torch.tensor([c["right_mano_vid"] for c in hh_pairs],
                              device=device, dtype=torch.long)
        l_vids = torch.tensor([c["left_mano_vid"]  for c in hh_pairs],
                              device=device, dtype=torch.long)
        weights = torch.tensor([c["weight"] for c in hh_pairs],
                               device=device, dtype=right_hand.dtype)
        p_r = right_hand[:, r_vids, :]              # (B, K, 3)
        p_l = left_hand[:,  l_vids, :]              # (1 or B, K, 3) — broadcasts
        sq_dist = ((p_r - p_l) ** 2).sum(dim=-1)    # (B, K)
        # finetune 已经是 True 才到这里; lambda 直接用力闭合阶段的强度
        lambda_hh_predef = 1000.0
        loss_hh_predef = (sq_dist * weights[None, :]).mean()
        loss_contact = loss_contact + lambda_hh_predef * loss_hh_predef

        # ---- DEBUG: 排查 hh_predef 是否真的有效, 数值对比其它 loss 看是不是被淹没 ----
        # 看完拉走就好, 别留太久 (每个 forward 都打印, 日志会很长)
        with torch.no_grad():
            _per_pair_d = sq_dist.detach().mean(dim=0).cpu().numpy()  # 各对的平均 sq_dist
            _per_pair_dist_m = np.sqrt(np.maximum(_per_pair_d, 0))    # 转成米
        _pair_summary = ", ".join(
            f"({c['right_mano_vid']}↔{c['left_mano_vid']}: {d*100:.2f}cm)"
            for c, d in zip(hh_pairs, _per_pair_dist_m.tolist())
        )
        print(f"[hh_predef] n_pairs={len(hh_pairs)} "
              f"loss={loss_hh_predef.item():.6f} lambda={lambda_hh_predef} "
              f"contrib(加入loss)={lambda_hh_predef * loss_hh_predef.item():.6f} | "
              f"each_pair={_pair_summary}", flush=True)

    # obj_meta = scene.metadata_obj[data.obj_id]
    # sdf_grid = torch.from_numpy(obj_meta['obj_sdf_grid']).to(right_hand.device)
    # sdf_center = torch.from_numpy(obj_meta['obj_sdf_center']).to(right_hand.device)
    # sdf_extent = torch.tensor(obj_meta['obj_sdf_extent'], device=right_hand.device)

    # loss_attract, loss_collision, _, _ = compute_contact_sdf_grid_loss(
    #     right_hand,
    #     rot,
    #     trans,
    #     sdf_grid,
    #     sdf_center,
    #     sdf_extent,
    #     contact_thresh=0.01,
    #     sdf_trunc=0.02,
    # )

    # hand self-pen

    # self_collision_loss = mano_self_collision_loss(body_r['v'], torch.from_numpy(body_model_r.faces).to(obj_points.device), mano_finger_labels.to(obj_points.device))
    # loss_contact+=self_collision_loss

    # # fc loss
    def force_closure(right_hand, obj_verts, obj_normals, side):
        B = right_hand.shape[0]
        # Tip indices come from config/baseball_golf.json -> hand_attract_tips.<side>
        # (legacy hardcoded default lives in utils/contact_config.py)
        tip_idxs_list = get_attract_tips(side)
        tip_idxs = torch.tensor(tip_idxs_list,
                                device=obj_points.device).long()
        tip_index = tip_idxs.unsqueeze(0).expand(B, -1)
        tips = torch.gather(
            right_hand,
            1,
            tip_index.unsqueeze(-1).expand(-1, -1, 3)
        )  # B × N_tips × 3
        if obj_verts.dim() == 2:
            obj_verts = obj_verts.unsqueeze(0).expand(B, -1, -1)
        dist, idx, nn = knn_points(tips, obj_verts, K=1, return_nn=True)

        nearest_obj = nn.squeeze(2)     # B × N_tips × 3
        nearest_idx = idx.squeeze(-1)   # B × N_tips
        vec = tips - nearest_obj
        contact_distance = torch.norm(vec, dim=-1)   # B × N_tips

        contact_normal = torch.gather(
            obj_normals,
            1,
            nearest_idx.unsqueeze(-1).expand(-1, -1, 3)
        )  # B × N_tips × 3
        contact_normal = contact_normal / (torch.norm(contact_normal, dim=-1, keepdim=True) + 1e-8)
        l8a, l8b = FCLoss().fc_loss(tips, contact_normal)

        # 新增: 直接的"tip → 最近 obj 顶点" 平方距离 mean. 形式跟 hh_predef 一致,
        # 用来真正把指尖拉到杆表面 (vs FCLoss 那种力闭合判据).
        # 平方距离: sum((tip - nearest_obj)**2, dim=-1) = contact_distance**2
        # 单位米², .mean() over (B, N_tips).
        tip_attract_loss = (contact_distance ** 2).mean()

        # ---- DEBUG: 看每个 attract_tip 离最近 obj 顶点的距离 (per-tip cm) ----
        with torch.no_grad():
            _per_tip_cm = contact_distance.detach().mean(dim=0).cpu().numpy() * 100.0
            _fc_contrib = (l8a.mean() + l8b.mean()).item() * 1e-2
            _tip_summary = ", ".join(
                f"vid{int(v)}={d:.2f}cm"
                for v, d in zip(tip_idxs_list, _per_tip_cm.tolist())
            )
            print(f"[fc_attract][{side}] mean_dist={_per_tip_cm.mean():.2f}cm  "
                  f"max={_per_tip_cm.max():.2f}cm  "
                  f"tip_attract_loss(m^2)={tip_attract_loss.item():.6f}  "
                  f"fc_contrib(×1e-2)={_fc_contrib:.6f}  "
                  f"per_tip=[{_tip_summary}]",
                  flush=True)

        return l8a, l8b, tip_attract_loss

    mesh = Meshes(obj_verts, obj_faces.unsqueeze(0))
    obj_normals = mesh.verts_normals_packed()
    obj_normals = obj_normals.to(obj_points.device)
    obj_normals = obj_normals.unsqueeze(0).expand(right_hand.shape[0], -1, -1)
    l8a,   l8b,   tip_attract_r = force_closure(right_hand, obj_verts, obj_normals, side="right")
    l8a_l, l8b_l, tip_attract_l = force_closure(left_hand,  obj_verts, obj_normals, side="left")
    if finetune:
        # 原力闭合判据 (FCLoss), 权重 1e-2 — 保留, 让"抓握稳定性"信号继续在
        loss_contact += (l8a.mean()+l8b.mean()+l8a_l.mean()+l8b_l.mean()) * 1e-2

        # 新增: 直接把 6 个 tip 顶点拉到最近 obj 顶点的距离 loss.
        # 用和 hand_hand_contacts (hh_predef) 同一档的 lambda, 以"6 个指节点→杆"
        # 的强度跟"右手 vid↔左手 vid"配对的强度对齐.
        lambda_tip_attract = 10.0     # = lambda_hh_predef
        loss_tip_attract = tip_attract_r + tip_attract_l
        loss_contact = loss_contact + lambda_tip_attract * loss_tip_attract
    

    # print('loss_collision',lambda_collision * loss_collision)
    # print('loss_attract',lambda_attract* loss_attract)
    # print('self_collision_loss',self_collision_loss)
    # print('fc',(l8a.mean()+l8b.mean())*1e-2)


    loss_reg = {'contact':(
        loss_contact
    )}

    #return None, loss_reg

    
    # render normal, disparity, & silhouette
    
    # R = torch.eye(3).unsqueeze(0)
    R = data.R.unsqueeze(0)
    T = data.T.unsqueeze(0)
    # print(R)
    # print(T)
    # print(data.K)
    # print((data.image_height, data.image_width))

    #points[:,2]*=-1
    #points[:, 1] *= -1  # 反转 Y 坐标
    #points[:, 0] *= -1 

    
    R[:, 0, 0] = -1
    R[:, 1, 1] = -1
    #R[:, 2, 2] = -1

    camera = PerspectiveCameras(
        focal_length=((data.K[0][0], data.K[1][1]),),  # (fx, fy)
        principal_point=((data.K[0][2], data.K[1][2]),),  # (px, py)
        image_size=((data.image_height, data.image_width),),
        in_ndc=False,
        device=points.device,
        R=R.cuda(), T=T.cuda(),

    )
    

    blend_params = BlendParams(
        sigma=torch.tensor(1e-8, dtype=torch.float32, device=points.device),
        gamma=torch.tensor(1e-8, dtype=torch.float32, device=points.device),
    )
    # raster_settings = RasterizationSettings(
    #     image_size=(data.image_height, data.image_width),
    #     blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
    #     faces_per_pixel=1,
    #     bin_size=-1,
    #     max_faces_per_bin=None,
    # )
    # renderer = MeshRenderer(
    #     rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
    #     shader=PhongNormalShader(cameras=camera, blend_params=blend_params),
    # )
    # 总 faces = 右手(1538) + 左手(1538) + obj_mesh(取决于 obj_max_faces).
    # bin_size=None + max_faces_per_bin=None 让 pytorch3d 自动估; 当 obj_mesh 上调到
    # 几万面时容易溢出 → "Bin size was too small ..." warning + 轮廓有空缺.
    # 上调 max_faces_per_bin 解决; 还不够再加; 实在不行 bin_size=0 走 naive (慢但稳).
    silhoutte_raster_settings = RasterizationSettings(
        image_size=(data.image_height, data.image_width),
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
        faces_per_pixel=100,
        bin_size=None,
        max_faces_per_bin=50000,
    )
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=silhoutte_raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )

    # raster_settings = RasterizationSettings(
    #     image_size=(data.image_height, data.image_width),
    #     blur_radius=1e-4,
    #     faces_per_pixel=5,
    #     #bin_size=None,  
    #     )

    #rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings)
    
    #depth, normal, disparity, silhouette = mesh_rendering(points, faces, rasterizer)
    depth, normal, disparity = None, None, None
    meshes = Meshes(verts=points.unsqueeze(0), faces=faces.unsqueeze(0))
    silhouette = silhouette_renderer(meshes)[..., 3]

    if vis:
        prefix="render"
        if iteration==1 or iteration>= 14920:
            prefix = 'render_{}_{}'.format(data.frame_id, f'{loss_contact:.4f}')
        debug_dir = os.path.join(scene.save_dir, "debug")
        save_render_outputs(
        depth,
        normal,
        disparity,
        silhouette.unsqueeze(0),
        data.full_mask,
        out_dir=debug_dir,
        prefix=prefix,
        iteration=iteration
        )
        with open(os.path.join(debug_dir, 'pcl_{}.obj'.format(iteration)), "w") as f:
            for v in body_r['v'][0].detach().cpu().numpy():
                f.write(f"v {v[0]} {v[1]} {v[2]} 1.0 0.0 0.0\n")
            for v in points_l.detach().cpu().numpy():
                f.write(f"v {v[0]} {v[1]} {v[2]} 0.0 1.0 0.0\n")
            for v in obj_points_posed.detach().cpu().numpy():
                f.write(f"v {v[0]} {v[1]} {v[2]} 0.0 0.0 1.0\n")
            for face in faces.detach().cpu().numpy():
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        with open(os.path.join(debug_dir, 'penetr_verts_{}.obj'.format(iteration)), "w") as f:
            for v in penetr_verts:
                f.write(f"v {v[0]} {v[1]} {v[2]} 1.0 1.0 0.0\n")

        points_vis = points[::1]
        uv_points = project_points(points_vis.detach().cpu().numpy(), data.K.detach().cpu().numpy())
        img = data.full_image.permute(1,2,0).detach().cpu().numpy()
        for u, v in uv_points:
            # 过滤掉飞出画布的点
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (int(u), int(v)), 1, (0, 255, 0), -1)

        # --- 3. mask_gt
        data.obj_mask
        
        
        # --- 4. 保存结果 ---
        # 将原图、点云投影、BBox 投影拼在一起显示
        cv2.imwrite(os.path.join(debug_dir, 'point_project_{}.png'.format(iteration)), img)

        # debug hand finger label

        # 假设 verts, faces 来自 MANO
        verts = body_r['v'][0].detach().cpu().numpy()  # 替换成你的 MANO verts
        faces = body_model_r.faces  # 替换成你的 MANO faces
        #save_colored_obj('/home/cyc/pycharm/lxy/3DGS/SportGS/output/debug/mano_finger_label.obj', verts, faces, labels)

    

    return {"normal": normal,
            "disparity": disparity,
            "silhouette": silhouette,
            }, loss_reg



def mesh_rendering(hand_verts, hand_faces, rasterizer, eps=1e-6):
    hand_verts = hand_verts.unsqueeze(0)
    hand_faces = hand_faces.unsqueeze(0)
    # ========= input asserts =========
    assert isinstance(hand_verts, torch.Tensor), "hand_verts must be torch.Tensor"
    assert isinstance(hand_faces, torch.Tensor), "hand_faces must be torch.Tensor"

    assert hand_verts.ndim == 3, f"hand_verts.ndim = {hand_verts.ndim}, expected 3"
    assert hand_faces.ndim == 3, f"hand_faces.ndim = {hand_faces.ndim}, expected 3"

    meshes = Meshes(verts=hand_verts, faces=hand_faces)

    fragments = rasterizer(meshes)
    depth_map = fragments.zbuf  # 获取深度图
    
    # ori_depth = fragments.zbuf
    # ori_depth = torch.where(ori_depth.le(0), torch.ones_like(ori_depth).to(ori_depth.device) * 0, ori_depth)
    # resize_depth = ori_depth.permute(0, 3, 1, 2)

    # get normal, disparity, & silhouette

    # 1. Silhouette
    pix_to_face = fragments.pix_to_face[..., 0]          # (B, H, W)
    silhouette = (pix_to_face >= 0).float()               # (B, H, W)

    # 2. Depth
    depth = fragments.zbuf[..., 0]                         # (B, H, W)
    depth = depth * silhouette                             # 背景置 0
    depth = depth.unsqueeze(1)                             # (B, 1, H, W)
    
    # 3. Disparity（单目：1 / depth，再做尺度无关化）
    disparity = 1.0 / (depth + eps)

    # 尺度无关归一化
    # Ⅰ 基于前景均值的深度分布归一化
    # scale = disparity[silhouette.unsqueeze(1) > 0].mean()
    # disparity = disparity / (scale + eps)
    # Ⅱ min–max归一化
    # foreground mask
    fg = silhouette.unsqueeze(1).float()
   
    # min–max on foreground
    d = disparity * fg
    d_min = d[fg > 0].min()
    d_max = d[fg > 0].max()

    # normalize
    disparity_norm = (d - d_min) / (d_max - d_min + 1e-6)
    disparity_norm = disparity_norm * fg

    
    # 4. Normal
    verts = meshes.verts_packed()          # (V_all, 3)
    faces = meshes.faces_packed()          # (F_all, 3)

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    face_normals = F.normalize(face_normals, dim=1)       # (F_all, 3)

    B, H, W = pix_to_face.shape
    normal = torch.zeros(B, H, W, 3, device=verts.device)

    valid = pix_to_face >= 0
    face_ids = pix_to_face[valid]
    normal[valid] = face_normals[face_ids]

    normal = F.normalize(normal, dim=-1)
    normal = normal.permute(0, 3, 1, 2)                    # (B, 3, H, W)
    normal = normal * silhouette.unsqueeze(1)

    silhouette = silhouette.unsqueeze(1)

    # --- 左右镜像 ---
    # 假设 shape: depth (B, 1, H, W), normal (B, 3, H, W), disparity (B, 1, H, W), silhouette (B, 1, H, W)
    # depth = depth.flip(-1).flip(-2)       # 左右 + 上下
    # disparity = disparity.flip(-1).flip(-2)
    # silhouette = silhouette.flip(-1).flip(-2)
    # normal = normal.flip(-1).flip(-2)
    # normal[:, 0, :, :] *= -1              # 左右镜像 X 分量取反
    # normal[:, 1, :, :] *= -1              # 上下镜像 Y 分量取反


    return depth, normal, disparity, silhouette

def save_colored_obj(filename, verts, faces, mano_finger_labels):
            """
            保存带颜色的 OBJ 文件
            verts: (N,3) 顶点
            faces: (F,3) 三角面索引
            labels: (N,) finger/palm label
            """
            # label -> RGB
            color_map = {
                0: [0.6, 0.6, 0.6],  # palm, 范围 0~1
                1: [1.0, 0.0, 0.0],  # thumb
                2: [0.0, 1.0, 0.0],  # index
                3: [0.0, 0.0, 1.0],  # middle
                4: [1.0, 0.65, 0.0], # ring
                5: [0.63, 0.13, 0.94]# pinky
            }

            with open(filename, "w") as f:
                # 写顶点
                for v, l in zip(verts, labels):
                    c = color_map[int(l)]
                    f.write(f"v {v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n")
                
                # 写面，OBJ 文件索引从1开始
                for face in faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

            print(f"Saved OBJ to {filename}")


import torchvision.utils as vutils
import cv2

def save_render_outputs(
    depth,
    normal,
    disparity,
    silhouette,
    mask,
    out_dir,
    prefix="sample",
    iteration=0
):
    """
    normal:     (B, 3, H, W)
    disparity:  (B, 1, H, W)
    silhouette: (B, 1, H, W)
    """

    # # -------- shape asserts --------
    # assert normal.ndim == 4, f"normal.ndim = {normal.ndim}, expected 4"
    # assert disparity.ndim == 4, f"disparity.ndim = {disparity.ndim}, expected 4"
    assert silhouette.ndim == 4, f"silhouette.ndim = {silhouette.ndim}, expected 4"

    # B, Cn, H, W = normal.shape
    # Bd, Cd, Hd, Wd = disparity.shape
    Bs, Cs, Hs, Ws = silhouette.shape

    # assert Cn == 3, f"normal channel = {Cn}, expected 3"
    # assert Cd == 1, f"disparity channel = {Cd}, expected 1"
    # assert Cs == 1, f"silhouette channel = {Cs}, expected 1"

    # assert B == Bd == Bs, f"batch mismatch: {B}, {Bd}, {Bs}"
    # assert H == Hd == Hs, f"height mismatch: {H}, {Hd}, {Hs}"
    # assert W == Wd == Ws, f"width mismatch: {W}, {Wd}, {Ws}"

    # -------- save --------
    os.makedirs(out_dir, exist_ok=True)

    for i in [iteration]:
        if depth is not None:
            cv2.imwrite(os.path.join(out_dir, f"{prefix}_{i:05d}_depth.png"), 
                        (depth[0, 0] * 1000.0).detach().cpu().numpy().round().astype(np.uint16))

            # Normal: [-1,1] -> [0,1]
            n = torch.clamp((normal[0] + 1.0) * 0.5, 0, 1)

            vutils.save_image(
                n,
                os.path.join(out_dir, f"{prefix}_{i:05d}_normal.png")
            )

            # Disparity: per-image min-max
            d = disparity[0]
            d = d - d.min()
            d = d / (d.max() + 1e-6)

            vutils.save_image(
                d,
                os.path.join(out_dir, f"{prefix}_{i:05d}_disp.png")
            )

        # Silhouette: hard mask
        s = (silhouette[0] > 0).float()

        vutils.save_image(
            s,
            os.path.join(out_dir, f"{prefix}_{i:05d}_mask.png")
        )

        vutils.save_image(
            mask,
            os.path.join(out_dir, f"{prefix}_{i:05d}_mask_gt.png")
        )



def project_points(points_3d, K):
    """
    核心投影函数
    points_3d: (N, 3)
    K: (3, 3) intrinsic matrix
    """
    # 1. 转齐次坐标 (N, 3) -> (N, 4)
    ones = np.ones((points_3d.shape[0], 1))
    points = np.hstack((points_3d, ones))
    
    # 2. 刚体变换 (Object -> Camera)
    # math: P_cam = T @ P_obj
    #points = (T @ points.T).T  # (N, 4)
    
    # 取出前三维 xyz
    xyz_cam = points[:, :3]
    
    # 3. 检查 Z > 0 (只保留相机前方的点)
    # 这里不做剔除，只是为了后续除法安全
    z = xyz_cam[:, 2:3]
    z[z == 0] = 1e-5 # 防止除零
    
    # 4. 投影 (Camera -> Pixel)
    # math: p_uv = K @ P_cam
    uv_z = (K @ xyz_cam.T).T
    
    # 5. 归一化
    uv = uv_z[:, :2] / z
    
    return uv





