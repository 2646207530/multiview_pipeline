import os

from gaussian_renderer.render_mesh import render_mesh
from right_hand_model import MANO
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import shutil
from omegaconf import OmegaConf
import swanlab
from gaussian_renderer import render
from utils.general_utils import fix_random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import pyiqa 
import hydra
from random import randint
from scene import Scene


def C(iteration, value):
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = OmegaConf.to_container(value)
        if not isinstance(value, list):
            raise TypeError('Scalar specification only supports list, got', type(value))
        value_list = [0] + value
        i = 0
        current_step = iteration
        while i < len(value_list):
            if current_step >= value_list[i]:
                i += 2
            else:
                break
        value = value_list[i - 1]
    return value


#@profile
def training(config):
    model = config.model
    dataset = config.dataset
    opt = config.opt
    pipe = config.pipeline
    checkpoint_iterations = config.checkpoint_iterations
    debug_from = config.debug_from
    gaussians_hand_group = {}
    gaussians_obj_group = {}


    scene = Scene(config, gaussians_hand_group, gaussians_obj_group, config.exp_dir)
    scene.train()
    #scene.eval()
    print("training_samples:", len(scene.train_dataset))
   
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    _MANO_DIR = '/data2/fubingshuai/golf/golf-hand-object/MANO'
    body_model_r = MANO(model_path=_MANO_DIR, flat_hand_mean=True).cuda()
    body_model_l = MANO(model_path=_MANO_DIR, is_rhand=True, flat_hand_mean=True).cuda()
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # ── 可配置的优化帧区间 ──
    opt_frame_start = config.dataset.get('opt_frame_start', None)
    opt_frame_end = config.dataset.get('opt_frame_end', None)
    export_data_path = config.dataset.get('export_data_path', None)
    export_output_path = config.dataset.get('export_output_path', None)

    # ── 可选: 周期性导出当前优化状态作为 debug 快照 ──
    opt_debug_every = int(config.dataset.get('opt_debug_every', 0) or 0)
    opt_debug_dir = config.dataset.get('opt_debug_dir', None)
    opt_debug_stage = config.dataset.get('opt_debug_stage', 'contact')
    if opt_debug_every > 0 and opt_debug_dir:
        import os as _os
        _os.makedirs(opt_debug_dir, exist_ok=True)
        print(f"[opt_debug] 启用周期 export: 每 {opt_debug_every} iter 导出到 "
              f"{opt_debug_dir}/{opt_debug_stage}_iter*.npy")

    data_stack = None
    ema_loss_for_log = 0.0
    first_iter = 0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # tracemalloc.start()

    lbs_weights_r = np.load('./hand_models/misc/skinning_weights_all.npz')['rightHand']
    # joint index -> finger index (0~4 fingers, 5=palm)
    joint_to_finger = {
        0: 5,      # wrist/palm
        1: 0, 2:0, 3:0,  # thumb joints
        4:1, 5:1, 6:1,   # index joints
        7:2, 8:2, 9:2,   # middle joints
        10:3, 11:3, 12:3, # ring joints
        13:4, 14:4, 15:4  # pinky joints
    }
    # 转成 torch tensor
    weights = torch.tensor(lbs_weights_r, dtype=torch.float32)  # (778,16)
    # argmax 找到每个顶点最主要的 joint
    joint_max = torch.argmax(weights, dim=1)  # (778,)
    # 映射到 finger
    mano_finger_labels = torch.tensor([joint_to_finger[int(j)] for j in joint_max], dtype=torch.long)

    for iteration in range(first_iter, opt.iterations + 1):
        if iteration == 5 or iteration == 15000:
            scene.converter.pose_correction.export('opt.npy',
                data_path=export_data_path, output_path=export_output_path)

        # 周期性 debug 快照: 当前 pose/obj 写到 <opt_debug_dir>/<stage>_iter<NN>.npy
        if (opt_debug_every > 0 and opt_debug_dir
                and iteration > 0 and iteration % opt_debug_every == 0):
            import os as _os
            _dbg_name = f"{opt_debug_stage}_iter{iteration:06d}.npy"
            _dbg_path = _os.path.join(opt_debug_dir, _dbg_name)
            try:
                scene.converter.pose_correction.export(
                    _dbg_name,
                    data_path=export_data_path,
                    output_path=_dbg_path,
                )
            except Exception as _e:
                print(f"[opt_debug] iter {iteration} export 失败: {_e}")
        
        iter_start.record()

        for sub_id in gaussians_hand_group:
            gaussians_hand_group[sub_id]['right'].update_learning_rate(iteration)
            gaussians_hand_group[sub_id]['left'].update_learning_rate(iteration)

        for obj_id in gaussians_obj_group:
            gaussians_obj_group[obj_id].update_learning_rate(iteration)

     
        # Every 1000 its we increase the levels of SH up to a maximum degree
        sh_step = len(gaussians_obj_group) + len(gaussians_hand_group) // 2
        if sh_step > 0 and iteration % (1000 * sh_step) == 0:
            for sub_id in gaussians_hand_group:
                gaussians_hand_group[sub_id]['right'].oneupSHdegree()
                gaussians_hand_group[sub_id]['left'].oneupSHdegree()
            for obj_id in gaussians_obj_group:
                gaussians_obj_group[obj_id].oneupSHdegree()
        # Pick a random data point
        if not data_stack:
            if opt_frame_start is not None and opt_frame_end is not None:
                data_stack = list(range(int(opt_frame_start), int(opt_frame_end)))
            else:
                data_stack = list(range(len(scene.train_dataset)))
        data_idx = data_stack.pop(randint(0, len(data_stack) - 1))
        #data_idx=0
        data = scene.train_dataset[data_idx]
        #prev_data = scene.train_dataset[max(0, data_idx - 1)]
        prev_data = None

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        lambda_mask = C(iteration, config.opt.lambda_mask)
        use_mask = lambda_mask > 0.

        render_pkg = render(data, iteration, scene, pipe, background, compute_loss=True, return_opacity=True,
                            white_bg=dataset.white_background)
       
        render_mesh_pkg, loss_reg_mesh = render_mesh(render_pkg["updated_camera"], iteration, scene, body_model_r, body_model_l, mano_finger_labels, 
                                      vis=(iteration <5 or iteration%1000==0 or iteration>=14920))
        
        # Loss
        loss = 0.
        loss_mask = 0.
        gt_image = data.original_image.cuda()
        obj_gt_image = data.obj_image.cuda()
        full_gt_image = data.full_image.cuda()

        gt_mask = data.original_mask.cuda()
        obj_mask = data.obj_mask.cuda()
        full_mask = data.full_mask.cuda()
        if render_mesh_pkg is not None:
            
            loss_mask = F.l1_loss(render_mesh_pkg["silhouette"], full_mask)
            #print(lambda_mask)
            loss += lambda_mask * loss_mask


        loss_reg = render_pkg["loss_reg"]
        for name, value in loss_reg.items():
            #print(name)
            #print(loss_reg[name])
            
            lbd = opt.get(f"lambda_{name}", 0.)
            #print(lbd)
            lbd = C(iteration, lbd)
            
            loss_reg[name] *= lbd
            loss += loss_reg[name]

        
        for name, _ in loss_reg_mesh.items():
            
            lbd = opt.get(f"lambda_{name}", 0.)
            #print(lbd)
            lbd = C(iteration, lbd)
            loss_reg_mesh[name] *= lbd
            loss += loss_reg_mesh[name]

            

        loss.backward()

        # for name, p in scene.converter.pose_correction.named_parameters():
        #     if p.grad is None:
        #         print(f'{name}: grad = None ❌')
        #     else:
        #         print(f'{name}: grad mean = {p.grad.abs().mean().item():.6f} ✅')


        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            elapsed = iter_start.elapsed_time(iter_end)
            log_loss = {
                
                'loss/mask_loss': loss_mask,
                #'loss/smooth_loss': loss_reg['smooth'],
                'loss/contact_loss': loss_reg_mesh['contact'],

                'loss/total_loss': loss.item(),
                'iter_time': elapsed,
            }
            log_loss.update({
                'loss/loss_' + k: v for k, v in loss_reg.items()
            })
            swanlab.log(log_loss)

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            #validation(iteration, testing_iterations, testing_interval, scene, evaluator, (pipe, background))

            scene.optimize(iteration)

            if iteration in checkpoint_iterations:
                scene.save_checkpoint(iteration)




@hydra.main(version_base=None, config_path="configs", config_name="config_golf")
def main(config):
    # print(OmegaConf.to_yaml(config))
    OmegaConf.set_struct(config, False)  # allow adding new values to config
    # print(config.name)
    _DEFAULT_RESULTS = '/data2/fubingshuai/golf/golf-hand-object/out/SportGS_results'
    config.exp_dir = config.get('exp_dir') or os.path.join(_DEFAULT_RESULTS, config.dataset._YCB_CLASSES[0], config.name)

    os.makedirs(config.exp_dir, exist_ok=True)
    config.checkpoint_iterations.append(config.opt.iterations)
    os.makedirs(os.path.join(config.exp_dir,'code'), exist_ok=True)
    if not config.wandb_disable:
        try:
            shutil.copyfile('train_golf.py', config.exp_dir + '/code/train_golf.py')
            shutil.copytree('./scene', config.exp_dir + '/code/scene')
            shutil.copytree('./models', config.exp_dir + '/code/models')
            shutil.copytree('./configs', config.exp_dir + '/code/configs')
            shutil.copytree('./dataset', config.exp_dir + '/code/dataset')
            shutil.copytree('./utils', config.exp_dir + '/code/utils')
        except Exception as e:
            print(f"[Warning] Failed to save codes: {e}")

    wandb_name = config.name
    enable_swanlab = not getattr(config, "wandb_disable", False)

    #swanlab_log = os.path.join('/mnt/sda2/lxy/ARGS_results/', config.dataset._YCB_CLASSES[0],'swanlab')
    swanlab_log = os.path.join(_DEFAULT_RESULTS, 'swanlab', 'SportGS')
    os.makedirs(swanlab_log, exist_ok=True) 
    swanlab.init(
        name=wandb_name,
        project='SportGS_228',
        config=OmegaConf.to_container(config, resolve=True),
        logdir=swanlab_log,
        mode='local' if enable_swanlab else 'disabled'
    )


    print("Optimizing " + config.exp_dir)

    # Initialize system state (RNG)
    fix_random(config.seed)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(config.detect_anomaly)

    training(config)

    # All done
    print("\nTraining complete.")


if __name__ == "__main__":
    main()  #
