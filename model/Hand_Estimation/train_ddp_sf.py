import os
import random
from argparse import Namespace
from time import time

import lib.models
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from lib.datasets import create_dataset
from lib.external import EXT_PACKAGE
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.io_utils import load_model
# from lib.viztools.draw import draw_batch_joint_images_all
# from lib.utils.transform_seq import batch_save_transformed_coords_json, batch_transform_coords
from lib.utils.config import get_config
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.misc import CONST, bar_perfixes, format_args_cfg
from lib.utils.net_utils import build_optimizer, build_scheduler, clip_gradient, setup_seed
from lib.utils.recorder import Recorder
from lib.utils.summary_writer import DDPSummaryWriter
from lib.utils.triangulation import batch_triangulate_dlt_torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from lib.utils.config import CN

# import logging
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # 可选："INFO"（基础）或"DETAIL"（详细）
# os.environ["NCCL_DEBUG"] = "INFO"  # 若用NCCL后端，开启NCCL日志
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def _init_fn(worker_id):
    seed = ((worker_id + 1) * int(torch.initial_seed())) % CONST.INT_MAX
    np.random.seed(seed)
    random.seed(seed)


def move_to_device(x, device, non_blocking=True):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=non_blocking)
    if isinstance(x, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [move_to_device(v, device, non_blocking) for v in x]
        return type(x)(t) if isinstance(x, tuple) else t
    return x  # 其它类型原样返回


def batch_cam2world(cam_coords, extrinsic):
    """
    批量将相机坐标系中的点转换到世界坐标系
    
    参数:
    cam_coords : torch.Tensor, 形状为 (batch, n, 3) - 相机坐标系中的点
    extrinsic : torch.Tensor, 形状为 (batch, 4, 4) - 齐次变换矩阵
    
    返回:
    world_coords : torch.Tensor, 形状为 (batch, n, 3) - 世界坐标系中的点
    """
    # 提取旋转矩阵和平移向量
    R = extrinsic[:, :3, :3]  # (batch, 3, 3)
    t = extrinsic[:, :3, 3]   # (batch, 3)
    
    # 计算逆旋转矩阵 (等价于转置，因为旋转矩阵是正交矩阵)
    R_inv = R.transpose(1, 2)  # (batch, 3, 3)
    
    # 扩展平移向量以匹配点集维度
    t_expanded = t.unsqueeze(1)  # (batch, 1, 3)
    
    # 执行坐标转换: world_coord = R_inv * (cam_coord - t)
    centered = cam_coords - t_expanded
    world_coords = torch.einsum('bij,bkj->bik', centered, R_inv)
    
    return world_coords


def main_worker(gpu_id: int, cfg: CN, arg: Namespace, time_f: float):

    # if the model is from the external package
    if cfg.MODEL.TYPE in EXT_PACKAGE:
        pkg = EXT_PACKAGE[cfg.MODEL.TYPE]
        exec(f"from lib.external import {pkg}")

    if arg.distributed:
        rank = arg.n_gpus * arg.node_rank + gpu_id
        torch.distributed.init_process_group(arg.dist_backend, rank=rank, world_size=arg.world_size)
        assert rank == torch.distributed.get_rank(), "Something wrong with nodes or gpus"
        torch.cuda.set_device(rank)
    else:
        rank = None  # only one process.

    setup_seed(cfg.TRAIN.MANUAL_SEED + rank, cfg.TRAIN.CONV_REPEATABLE)
    recorder = Recorder(arg.exp_id, cfg, rank=rank, time_f=time_f)
    summary = DDPSummaryWriter(log_dir=recorder.tensorboard_path, rank=rank)
    # summarizer = Summarizer(arg.exp_id, cfg, rank=rank, time_f=time_f)

    # add a barrier, to make sure all recorders are created
    torch.distributed.barrier()

    train_data = create_dataset(cfg.DATASET.TRAIN, data_preset=cfg.DATA_PRESET)
    train_sampler = DistributedSampler(train_data, num_replicas=arg.world_size, rank=rank, shuffle=True)
    _n_workers = int(arg.workers)
    train_loader = DataLoader(train_data,
                              batch_size=arg.batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=_n_workers,
                              pin_memory=True,
                              drop_last=True,
                              sampler=train_sampler,
                              worker_init_fn=_init_fn,
                              # persistent_workers 必须 num_workers>0 才能开
                              persistent_workers=(_n_workers > 0))

    if rank == 0:
        val_data = create_dataset(cfg.DATASET.TEST, data_preset=cfg.DATA_PRESET)
        val_loader = DataLoader(val_data,
                                batch_size=arg.val_batch_size,
                                shuffle=False,
                                num_workers=int(arg.workers),
                                pin_memory=True,
                                drop_last=False,
                                worker_init_fn=_init_fn)
    else:
        val_loader = None
    # val_data = create_dataset(cfg.DATASET.TEST, data_preset=cfg.DATA_PRESET)
    # val_sampler = DistributedSampler(val_data, num_replicas=arg.world_size, rank=rank, shuffle=True)
    # val_loader = DataLoader(val_data,
    #                         batch_size=arg.val_batch_size,
    #                         shuffle=(val_sampler is None),
    #                         num_workers=int(arg.workers),
    #                         pin_memory=True,
    #                         drop_last=False,
    #                         sampler=val_sampler,
    #                         worker_init_fn=_init_fn,
    #                         persistent_workers=True)



    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    model.setup(summary_writer=summary)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=cfg.TRAIN.FIND_UNUSED_PARAMETERS, static_graph=True)

    # for idx, (name, param) in enumerate(model.named_parameters()):
    #     print(f"Parameter {name} (index {idx}: {param.shape}")

    backbone_lr_ratio = 0.1
    backbone_params, head_params = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name.lower():
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {'params': backbone_params, 'lr': cfg.TRAIN.LR * backbone_lr_ratio},
        {'params': head_params, 'lr': cfg.TRAIN.LR}
    ]

    # 2. 直接调用你原有的函数，不需要修改它的内部逻辑！
    optimizer = build_optimizer(param_groups, cfg=cfg.TRAIN)

    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    #optimizer = build_optimizer(model.parameters(), cfg=cfg.TRAIN)
    scheduler = build_scheduler(optimizer, cfg=cfg.TRAIN)

    if arg.ft:
        map_location = f"cuda:{rank}" if rank is not None else "cuda"
        # --reload 给了就用它做 ft 起点 (我们 mv_finetune 走这条路);
        # 否则回退到原始硬编码路径 (历史训练流程).
        ft_path = arg.reload if getattr(arg, 'reload', None) else 'exp/Sim_ResNet/checkpoints/checkpoint_40'
        load_model(model, resume_path=ft_path, map_location=map_location)

    if arg.resume:
        epoch = recorder.resume_checkpoints(model, optimizer, scheduler, arg.resume, arg.resume_epoch)
    else:
        epoch = 0

    # Make sure model is created, resume is finished
    torch.distributed.barrier()

    # for idx, (name, param) in enumerate(model.named_parameters()):
    #     print(f"Index {idx}: {name}")

    logger.warning(f"############## start training from {epoch} to {cfg.TRAIN.EPOCH} ##############")
    for epoch_idx in range(epoch, cfg["TRAIN"]["EPOCH"]):
        if arg.distributed:
            train_sampler.set_epoch(epoch_idx)
        # if epoch_idx == cfg.TRAIN.PHASE1_EPOCHS:
        #     model.module.set_phase(1)
        # if epoch_idx == cfg.TRAIN.PHASE2_EPOCHS:
        #     model.module.set_phase(2)  # 进入阶段2
            # optimizer = build_optimizer(filter(lambda p: p.requires_grad, model.parameters()), cfg=cfg.TRAIN)
            # scheduler = build_scheduler(optimizer, cfg=cfg.TRAIN)
        # elif epoch_idx == cfg.TRAIN.PHASE3_EPOCHS:
        # model.module.set_phase(3)  # 进入阶段3
        # optimizer = build_optimizer(filter(lambda p: p.requires_grad, model.parameters()), cfg=cfg.TRAIN)
        # scheduler = build_scheduler(optimizer, cfg=cfg.TRAIN)

        model.train()
        trainbar = etqdm(train_loader, rank=rank)
        for bidx, batch in enumerate(trainbar):
            optimizer.zero_grad()
            step_idx = epoch_idx * len(train_loader) + bidx

            preds, loss_dict = model(batch, step_idx, epoch_idx, "train")
            loss = loss_dict["loss"]

            loss.backward()
            if cfg.TRAIN.GRAD_CLIP_ENABLED:
                clip_gradient(optimizer, cfg.TRAIN.GRAD_CLIP.NORM, cfg.TRAIN.GRAD_CLIP.TYPE)

            optimizer.step()
            optimizer.zero_grad()


            trainbar.set_description(f"{bar_perfixes['train']} Epoch {epoch_idx} "
                                     f"{model.module.format_metric('train', epoch_idx)}")

        scheduler.step()
        logger.info(f"Current LR: {[group['lr'] for group in optimizer.param_groups]}")

        recorder.record_checkpoints(model, optimizer, scheduler, epoch_idx, arg.snapshot)
        torch.distributed.barrier()
        model.module.on_train_finished(recorder, epoch_idx)

        '''
        if (epoch_idx % arg.eval_interval == 0 or epoch_idx == 0) and rank == 0:
            logger.info("do validation and save results")
            with torch.no_grad():
                model.eval()
                net = model.module
                device = next(net.parameters()).device
                valbar = etqdm(val_loader, rank=rank)
                for bidx, batch in enumerate(valbar):
                    batch = move_to_device(batch, device, non_blocking=True)
                    step_idx = epoch_idx * len(val_loader) + bidx                   
                    preds, res = net(batch, step_idx, epoch_idx, "val")                  
                    valbar.set_description(f"{bar_perfixes['val']} Epoch {epoch_idx} "
                                        f"{net.format_metric('val', epoch_idx)}")

            model.module.on_val_finished(recorder, epoch_idx)
            '''
    if arg.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    exp_time = time()
    arg, _ = parse_exp_args()
    if arg.resume:
        logger.warning(f"config will be reloaded from {os.path.join(arg.resume, 'dump_cfg.yaml')}")
        arg.cfg = os.path.join(arg.resume, "dump_cfg.yaml")
        cfg = get_config(config_file=arg.cfg, arg=arg, merge=False)
    else:
        cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)

    # ────────────── mv_finetune 兼容性 patch ──────────────
    # 我们在 wrapper 端用 --ft + --reload <pretrained_dir> 加载 ckpt 目录给 load_model.
    # 但 lib/utils/config.py:103 会把 --reload 也同步赋给 cfg.MODEL.PRETRAINED,
    # 接着 init_weights() 拿目录路径去 os.path.isfile() 必然 False -> FileNotFoundError.
    # 这里 reset 一下让 init_weights 走 "随机初始化 + 后续 load_model" 路径.
    if arg.ft and getattr(arg, 'reload', None):
        try:
            cfg.defrost()
        except Exception:
            pass
        cfg.MODEL.PRETRAINED = None

    os.environ["MASTER_ADDR"] = arg.dist_master_addr
    os.environ["MASTER_PORT"] = arg.dist_master_port
    # must have equal gpus on each node.
    arg.world_size = arg.n_gpus * arg.nodes
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    arg.batch_size = int(arg.batch_size / arg.n_gpus)
    if arg.val_batch_size is None:
        arg.val_batch_size = arg.batch_size

    arg.workers = int((arg.workers + arg.n_gpus - 1) / arg.n_gpus)

    logger.warning(f"final args and cfg: \n{format_args_cfg(arg, cfg)}")
    # input("Confirm (press enter) ?")

    logger.info("====> Use Distributed Data Parallel <====")
    torch.multiprocessing.spawn(main_worker, args=(cfg, arg, exp_time), nprocs=arg.n_gpus)

