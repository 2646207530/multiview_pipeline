import os
import random
import sys
import time
from pprint import pformat
from typing import Dict, List, Optional, TypeVar, Union

import numpy as np
import torch
from git import Repo
from termcolor import colored

from torch.optim.optimizer import Optimizer

from lib.metrics.basic_metric import LossMetric

from ..metrics import Metric
from .dist_utils import master_only
from .io_utils import load_model, load_random_state, load_train_param, save_states
from .logger import logger
from .misc import RandomState

T = TypeVar("T", bound="Recorder")


class Recorder:

    def __init__(
        self: T,
        exp_id: str,
        cfg: Dict,
        root_path: str = "exp",
        rank: Optional[int] = None,
        time_f: Optional[float] = None,
        eval_only: bool = False,
    ):
        if not eval_only:
            # 原来这里强制要求是 git repo 且无未提交修改; 我们的部署树 Hand_Estimation/
            # 本身不是独立 git repo, 所以放宽成 warning, 继续往下走.
            if exp_id not in ["default", "debug"]:
                try:
                    _commit = self.get_git_commit()
                except Exception as _e:
                    _commit = None
                if not _commit:
                    logger.warning(
                        f"[Recorder] exp_id={exp_id} 既不是 default/debug, "
                        f"也拿不到 git commit hash (非 git repo 或有未提交修改); "
                        f"继续运行, 不再 raise.")
        self.eval_only = eval_only
        self.timestamp = time.strftime("%Y_%m%d_%H%M_%S", time.localtime(time_f if time_f else time.time()))
        self.exp_id = exp_id
        self.cfg = cfg
        self.dump_path = os.path.join(root_path, f"{exp_id}_{self.timestamp}")
        self.eval_dump_path = os.path.join(self.dump_path, "evaluations")
        self.tensorboard_path = os.path.join(self.dump_path, "runs")
        self.rank = rank
        self._record_init_info()

    @master_only
    def _record_init_info(self: T):
        assert self.rank == 0, "Only master process can record init info!"
        if not os.path.exists(self.dump_path):
            os.makedirs(self.dump_path)
        if not os.path.exists(self.eval_dump_path):
            os.makedirs(self.eval_dump_path)
        assert logger.filehandler is None, "log file path has been set"
        logger.set_log_file(path=self.dump_path, name=f"{self.exp_id}_{self.timestamp}")
        logger.info(f"run command: {' '.join(sys.argv)}")
        if not self.eval_only and self.exp_id not in ["default", "debug"]:
            logger.info(f"git commit: {self.get_git_commit()}")
        with open(os.path.join(self.dump_path, "dump_cfg.yaml"), "w") as f:
            f.write(self.cfg.dump(sort_keys=False))
        f.close()
        msg = colored(f">>> Dump cfg file to {os.path.join(self.dump_path, 'dump_cfg.yaml')}", "magenta")
        logger.info(msg)

    @master_only
    def record_checkpoints(self: T, model, optimizer: Union[Dict[str, Optimizer], Optimizer],
                           scheduler, epoch: int, snapshot: int):

        assert self.rank == 0, "only master process can record loss"
        checkpoints_path = os.path.join(self.dump_path, "checkpoints")
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)

        # construct RandomState tuple
        random_state = RandomState(
            torch_rng_state=torch.get_rng_state(),
            torch_cuda_rng_state=torch.cuda.get_rng_state(),
            torch_cuda_rng_state_all=torch.cuda.get_rng_state_all(),
            numpy_rng_state=np.random.get_state(),
            random_rng_state=random.getstate(),
        )

        save_states(
            {
                "epoch": epoch + 1,
                "model": model,
                "optimizer": (optimizer.state_dict() if type(optimizer) is not dict else {
                    k: v.state_dict() for k, v in optimizer.items()
                }),
                "scheduler": (scheduler.state_dict() if type(scheduler) is not dict else {
                    k: v.state_dict() for k, v in scheduler.items()
                }),
                "random_state": random_state,
            },
            is_best=False,
            checkpoint=checkpoints_path,
            snapshot=snapshot,
        )
        logger.info(f"record checkpoints to {checkpoints_path}")

    def resume_checkpoints(self: T,
                           model,
                           optimizer: Union[Dict[str, Optimizer], Optimizer],
                           scheduler,
                           resume_path: str,
                           resume_epoch: Optional[int] = None):
        """resume checkpoints from resume_path, all process are synchronized

        Args:
            self (T): _description_
            model (_type_): _description_
            optimizer (Union[Dict[str, Optimizer], Optimizer]): _description_
            scheduler (Union[Dict[str, _LRScheduler], _LRScheduler]): _description_
            resume_path (str): _description_
            resume_epoch (Optional[int], optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        map_location = f"cuda:{self.rank}" if self.rank is not None else "cuda"
        resume_path = os.path.join(resume_path, "checkpoints",
                                   f"checkpoint_{resume_epoch}" if resume_epoch else "checkpoint")
        epoch = load_train_param(optimizer,
                                 scheduler,
                                 os.path.join(resume_path, "train_param.pth.tar"),
                                 map_location=map_location)

        load_random_state(os.path.join(resume_path, "random_state.pkl"))
        load_model(model, resume_path, map_location=map_location)
        return epoch

    @master_only
    def record_loss(self, loss_metric: LossMetric, epoch: int, comment=""):
        assert self.rank == 0, "only master process can record loss"
        loss_dump_path = os.path.join(self.eval_dump_path, f"{comment}_Loss.txt")
        with open(loss_dump_path, "a") as f:
            f.write(f"Epoch {epoch} | {comment} loss metric:\n {pformat(loss_metric.get_measures())}\n\n")

    @master_only
    def record_metric(self, metrics: List, epoch: int, comment=""):
        assert self.rank == 0, "only master process can record metirc"
        metric_dump_path = os.path.join(self.eval_dump_path, f"{comment}_Metric.txt")

        if type(metrics) is Metric:
            metrics = [metrics]

        with open(metric_dump_path, "a") as f:
            f.write(f"Epoch {epoch} | {comment} metric:\n")
            for M in metrics:
                f.write(f"{pformat(M.get_measures())}\n")
            f.write("\n")

    @staticmethod
    def get_git_commit() -> Optional[str]:
        # get current git report
        proj_root = os.environ.get("PROJECT_ROOT")
        try:
            if proj_root is not None:
                repo = Repo(proj_root)
            else:
                repo = Repo(".")
        except Exception as _e:
            # 我们的部署树 Hand_Estimation/ 不是独立 git repo, 返回 None 而不是 raise.
            # 上层 _record_init_info / __init__ 的 logger.warning 会接住.
            return None

        modified_files = [item.a_path for item in repo.index.diff(None)]
        staged_files = [item.a_path for item in repo.index.diff("HEAD")]
        untracked_files = repo.untracked_files

        if len(modified_files):
            logger.error(f"modified_files: {' '.join(modified_files)}")
        if len(staged_files):
            logger.error(f"staged_files: {' '.join(staged_files)}")
        if len(untracked_files):
            logger.error(f"untracked_files: {' '.join(untracked_files)}")

        return (repo.head.commit.hexsha
                if not (len(modified_files) or len(staged_files) or len(untracked_files)) else None)
