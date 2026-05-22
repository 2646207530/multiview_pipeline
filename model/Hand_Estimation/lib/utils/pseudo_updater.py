import torch
import logging
import os
import json
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 设置Matplotlib使用非交互式后端（适用于无GUI环境）
plt.switch_backend('Agg')

class PseudoLabelUpdater:
    def __init__(self, update_threshold=0.2, conf_threshold=0.2):
        """
        伪标签更新统计管理器
        
        参数:
            update_threshold: 不确定性更新阈值 (默认前20%)
            conf_threshold: 置信度更新阈值 (默认0.2)
        """
        self.update_threshold = update_threshold
        self.conf_threshold = conf_threshold
        self.reset_stats()
        
    def reset_stats(self):
        """重置统计信息"""
        self.total_points = 0
        self.total_updated = 0
        self.epoch_updates = []
        self.batch_stats = []
        
    def update(self, pgt_2d, pse_conf, joint_hm, pred_conf, pred_unc, epoch):
        """
        执行伪标签更新并收集统计信息
        
        返回:
            updated_pgt_2d: 更新后的伪标签关键点
            updated_pse_conf: 更新后的伪标签置信度
        """
        # 确保所有张量在相同设备上
        device = pgt_2d.device
        pse_conf = pse_conf.to(device)
        joint_hm = joint_hm.to(device)
        pred_conf = pred_conf.to(device)
        pred_unc = pred_unc.to(device)
        
        # 获取批次信息
        B, N, _ = pred_unc.shape
        batch_points = B * N
        
        # 1. 创建更新掩码
        flat_unc = pred_unc.reshape(-1)
        k = max(1, int(self.update_threshold * flat_unc.numel()))
        
        # 处理全零不确定性的特殊情况
        if flat_unc.max() == 0:
            unc_mask = torch.zeros_like(pred_unc, dtype=torch.bool)
        else:
            topk_values, _ = torch.topk(flat_unc, k, largest=True)
            threshold = topk_values[-1] if k > 0 else 0
            unc_mask = (pred_unc >= threshold)
        
        conf_mask = (pse_conf <= self.conf_threshold)
        update_mask = unc_mask & conf_mask
        
        # 2. 创建更新后的伪标签和置信度
        updated_pgt_2d = pgt_2d.clone()
        updated_pse_conf = pse_conf.clone()
        
        if epoch == 0:
            update_mask = torch.zeros_like(update_mask.squeeze(-1))

        # 3. 执行更新
        num_updated = 0
        if update_mask.any():
            # 使用向量化操作提高效率
            update_indices = torch.where(update_mask.squeeze(-1))
            
            # 获取需要更新的点
            current_pgt = pgt_2d[update_indices]
            current_conf = pse_conf[update_indices].squeeze(-1)
            hm_points = joint_hm[update_indices]
            hm_confs = pred_conf[update_indices].squeeze(-1)
            
            # 计算加权平均
            total_weights = current_conf + hm_confs
            weights_pgt = current_conf / total_weights
            weights_hm = hm_confs / total_weights
            
            # 处理除零情况
            zero_weight_mask = (total_weights < 1e-6)
            weights_pgt[zero_weight_mask] = 0
            weights_hm[zero_weight_mask] = 1
            
            # 计算新点
            new_points = weights_pgt.unsqueeze(-1) * current_pgt + weights_hm.unsqueeze(-1) * hm_points
            
            # 更新伪标签
            updated_pgt_2d[update_indices] = new_points
            
            # 更新置信度 (求和平均)
            new_confs = (current_conf + hm_confs) / 2
            updated_pse_conf[update_indices] = new_confs.unsqueeze(-1)
            
            num_updated = len(update_indices[0])
        
        # 4. 更新统计信息
        self.total_points += batch_points
        self.total_updated += num_updated
        self.epoch_updates.append(num_updated)
        
        # 计算置信度变化
        orig_conf_mean = pse_conf.mean().item()
        updated_conf_mean = updated_pse_conf.mean().item()
        updated_points_conf = updated_pse_conf[update_mask].mean().item() if num_updated > 0 else 0
        
        # 记录批次统计
        batch_stat = {
            "batch_points": batch_points,
            "num_updated": num_updated,
            "update_ratio": num_updated / batch_points if batch_points > 0 else 0,
            "avg_uncertainty": pred_unc.mean().item(),
            "avg_pse_conf": orig_conf_mean,
            "avg_updated_conf": updated_conf_mean,
            "avg_updated_points_conf": updated_points_conf,
            "conf_change": updated_conf_mean - orig_conf_mean
        }
        self.batch_stats.append(batch_stat)
        
        return updated_pgt_2d, updated_pse_conf
    
    def get_epoch_summary(self):
        """获取整个epoch的统计摘要"""
        if self.total_points == 0:
            return {
                "total_points": 0,
                "total_updated": 0,
                "update_ratio": 0,
                "avg_batch_update": 0,
                "max_batch_update": 0,
                "min_batch_update": 0
            }
        
        update_ratio = self.total_updated / self.total_points
        avg_batch_update = sum(self.epoch_updates) / len(self.epoch_updates)
        
        return {
            "total_points": self.total_points,
            "total_updated": self.total_updated,
            "update_ratio": update_ratio,
            "avg_batch_update": avg_batch_update,
            "max_batch_update": max(self.epoch_updates),
            "min_batch_update": min(self.epoch_updates),
            "batch_stats": self.batch_stats
        }
    
    def log_epoch_summary(self, epoch, logger=None):
        """记录epoch摘要到日志"""
        summary = self.get_epoch_summary()
        
        if logger is None:
            logger = logging.getLogger()
        
        logger.info(
            f"Epoch {epoch} 伪标签更新统计: "
            f"总共更新 {summary['total_updated']}/{summary['total_points']} 个关键点 "
            f"({summary['update_ratio']:.2%}) | "
            f"平均每批更新: {summary['avg_batch_update']:.1f} 点 "
            f"(最小: {summary['min_batch_update']}, 最大: {summary['max_batch_update']})"
        )
        
        # 重置统计
        self.reset_stats()
        
        return summary
    
    def get_epoch_summary(self) -> Dict:
        """获取整个epoch的统计摘要"""
        if self.total_points == 0:
            return {
                "total_points": 0,
                "total_updated": 0,
                "update_ratio": 0,
                "avg_batch_update": 0,
                "max_batch_update": 0,
                "min_batch_update": 0,
                "batch_stats": []
            }
        
        update_ratio = self.total_updated / self.total_points
        avg_batch_update = sum(self.epoch_updates) / len(self.epoch_updates) if self.epoch_updates else 0
        
        return {
            "epoch": len(self.epoch_updates),
            "total_points": self.total_points,
            "total_updated": self.total_updated,
            "update_ratio": update_ratio,
            "avg_batch_update": avg_batch_update,
            "max_batch_update": max(self.epoch_updates) if self.epoch_updates else 0,
            "min_batch_update": min(self.epoch_updates) if self.epoch_updates else 0,
            "batch_stats": self.batch_stats
        }
    
    def log_epoch_summary(self, epoch: int, logger: Optional[logging.Logger] = None) -> Dict:
        """记录epoch摘要到日志"""
        summary = self.get_epoch_summary()
        
        if logger is None:
            logger = logging.getLogger(__name__)
        
        logger.info(
            f"Epoch {epoch} 伪标签更新统计: "
            f"总共更新 {summary['total_updated']}/{summary['total_points']} 个关键点 "
            f"({summary['update_ratio']:.2%}) | "
            f"平均每批更新: {summary['avg_batch_update']:.1f} 点 "
            f"(最小: {summary['min_batch_update']}, 最大: {summary['max_batch_update']}) | "
            f"平均置信度变化: {summary['batch_stats'][-1]['conf_change'] if summary['batch_stats'] else 0:.4f}"
        )
        
        # 重置统计
        self.reset_stats()
        
        return summary


def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO) -> logging.Logger:
    """配置全局日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pseudo_label_update_{timestamp}.log")
    
    # 创建记录器
    logger = logging.getLogger("pseudo_label")
    logger.setLevel(log_level)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"日志系统已初始化，日志文件: {log_file}")
    return logger


def save_stats(epoch: int, summary: Dict, stats_dir: str = "stats") -> None:
    """保存详细统计信息到文件"""
    os.makedirs(stats_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存JSON格式
    json_file = os.path.join(stats_dir, f"pseudo_stats_ep{epoch}_{timestamp}.json")
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 保存CSV格式（用于Excel分析）
    csv_file = os.path.join(stats_dir, f"pseudo_stats_ep{epoch}_{timestamp}.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Batch", "Points", "Updated", "Ratio", 
            "Avg_Unc", "Avg_Conf", "Avg_Updated_Conf", 
            "Conf_Change"
        ])
        
        for i, stat in enumerate(summary["batch_stats"]):
            writer.writerow([
                i,
                stat["batch_points"],
                stat["num_updated"],
                stat["update_ratio"],
                stat["avg_uncertainty"],
                stat["avg_pse_conf"],
                stat["avg_updated_conf"],
                stat["conf_change"]
            ])
    
    logging.getLogger("pseudo_label").info(f"保存详细统计到: {json_file} 和 {csv_file}")


def plot_pseudo_stats(epoch: int, summary: Dict, save_dir: str = "plots") -> str:
    """绘制伪标签更新统计图"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建DataFrame
    stats = summary["batch_stats"]
    if not stats:
        return ""
    
    df = pd.DataFrame(stats)
    df["Batch"] = df.index
    
    # 创建图表
    plt.figure(figsize=(15, 12))
    
    # 更新比例 (修改点1)
    plt.subplot(3, 2, 1)
    plt.plot(df["Batch"].to_numpy(), df["update_ratio"].to_numpy(), 'b-', linewidth=2)
    plt.title("Batch Update Ratio", fontsize=14)
    plt.xlabel("Batch Index", fontsize=12)
    plt.ylabel("Update Ratio", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 更新数量 (修改点2)
    plt.subplot(3, 2, 2)
    plt.bar(df["Batch"].to_numpy(), df["num_updated"].to_numpy(), color='orange', alpha=0.7)
    plt.title("Number of Updated Points per Batch", fontsize=14)
    plt.xlabel("Batch Index", fontsize=12)
    plt.ylabel("Updated Points", fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 置信度变化 (修改点3)
    plt.subplot(3, 2, 3)
    plt.plot(df["Batch"].to_numpy(), df["avg_pse_conf"].to_numpy(), 'r-', label="Original Conf", linewidth=2)
    plt.plot(df["Batch"].to_numpy(), df["avg_updated_conf"].to_numpy(), 'g-', label="Updated Conf", linewidth=2)
    plt.title("Confidence Comparison", fontsize=14)
    plt.xlabel("Batch Index", fontsize=12)
    plt.ylabel("Confidence", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 不确定性 (修改点4)
    plt.subplot(3, 2, 4)
    plt.plot(df["Batch"].to_numpy(), df["avg_uncertainty"].to_numpy(), 'm-', linewidth=2)
    plt.title("Average Uncertainty", fontsize=14)
    plt.xlabel("Batch Index", fontsize=12)
    plt.ylabel("Uncertainty", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 置信度变化 (修改点5)
    plt.subplot(3, 2, 5)
    plt.plot(df["Batch"].to_numpy(), df["conf_change"].to_numpy(), 'c-', linewidth=2)
    plt.title("Confidence Change", fontsize=14)
    plt.xlabel("Batch Index", fontsize=12)
    plt.ylabel("Confidence Change", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(0, color='gray', linestyle='--')
    
    # 总览统计 (无需修改，因为使用列表数据)
    plt.subplot(3, 2, 6)
    overview_data = [
        summary["total_points"],
        summary["total_updated"],
        summary["update_ratio"],
        df["avg_uncertainty"].mean(),
        df["avg_pse_conf"].mean(),
        df["avg_updated_conf"].mean()
    ]
    overview_labels = [
        "Total Points", "Total Updated", "Update Ratio", 
        "Avg Uncertainty", "Avg Orig Conf", "Avg Updated Conf"
    ]
    
    plt.barh(overview_labels, overview_data, color='purple', alpha=0.6)
    plt.title("Epoch Overview", fontsize=14)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for i, v in enumerate(overview_data):
        plt.text(v, i, f"{v:.4f}", va='center', fontsize=10)
    
    # 保存图表
    plt.tight_layout()
    plot_file = os.path.join(save_dir, f"pseudo_stats_ep{epoch}.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_file