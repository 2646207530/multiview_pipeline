"""从 Hugging Face Hub 把权重拉到本地正确位置.

新机器 clone 完代码后:
    pip install -U huggingface_hub
    # 私有 repo 需要先登录:
    huggingface-cli login    # 粘 HF token, 至少 read 权限
    # 或者 export HF_TOKEN=...
    python scripts/fetch_weights.py

跑完会自动按目录布局摆好. 跑完再用 README 里的 Python 验证脚本确认齐了.

MANO 文件因为 license 没在 HF repo 里 (除非你 upload 时启了 INCLUDE_MANO).
没有 MANO 的话, 单独去 https://mano.is.tue.mpg.de 注册下载, 解压到:
    MANO/{MANO_LEFT,MANO_RIGHT,MANO_PART,v_color}.pkl
    model/Hand_Estimation/mano_data/MANO_RIGHT.pkl
    model/Hand_Estimation/mano/models/{MANO_LEFT,MANO_RIGHT}.pkl
    model/hamer/_DATA/data/mano/{MANO_LEFT,MANO_RIGHT}.pkl
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# 默认 HF repo (覆盖请用 --repo 参数 或 env HF_REPO).
DEFAULT_HF_REPO = "lilfiiiiish/pipeline"
DEFAULT_REPO_TYPE = "model"

# repo_path → 本地 (相对 pipeline/ 根) 目标
ITEMS: list[tuple[str, str]] = [
    # HE ckpt (师兄)
    ("Hand_Estimation/exp/new/checkpoints/checkpoint_30",
     "model/Hand_Estimation/exp/new/checkpoints/checkpoint_30"),
    # DINOv3
    ("Hand_Estimation/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
     "model/Hand_Estimation/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth"),
    ("Hand_Estimation/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
     "model/Hand_Estimation/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"),
    # rootnet
    ("rootnet/SAR-convnext-root.pth",
     "model/rootnet/SAR-convnext-root.pth"),
    ("rootnet/SAR-resnet34-Root.pth",
     "model/rootnet/SAR-resnet34-Root.pth"),
    # yolov7
    ("config/checkpoints/yolov7_best.pt",
     "model/config/checkpoints/yolov7_best.pt"),
    # HaMER
    ("hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt",
     "model/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"),
    ("hamer/_DATA/hamer_ckpts/dataset_config.yaml",
     "model/hamer/_DATA/hamer_ckpts/dataset_config.yaml"),
    ("hamer/_DATA/hamer_ckpts/model_config.yaml",
     "model/hamer/_DATA/hamer_ckpts/model_config.yaml"),
    ("hamer/_DATA/hamer_ckpts/onnx/hamer_inferpy.onnx",
     "model/hamer/_DATA/hamer_ckpts/onnx/hamer_inferpy.onnx"),
    ("hamer/_DATA/data/mano_mean_params.npz",
     "model/hamer/_DATA/data/mano_mean_params.npz"),
    ("Hand_Estimation/mano_data/mano_mean_params.npz",
     "model/Hand_Estimation/mano_data/mano_mean_params.npz"),
    # WiLoR (原始源: huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/*)
    ("WiLoR/detector.pt",
     "model/WiLoR/pretrained_models/detector.pt"),
    ("WiLoR/wilor_final.ckpt",
     "model/WiLoR/pretrained_models/wilor_final.ckpt"),
    # WiLoR mano_data (MANO_RIGHT 受 license 约束, 按用户要求统一拉)
    ("WiLoR/mano_data/mano_mean_params.npz",
     "model/WiLoR/mano_data/mano_mean_params.npz"),
    ("WiLoR/mano_data/MANO_RIGHT.pkl",
     "model/WiLoR/mano_data/MANO_RIGHT.pkl"),
]

# 如果你上传时启了 INCLUDE_MANO, 加 --with-mano 也会拉这些
MANO_ITEMS: list[tuple[str, str]] = [
    ("MANO/MANO_LEFT.pkl",   "MANO/MANO_LEFT.pkl"),
    ("MANO/MANO_RIGHT.pkl",  "MANO/MANO_RIGHT.pkl"),
    ("MANO/MANO_PART.pkl",   "MANO/MANO_PART.pkl"),
    ("MANO/v_color.pkl",     "MANO/v_color.pkl"),
    ("hamer/_DATA/data/mano/MANO_LEFT.pkl",
     "model/hamer/_DATA/data/mano/MANO_LEFT.pkl"),
    ("hamer/_DATA/data/mano/MANO_RIGHT.pkl",
     "model/hamer/_DATA/data/mano/MANO_RIGHT.pkl"),
    ("Hand_Estimation/mano_data/MANO_RIGHT.pkl",
     "model/Hand_Estimation/mano_data/MANO_RIGHT.pkl"),
    ("Hand_Estimation/mano/models/MANO_LEFT.pkl",
     "model/Hand_Estimation/mano/models/MANO_LEFT.pkl"),
    ("Hand_Estimation/mano/models/MANO_RIGHT.pkl",
     "model/Hand_Estimation/mano/models/MANO_RIGHT.pkl"),
    ("Hand_Estimation/assets/mano_v1_2.zip",
     "model/Hand_Estimation/assets/mano_v1_2.zip"),
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _download_one(repo_path: str, local_path: Path, repo_id: str,
                  repo_type: str, token: str | None, force: bool):
    """从 HF repo 拉一个文件或子目录到 local_path."""
    from huggingface_hub import hf_hub_download, snapshot_download
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # 子目录: 用 snapshot_download + allow_patterns
    is_dir_like = not Path(repo_path).suffix  # 没扩展名当目录
    if is_dir_like:
        if local_path.exists() and any(local_path.iterdir()) and not force:
            print(f"  skip (exists): {local_path}")
            return
        print(f"  ↓ dir  {repo_id}:{repo_path}/  →  {local_path}")
        snap = snapshot_download(
            repo_id=repo_id, repo_type=repo_type, token=token,
            allow_patterns=[f"{repo_path}/*", f"{repo_path}/**/*"],
        )
        src = Path(snap) / repo_path
        if not src.is_dir():
            raise FileNotFoundError(f"HF repo 里没找到目录: {repo_path}")
        if local_path.exists():
            shutil.rmtree(local_path)
        shutil.copytree(src, local_path)
        return

    # 单文件
    if local_path.is_file() and not force:
        print(f"  skip (exists): {local_path}")
        return
    print(f"  ↓ file {repo_id}:{repo_path}  →  {local_path}")
    cached = hf_hub_download(
        repo_id=repo_id, filename=repo_path, repo_type=repo_type,
        token=token,
    )
    if local_path.exists() or local_path.is_symlink():
        local_path.unlink()
    shutil.copy2(cached, local_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("HF_REPO", DEFAULT_HF_REPO),
                    help=f"HF repo (默认 {DEFAULT_HF_REPO}, env HF_REPO)")
    ap.add_argument("--repo-type", default=DEFAULT_REPO_TYPE,
                    choices=["model", "dataset"])
    ap.add_argument("--with-mano", action="store_true",
                    help="也拉 MANO 文件 (你的 HF repo 里必须有)")
    ap.add_argument("--force", action="store_true",
                    help="目标已存在也覆盖重下")
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download  # noqa: F401
    except ImportError:
        print("缺 huggingface_hub: pip install -U huggingface_hub", file=sys.stderr)
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")  # 也可以提前 huggingface-cli login

    items = list(ITEMS)
    if args.with_mano:
        items.extend(MANO_ITEMS)

    print(f"HF repo: {args.repo} (type={args.repo_type})")
    print(f"项目根: {PROJECT_ROOT}")
    print(f"共 {len(items)} 项, with_mano={args.with_mano}\n")

    failed = []
    for repo_path, local_rel in items:
        local_path = PROJECT_ROOT / local_rel
        try:
            _download_one(repo_path, local_path, args.repo, args.repo_type,
                          token, args.force)
        except Exception as e:
            print(f"  ⚠️  失败: {repo_path} → {e}")
            failed.append((repo_path, str(e)))

    if failed:
        print(f"\n❌ {len(failed)} 项失败:")
        for rp, e in failed:
            print(f"  - {rp}: {e}")
        sys.exit(1)
    print("\n✅ 全部下载完毕.")


if __name__ == "__main__":
    main()
