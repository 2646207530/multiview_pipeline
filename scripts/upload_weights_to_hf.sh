#!/usr/bin/env bash
# 把本地权重传到 Hugging Face Hub.
#
# 用法:
#   1. 改下面的 HF_REPO 为你的 repo (要先在 HF 网页 create empty model repo)
#   2. huggingface-cli login    # 粘 HF token (https://huggingface.co/settings/tokens, 至少 write 权限)
#   3. cd /data2/fubingshuai/golf/pipeline && bash scripts/upload_weights_to_hf.sh
#
# 默认跳过 MANO/SMPLH (license 限制不允许再分发).
# 想包括 MANO, 设 INCLUDE_MANO=1 再跑, 但务必确认你的 HF repo 是 private.

set -e

# ─── 配置 ────────────────────────────────────────────────────────────
HF_REPO="${HF_REPO:-lilfiiiiish/pipeline}"   # 改成你自己的
REPO_TYPE="${REPO_TYPE:-model}"                            # model | dataset
INCLUDE_MANO="${INCLUDE_MANO:-0}"                          # 1 = 包括 MANO (注意 license)

# ─── 上传清单 ────────────────────────────────────────────────────────
# 每项: <repo_path>:<local_path>
declare -a ITEMS=(
  # Hand_Estimation 训练 ckpt (师兄)
  "Hand_Estimation/exp/new/checkpoints/checkpoint_30:model/Hand_Estimation/exp/new/checkpoints/checkpoint_30"
  # DINOv3 (公开)
  "Hand_Estimation/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth:model/Hand_Estimation/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth"
  "Hand_Estimation/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth:model/Hand_Estimation/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"
  # rootnet (师兄)
  "rootnet/SAR-convnext-root.pth:model/rootnet/SAR-convnext-root.pth"
  "rootnet/SAR-resnet34-Root.pth:model/rootnet/SAR-resnet34-Root.pth"
  # yolov7 (师兄 fine-tune)
  "config/checkpoints/yolov7_best.pt:model/config/checkpoints/yolov7_best.pt"
  # HaMER (官方)
  "hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt:model/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
  "hamer/_DATA/hamer_ckpts/dataset_config.yaml:model/hamer/_DATA/hamer_ckpts/dataset_config.yaml"
  "hamer/_DATA/hamer_ckpts/model_config.yaml:model/hamer/_DATA/hamer_ckpts/model_config.yaml"
  "hamer/_DATA/hamer_ckpts/onnx/hamer_inferpy.onnx:model/hamer/_DATA/hamer_ckpts/onnx/hamer_inferpy.onnx"
  "hamer/_DATA/data/mano_mean_params.npz:model/hamer/_DATA/data/mano_mean_params.npz"
  "Hand_Estimation/mano_data/mano_mean_params.npz:model/Hand_Estimation/mano_data/mano_mean_params.npz"
)

# MANO (license restricted) — 只在 INCLUDE_MANO=1 时加入
if [[ "$INCLUDE_MANO" == "1" ]]; then
  echo "⚠️  你启用了 INCLUDE_MANO=1, 上传 MANO 文件. 务必确认 HF repo 是 private,"
  echo "   且你的下游使用者也都拿到了 MANO 注册资格."
  ITEMS+=(
    "MANO/MANO_LEFT.pkl:MANO/MANO_LEFT.pkl"
    "MANO/MANO_RIGHT.pkl:MANO/MANO_RIGHT.pkl"
    "MANO/MANO_PART.pkl:MANO/MANO_PART.pkl"
    "MANO/v_color.pkl:MANO/v_color.pkl"
    "hamer/_DATA/data/mano/MANO_LEFT.pkl:model/hamer/_DATA/data/mano/MANO_LEFT.pkl"
    "hamer/_DATA/data/mano/MANO_RIGHT.pkl:model/hamer/_DATA/data/mano/MANO_RIGHT.pkl"
    "Hand_Estimation/mano_data/MANO_RIGHT.pkl:model/Hand_Estimation/mano_data/MANO_RIGHT.pkl"
    "Hand_Estimation/mano/models/MANO_LEFT.pkl:model/Hand_Estimation/mano/models/MANO_LEFT.pkl"
    "Hand_Estimation/mano/models/MANO_RIGHT.pkl:model/Hand_Estimation/mano/models/MANO_RIGHT.pkl"
    "Hand_Estimation/assets/mano_v1_2.zip:model/Hand_Estimation/assets/mano_v1_2.zip"
  )
fi

# ─── 执行 ────────────────────────────────────────────────────────────
echo "上传到 HF repo: $HF_REPO  (type=$REPO_TYPE)"
echo "共 ${#ITEMS[@]} 个 item"

for entry in "${ITEMS[@]}"; do
  repo_path="${entry%%:*}"
  local_path="${entry#*:}"
  if [[ ! -e "$local_path" ]]; then
    echo "  ⚠️  missing, skip: $local_path"
    continue
  fi
  echo "  → $local_path  →  $HF_REPO:$repo_path"
  huggingface-cli upload "$HF_REPO" "$local_path" "$repo_path" \
      --repo-type="$REPO_TYPE"
done

echo "✅ done. https://huggingface.co/$HF_REPO"
