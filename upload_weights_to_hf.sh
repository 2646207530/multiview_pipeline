#!/bin/bash
set -e

# ========================================
# 上传 run_hamer_to_npy.py 所需权重到 Hugging Face
# ========================================
# 覆盖:
#   • HaMER: ckpt + yaml + (可选) ONNX + _DATA/data 下 MANO/mean
#   • YOLOv7 手部检测
#   • RootNet SAR (resnet34，run_hamer 默认)
#   • MANO（smplx 手物重放 + SportGS Contact/力闭合）
# 用法:
#   1. https://huggingface.co/settings/tokens 创建 Write Token
#   2. bash upload_weights_to_hf.sh
# ========================================
# MANO 模型需遵守 MANO/SMPL 许可，仅用于研究用途。
# ========================================

HF_REPO="lilfiiiiish/golf-weights"
WORK_DIR="/home/pt/fbs"

pip install -q huggingface_hub

echo "请输入你的 Hugging Face Token (Write 权限):"
huggingface-cli login

echo ""
echo "========== 开始上传（单文件）=========="

# --- run_hamer_to_npy 主路径 ---
FILES=(
    # HaMER PyTorch（与 model/config/hamer_config.py 一致）
    "model/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
    "model/hamer/_DATA/hamer_ckpts/checkpoints/model_config.yaml"
    # 部分安装里 model_config 在上级目录
    "model/hamer/_DATA/hamer_ckpts/model_config.yaml"
    "model/hamer/_DATA/hamer_ckpts/dataset_config.yaml"
    # 可选：ONNX 推理（hamer_opt.use_onnx=True 时）
    "model/hamer/_DATA/hamer_ckpts/onnx/hamer_inferpy.onnx"
    # 可选：TorchScript
    "model/hamer/traced_model.pt"
    # YOLO（与 model/config/yolo_config.py 一致）
    "model/config/checkpoints/yolov7_best.pt"
    "model/yolo/yolov7/traced_model.pt"
    # SAR / RootNet（get_model 默认 SAR-resnet34-Root.pth）
    "model/rootnet/SAR-resnet34-Root.pth"
    "model/rootnet/SAR-convnext-root.pth"
)

for f in "${FILES[@]}"; do
    FULL_PATH="${WORK_DIR}/${f}"
    if [ -f "$FULL_PATH" ]; then
        echo ""
        echo ">>> 上传: $f ($(du -h "$FULL_PATH" | cut -f1))"
        hf upload "$HF_REPO" "$FULL_PATH" "$f"
    else
        echo ">>> 跳过 (不存在): $f"
    fi
done

# --- 整目录上传（HaMER 内置 MANO 与 mano_mean_params 等）---
upload_dir_if_exists() {
    local rel="$1"
    local base="${WORK_DIR}/${rel}"
    if [ ! -d "$base" ]; then
        echo ">>> 跳过目录 (不存在): ${rel}/"
        return 0
    fi
    local cnt
    cnt=$(find "$base" -type f 2>/dev/null | wc -l)
    if [ "$cnt" -eq 0 ]; then
        echo ">>> 跳过空目录: ${rel}/"
        return 0
    fi
    echo ""
    echo ">>> 上传目录 ${rel}/ （共 ${cnt} 个文件）..."
    while IFS= read -r -d '' fp; do
        rp="${fp#${WORK_DIR}/}"
        echo "    ... $rp"
        hf upload "$HF_REPO" "$fp" "$rp"
    done < <(find "$base" -type f -print0)
}

echo ""
echo "========== 目录同步（若本地有则上传）=========="
# HaMER 配置里 MANO.DATA_DIR = _DATA/data/（含 mano/、mano_mean_params.npz 等）
upload_dir_if_exists "model/hamer/_DATA/data"
# smplx 与 SportGS 用的 MANO（MANO_MODEL_PATH 默认 <仓库根>/MANO）
upload_dir_if_exists "MANO"

echo ""
echo "========== 可选：其它高尔夫管线权重（与 run_hamer 无直接关系）=========="
OPTIONAL=(
    "model/rootnet/KeypointFusion/checkpoint/dexycb/KPFusion_Dexycb_s0.pth"
)
for f in "${OPTIONAL[@]}"; do
    FULL_PATH="${WORK_DIR}/${f}"
    if [ -f "$FULL_PATH" ]; then
        echo ""
        echo ">>> 上传(可选): $f"
        hf upload "$HF_REPO" "$FULL_PATH" "$f"
    else
        echo ">>> 跳过(可选): $f"
    fi
done

echo ""
echo "========== 完成 =========="
echo "仓库: https://huggingface.co/${HF_REPO}"
echo ""
echo "在新机器恢复 run_hamer 所需文件（保持目录结构）:"
echo "  pip install -U huggingface_hub && huggingface-cli login"
echo "  hf download ${HF_REPO} --local-dir ${WORK_DIR}"
echo ""
echo "然后确认:"
echo "  • HaMER: model/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt + model_config.yaml"
echo "  • YOLO:  model/config/checkpoints/yolov7_best.pt"
echo "  • SAR:   model/rootnet/SAR-resnet34-Root.pth"
echo "  • MANO:  MANO/MANO_RIGHT.pkl（及 HaMER _DATA/data/ 若已上传）"
