#!/usr/bin/env bash
# =============================================================================
# 批量跑高尔夫多视角手部 pipeline: raw 抽帧 → 去畸变 → (人体姿态/vitpose) →
# bbox 检测 → WiLoR 伪标 → UST-hand 多视角推理. 不做最后的 way_vis 可视化.
#
# 适配新数据布局: camera_params.json 放在「所有序列的上级目录」(共享标定),
# 每个序列目录下只有各相机的 .raw. 脚本会把共享标定软链进每个序列目录,
# 让 pipeline 的 step0 能就地找到 camera_params.json.
#
# 涉及两个 conda 环境:
#   golf_pipeline : 本仓库的 pipeline 步骤 (抽帧/去畸变/检测/伪标/推理)
#   gvhmr         : human-dataset-tools 的人体姿态, 产出 preprocess/<cam>/vitpose.pt
#
# 用法:
#   bash scripts/batch_run.sh <dataset_root> [bbox_patch] [gpu]
#     dataset_root : 含 camera_params.json + 多个 <seq>/ 子目录的根目录
#     bbox_patch   : step2 bbox 扩张系数 (中心不变长宽各 ×k), 默认 1.0
#     gpu          : CUDA_VISIBLE_DEVICES, 默认 0
#
# 可选环境变量:
#   FORCE=1          各步强制重跑 (覆盖已有结果)
#   ONLY_SEQ=<seq>   只跑指定的一个序列
#   GVHMR_FPS=120    人体姿态采样 fps (默认 120)
#   GVHMR_VIS_FPS=30 人体姿态可视化 fps (默认 30)
#   PIPELINE_DIR / HDT_DIR / GOLF_ENV / GVHMR_ENV  覆盖默认路径/环境名
#
# 例:
#   bash scripts/batch_run.sh /data2/fubingshuai/golf/data/20260529 1.5 0
#   FORCE=1 ONLY_SEQ=20260529114030110 bash scripts/batch_run.sh /data2/.../20260529
# =============================================================================

# 注意: 不开 `set -e`. 单个序列出错只跳过它, 不中断整个批处理.
# 也不开 `set -u`: conda 的 activate.d 钩子(如 libblas_mkl_activate.sh)引用未设
# 变量, 在 nounset 下会报 "MKL_INTERFACE_LAYER: unbound variable" 而中断.
set -o pipefail

# ── 参数 ────────────────────────────────────────────────────────────────
DATASET_ROOT="${1:-}"
BBOX_PATCH="${2:-1.0}"
GPU="${3:-0}"

if [ -z "$DATASET_ROOT" ]; then
    echo "用法: bash scripts/batch_run.sh <dataset_root> [bbox_patch] [gpu]" >&2
    exit 1
fi
DATASET_ROOT="$(cd "$DATASET_ROOT" && pwd)"   # 绝对化

# ── 可覆盖配置 ────────────────────────────────────────────────────────────
PIPELINE_DIR="${PIPELINE_DIR:-/data2/fubingshuai/golf/pipeline_release}"
HDT_DIR="${HDT_DIR:-/data2/fubingshuai/golf/human-dataset-tools}"
GOLF_ENV="${GOLF_ENV:-golf_pipeline}"
GVHMR_ENV="${GVHMR_ENV:-gvhmr}"
GVHMR_FPS="${GVHMR_FPS:-120}"
GVHMR_VIS_FPS="${GVHMR_VIS_FPS:-30}"
ONLY_SEQ="${ONLY_SEQ:-}"

FORCE_ARG=""
[ "${FORCE:-0}" = "1" ] && FORCE_ARG="--force"

DRIVER="$PIPELINE_DIR/scripts/run_pipeline_stage.py"
CAM_JSON="$DATASET_ROOT/camera_params.json"
LOG_DIR="$DATASET_ROOT/_batch_logs"
mkdir -p "$LOG_DIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"

# ── 前置检查 ──────────────────────────────────────────────────────────────
[ -f "$CAM_JSON" ]  || { echo "[fatal] 找不到共享标定: $CAM_JSON" >&2; exit 1; }
[ -f "$DRIVER" ]    || { echo "[fatal] 找不到驱动脚本: $DRIVER" >&2; exit 1; }
[ -d "$HDT_DIR" ]   || { echo "[fatal] 找不到 human-dataset-tools: $HDT_DIR" >&2; exit 1; }

# ── 激活 conda ────────────────────────────────────────────────────────────
# 非交互 bash 里 `conda` 通常不是命令(它是 .bashrc 注入的 shell 函数), 因此
# 优先用激活态会继承的环境变量推 conda base, 再退回命令 / 常见默认路径.
CONDA_BASE=""
if [ -n "${CONDA_EXE:-}" ]; then
    CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null)"
elif [ -n "${CONDA_PREFIX:-}" ]; then
    CONDA_BASE="${CONDA_PREFIX%/envs/*}"
else
    for c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda \
             /data2/fubingshuai/miniconda3; do
        [ -f "$c/etc/profile.d/conda.sh" ] && CONDA_BASE="$c" && break
    done
fi
if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "[fatal] 没找到 conda (CONDA_BASE='$CONDA_BASE'). 可手动 export CONDA_EXE=<.../bin/conda> 再跑" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "================================================================"
echo " dataset   : $DATASET_ROOT"
echo " bbox patch: $BBOX_PATCH    gpu: $GPU    force: ${FORCE:-0}"
echo " envs      : pipeline=$GOLF_ENV  human-pose=$GVHMR_ENV"
echo " logs      : $LOG_DIR/<seq>.log"
echo "================================================================"

n_ok=0; n_fail=0; failed_seqs=()

for seq_dir in "$DATASET_ROOT"/*/; do
    seq="$(basename "$seq_dir")"
    seq_dir="${seq_dir%/}"   # 去掉结尾斜杠

    # 跳过非序列目录
    case "$seq" in
        _batch_logs|.*) continue ;;
    esac
    [ -n "$ONLY_SEQ" ] && [ "$seq" != "$ONLY_SEQ" ] && continue

    # 必须有完整的 .raw (排除还在写入的 .xxx.raw.tmp 之类隐藏临时文件)
    if ! ls "$seq_dir"/*.raw >/dev/null 2>&1; then
        echo "[skip] $seq : 无 .raw, 跳过"
        continue
    fi

    log="$LOG_DIR/$seq.log"
    undist_data="$seq_dir/.undistorted/$seq"
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>> $seq <<<<<<<<<<<<<<<<<<<<"
    echo "   日志: $log"
    : > "$log"

    # 1) 共享标定就位 (上级 camera_params.json → 复制进序列目录; 已存在则不动)
    if [ ! -e "$seq_dir/camera_params.json" ]; then
        cp "$CAM_JSON" "$seq_dir/camera_params.json"
        echo "   [calib] 复制 camera_params.json ← 上级共享标定" | tee -a "$log"
    fi

    # 2) golf_pipeline: setup (raw 抽帧) + undistort
    conda activate "$GOLF_ENV"
    echo "   [1/3] setup + undistort (env=$GOLF_ENV)" | tee -a "$log"
    python "$DRIVER" --capture_dir "$seq_dir" --seq "$seq" \
        --stages setup,undistort --bbox_size "$BBOX_PATCH" --gpu_id "$GPU" \
        $FORCE_ARG >>"$log" 2>&1
    rc=$?
    if [ $rc -ne 0 ] || [ ! -d "$undist_data" ]; then
        echo "   [fail] setup/undistort 失败 (rc=$rc), 跳过 $seq" | tee -a "$log"
        n_fail=$((n_fail+1)); failed_seqs+=("$seq"); continue
    fi

    # 3) gvhmr: 人体姿态 → vitpose.pt (在 human-dataset-tools 下跑)
    conda activate "$GVHMR_ENV"
    echo "   [2/3] human pose / vitpose (env=$GVHMR_ENV) DATA=$undist_data" | tee -a "$log"
    ( cd "$HDT_DIR" && python -m multi_view_smpl_optimizer.dataset.ours \
        --dataset_dir "$undist_data" --stage vis_world \
        --fps "$GVHMR_FPS" --vis_fps "$GVHMR_VIS_FPS" ) >>"$log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "   [fail] 人体姿态 stage 失败 (rc=$rc), 跳过 $seq" | tee -a "$log"
        n_fail=$((n_fail+1)); failed_seqs+=("$seq"); continue
    fi

    # 4) golf_pipeline: detect (vitpose) + pseudo (wilor) + infer (UST-hand)
    conda activate "$GOLF_ENV"
    echo "   [3/3] detect + pseudo + infer (env=$GOLF_ENV)" | tee -a "$log"
    python "$DRIVER" --capture_dir "$seq_dir" --seq "$seq" \
        --stages detect,pseudo,infer --bbox_size "$BBOX_PATCH" --gpu_id "$GPU" \
        $FORCE_ARG >>"$log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "   [fail] detect/pseudo/infer 失败 (rc=$rc), 跳过 $seq" | tee -a "$log"
        n_fail=$((n_fail+1)); failed_seqs+=("$seq"); continue
    fi

    echo "   [ok] $seq → npy: $seq_dir/.pipeline/$seq.npy" | tee -a "$log"
    n_ok=$((n_ok+1))
done

echo ""
echo "================================================================"
echo " 完成: 成功 $n_ok, 失败 $n_fail"
[ ${#failed_seqs[@]} -gt 0 ] && echo " 失败序列: ${failed_seqs[*]}"
echo " 每个序列产物: <seq>/.pipeline/<seq>.npy + 各步可视化"
echo "   - 检测 overlay : <seq>/.pipeline/_detect_vis/"
echo "   - 伪标 overlay : <seq>/.undistorted/_pseudo_vis/"
echo "   - 推理 hand mp4: <seq>/.undistorted/_he_output/"
echo "================================================================"
[ $n_fail -eq 0 ]
