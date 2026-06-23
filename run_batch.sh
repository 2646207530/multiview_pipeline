#!/bin/bash
# pipeline_ik 批处理: 对 <DATA_DIR> 下每个采集序列跑 5 步:
#   1 抽RGB  2 去畸变  3 人体姿态(GVHMR)  4 球杆追踪(ArUco)  5 手跟杆 + 人体手臂IK
# 球杆型号 = <DATA_DIR> 目录名: 自动用 standard_pose/<型号> 的预定义手 + club-assets/<型号> 的 mesh/rig.
# 幂等 (产物在则跳过), 单序列某步失败记日志继续下一序列.
#
# 用法: bash run_batch.sh <DATA_DIR> [GPU] [seq ...]
#   例: bash run_batch.sh /data2/fubingshuai/golf/data/35_wood_8_01 2
set -u

DATA="${1:?用法: bash run_batch.sh <DATA_DIR> [GPU] [seq ...]}"; shift
GPU="${1:-2}"; shift || true
DATA="${DATA%/}"

PR=/data2/fubingshuai/golf/pipeline_ik
GOLF=/data2/fubingshuai/golf
HDT=$GOLF/human-dataset-tools            # 外部: GVHMR 人体姿态 (gvhmr 环境)
OMNI=$GOLF/omni-club-tracking            # 外部: ArUco 球杆追踪 (C++ club_tracker)
CLUB_ASSETS=$GOLF/data/club-assets       # aruco rig + 球杆 mesh
GOLF_PY=/data2/fubingshuai/miniconda3/envs/golf_pipeline/bin/python
GVHMR_PY=/data2/fubingshuai/miniconda3/envs/gvhmr/bin/python
GOLF_LD=/data2/fubingshuai/miniconda3/envs/golf_pipeline/lib
CLUB_TRACKER=$OMNI/cmake-build-Release/club_tracker

CLUB=$(basename "$DATA")
ARUCO_RIG=$(ls "$CLUB_ASSETS/$CLUB"/aruco_tags*.json 2>/dev/null | head -1)
LOG=$DATA/_pipeline_ik_log.txt
export EGL_DEVICE_ID=$GPU
echo "[batch] club=$CLUB  data=$DATA  aruco_rig=$ARUCO_RIG" | tee -a "$LOG"

SEQS=("$@")
if [ ${#SEQS[@]} -eq 0 ]; then
    SEQS=(); for d in "$DATA"/*/; do d=$(basename "$d"); ls "$DATA/$d"/*.raw >/dev/null 2>&1 && SEQS+=("$d"); done
fi
echo "=== pipeline_ik start $(date) GPU=$GPU seqs=${SEQS[*]} ===" | tee -a "$LOG"

for SEQ in "${SEQS[@]}"; do
  S="$DATA/$SEQ"; DS="$S/.undistorted/$SEQ"
  echo "" | tee -a "$LOG"; echo "########## $SEQ $(date) ##########" | tee -a "$LOG"

  # 0) calib
  [ -f "$S/camera_params.json" ] || cp "$DATA/camera_params.json" "$S/" 2>/dev/null

  # 1-2) 抽彩色RGB + 去畸变
  if [ ! -d "$DS/0/images_undistorted" ]; then
    echo "[1-2] setup+undistort" | tee -a "$LOG"
    LD_LIBRARY_PATH=$GOLF_LD $GOLF_PY "$PR/scripts/run_stage.py" \
      --capture_dir "$S" --seq "$SEQ" --stages setup,undistort >>"$LOG" 2>&1 \
      || { echo "[FAIL 1-2] $SEQ" | tee -a "$LOG"; continue; }
  else echo "[1-2] skip (done)" | tee -a "$LOG"; fi
  VIS="$S/_pipeline_vis"; mkdir -p "$VIS"
  [ -f "$VIS/step12_undistorted_cam0_cam1.jpg" ] || ffmpeg -y \
     -i "$DS/0/images_undistorted/000300.jpg" -i "$DS/1/images_undistorted/000300.jpg" \
     -filter_complex hstack "$VIS/step12_undistorted_cam0_cam1.jpg" -loglevel error 2>/dev/null

  # 1b) 抽 mono 相机 (球杆多视角追踪用)
  TMPM=$(mktemp -d)
  for r in "$S"/*Mono8*.raw; do [ -e "$r" ] || continue
    stem=$(basename "$r" .raw)
    [ -d "$S/.tmp_images/$stem" ] || ln -sf "$r" "$TMPM/"
  done
  if [ -n "$(ls -A "$TMPM" 2>/dev/null)" ]; then
    echo "[1b] extract mono $(ls "$TMPM" | wc -l)" | tee -a "$LOG"
    LD_LIBRARY_PATH=$GOLF_LD $GOLF_PY "$OMNI/scripts/raw_to_images.py" \
      --data_dir "$TMPM" --output_base_dir "$S/.tmp_images" >>"$LOG" 2>&1
  fi
  rm -rf "$TMPM"

  # 3) 人体姿态 (GVHMR, gvhmr 环境)
  if [ ! -f "$DS/world_params/smplx_params_world.pt" ]; then
    echo "[3] human pose (GVHMR)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$HDT $GVHMR_PY -m multi_view_smpl_optimizer.dataset.ours \
      --dataset_dir "$DS" --views 0 1 --fps 120 --stage all >>"$LOG" 2>&1
    [ -f "$DS/world_params/smplx_params_world.pt" ] || echo "[FAIL 3] $SEQ" | tee -a "$LOG"
  else echo "[3] skip (done)" | tee -a "$LOG"; fi
  if [ -f "$DS/world_params/smplx_params_world.pt" ] && [ ! -f "$VIS/step3_humanpose_view0.mp4" ]; then
    CUDA_VISIBLE_DEVICES=$GPU LD_LIBRARY_PATH=$GOLF_LD $GOLF_PY "$PR/scripts/render_world_body.py" \
      --capture_dir "$S" --seq "$SEQ" --smplx_pt "$DS/world_params/smplx_params_world.pt" \
      --out_video "$VIS/step3_humanpose.mp4" --stride 4 >>"$LOG" 2>&1
  fi

  # 4) 球杆追踪 (字典 4X4/6X6 自动判别)
  if [ ! -f "$S/trajectory_output/results.json" ]; then
    echo "[4] club tracking" | tee -a "$LOG"
    CFG="$OMNI/config/_pipeline_ik_$SEQ.json"
    LD_LIBRARY_PATH=$GOLF_LD $GOLF_PY "$PR/scripts/gen_club_config.py" \
      --seq_dir "$S" --aruco_rig "$ARUCO_RIG" --out_config "$CFG" >>"$LOG" 2>&1
    (cd "$OMNI" && "$CLUB_TRACKER" --config "$CFG") >>"$LOG" 2>&1
    [ -f "$S/trajectory_output/results.json" ] || echo "[FAIL 4] $SEQ" | tee -a "$LOG"
  else echo "[4] skip (done)" | tee -a "$LOG"; fi

  # 5) 手跟杆 + 人体手臂IK -> 完整带手人体 + 球杆点云 overlay (按 --club 加载预定义手)
  if [ -f "$VIS/step5_complete_body_cam0.mp4" ]; then
    echo "[5] skip (done)" | tee -a "$LOG"
  elif [ -f "$DS/world_params/smplx_params_world.pt" ] && [ -f "$S/trajectory_output/results.json" ]; then
    echo "[5] club-grasp body IK (club=$CLUB)" | tee -a "$LOG"
    cd "$PR" && CUDA_VISIBLE_DEVICES=$GPU LD_LIBRARY_PATH=$GOLF_LD $GOLF_PY -m pipeline.club_grasp_ik \
      --capture_dir "$S" --seq "$SEQ" --results_json "$S/trajectory_output/results.json" \
      --club "$CLUB" --render_stride 2 --crf 28 >>"$LOG" 2>&1 \
      && { cp "$S/.pipeline/vis/${SEQ}_body_view0.mp4" "$VIS/step5_complete_body_cam0.mp4" 2>/dev/null
           cp "$S/.pipeline/vis/${SEQ}_body_view1.mp4" "$VIS/step5_complete_body_cam1.mp4" 2>/dev/null; } \
      || echo "[FAIL 5] $SEQ" | tee -a "$LOG"
  fi
  echo "[done] $SEQ $(date)" | tee -a "$LOG"
done
echo "=== pipeline_ik done $(date) ===" | tee -a "$LOG"
