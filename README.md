# pipeline_ik

球杆 + 人体一条龙批处理（从 `pipeline_release` 抽出的精简独立项目，只保留这条链路）：

```
raw → 抽帧 → 去畸变 → 球杆姿态(ArUco) → 人体姿态(GVHMR) → 按球杆型号用 standard_pose 预定义手做IK
```

对 `data/<球杆型号>/` 下每个采集序列, 顺序跑 5 步, 产出**完整带手人体** + 各步可视化。

## 用法

```bash
bash run_batch.sh <DATA_DIR> [GPU] [seq ...]
# 例: 处理 35_wood_8_01 全部序列, 用 GPU 2
bash run_batch.sh /data2/fubingshuai/golf/data/35_wood_8_01 2
# 只处理某几个序列
bash run_batch.sh /data2/fubingshuai/golf/data/35_wood_8_01 2 20260424165613363
```

- **球杆型号 = `<DATA_DIR>` 目录名**：自动用 `standard_pose/<型号>_canonical_grasp.npy`（预定义手）
  + `club-assets/<型号>/`（aruco rig + mesh）。换球杆只要把数据放到 `data/<别的型号>/` 下即可。
- **幂等**：每步产物已存在就跳过；单序列某步失败记日志、继续下一个。
- 日志：`<DATA_DIR>/_pipeline_ik_log.txt`。

## 单序列 IK（`single_ik.py`）

只对**一段已有的人体**做「手跟杆 + 仅手臂 IK + 灌手指」，直接给四个绝对路径，不跑前面 1~4 步：

```bash
python single_ik.py \
  --club_traj   /abs/trajectory.csv \             # 球杆 6DoF 轨迹 (omni-club-tracking 输出)
  --smplx_world /abs/smplx_params_world.pt \       # 这段人体 (GVHMR 输出)
  --grasp       /abs/3-1-wood-03_canonical_grasp.npy \  # 该球杆预定义握杆 (standard_pose)
  --out         /abs/smplx_params_world_ik.pt      # 输出: 完整带手 SMPLX
```

把 grasp 两手按每帧球杆位姿刚性搬到世界系 → 作为腕目标对人体做仅手臂 IK，并把 MANO 手指
姿态灌进 SMPLX 自己的手，存到 `--out`（含 `body_pose/betas/global_orient/transl/left_hand_pose/right_hand_pose`）。
核心是 `pipeline/step5b_body_ik.py:solve_body_ik`（与上面 5 步链路共用同一份 IK）。

可选 **`--calib`**：传一个去畸变标定 yaml（如 `.../calib_undistorted/0.yaml`，含 K + world→cam R/t），
就额外渲染**该视角下的完整带手人体 overlay 视频**（背景图自动从 yaml 旁边的 `<N>/images_undistorted/`
找，找不到就黑底）。输出默认 `<out>_vis_view0.mp4`，可用 `--vis_out` 指定、`--vis_stride`/`--crf` 调帧/体积。

```bash
python single_ik.py --club_traj ... --smplx_world ... --grasp ... --out ... \
    --calib /abs/.../calib_undistorted/0.yaml          # 出该视角可视化视频
```

## 5 步

| 步 | 干什么 | 实现 | 环境 |
|---|---|---|---|
| 1 抽RGB | raw → `.tmp_images/<cam>/frame_*.jpg`（彩色 + mono） | `scripts/run_stage.py setup` + `omni-club-tracking/raw_to_images.py` | golf_pipeline |
| 2 去畸变 | → `.undistorted/<seq>/{0,1}/images_undistorted` + `calib_undistorted` | `scripts/run_stage.py undistort` (`pipeline/undistort_core.py`) | golf_pipeline |
| 3 人体姿态 | GVHMR 多视角 → `world_params/smplx_params_world.pt` | `human-dataset-tools` 的 `multi_view_smpl_optimizer.dataset.ours --stage all` | **gvhmr** |
| 4 球杆追踪 | ArUco + Ceres 6DoF → `trajectory_output/results.json` | `omni-club-tracking` 的 C++ `club_tracker`（字典 4X4/6X6 自动判别, `scripts/gen_club_config.py`） | C++ 二进制 |
| 5 手跟杆+人体IK | 预定义手 ⊗ 球杆轨迹 → 腕目标 → 人体仅手臂IK + 灌MANO手指 → 完整带手SMPLX + 球杆点云overlay | `pipeline/club_grasp_ik.py` + `pipeline/step5b_body_ik.py` | golf_pipeline |

世界系 = cam0（低 id 彩色相机），人体与球杆同系。每步可视化落 `<seq>/_pipeline_vis/`。

## 输出（每序列）

```
<seq>/.undistorted/<seq>/world_params/smplx_params_world.pt      # GVHMR 人体
<seq>/.undistorted/<seq>/world_params/smplx_params_world_ik.pt   # IK后 完整带手人体
<seq>/trajectory_output/results.json                            # 球杆 6DoF 轨迹
<seq>/_pipeline_vis/step12_undistorted_cam0_cam1.jpg            # 去畸变
<seq>/_pipeline_vis/step3_humanpose_view{0,1}.mp4              # 人体姿态
<seq>/_pipeline_vis/step5_complete_body_cam{0,1}.mp4           # 完整带手人体 + 球杆点云
<seq>/trajectory_output/overlay_*.mp4                          # 球杆追踪 overlay
```

## 外部依赖（不在本项目内, 按现有路径调用）

- **GVHMR 人体姿态**: `/data2/fubingshuai/golf/human-dataset-tools`（gvhmr conda 环境）。
- **球杆追踪**: `/data2/fubingshuai/golf/omni-club-tracking`（已编译的 `cmake-build-Release/club_tracker`）。
- **预定义握杆**: `/data2/fubingshuai/golf/standard_pose/<型号>_canonical_grasp.npy`
  （由 `pipeline_release/scripts/standard_pose_bind.py` 生成）。
- **球杆 mesh + aruco rig**: `/data2/fubingshuai/golf/data/club-assets/<型号>/`。
- **SMPLX 模型**: `assets/body_models` → 软链到 `../pipeline/assets/body_models`。
- **MANO 模型**: 本项目自带 `MANO/`。
- conda 环境: `golf_pipeline`（去畸变/IK/渲染）、`gvhmr`（人体姿态）。路径写死在 `run_batch.sh` 顶部, 换机器改那几行。

## 结构

```
pipeline_ik/
├── run_batch.sh              # 主驱动 (5 步)
├── pipeline/
│   ├── step0_setup.py        # 抽帧 (raw->jpg)
│   ├── undistort_core.py     # 去畸变核心
│   ├── step1_undistort.py    # 去畸变 step 包装
│   ├── step5b_body_ik.py     # 人体仅手臂IK + 灌手指 -> 完整带手SMPLX
│   ├── club_grasp_ik.py      # 手跟杆 + 调 step5b + 渲染 (--club 自动加载预定义手)
│   ├── raw_extract.py / raw_to_images.py / workspace.py / state.py
├── utils/{camera_npy.py, body_vis.py}   # 相机块 / overlay 渲染
├── scripts/{run_stage.py, gen_club_config.py, render_world_body.py}
├── assets/body_models -> ../pipeline/assets/body_models   # SMPLX (软链)
└── MANO/                                                  # MANO 模型
```
