# Golf Multi-View Pipeline (web wizard)

把 `golf-hand-object` 的多视角手物估计 pipeline 拆成独立子模块, 加上 Gradio
web 界面, 用户**选完 capture 文件夹之后点按钮一步一步跑**, 中间结果可以预览,
状态自动持久化 (`<capture>/.pipeline/state.json`), 关掉浏览器再开能续上.

## 7 个 wizard step

| # | 名字 | 干什么 | 关键 wrap | UI 预览 |
|---|---|---|---|---|
| 0 | Setup | 选 `capture_dir` + `seq_name`, 探测相机 (cam0/cam1); 若 capture 下只有 `.raw` 自动解码出 `.tmp_images/<cam>/frame_*.jpg` | `run_golf_capture_to_npy._load_camera_params/_resolve_color_cams` + `pipeline.raw_extract.extract_all` | 相机 + raw 抽帧 JSON |
| 1 | Undistort | 算 newK + 写去畸变 jpg + `calib_undistorted` yaml | `multiview_hand_init.prepare_undistort_dir` | slider 浏览 cam0 / cam1 任意一帧 |
| 2 | Detect | YOLO 检测每帧两手 bbox, 存 `detections.json` + 每个相机一个完整 overlay mp4 | `yolo.detector.Detector` + `parse_detections` | cam0 / cam1 bbox overlay mp4 |
| 3 | Pseudo | 读 step2 的 bbox, 喂 HaMER 拿 21 关节 → `pseudo_label_wilor/*.npz`; 同时生成 `_pseudo_vis/*.mp4` | `model.hamer.infer.hamer_inference` + `multiview_hand_init.make_pseudo_video` | overlay mp4 |
| 4 | Finetune (opt) | 自监督 finetune Hand_Estimation 权重, ckpt 直接保存到 `<workspace>/_finetune/<id>/` | `multiview_hand_init.run_hand_estimation_finetune_subprocess` | log |
| 5 | Infer | 跑 HE 多视角推理 → `_mano.json`, 组装最终 npy. ckpt 可下拉自选 (默认: finetune 跑过用它, 否则 `exp/new/checkpoints/checkpoint_30`). **没物体轨迹 csv 时只输出手, 跳过 object 字段** | `multiview_hand_init.run_hand_estimation_subprocess` + `parse_mano_json_to_arrays` + `_build_camera_block` + `_object_poses_to_world` | hand0/hand1 mp4 + npy |
| 6 | Visualize | `way_vis` 渲染 3D 轨迹 mp4. 多视角 overlay 在原图按 pipeline 目录约定 (`<undist>/<capture_id>/<view>/images_undistorted/`); npy 没 object 时自动切 `show=hand` | `utils.way_vis.main` (monkey-patched globals + view-overlay 路径) | trajectory_view0 / view1 mp4 |

v1 不含: SAM2 交互式手分割, SportGS 接触优化, 单帧 standard_pose 抽取.

## 启动

```bash
cd /data2/fubingshuai/golf/pipeline
python app.py [--port 7860] [--host 0.0.0.0]
```

如果出现 `PermissionError: [Errno 13] Permission denied: '/tmp/gradio/...'`,
是因为 `/tmp/gradio` 被别人创建过, 设个你能写的目录:

```bash
export GRADIO_TEMP_DIR=$HOME/.gradio_tmp
python app.py
```

**填 capture_dir 注意:** 写到**采集会话根目录** (含 `camera_params.json` 那一层),
**不要**带上 `.tmp_images` / `.undistorted` 这种子目录后缀, 否则路径会嵌套出问题.
例子: `/data2/fubingshuai/golf/data/35_wood_8_01_fbs/20260424170424563`
而不是 `.../20260424170424563/.tmp_images`. step0 自带 sanity 会自动剥这种后缀.

浏览器开 `http://<server>:7860`. wizard 风格, 一个 Tab 一个 step.

## 第一次使用流程

1. **Setup** tab: 填 `Capture dir` + `Seq name`, 点 `Initialize workspace`.
   - 没解过的 `.raw` 文件会自动转 jpg 到 `<capture>/.tmp_images/`.
   - 关掉勾再点 Initialize 也可以**从已有 state.json 恢复所有 step 的预览**.
2. **1. Undistort**: 点 `Run undistort` (秒级). slider 拖动浏览任意一帧的 cam0 / cam1 去畸变结果.
3. **2. Hand Detection**: 点 `Run hand detection`, 完成后两个相机各出一个完整 overlay mp4.
4. **3. HaMER Pseudo Label**: 点 `Generate pseudo labels` (慢, 约 0.5–2h depending on 帧数). 看到伪标 overlay mp4.
5. **4. Self-supervised Finetune**: 默认不勾选, 跳过直接用 `checkpoint_30`. 想 finetune 就勾上 + 调 epochs.
   - finetune 完产物存在 `<capture>/.pipeline/_finetune/<exp_name>__<inner>/`, **不污染**
     `pipeline/model/Hand_Estimation/exp/`.
6. **5. Multi-view Inference + npy**: 在 Checkpoint 下拉里选权重 (默认就是 finetune 产物 / 否则官方 ckpt), 点 `Run HE inference`. 看到 hand0/hand1 mp4 + 拿到 npy 文件.
7. **6. 3D Visualization**: 点 `Render way_vis`. 看到 3D 轨迹 mp4 (两个视角各一个).
   - 没物体轨迹 (`<capture>/trajectory_output/trajectory.csv` 不存在) 时, step5 只输出手, step6 自动切 hand-only.

## 项目结构

```
pipeline/
├── app.py                          # Gradio 入口
├── PIPELINE_README.md              # 本文件
├── pipeline/                       # 解耦后的子模块
│   ├── state.py                    # 持久化状态
│   ├── workspace.py                # 路径管理
│   ├── raw_extract.py              # .raw → .jpg
│   ├── raw_to_images.py            # 单文件 raw decoder
│   ├── step0_setup.py
│   ├── step1_undistort.py
│   ├── step2_detect.py
│   ├── step3_pseudo_label.py
│   ├── step4_finetune.py
│   ├── step5_inference.py
│   └── step6_visualize.py
│
└── (cp 自 golf-hand-object/)
    ├── model/                      # Hand_Estimation, hamer, yolo, ...
    ├── config/                     # baseball_golf.json
    ├── utils/                      # way_vis.py, ...
    ├── multiview_hand_init.py      # orchestration, 子模块大量复用其中函数
    ├── run_golf_capture_to_npy.py  # 老 CLI 入口, 保留, 复用辅助函数
    └── ...
```

## State / Workspace

```
<capture_dir>/
├── camera_params.json          # 用户提供
├── <cam>_w1440_h1080_.../      # 用户原始帧目录 (如有 jpg) 或 .raw 文件
├── .tmp_images/                # ← step0 解 .raw 得到的 frame_*.jpg
├── trajectory_output/          # 用户可选提供 (物体位姿 csv); 没有就 hand-only
│
├── .undistorted/               # ← step1 输出 (跟现有 wrapper 保持一致)
│   ├── <seq>/<cam>/images_undistorted/
│   ├── <seq>/calib_undistorted/<cam>.yaml
│   ├── pseudo_label_wilor/     # ← step3 输出
│   ├── _pseudo_vis/            # ← step3 视频
│   └── _he_output/             # ← step5 输出
│
└── .pipeline/                  # ← 本项目独有的 workspace
    ├── state.json              # 各 step 当前状态
    ├── detections.json         # ← step2 输出
    ├── _detect_vis/            # ← step2 预览图 + 完整 overlay mp4
    ├── _finetune/              # ← step4 微调 ckpt (含 TestMultiviewStereo.pth.tar)
    ├── <seq>.npy               # ← step5 输出
    └── vis/                    # ← step6 way_vis 输出
        └── <seq>_trajectory_view{0,1}.mp4
```

## 状态恢复

`<capture_dir>/.pipeline/state.json` 是真理源. 每个 step 完成后写入:

```json
{
  "capture_dir": "...",
  "seq_name":    "...",
  "steps": {
     "setup":     {"status": "done", "ts": "...", "outputs": {...}},
     "undistort": {"status": "done", "ts": "...", "outputs": {...}},
     ...
  }
}
```

关掉浏览器再开 web, **再点一下 Setup 用同一个 capture_dir + seq_name 重新 init**,
就会从 state.json reload, 已完成的 step 都显示 `done` 状态, **所有预览图/视频/npy/ckpt 下拉框也会从 state 恢复回来**.

## 依赖

跟 `golf-hand-object` 完全一致, 多一个 `gradio`:

```bash
pip install gradio
```

GPU + CUDA, 跟 `golf-hand-object/environment.yml` 同环境.

---

## 权重 / 大文件 setup (clone 仓库后必看)

仓库**只含代码 + 配置**, 不含任何 `.pth` / `.ckpt` / `.pkl` / `.zip` 等权重 (`.gitignore`
里全部忽略). 装好仓库后还要自己把权重就位才能跑.

### 清单

| 路径 | 大小 | 来源 |
|---|---|---|
| `model/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt` | 2.6G | [HaMER 官方 release](https://github.com/geopavlakos/hamer) (`fetch_demo_data.sh`) |
| `model/hamer/_DATA/hamer_ckpts/{onnx,dataset_config.yaml,model_config.yaml}` | 1M | HaMER 官方 |
| `model/hamer/_DATA/data/mano/{MANO_LEFT,MANO_RIGHT}.pkl` | 4M each | [MPI MANO 注册下载](https://mano.is.tue.mpg.de) (license-restricted) |
| `model/hamer/_DATA/data/mano_mean_params.npz` | <1M | HaMER 官方 |
| `model/Hand_Estimation/exp/new/checkpoints/checkpoint_30/{TestMultiviewStereo.pth.tar,train_param.pth.tar,random_state.pkl}` | 879M | 师兄 train (内部) |
| `model/Hand_Estimation/mano_data/MANO_RIGHT.pkl` + `mano_mean_params.npz` | <5M | MPI MANO + HaMER |
| `model/Hand_Estimation/mano/models/{MANO_LEFT,MANO_RIGHT}.pkl` | 4M each | MPI MANO |
| `model/Hand_Estimation/assets/mano_v1_2/` (完整) | ~300M | MPI MANO 原始压缩包解压 |
| `model/Hand_Estimation/dinov3_convnext_*.pth` | 108M+193M | [DINOv3 官方 release](https://github.com/facebookresearch/dinov3) |
| `model/rootnet/SAR-convnext-root.pth` | 466M | 师兄 train (内部) |
| `model/rootnet/SAR-resnet34-Root.pth` | 115M | 师兄 train (内部) |
| `model/config/checkpoints/yolov7_best.pt` | 72M | 师兄 train (内部) |
| `MANO/{MANO_LEFT,MANO_RIGHT,MANO_PART,v_color}.pkl` | 4M each | MPI MANO + 项目自带 segmentation |
| `model/sam2_ckpts/sam2.1_hiera_small.pt` | 176M | [SAM2 官方](https://github.com/facebookresearch/sam2) (仅 SAM2 分支用, 本 pipeline 不需要) |

### 获取流程

#### 一键下载 (除 MANO 外)

所有非 MANO 权重已传到 Hugging Face Hub: **`lilfiiiiish/pipeline`**
(私有 repo, 要 HF token 才能拉).

```bash
# 1) 装 hf_hub
pip install -U huggingface_hub

# 2) 登录 (粘 token, https://huggingface.co/settings/tokens, 至少 read 权限)
huggingface-cli login
# 或者: export HF_TOKEN=<your_token>

# 3) clone 完代码后, 在项目根跑
cd /path/to/pipeline
python scripts/fetch_weights.py
# 想换 repo: python scripts/fetch_weights.py --repo <user>/<repo>
# 想强制重下: python scripts/fetch_weights.py --force
```

脚本会自动把权重摆到正确位置:
- HE checkpoint_30 → `model/Hand_Estimation/exp/new/checkpoints/checkpoint_30/`
- HaMER ckpt → `model/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt`
- rootnet → `model/rootnet/SAR-*.pth`
- yolov7 → `model/config/checkpoints/yolov7_best.pt`
- DINOv3 → `model/Hand_Estimation/dinov3_convnext_*.pth`

#### MANO 文件 (license-restricted, 单独获取)

SMPL/MANO 注册协议禁止再分发, **不在 HF repo 里**. 自己去 [MPI MANO](https://mano.is.tue.mpg.de)
注册下 `mano_v1_2.zip`, 解压后按以下路径 cp 就位:

```text
mano_v1_2/models/MANO_LEFT.pkl   → MANO/MANO_LEFT.pkl
mano_v1_2/models/MANO_RIGHT.pkl  → MANO/MANO_RIGHT.pkl
                                 → model/hamer/_DATA/data/mano/MANO_LEFT.pkl
                                 → model/hamer/_DATA/data/mano/MANO_RIGHT.pkl
                                 → model/Hand_Estimation/mano/models/MANO_LEFT.pkl
                                 → model/Hand_Estimation/mano/models/MANO_RIGHT.pkl
mano_v1_2/models/MANO_RIGHT.pkl  → model/Hand_Estimation/mano_data/MANO_RIGHT.pkl
```

`MANO/{MANO_PART,v_color}.pkl` 是本项目自带的 segmentation/可视化辅助 pkl, 跟 MANO_RIGHT 配套, 找师兄要原始版本.

`model/Hand_Estimation/assets/mano_v1_2.zip` 整个解压版可选 (训练 finetune 时用得到, 推理不需要).

#### 上传你自己版本 (如果你 fork 后想换 repo)

参见 `scripts/upload_weights_to_hf.sh`. 改头部 `HF_REPO` 为你自己的 repo, `huggingface-cli login` 后 `bash scripts/upload_weights_to_hf.sh`. 想顺带传 MANO 用 `INCLUDE_MANO=1 bash scripts/upload_weights_to_hf.sh`, 但**只能传到 private repo**, 别公开.

### 验证

权重就位后, 跑下面这段检查所有关键文件齐了:

```bash
python -c "
from pathlib import Path
paths = [
    'MANO/MANO_RIGHT.pkl',
    'model/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt',
    'model/hamer/_DATA/data/mano/MANO_RIGHT.pkl',
    'model/rootnet/SAR-resnet34-Root.pth',
    'model/rootnet/SAR-convnext-root.pth',
    'model/config/checkpoints/yolov7_best.pt',
    'model/Hand_Estimation/exp/new/checkpoints/checkpoint_30/TestMultiviewStereo.pth.tar',
    'model/Hand_Estimation/mano_data/MANO_RIGHT.pkl',
]
for p in paths:
    print(('OK   ' if Path(p).is_file() else 'MISS '), p)
"
```

全部 `OK` 才能跑 `python app.py`.

---

## 跟原 CLI (`run_golf_capture_to_npy.py`) 的对应

| CLI flag | wizard 对应 |
|---|---|
| `--capture_dir` | Setup tab |
| `--seq_name` | Setup tab |
| `--init_method auto/multiview` | 自动走多视角 (v1 不暴露选项) |
| `--mv_finetune_epochs` / `_lr` / `_bs` | Step 4 |
| `--output` | 自动写到 `<workspace>/<seq>.npy` |
| 优化相关 (`--opt_range` 等) | 不在 v1 范围 |

## 已知限制

- v1 只支持**2 个 1440×1080 彩色相机**的高尔夫采集格式 (跟现有 wrapper 同).
- Step 4 (finetune) 现在 web 没流式 log, 看 stderr 要去启动 web 的终端窗口.
- Step 3 (HaMER) 耗时长, gradio 进度条按 batch 粗粒度更新.
- Step 6 (way_vis) 通过 monkey-patch globals + `_build_view_overlay_sources` 函数调用, 不影响 CLI 兼容, 但**多用户并发会冲突** (模块全局共享). 单用户场景不影响.
- `config/baseball_golf.json` 里的 `club_mesh_path` 是绝对路径, 迁移到新机器后要改 (或把 mesh 也搬过去).
