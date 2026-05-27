# Golf Multi-View Pipeline (web wizard)

Gradio web 界面包装的多视角高尔夫手部估计 pipeline: 用户选完 capture 文件夹之后**一步一步点按钮跑** (raw 抽帧 → 去畸变 → 双手 bbox 检测 → 双手 2D 关节伪标 → (可选) 自监督 finetune → 多视角 3D 推理 → 3D 轨迹可视化), 中间结果可以预览, 状态自动持久化 (`<capture>/.pipeline/state.json`), 关掉浏览器再开能续上.

输出视频统一 **60fps**, 每一步都带**单帧浏览** (slider 拖到任意一帧看 bbox / 21 关节 / 双手 3D 投影).

## 7 个 wizard step (标准 pipeline)

| # | 名字 | 干什么 | 后端选项 | UI 预览 |
|---|---|---|---|---|
| 0 | Setup | 选 `capture_dir` + `seq_name`, 探测相机 (cam0/cam1); 若 capture 下只有 `.raw` 自动解码出 `.tmp_images/<cam>/frame_*.jpg` | — | 相机 + raw 抽帧 JSON |
| 1 | Undistort | 算 newK + 写去畸变 jpg + `calib_undistorted` yaml (含 K + R/t, 全部基于 cam0 master frame) | — | slider 浏览 cam0 / cam1 任意一帧 |
| 2 | Detect | 每帧每只手出一个 bbox, 存 `detections.json` + 每个相机一个 60fps overlay mp4. 可选 `bbox patch` 把 bbox 中心不变长宽各 ×k 扩张 | **yolo** (默认逐帧检测) / **sam2** (首帧/多帧手动标点 → 视频分割 propagate, 适合 YOLO 漏检或遮挡场景) | cam0/cam1 mp4 + 单帧 slider |
| 3 | Pseudo | 读 step2 的 bbox, 喂 2D 估计后端拿 21 关节 → `pseudo_label_wilor/*.npz`; 生成 `_pseudo_vis/*.mp4` + 全帧 jpg | **hamer** / **wilor** (新, 通常更准, npz 格式一致) | cam0\|cam1 拼接 overlay mp4 + 单帧 slider |
| 4 | Finetune (opt) | 自监督 finetune Hand_Estimation 权重 (FlipModel + `FLIP_GOLF_DINO.yaml`), ckpt 存到 `<workspace>/_finetune/<id>/` | — | log + ckpt 路径 |
| 5 | Infer | 跑 HE 多视角推理 (FlipModel + `FLIP_GOLF_Inference.yaml`, 左右手都参与) → `_mano.json`, 组装最终 npy. ckpt 下拉自选: finetune 跑过默认用 FT 产物, 否则用 `exp/new/checkpoints/checkpoint_30`. **没物体轨迹 csv 时只输出手, 跳过 object 字段** | — | hand0/hand1 mp4 + 单帧 slider + npy |
| 6 | Visualize | `way_vis` 渲染 3D 轨迹 mp4. npy 没 object 时自动切 `show=hand` | — | trajectory_view0 / view1 mp4 |

标准跑法: **0 → 1 → 2 → 3 → 4 (可选) → 5 → 6**. 每一步完成后 `state.json` 写一次, 中间结果都能在 UI 单帧浏览, 不满意可以原地改参数勾 `Force` 重跑那一步, **后续步骤无需重跑** (除非那一步的输入也变了).

### 模型组合默认值

| 步骤 | 默认后端 | 备选 |
|---|---|---|
| Step 2 (Detect) | YOLO | SAM2 (首帧/多帧标正负点, 自动 propagate, 推荐处理遮挡场景) |
| Step 3 (Pseudo) | HaMER | WiLoR |
| Step 4 (FT) | FlipModel (TestFlipMultiviewStereo) | — |
| Step 5 (Infer) | FlipModel | — |

### 不在标准 pipeline 里的 mega-stage / 外挂

下面这些**不属于** 0~6 的标准链路, 用到时单独触发, 不在 web wizard 主路径里:

- **SportGS 接触优化** — 老 `golf-hand-object` 仓库里的 stage, 在 npy 之外做物体-手接触约束. 当前 pipeline **不包含**, 想用要切到原仓库.
- **单帧 standard_pose 抽取** — 老 CLI `run_golf_capture_to_npy.py` 的某个旁路功能, **不在 wizard 里**, 直接 CLI 跑.
- **way_vis 单视角导出 / 自定义 club mesh** — 都还在 Step 6, 但 club mesh 路径写死在 `config/baseball_golf.json`, 想换 mesh 手动改 json.

---

## 快速开始

```bash
# 1) clone
git clone <this-repo> pipeline
cd pipeline

# 2) 装环境 (见下文「环境配置」)
conda env create -f environment.yml
conda activate golf_pipeline
pip install --no-build-isolation -r requirements.txt
# 安pytorch3d
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
# 安 SAM2 (editable). 务必加 --no-deps, 否则会把 torch / numpy / Pillow 升到更新版本,
# 跟 environment.yml 锁的 torch 2.1.2+cu118 不兼容, 老 NVIDIA 驱动会直接 CUDA init 失败.
pip install -e model/sam2/ --no-deps


# 3) 拉权重 (见下文「权重 setup」)
huggingface-cli login   # 粘 HF token
python scripts/fetch_weights.py

# 4) MANO 文件单独从 mpi 网站注册下载 (license-restricted), 见下面 [MANO 文件] 段

# 5) 启动 wizard
python app.py [--port 7860] [--host 0.0.0.0]
# 浏览器开 http://<server>:7860
```

**填 capture_dir 注意:** 写到**采集会话根目录** (含 `camera_params.json` 那一层),
**不要**带上 `.tmp_images` / `.undistorted` 这种子目录后缀, 否则路径会嵌套出问题.
例子: `/data/.../35_wood_8_01/20260424170424563`
而不是 `.../20260424170424563/.tmp_images`. step0 自带 sanity 会自动剥这种后缀.

---

## 第一次使用流程

1. **Setup** tab: 填 `Capture dir` + `Seq name`, 点 `Initialize workspace`.
   - 没解过的 `.raw` 文件会自动转 jpg 到 `<capture>/.tmp_images/`.
   - 浏览器关了再开, 重新点 Initialize 就能**从已有 state.json 恢复所有 step 的预览** (含每个 step 的视频 + 单帧 slider + ckpt 下拉).
2. **1. Undistort**: 点 `Run undistort` (秒级). slider 拖动浏览任意一帧的 cam0 / cam1 去畸变结果.
3. **2. Hand Detection**: 选检测后端.
   - **YOLO** (默认): 直接点 `Run hand detection` 逐帧检. 完成出 cam0/cam1 各一个 60fps overlay mp4 + 拖单帧 slider 看任意一帧.
   - **SAM2** (推荐处理 YOLO 漏检 / 遮挡场景): 点 `Load first frames` 加载 SAM2 (~5-10s) → 拖 `frame slider` 到任意一帧 → 选**当前点类型** (右手正/右手负/左手正/左手负) → 在图上点击加点 (即时看 mask) → 可以换到别的帧继续标 (跨帧 anchor 持久化) → 点 `Run hand detection` → SAM2 从 frame 0 把所有 anchor 当 memory 一口气 propagate 到末尾.
   - **bbox patch** (可选): 默认 `1.0` (紧 bbox). 想给下游 HaMER/WiLoR/USThand 看到更大的 crop 区域可调到 `1.2~2.0`, 中心不变长宽各乘 k, clip 到原图. 改了要勾 `Force re-run`.
4. **3. Pseudo Label**: 选 2D 估计后端.
   - **HaMER** (默认, 稳): rescale=2.5, 慢.
   - **WiLoR** (通常更准): rescale=2.0, 快一些.
   - 完成出 `_pseudo_vis/<seq>_pseudo_overlay.mp4` (cam0|cam1 拼接, 60fps) + 单帧 slider. 落地 `pseudo_label_wilor/*.npz` (目录名是历史命名, 实际内容就是你选的 backend 输出).
5. **4. Self-supervised Finetune** (optional): 默认不勾选, 跳过直接用 `checkpoint_30` 的 FlipModel 权重. 想 finetune 就勾 `Enable` + 调 epochs.
   - 用 `FLIP_GOLF_DINO.yaml` 模板, 模型类 `TestFlipMultiviewStereo`.
   - ckpt 存到 `<capture>/.pipeline/_finetune/<exp>__<inner>/`, **不污染**项目目录. step5 的 Checkpoint 下拉自动列出.
6. **5. Multi-view Inference + npy**: 在 Checkpoint 下拉里选权重 (默认: finetune 跑过用 FT 产物, 否则官方 `checkpoint_30`).
   - 用 `FLIP_GOLF_Inference.yaml` (FlipModel + FlipDataset, `INCLUDE_RIGHT_HAND=true` 让左右手都参与推理).
   - 完成出 `hand0.mp4` (右) / `hand1.mp4` (左) 各一个 60fps overlay mp4, 拖单帧 slider 看 cam0|cam1 拼接的双手 3D 投影. npy 同时落到 `<capture>/.pipeline/<seq>.npy`.
7. **6. 3D Visualization**: 点 `Render way_vis`. 看到 3D 轨迹 mp4 (cam0 / cam1 各一个).
   - 没物体轨迹 (`<capture>/trajectory_output/trajectory.csv` 不存在) 时, step5 只输出手, step6 自动切 hand-only.

---

## 项目结构

```
pipeline/
├── app.py                          # Gradio 入口
├── README.md                       # 本文件
├── environment.yml                 # conda env
├── requirements.txt                # pip 增量依赖
├── scripts/
│   ├── upload_weights_to_hf.sh     # 把本地权重上传到 HF Hub
│   └── fetch_weights.py            # 从 HF Hub 拉权重到正确位置
├── pipeline/                       # 解耦后的 wizard 子模块
│   ├── state.py                    # 持久化状态
│   ├── workspace.py                # 路径管理
│   ├── raw_extract.py              # .raw → .jpg
│   ├── raw_to_images.py            # 单文件 raw decoder
│   ├── step0_setup.py
│   ├── step1_undistort.py
│   ├── step2_detect.py
│   ├── detect_backends.py          # YOLO + SAM2 检测后端抽象 + bbox patch
│   ├── sam2_interactive.py         # SAM2 image predictor 交互式标点
│   ├── step3_pseudo_label.py
│   ├── pseudo_backends.py          # HaMER + WiLoR 伪标后端抽象
│   ├── step4_finetune.py
│   ├── step5_inference.py
│   └── step6_visualize.py
├── model/
│   ├── Hand_Estimation/            # USThand (FlipModel + FlipDataset)
│   ├── hamer/                      # HaMER (Step 3 备选)
│   ├── WiLoR/                      # WiLoR (Step 3 备选, vendored, 不嵌套 git)
│   ├── sam2/                       # SAM2 (Step 2 备选)
│   ├── config/                     # YOLO 配置 + yolov7 ckpt 位置
│   └── rootnet/                    # 老的 rootnet (现在 step 5 不直接用了)
├── config/                         # baseball_golf.json (球杆 mesh 路径等)
├── utils/                          # way_vis.py 等
├── MANO/                           # MANO 模型文件 (license-restricted, 用户自备)
├── multiview_hand_init.py          # orchestration, pipeline 大量复用其中函数
├── run_golf_capture_to_npy.py      # 老 CLI 入口, 保留, 复用辅助函数
└── run_hamer_to_npy.py             # 老 CLI 入口, 复用 parse_detections 等
```

---

## State / Workspace

```
<capture_dir>/
├── camera_params.json          # 用户提供
├── <cam>_w1440_h1080_.../      # 用户原始帧目录 (jpg) 或 .raw 文件
├── .tmp_images/                # ← step0 解 .raw 得到的 frame_*.jpg
├── trajectory_output/          # 用户可选提供 (物体位姿 csv); 没有就 hand-only
│
├── .undistorted/               # ← step1 输出
│   ├── <seq>/<cam>/images_undistorted/
│   ├── <seq>/calib_undistorted/<cam>.yaml
│   ├── pseudo_label_wilor/     # ← step3 输出
│   ├── _pseudo_vis/            # ← step3 视频
│   └── _he_output/             # ← step5 输出
│
└── .pipeline/                  # ← 本项目独有的 workspace
    ├── state.json              # 各 step 当前状态
    ├── detections.json         # ← step2 输出
    ├── _detect_vis/            # ← step2 完整 overlay mp4
    ├── _finetune/              # ← step4 微调 ckpt
    ├── <seq>.npy               # ← step5 输出
    └── vis/                    # ← step6 way_vis 输出
        └── <seq>_trajectory_view{0,1}.mp4
```

**状态恢复**: `<capture_dir>/.pipeline/state.json` 是真理源. 关掉浏览器再开 web,
**再点一下 Setup 用同一个 capture_dir + seq_name 重新 init**, 就会从 state.json reload,
已完成的 step 都显示 `done` 状态, **所有预览图/视频/npy/ckpt 下拉框也会从 state 恢复回来**.

---

## 环境配置

GPU + CUDA. PyTorch 2.1 + CUDA 11.8, 跟 HaMER / Hand_Estimation 同栈.

```bash
conda env create -f environment.yml
conda activate golf_pipeline
```

`environment.yml` 只装 conda 必装的 (python / pytorch / pytorch3d), 其余 pip 包从 `requirements.txt` 一并装上 (含 chumpy, 已加 `--no-build-isolation` 绕开它的 `import pip` 坑).

或者纯 pip (假设 PyTorch + CUDA + pytorch3d 已经独立装好):

```bash
pip install --no-build-isolation -r requirements.txt
```

`environment.yml` 已经把 SportGS / SAM2 相关重型依赖砍掉 (v1 不含). 想要 SportGS / 接触优化, 见 `golf-hand-object` 仓库原始 environment.yml.

---

## 权重 setup (clone 仓库后必看)

仓库**只含代码 + 配置**, 不含任何 `.pth` / `.ckpt` / `.pkl` / `.zip` 等权重 (`.gitignore` 全部忽略). 装好仓库后还要自己把权重就位才能跑.

### 一键下载 (除 MANO 外)

所有非 MANO 权重已传到 Hugging Face Hub: **`lilfiiiiish/pipeline`** (私有 repo, 要 HF token 才能拉).

```bash
# 1) 装 hf_hub (已装在 environment.yml 里, 重复装无害)
pip install -U huggingface_hub

# 2) 登录 (粘 HF token, https://huggingface.co/settings/tokens, 至少 read 权限)
huggingface-cli login
# 或者: export HF_TOKEN=<your_token>

# 3) 在项目根跑
python scripts/fetch_weights.py
#  --repo <user>/<repo>   换 repo
#  --force                目标已存在也覆盖
#  --with-mano            也拉 MANO (HF repo 里得有)
```

脚本会自动把权重摆到正确位置:
- HE checkpoint_30 → `model/Hand_Estimation/exp/new/checkpoints/checkpoint_30/`
- HaMER ckpt → `model/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt`
- rootnet → `model/rootnet/SAR-*.pth`
- yolov7 → `model/config/checkpoints/yolov7_best.pt`
- DINOv3 → `model/Hand_Estimation/dinov3_convnext_*.pth`

### MANO 文件 (license-restricted)

SMPL/MANO 注册协议禁止再分发, **不在 HF repo 里**. 自己去 [MPI MANO](https://mano.is.tue.mpg.de) 注册下 `mano_v1_2.zip`, 解压后按以下路径 cp 就位:

```text
mano_v1_2/models/MANO_LEFT.pkl   → MANO/MANO_LEFT.pkl
mano_v1_2/models/MANO_RIGHT.pkl  → MANO/MANO_RIGHT.pkl
                                 → model/hamer/_DATA/data/mano/MANO_LEFT.pkl
                                 → model/hamer/_DATA/data/mano/MANO_RIGHT.pkl
                                 → model/Hand_Estimation/mano/models/MANO_LEFT.pkl
                                 → model/Hand_Estimation/mano/models/MANO_RIGHT.pkl
mano_v1_2/models/MANO_RIGHT.pkl  → model/Hand_Estimation/mano_data/MANO_RIGHT.pkl
```

`MANO/{MANO_PART,v_color}.pkl` 是本项目自带的 segmentation/可视化辅助 pkl, 跟 MANO_RIGHT 配套, 找作者要原始版本.

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
    'model/Hand_Estimation/exp/new/checkpoints/checkpoint_30/TestFlipMultiviewStereo.pth.tar',
    'model/WiLoR/pretrained_models/wilor_final.ckpt',
    'model/WiLoR/pretrained_models/detector.pt',
    'model/sam2/checkpoints/sam2.1_hiera_large.pt',
    'model/Hand_Estimation/mano_data/MANO_RIGHT.pkl',
]
for p in paths:
    print(('OK   ' if Path(p).is_file() else 'MISS '), p)
"
```

全部 `OK` 才能跑 `python app.py`.

### 上传你自己版本 (维护者)

见 `scripts/upload_weights_to_hf.sh`. 改头部 `HF_REPO` 为你自己的 repo, `huggingface-cli login` 后 `bash scripts/upload_weights_to_hf.sh`. 想顺带传 MANO 用 `INCLUDE_MANO=1 bash scripts/upload_weights_to_hf.sh`, 但**只能传到 private repo**, 别公开.

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

---

## 已知限制

- 只支持**2 个 1440×1080 彩色相机**的高尔夫采集格式.
- Step 4 (finetune) 现在 web 没流式 log, 看 stderr 要去启动 web 的终端窗口.
- Step 3 (HaMER) 耗时长, gradio 进度条按 batch 粗粒度更新. 想快可以换 WiLoR 后端.
- Step 2 SAM2 后端的 `propagate_in_video` 是端到端逐帧推理, 长序列 + 高分辨率慢; 已经强制 `offload_video_to_cpu=True, offload_state_to_cpu=True` 防 OOM, 但慢 15-25%.
- Step 6 (way_vis) 通过 monkey-patch globals + `_build_view_overlay_sources` 函数调用, 不影响 CLI 兼容, 但**多用户并发会冲突** (模块全局共享). 单用户场景不影响.
- `config/baseball_golf.json` 里的 `club_mesh_path` 是绝对路径, 迁移到新机器后要改 (或把 mesh 也搬过去).
- USThand 切到 FlipModel 后, 内部约定为 "训练只见右手, 左手镜像后再翻回"; 老的 `TestMultiviewStereo` 权重和老的 `GolfInfraDataset` 推理 yaml **已不兼容**, 加载会 weight key 不匹配崩. FT / 推理统一走 `FLIP_GOLF_DINO.yaml` / `FLIP_GOLF_Inference.yaml`.

---

## License

代码部分: 自定 / 内部使用. MANO 等人体模型请遵守官方许可, 仅限研究用途.
