# Golf Multi-View Pipeline (web wizard)

Gradio web 界面包装的多视角高尔夫手部估计 pipeline: 用户选完 capture 文件夹之后**一步一步点按钮跑** (raw 抽帧 → 去畸变 → 双手 bbox 检测 → 双手 2D 关节伪标 → 多视角 3D 推理 → 3D 轨迹可视化), 中间结果可以预览, 状态自动持久化 (`<capture>/.pipeline/state.json`), 关掉浏览器再开能续上.

输出视频统一 **60fps**, 每一步都带**单帧浏览** (slider 拖到任意一帧看 bbox / 21 关节 / 双手 3D 投影).

## 6 个 wizard step (标准 pipeline)

| # | 名字 | 干什么 | 后端 | UI 预览 |
|---|---|---|---|---|
| 0 | Setup | 选 `capture_dir` + `seq_name`, 探测相机 (cam0/cam1); 若 capture 下只有 `.raw` 自动解码出 `.tmp_images/<cam>/frame_*.jpg` | — | 相机 + raw 抽帧 JSON |
| 1 | Undistort | 算 newK + 写去畸变 jpg + `calib_undistorted` yaml (含 K + R/t, 全部基于 cam0 master frame) | — | slider 浏览 cam0 / cam1 任意一帧 |
| 2 | Detect | 每只手出一个 bbox, 存 `detections.json` + 每个相机一个 60fps overlay mp4. 可选 `bbox patch` 把 bbox 中心不变长宽各 ×k 扩张 | **vitpose** (从 `preprocess/<cam>/vitpose.pt` 取 COCO17 手腕做 bbox 中心) | cam0/cam1 mp4 + 单帧 slider |
| 3 | Pseudo | 读 step2 的 bbox, 喂 2D 估计后端拿 21 关节 → `pseudo_label_wilor/*.npz`; 生成 `_pseudo_vis/*.mp4` + 全帧 jpg | **wilor** | cam0\|cam1 拼接 overlay mp4 + 单帧 slider |
| 4 | Infer | 跑 HE 多视角推理 (FlipModel + `FLIP_GOLF_Inference.yaml`, 左右手都参与) → `_mano.json`, 组装最终 npy. 默认用 `exp/new/checkpoints/checkpoint_30`, Checkpoint 下拉框可手填别的 ckpt 路径. **没物体轨迹 csv 时只输出手, 跳过 object 字段** | — | hand0/hand1 mp4 + 单帧 slider + npy |
| 5 | Visualize | `way_vis` 渲染 3D 轨迹 mp4. npy 没 object 时自动切 `show=hand` | — | trajectory_view0 / view1 mp4 |

标准跑法: **0 → 1 → 2 → 3 → 4 → 5**. 每一步完成后 `state.json` 写一次, 中间结果都能在 UI 单帧浏览, 不满意可以原地改参数勾 `Force` 重跑那一步, **后续步骤无需重跑** (除非那一步的输入也变了).

> Step 2 (vitpose) 依赖上游 preprocess 产出的 `<capture>/.undistorted/<seq>/preprocess/<cam>/vitpose.pt` (COCO17 关节). 这一步在跑本 pipeline 之前由 human-dataset-tools / multi_view_smpl_optimizer 的 vitpose stage 产生.

> <capture>/.undistorted/<seq>文件夹为去畸变图像及标定目录，已整理为human-dataset-tools所需数据目录形式，直接作为根目录跑人体姿态结果即可

### 不在标准 pipeline 里的 mega-stage / 外挂

下面这些**不属于** 0~5 的标准链路, 用到时单独触发, 不在 web wizard 主路径里:

- **way_vis 单视角导出 / 自定义 club mesh** — 都还在 Step 5, 但 club mesh 路径写死在 `config/baseball_golf.json`, 想换 mesh 手动改 json.

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
   - 浏览器关了再开, 重新点 Initialize 就能**从已有 state.json 恢复所有 step 的预览** (含每个 step 的视频 + 单帧 slider).
2. **1. Undistort**: 点 `Run undistort` (秒级). slider 拖动浏览任意一帧的 cam0 / cam1 去畸变结果.
3. **2. Hand Detection** (vitpose): 直接点 `Run hand detection`. 从上游 `preprocess/<cam>/vitpose.pt` 取 COCO17 手腕 (9=左, 10=右) 当 bbox 中心 (固定 80px 方框, conf<0.3 丢弃). 完成出 cam0/cam1 各一个 60fps overlay mp4 + 单帧 slider.
   - **bbox patch** (可选): 默认 `1.0` (紧 bbox). 想给下游 WiLoR 看到更大的 crop 区域可调到 `1.2~2.0`, 中心不变长宽各乘 k, clip 到原图. 改了要勾 `Force re-run`.
4. **3. Pseudo Label** (wilor): 点 `Generate pseudo labels`, 用 step2 的 bbox 估计 21 个 2D 关节.
   - 完成出 `_pseudo_vis/<seq>_pseudo_overlay.mp4` (cam0|cam1 拼接, 60fps) + 单帧 slider. 落地 `pseudo_label_wilor/*.npz`.
5. **4. Multi-view Inference + npy**: 默认用官方 `checkpoint_30`; 想换权重直接在 Checkpoint 框里填 ckpt 目录路径.
   - 用 `FLIP_GOLF_Inference.yaml` (FlipModel + FlipDataset, `INCLUDE_RIGHT_HAND=true` 让左右手都参与推理).
   - 完成出 `hand0.mp4` (右) / `hand1.mp4` (左) 各一个 60fps overlay mp4, 拖单帧 slider 看 cam0|cam1 拼接的双手 3D 投影. npy 同时落到 `<capture>/.pipeline/<seq>.npy`.
6. **5. 3D Visualization**: 点 `Render way_vis`. 看到 3D 轨迹 mp4 (cam0 / cam1 各一个).
   - 没物体轨迹 (`<capture>/trajectory_output/trajectory.csv` 不存在) 时, step4 只输出手, step5 自动切 hand-only.

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
│   ├── detect_backends.py          # ViTPose 检测后端 + bbox patch
│   ├── step3_pseudo_label.py
│   ├── pseudo_backends.py          # WiLoR 伪标后端
│   ├── step5_inference.py          # 多视角推理 + npy 组装 (UI 上是 "4. Infer")
│   └── step6_visualize.py          # way_vis 渲染 (UI 上是 "5. Visualize")
├── model/
│   ├── Hand_Estimation/            # USThand (FlipModel + FlipDataset)
│   └── WiLoR/                      # WiLoR (Step 3, vendored, 不嵌套 git)
├── config/                         # baseball_golf.json (球杆 mesh 路径等)
├── utils/                          # way_vis.py 等
├── MANO/                           # MANO 模型文件 (license-restricted, 用户自备)
├── multiview_hand_init.py          # orchestration, pipeline 大量复用其中函数
├── run_golf_capture_to_npy.py      # 老 CLI 入口, 保留, 复用辅助函数
└── run_hamer_to_npy.py             # 老 CLI 入口, 复用 _interpolate_nan 等辅助函数
```

> `step5_inference.py` / `step6_visualize.py` 是历史文件名 (沿用旧的步骤编号); 现在 UI 上分别是第 4 / 第 5 个 Tab.
> 两个 `run_*.py` 是旧 CLI 入口, 现在只作为辅助函数库被 step0/1/4 复用; 它们内部用到 HaMER/YOLO 的旧处理路径已改成函数内 lazy import, 删掉对应 model 目录后仍能正常 import.

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
│   ├── <seq>/preprocess/<cam>/vitpose.pt  # ← 上游产出, step2 读它
│   ├── pseudo_label_wilor/     # ← step3 输出
│   ├── _pseudo_vis/            # ← step3 视频
│   └── _he_output/             # ← step4 (Infer) 输出
│
└── .pipeline/                  # ← 本项目独有的 workspace
    ├── state.json              # 各 step 当前状态
    ├── detections.json         # ← step2 输出
    ├── _detect_vis/            # ← step2 完整 overlay mp4
    ├── <seq>.npy               # ← step4 (Infer) 输出
    └── vis/                    # ← step5 (Visualize) way_vis 输出
        └── <seq>_trajectory_view{0,1}.mp4
```

**状态恢复**: `<capture_dir>/.pipeline/state.json` 是真理源. 关掉浏览器再开 web,
**再点一下 Setup 用同一个 capture_dir + seq_name 重新 init**, 就会从 state.json reload,
已完成的 step 都显示 `done` 状态, **所有预览图/视频/npy 也会从 state 恢复回来**.

---

## 环境配置

GPU + CUDA. PyTorch 2.1 + CUDA 11.8.

```bash
conda env create -f environment.yml
conda activate golf_pipeline
```

`environment.yml` 只装 conda 必装的 (python / pytorch / pytorch3d), 其余 pip 包从 `requirements.txt` 一并装上 (含 chumpy, 已加 `--no-build-isolation` 绕开它的 `import pip` 坑).

或者纯 pip (假设 PyTorch + CUDA + pytorch3d 已经独立装好):

```bash
pip install --no-build-isolation -r requirements.txt
```

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
- DINOv3 → `model/Hand_Estimation/dinov3_convnext_*.pth`
- WiLoR → `model/WiLoR/pretrained_models/{wilor_final.ckpt,detector.pt}`

### MANO 文件 (license-restricted)

SMPL/MANO 注册协议禁止再分发, **不在 HF repo 里**. 自己去 [MPI MANO](https://mano.is.tue.mpg.de) 注册下 `mano_v1_2.zip`, 解压后按以下路径 cp 就位:

```text
mano_v1_2/models/MANO_LEFT.pkl   → MANO/MANO_LEFT.pkl
mano_v1_2/models/MANO_RIGHT.pkl  → MANO/MANO_RIGHT.pkl
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
    'model/Hand_Estimation/exp/new/checkpoints/checkpoint_30/TestFlipMultiviewStereo.pth.tar',
    'model/WiLoR/pretrained_models/wilor_final.ckpt',
    'model/WiLoR/pretrained_models/detector.pt',
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

## 已知限制

- 只支持**2 个 1440×1080 彩色相机**的高尔夫采集格式.
- Step 2 (vitpose) 依赖上游 preprocess 产出的 `vitpose.pt`; 没有它会直接报错.
- Step 5 (way_vis) 通过 monkey-patch globals + `_build_view_overlay_sources` 函数调用, 不影响 CLI 兼容, 但**多用户并发会冲突** (模块全局共享). 单用户场景不影响.
- `config/baseball_golf.json` 里的 `club_mesh_path` 是绝对路径, 迁移到新机器后要改 (或把 mesh 也搬过去).
- USThand 切到 FlipModel 后, 内部约定为 "训练只见右手, 左手镜像后再翻回"; 老的 `TestMultiviewStereo` 权重和老的 `GolfInfraDataset` 推理 yaml **已不兼容**, 加载会 weight key 不匹配崩. 推理统一走 `FLIP_GOLF_Inference.yaml`.

---

## License

代码部分: 自定 / 内部使用. MANO 等人体模型请遵守官方许可, 仅限研究用途.
