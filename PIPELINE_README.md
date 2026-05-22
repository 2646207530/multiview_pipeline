# Golf Multi-View Pipeline (web wizard)

把 `golf-hand-object` 的多视角手物估计 pipeline 拆成独立子模块, 加上 Gradio
web 界面, 用户**选完 capture 文件夹之后点按钮一步一步跑**, 中间结果可以预览,
状态自动持久化 (`<capture>/.pipeline/state.json`), 关掉浏览器再开能续上.

## 7 个 wizard step

| # | 名字 | 干什么 | 关键 wrap | UI 预览 |
|---|---|---|---|---|
| 0 | Setup | 选 `capture_dir` + `seq_name`, 探测相机 (cam0/cam1) | `run_golf_capture_to_npy._load_camera_params/_resolve_color_cams` | 相机 JSON |
| 1 | Undistort | 算 newK + 写去畸变 jpg + calib_undistorted yaml | `multiview_hand_init.prepare_undistort_dir` | cam0 第一帧预览 |
| 2 | Detect | YOLO 检测每帧两手 bbox, 存 `detections.json` | `yolo.detector.Detector` + `parse_detections` | 带 bbox overlay 的 cam0 第一帧 |
| 3 | Pseudo | 读 step2 的 bbox, 喂 HaMER 拿 21 关节 → `pseudo_label_wilor/*.npz`; 同时生成 `_pseudo_vis/*.mp4` | `model.hamer.infer.hamer_inference` + `multiview_hand_init.make_pseudo_video` | overlay mp4 |
| 4 | Finetune (opt) | 自监督 finetune Hand_Estimation 权重 | `multiview_hand_init.run_hand_estimation_finetune_subprocess` | log |
| 5 | Infer | 跑 HE 多视角推理 → `_mano.json`, 组装最终 npy | `multiview_hand_init.run_hand_estimation_subprocess` + `parse_mano_json_to_arrays` + `run_golf_capture_to_npy._build_camera_block` + `_object_poses_to_world` | hand0/hand1 mp4 + npy |
| 6 | Visualize | `way_vis` 渲染 3D 轨迹 mp4 (含可选 overlay 在原图) | `utils.way_vis.main` (monkey-patched globals) | trajectory_view0/view1 mp4 |

v1 不含: SAM2 交互式手分割, SportGS 接触优化, 单帧 standard_pose 抽取.

## 启动

```bash
cd /data2/fubingshuai/golf/pipeline
python app.py [--port 7860] [--host 0.0.0.0]
```

如果出现 `PermissionError: [Errno 13] Permission denied: '/tmp/gradio/...'`,
是因为 `/tmp/gradio` 被别人创建过, 设个你能写的目录:

```bash
export GRADIO_TEMP_DIR=$HOME/.gradio_tmp   # 或者 /data2/fubingshuai/.gradio_tmp
python app.py
```

**填 capture_dir 注意:** 写到**采集会话根目录** (含 `camera_params.json` 那一层),
**不要**带上 `.tmp_images` / `.undistorted` 这种子目录后缀, 否则路径会嵌套出问题.
例子: `/data2/fubingshuai/golf/data/35_wood_8_01_fbs/20260424170424563`
而不是 `.../20260424170424563/.tmp_images`. step0 自带 sanity 会自动剥这种后缀.

浏览器开 `http://<server>:7860`. wizard 风格, 一个 Tab 一个 step.

## 第一次使用流程

1. **Setup** tab: 填 `Capture dir` (e.g. `/data2/fubingshuai/golf/data/35_wood_8_01_fbs/20260424170424563`) + `Seq name` (通常跟文件夹名一致), 点 `Initialize workspace`. 看到识别出的两个相机 + workspace 路径.
2. **1. Undistort**: 点 `Run undistort` (秒级). 看到 cam0 第一帧预览.
3. **2. Hand Detection**: 点 `Run hand detection`. 看到带 bbox 的 cam0 预览.
4. **3. HaMER Pseudo Label**: 点 `Generate pseudo labels` (慢, 约 0.5-2h depending on 帧数). 看到伪标 overlay mp4.
5. **4. Self-supervised Finetune**: 默认不勾选, 跳过直接用 `checkpoint_30`. 想 finetune 就勾上 + 调 epochs.
6. **5. Multi-view Inference + npy**: 点 `Run HE inference`. 看到 hand0/hand1 mp4 + 拿到 npy 文件.
7. **6. 3D Visualization**: 点 `Render way_vis`. 看到 3D 轨迹 mp4 (两个视角各一个).

## 项目结构

```
pipeline/
├── app.py                          # Gradio 入口
├── PIPELINE_README.md              # 本文件
├── pipeline/                       # 解耦后的子模块
│   ├── state.py                    # 持久化状态
│   ├── workspace.py                # 路径管理
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
├── <cam>_w1440_h1080_.../      # 用户原始帧目录
├── trajectory_output/          # 用户提供 (物体位姿 csv)
│
├── .undistorted/               # ← step1 输出 (跟现有 wrapper 保持一致)
│   ├── <seq>/<cam>/images_undistorted/
│   ├── <seq>/calib_undistorted/<cam>.yaml
│   ├── pseudo_label_wilor/       # ← step3 输出
│   ├── _pseudo_vis/              # ← step3 视频
│   └── _he_output/               # ← step5 输出
│
└── .pipeline/                  # ← 本项目独有的 workspace
    ├── state.json              # 各 step 当前状态
    ├── detections.json         # ← step2 输出
    ├── _detect_vis/            # ← step2 预览图
    ├── _finetune/              # ← step4 (可选)
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
就会从 state.json reload, 已完成的 step 都显示 `done` 状态, 下次跑 step N 时
会自动读上游 step 的 outputs.

## 依赖

跟 `golf-hand-object` 完全一致, 多一个 `gradio`:

```bash
pip install gradio
```

GPU + CUDA, 跟 `golf-hand-object/environment.yml` 同环境.

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
- Step 6 (way_vis) 通过 monkey-patch globals 调用, 不影响 CLI 兼容, 但**多用户并发会冲突** (模块全局共享). 单用户场景不影响.
