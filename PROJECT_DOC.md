# golf-hand-object 项目文档

本文档面向"想知道每段代码做什么、谁调谁、输入输出长啥样"的开发者。
按数据流串起整条 pipeline，每节给文件路径、职责、输入、输出、关键参数。

---

## 0. 一图速览

```
                       capture_dir/
        ┌──── camera_params.json ─────┐
        │                              │
        ▼                              ▼
  trajectory.csv               .tmp_images/<cam>/frame_*.jpg
        │                              │
        │            ┌─────────────────┴───── (≥2 个相机) ─────► multiview_hand_init.py
        │            │                                              │ ① undistort
        │            │ (1 个相机)                                   │ ② per-cam HaMER → 2D 伪标
        │            ▼                                              │ ③ Hand_Estimation/visualize_mano.py 子进程
        │  HaMER + YOLO 单视角                                     │ ④ <seq>_mano.json
        │  (run_hamer_to_npy 的助手)                              │
        ▼                              ▼                          ▼
  obj_rot/obj_trans              right/left hand MANO 4 件套 (rot, pose, trans, shape)
   (世界系 / m)                  per-frame, 缺失 → NaN 占位
        │                              │
        └──────────────┬───────────────┘
                       ▼
            root dict (内存)
                       │
        ┌──────────────┼───────────────┐
        │              │               │
        ▼              ▼               ▼
  NaN 插值        concat_hand_to_object   smooth_object_pose_temporal
  (SLERP /       (--point_r/--point_l)    (--obj_pose_sigma>0 时)
   线性)
                       │
                       ▼
        save → <output>_post_concat.npy   ← Concat 后/优化前 快照
                       │
                       ▼
        SportGS Contact 优化 (--opt_range)            ┐
        train_contact.py 子进程                        │
        (修改 hand pose + obj pose 同时调; 通用碰撞   │
         /吸引 + 你定义的 hand_hand_contacts)         │
                       │                                │
                       ▼                                │ render_mesh.py 装配 loss
        save → <output>_opt_contact.npy                │ + utils/loss_utils.py 计算
                       │                                │ + models/pose_correction.py 给参数
                       ▼                                │
        SportGS 力闭合优化 (--force_closure_range)     │
        finetune_force_closure.py 子进程               │
        (冻结手腕+物体位姿; 微调手指; FCLoss 用       │
         hand_attract_tips 拉近最近 obj 顶点)          │
                       │                                ┘
                       ▼
        save → <output>_opt_force_closure.npy
                       │
                       ▼
        rehand_by_object (--ref_frame)
                       │
                       ▼
        save → <output>.npy   (最终态)
```

---

## 1. 顶层入口 `run_golf_capture_to_npy.py`

**职责**：编排整条 pipeline——读 capture → 初始化手 → 对齐物体 → 写 mask → 调用 SportGS 子进程 → rehand → 写 npy。

**输入**（CLI）：

| 参数 | 必需 | 说明 |
|---|:---:|---|
| `--capture_dir` | ✓ | 例如 `/data2/.../35_wood_8_01_fbs/20260424170424563`，自动识别布局 A（帧在根下）/布局 B（帧在 `.tmp_images/` 下）|
| `--output` | ✓ | 输出 npy 路径 |
| `--seq_name` | ✓ | 写入 `data_dict` 的 key |
| `--hamer_cam` / `--other_cam` |  | 指定 cam0 / cam1（默认按 `camera_id` 自动挑两台 1440×1080 彩色）|
| `--obj_pose_sigma` |  | 物体位姿时序高斯滤波 σ（帧），0=不滤 |
| `--point_r` `--point_l` |  | 各 3 个 float (x y z m)。两者必须**同时**给。`concat_hand_to_object` 把手平移到这两个点 |
| `--opt_range START END` |  | Contact 优化的帧区间（左闭右开）|
| `--force_closure_range START END` |  | 力闭合优化的帧区间 |
| `--ref_frame` |  | 优化结束后用这一帧把手"重新挂"回物体 |
| `--club_stl` |  | 球杆 STL（mm）。默认按规则在 `<data_root>/club-assets/<club_name>/*.stl` 找 |
| `--contact_config` |  | 预定义接触点 JSON。默认走 `config/baseball_golf.json`；通过 `CONTACT_CONFIG_PATH` 环境变量传给 SportGS 子进程 |
| `--init_method` |  | `auto`（默认）/ `hamer` / `multiview`。`auto` 看录制相机数 ≥2 就走 multiview |

**输出**：在 `--output` 同目录产出多份 npy（中间快照都保留）：

```
<output>_post_concat.npy           ← 仅 concat 后（如果传了 --point_r/--point_l）
<output>_opt_contact.npy           ← Contact 优化后
<output>_opt_force_closure.npy     ← 力闭合后
<output>.npy                       ← 最终（含 rehand）
```

**关键调用链**（`run()` 函数）：

1. `_load_camera_params(capture_dir)` 读 `camera_params.json`
2. `_resolve_color_cams(...)` 选 hamer_cam / other_cam（仅从有帧目录的彩色相机里挑）
3. `_load_trajectory(...)` 读 `trajectory_output/trajectory.csv`，得到 `(ref_camera_name, [qw,qx,qy,qz,tx,ty,tz]_per_frame)`
4. `_object_poses_to_world(...)` 把物体位姿从 ref_camera_sensor 变到 hamer_cam_sensor（= world），单位 mm→m
5. **手部初始化分叉**（`init_method` + `n_color_cams_recorded` 联合决定）：
   - 单视角分支：本文件内的 HaMER 主循环
   - 多视角分支：`multiview_hand_init.init_hands_from_multiview(...)`
6. NaN 插值：`_slerp_interpolate_nan(rot)` + `_interpolate_nan(其他)`
7. 组装 `root` dict（schema 见 §10）
8. 物体平滑（可选）：`smooth_object_pose_temporal`
9. `concat_hand_to_object`（如果传了两个 point）→ 写 `_post_concat.npy`
10. `generate_masks_sam2.ensure_masks(...)`（如果要优化）
11. SportGS 子进程：先 `train_contact.py`，再 `finetune_force_closure.py`
12. `rehand_by_object`（如果传了 `--ref_frame`）
13. 写最终 npy

**环境变量**：`PYOPENGL_PLATFORM=egl`（rendering 头文件需要）、`CONTACT_CONFIG_PATH`（仅给 SportGS 子进程）。

---

## 2. 单视角手部初始化（HaMER）`run_hamer_to_npy.py`

被 `run_golf_capture_to_npy.py` 当成"工具库"复用：

| 函数 | 职责 |
|---|---|
| `parse_detections(dets)` | YOLO 输出 → `[[hand_label, x1, y1, x2, y2], ...]` |
| `extract_hand_params(output, mano_params, hand_label)` | 从 HaMER 输出抽 `pose_global (3,)`, `pose_hand (45,)`, `betas (10,)`, `cam_t (3,)` |
| `NAN_RIGHT` / `NAN_LEFT` | 缺失帧的 NaN 占位字典 |
| `_interpolate_nan(arr)` | 线性插值 NaN（用于 pose / shape / trans）|
| `_slerp_interpolate_nan(rot_arr)` | 旋转 SLERP 插值 |
| `concat_hand_to_object(root, seq_name, point_r, point_l)` | 把每帧手关节质心整体平移到 `point_r`/`point_l`（世界系，米）。优化前的硬对齐 |
| `rehand_by_object(root, seq_name, ref_frame)` | 用 `ref_frame` 那一帧的物体位姿 + 该帧手与物体的相对关系，把整段手变换"挂"回物体上 |
| `smooth_object_pose_temporal(root, seq_name, sigma)` | 物体 rot（quat 域）/ trans 时序高斯平滑 |

**HaMER 主循环**（在 `run_golf_capture_to_npy.py:run()` 单视角分支）：

```
for each frame i in 0..total_frames-1:
    image = imread(frame_map[i])
    if image is None: stats['missing_file'] += 1; 占位 NaN; continue
    dets = detector.detect(image)               # YOLO bbox
    for bbox in dets:
        output = hamer.estimate_from_rgb(image, [bbox], k_use)
        hand_data = extract_hand_params(output, ...)  # (rot, pose, betas, cam_t)
    若某只手没检出 → 占位 NaN, stats['missing_*'] += 1
```

**输出**：8 个 numpy 数组（右手/左手 × {rot (T,3), pose (T,45), shape (T,10), trans (T,3)}），缺失帧为 NaN。

---

## 3. 多视角手部初始化 `multiview_hand_init.py`

**目的**：当有 ≥2 个 1440×1080 彩色相机时，把手部位姿从 HaMER 单视角换成 `model/Hand_Estimation/` 多视角网络的输出。

**职责拆解**（4 步，全部封在 `init_hands_from_multiview(...)`）：

### 3.1 `prepare_undistort_dir(capture_dir, cams_dict, cam_names, world_name, frame_dirs, force=False)`

- **做什么**：构建 `<capture_dir>/.undistorted/` 缓存目录（**幂等**，已 undistort 过的 jpg 会跳过）
- **输出布局**：

```
<capture_dir>/.undistorted/
├── <capture_id>/
│   ├── 0/images_undistorted/frame_NNNNNN.jpg     ← cam_idx=0 (= hamer_cam = world)
│   ├── 1/images_undistorted/frame_NNNNNN.jpg     ← cam_idx=1
│   └── calib_undistorted/
│       ├── 0.yaml   { K: 3x3, R: 3x3 (= I), t: [0,0,0] }
│       └── 1.yaml   { K, R: world->cam1 旋转, t: world->cam1 平移 (米) }
└── pseudo_label_wilor/                            (空目录，下一步填)
```

- **关键技术**：`cv2.getOptimalNewCameraMatrix` 算 newK，`cv2.initUndistortRectifyMap` + `cv2.remap` 去畸变；newK 比原 K 略大（边缘 alpha=0 拉直）

### 3.2 `write_pseudo_labels_from_hamer(undist_root, ..., hamer, detector, parse_detections, force=False)`

- **做什么**：每相机 × 每帧 × 每检出手 → HaMER 推理 → 提取 21 关节 2D（`output['pred_keypoints_2d_full']`）→ 写 npz
- **输出文件名规则**：`<undist_root>/pseudo_label_wilor/<seq>_<cam_idx>_<frame_id>_<hand_id>.npz`
  - `<hand_id>`：0=right, 1=left（Hand_Estimation 约定）
- **每个 npz 的字段**：
  - `is_right` shape (1,) float32：1.0/0.0
  - `joints_2d` shape (21, 2) float32：在 undistorted 图像坐标系
- **幂等**：已存在的 npz 跳过（除非 `force=True`）

### 3.3 `run_hand_estimation_subprocess(project_root, undist_root, output_dir, gpu_id="0")`

- **做什么**：spawn 子进程 `python visualize_mano.py --cfg <yaml> --input_dir <undist_root> --output_dir <tmp> --reload <ckpt>`
- **依赖**：
  - 配置：`model/Hand_Estimation/config/release/GOLF_Inference.yaml`
  - 检查点：`model/Hand_Estimation/exp/FT/checkpoints/checkpoint`
- **输出**：`<output_dir>/<seq>_mano.json`，schema：

```jsonc
{
  "right_hand": {
    "rot_r":   [[r0,r1,r2], ...],          // (T_r, 3)   全局朝向 axis-angle
    "pose_r":  [[..], ...],                 // (T_r, 45)  15 关节弯曲 axis-angle
    "trans_r": [[tx,ty,tz], ...],           // (T_r, 3)   世界系平移 (米)
    "shape_r": [[..], ...]                  // (T_r, 10)  betas
  },
  "left_hand": { ... }                       // 同上, _l 后缀
}
```

### 3.4 `parse_mano_json_to_arrays(json_path, total_frames)`

- **做什么**：把 JSON 转成 8 个 (T, *) 形状的 numpy 数组，缺帧填 NaN
- **⚠️ 已知缺陷**：当前 `visualize_mano.py` 只按 `sorted(frame_id)` 写出值，**没有同步写出 frame_id 列表**。本函数假设第 i 行就是第 i 帧；如果 Hand_Estimation 内部按 `WINDOW_SIZE=3, STEP_SIZE=3` 跳了帧，对齐会错位。生产前要么改 `visualize_mano.py` 同时导 frames，要么在 wrapper 里推导

### 3.5 入口 `init_hands_from_multiview(...)`

按顺序串起 3.1 → 3.2 → 3.3 → 3.4，返回与单视角分支同形状的 8 个数组。

---

## 4. 物体位姿读取（轨迹 CSV）

完全在 `run_golf_capture_to_npy.py` 内：

- `_load_trajectory(csv_path)` 读 `trajectory_output/trajectory.csv` 顶部 `# reference_camera: <name>` 注释 + 数据行 `qw,qx,qy,qz,tx,ty,tz`（mm）
- `_object_poses_to_world(poses_mm, cams, ref_name, world_name)` 把 obj→ref_camera_sensor 的位姿换算到 obj→world_camera_sensor。返回 `(obj_rot (T,3) axis-angle, obj_trans (T,3) meters)`
- 两端帧数对不齐就按 `min(n_obj, total_frames)` 截断

---

## 5. SAM2 mask 生成 `generate_masks_sam2.py`

**只在 `--opt_range` 或 `--force_closure_range` 触发时调**。完全交互式，没自动初始化路径。

**入口**：`ensure_masks(root, seq_name, frames, club_mesh_path=None, vis_dir=..., force=True, max_disp_w=1280)`

**工作流**（VideoPredictor 模式，2025 年 5 月起）：

1. 把请求帧符号链接到 `tempfile.mkdtemp()`，重命名为 `000000.jpg, 000001.jpg, ...`（SAM2 video 格式要求整数帧名）
2. 用 `build_sam2_video_predictor` 起多目标 video predictor
3. **Phase 1**：在第一帧上交互标注 3 个 obj_id
   - obj_id=2（右手）→ obj_id=3（左手）→ obj_id=1（球杆）
   - 操作：Shift+左键正点 / Shift+右键负点 / `u` 撤销 / `c` 清空 / `SPACE` 确认 / `n` 跳过 / `q` 退出
4. **Phase 2**：`predictor.propagate_in_video(state)` 自动传播全段
5. **Phase 3**：复查回路
   - `j`/`k` 翻帧；`J`/`K` ±10；`g` 跳指定 fi
   - 发现错帧 → `e` 进入编辑：`1`/`2`/`3` 切目标，加点修正，`SPACE` 重新传播
   - `s` 保存全部并退出；`q` 放弃

**输出**：

- `<root["imgpath"]>_masks/<basename>.npy` — uint8 (H,W)，0=背景, 1=club, 2=右手, 3=左手
- `<vis_dir>/<seq>_<basename>_mask.jpg` — 可视化 overlay

**输入要求**：`root["imgnames"]` 和 `root["imgpath"]` 已正确填写（HaMER cam0 的帧）。

---

## 6. SportGS 优化层

两个子进程，wrapper 用 hydra override 喂参数：

| 阶段 | 入口 | 触发 | 关键 hydra override |
|---|---|---|---|
| Contact 优化 | `model/SportGS/train_contact.py` | `--opt_range S E` | `dataset.pose_path`, `+dataset.opt_frame_start`, `+dataset.opt_frame_end`, `+dataset.export_data_path`, `+dataset.export_output_path`, `+dataset.seq_name`, `+dataset.obj_mesh_path`, `wandb_disable=true` |
| 力闭合 | `model/SportGS/finetune_force_closure.py` | `--force_closure_range S E` | 同上 |

子进程读 `CONTACT_CONFIG_PATH` 环境变量定位 contact 配置。

### 6.1 优化变量（`model/SportGS/models/pose_correction/pose_correction.py`）

```
DirectPoseOptimization(freeze_pose):
  betas_r/l        nn.Parameter            (10维)         → 始终可训
  rot_r/l          nn.Embedding(freeze=False)             → 始终可训 (手腕全局朝向)
  pose_r/l         nn.Embedding(freeze=False)             → 始终可训 (15 关节)
  trans_r/l        nn.Embedding(freeze=freeze_pose)       → 力闭合阶段冻结
  obj_rots         nn.Embedding(freeze=freeze_pose)       → 力闭合阶段冻结
  obj_trans        nn.Embedding(freeze=freeze_pose)       → 力闭合阶段冻结
  pose_r_ori, pose_l_ori, trans_r_ori, trans_l_ori        → 永久冻结，保留初始值供 prior 用
```

| | Contact 阶段 (`freeze_pose=False`) | 力闭合阶段 (`freeze_pose=True`) |
|---|:---:|:---:|
| 手腕位置 trans | 可训 | 冻结 |
| 球杆位姿 obj_rot/trans | 可训 | 冻结 |
| 手指 pose | 可训 | 可训 |
| 手腕朝向 rot | 可训 | 可训 |
| 手形 betas | 可训 | 可训 |

### 6.2 Loss 装配（`model/SportGS/gaussian_renderer/render_mesh.py`）

`render_mesh(data, iteration, scene, body_model_r, body_model_l, mano_finger_labels, vis, finetune)` 返回 `(render_pkg, loss_dict)`。`loss_dict["contact"]` 由如下项相加：

```python
loss_contact = lambda_collision * (loss_collision     # 右手 ↔ 杆 穿透
                                 + loss_collision_l   # 左手 ↔ 杆 穿透
                                 + loss_collision_h2h)# 左 ↔ 右 穿透
             + lambda_attract  * (loss_attract        # 右手 ↔ 杆 吸引 (contact_zones=zones)
                                 + loss_attract_l)    # 左手 ↔ 杆 吸引

loss_contact += lambda_hh_predef * loss_hh_predef     # ★ 你的 hand_hand_contacts (config 来源)

if finetune:
    loss_contact += (l8a + l8b + l8a_l + l8b_l) * 1e-2   # ★ force_closure (用 hand_attract_tips)
    lambda_collision *= 10                                # 力闭合阶段碰撞强 10 倍
    lambda_hh_predef *= 10                                # hand_hand 同步加强
```

| 子项 | 来自 | 实现位置 |
|---|---|---|
| `loss_collision`/`loss_attract` | 通用 | `utils/loss_utils.compute_contact_loss` (contact_zones="zones" 读 contact_zones.pkl) |
| `loss_collision_h2h` | 通用 | 同上，左手顶点 vs 右手三角面 |
| `loss_hh_predef` | **配置** | `render_mesh.py` 内联，读 `contact_config.get_hand_hand_contacts()` |
| `force_closure` (l8a/l8b) | **配置** | `render_mesh.force_closure(side)` → `contact_config.get_attract_tips(side)` → knn_points 找最近 obj 顶点 → `FCLoss().fc_loss(tips, contact_normal)` |

### 6.3 总 loss

`train_contact.py` 主循环里：

```python
loss = lambda_mask * mask_loss
     + Σ loss_reg["..."] * lambda_*               # 来自 render() 的光度/正则
     + Σ loss_reg_mesh["..."] * lambda_*          # render_mesh 返回, 主要就是 contact
loss.backward(); scene.optimize(iteration)
```

`finetune_force_closure.py` 几乎一样，只是 `Scene(..., freeze_pose=True)` + `render_mesh(..., finetune=True)`，并默认只跑 `range(120, 200)` 帧（**注意**：这是 SportGS 自己写死的，可能与你 `--force_closure_range` 不一致）。

---

## 7. 接触配置 `config/baseball_golf.json`

唯一的"业务配置"。所有需要按场景调整的接触点 / tip 顶点都在这里。

```jsonc
{
  "_README": [...],                 // 注释数组, 解析时被忽略
  "club_mesh_path": "/data2/.../20260421142122_meter.stl",

  "hand_attract_tips": {            // §6.2 force_closure 用的 MANO tip vid
    "right": [768, 342, 454, 565, 683, 77],
    "left":  [768, 342, 454, 565, 683, 77]
  },

  "hand_hand_contacts": [           // §6.2 hand_hand_predef 用的成对配置
    {"name": "auto_R_L_0",
     "right_mano_vid": 112, "left_mano_vid": 718, "weight": 1.0}
  ],

  "right_hand_object_contacts": [...],   // 预留, 当前未消费 (架子已搭, loss 项可补)
  "left_hand_object_contacts":  [...]    // 同上
}
```

**Loader**：`model/SportGS/utils/contact_config.py`

| 函数 | 返回 |
|---|---|
| `get_attract_tips(side)` → `List[int]` | side ∈ {"right","left"}; 缺失/越界回退 `DEFAULT_ATTRACT_TIPS = [768, 342, 454, 565, 683, 77]` |
| `get_hand_hand_contacts(n_mano_verts=778)` → `List[dict]` | 校验过的 `[{right_mano_vid, left_mano_vid, weight, name}, ...]`；缺字段/越界/坏 weight 都跳过并打告警 |
| `get_config()` → `dict` | 整份 JSON |
| `reload()` | 强制下次访问重读（用于 viewer 的 `r` 命令）|

**路径解析**：`CONTACT_CONFIG_PATH` 环境变量优先；否则 fallback 到 `<project_root>/config/baseball_golf.json`。

---

## 8. 接触点可视化 `utils/vis_predefined_contacts.py`

**做什么**：在 Open3D 弹窗里同时显示右手/左手/球杆 mesh，并把配置里的所有接触点画成 sphere（成对的还有圆柱连线）。支持手动从 JSON 重载（`r` 命令）和窗口内 Shift+pick 直接添加新接触点。

**用法**：
```bash
python utils/vis_predefined_contacts.py config/baseball_golf.json
```

**视觉对照**：

| 元素 | 颜色 | 半径 | 来源 |
|---|---|---|---|
| 右手 MANO mesh | 浅绿 | — | `MANO/MANO_RIGHT.pkl` v_template |
| 左手 MANO mesh | 浅蓝 | — | `MANO/MANO_LEFT.pkl` v_template |
| 球杆 mesh | 浅橙 | — | `club_mesh_path` |
| 配对 contact sphere | 调色板循环色 | 4mm | `*_hand_object_contacts` / `hand_hand_contacts` |
| 配对 contact 连线 | 同 sphere 色 | 0.8mm 圆柱 | 同上 |
| **attract tip sphere** | **金色** | **6mm** | `hand_attract_tips.right/left` |

**交互**（终端 stdin 输入 + Enter）：

| 命令 | 作用 |
|---|---|
| `r` | 从磁盘重新读 config, 重建几何 |
| `c` | 清掉当前 Shift+pick 但还没成对的点 |
| `q` | 退出 |
| 窗口里 Shift+左键点 2 个顶点 | 自动按落点 mesh 分类（右手+杆 / 左手+杆 / 双手），追加新条目到 JSON 并刷新 |

**Open3D 限制**：用 `VisualizerWithVertexSelection`（支持顶点拾取），它**只能 add_geometry 一次**，所以所有 mesh+sphere+连线圆柱都被合并成单个 TriangleMesh 添加（`_combine_meshes`）。

---

## 9. 模型与外部权重

| 路径 | 用途 | 来源 |
|---|---|---|
| `model/hamer/` | HaMER 单视角手部网络 | 上游论文，本项目入口 `infer.py:hamer_inference` |
| `yolo/` + `model/sam2_ckpts/sam2.1_hiera_small.pt` | YOLO 手 bbox 检测 + SAM2 分割 | 上游 |
| `model/Hand_Estimation/` | 多视角手部网络（替代 HaMER）| 子模块 |
| `model/Hand_Estimation/exp/FT/checkpoints/checkpoint` | Hand_Estimation 权重 | 训练产物 |
| `MANO/MANO_RIGHT.pkl` / `MANO_LEFT.pkl` | MANO 手模板（778 verts, 1538 faces）| MPI |
| `model/SportGS/` | 高斯泼溅 + 手物 contact 优化 | 改自 SportGS 论文 |

---

## 10. 输出 npy schema

整条 pipeline 各阶段产物的 `.npy` 都遵循同一 schema（`np.save(..., allow_pickle=True)` 存的是字典）：

```python
{
  "imgnames": ["frame_000000.jpg", "frame_000001.jpg", ...],   # HaMER cam0 的帧文件名
  "imgpath":  ".../<hamer_cam>_w1440_h1080_pBayerRG8_f120/",   # 帧目录
  "data_dict": {
    "<seq_name>": {
      "params": {
        "right hand": {
          "rot_r":   ndarray(T, 3)  float32,    # 手腕全局 axis-angle
          "pose_r":  ndarray(T, 45) float32,    # 15 关节 axis-angle
          "trans_r": ndarray(T, 3)  float32,    # 米, world (=cam0 sensor) 系
          "shape_r": ndarray(T, 10) float32,    # MANO betas
        },
        "left hand": {                          # 同上, *_l 后缀
          "rot_l":   ndarray(T, 3),
          "pose_l":  ndarray(T, 45),
          "trans_l": ndarray(T, 3),
          "shape_l": ndarray(T, 10),
        },
        "object": {
          "obj_rot":   ndarray(T, 3) float32,   # axis-angle
          "obj_trans": ndarray(T, 3) float32,   # 米
        },
        "camera": {
          "world2cam": [w2c_cam0_4x4 (= I), w2c_cam1_4x4],   # 4×4 float32
          "K":         [K_cam0_3x3, K_cam1_3x3],             # 3×3 float32
          "views":     ["cam0", "cam1"],
        },
      },
    },
  },
}
```

**坐标系/单位约束**：

- World = cam0 (= `hamer_cam`) sensor 系
- 所有平移单位米
- `world2cam[cam0]` 恒为单位阵
- `world2cam[cam1]` = `R_cam1.T @ R_cam0`, `t = R_cam1.T @ (t_cam0 - t_cam1)`，其中 R/t 是 sensor2rig

---

## 11. 各阶段 npy 文件对应表

`<output>` 是 `--output` 给的路径（不带 `_post_concat` 等后缀）。出现哪些文件取决于哪些 CLI 触发了：

| 文件 | 何时产出 | 内容 |
|---|---|---|
| `<output>_post_concat.npy` | 给了 `--point_r`+`--point_l` | HaMER/MV 推理 + obj_pose_smooth + concat 后, 优化前的快照 |
| `<output>_opt_contact.npy` | 给了 `--opt_range` | Contact 优化产物 (内部由 `train_contact.py` 写出) |
| `<output>_opt_force_closure.npy` | 给了 `--force_closure_range` | 力闭合优化产物 |
| `<output>.npy` | 永远 | 最终态: 上一步基础上再 rehand_by_object（如果 `--ref_frame`）|

---

## 12. 常用命令速查

```bash
# 最小: 仅手 + 物体注入, 自动选 init 方法
python run_golf_capture_to_npy.py \
  --capture_dir /data2/.../20260424170424563 \
  --output ./out/35_wood_8_01_fbs.npy \
  --seq_name 20260424170424563

# 强制单视角 HaMER (旧行为)
python run_golf_capture_to_npy.py ... --init_method hamer

# 完整: concat + Contact 优化 + 力闭合 + rehand
python run_golf_capture_to_npy.py \
  --capture_dir /data2/.../20260424170424563 \
  --output ./out/35_wood_8_01_fbs.npy \
  --seq_name 20260424170424563 \
  --point_r 0.0 0.82 0.0 --point_l 0.0 0.90 0.0 \
  --contact_config config/baseball_golf.json \
  --opt_range 1 50 --force_closure_range 1 50 --ref_frame 25

# 接触点 / tip vid 调整 (打开窗口, Shift+pick 添加, r 重载)
python utils/vis_predefined_contacts.py config/baseball_golf.json
```

---

## 13. 已知坑 & TODO

1. **`visualize_mano.py` 不导出 frame_id**：多视角分支 `parse_mano_json_to_arrays` 假设按帧顺序填充，若 Hand_Estimation 因 WINDOW_SIZE 跳帧会错位。修法：改 `visualize_mano.py` 在 JSON 里同时写一份 `frames` 列表
2. **`finetune_force_closure.py` 内写死了 `range(120, 200)`** ([train loop](model/SportGS/finetune_force_closure.py#L130-L131))：与 wrapper 传来的 `--force_closure_range` 解耦了，需要后续把 hydra override 真正接到这个循环
3. **`right_hand_object_contacts` / `left_hand_object_contacts`**：JSON 字段已定，但 SportGS 里**没有**对应的 loss 项，可视化也只画。如果想消费它们，参考 `loss_hh_predef` 的写法在 `render_mesh.py` 加一段
4. **HaMER 用 K 与 undistort 一致性**：多视角分支里 HaMER 推理用的是 newK + undistorted 图。如发现 2D 关键点偏差大，可改回 origK + 原图，再在写盘前把 2D 点 undistort 一遍
5. **`contact_zones.pkl` 路径硬编码**：[loss_utils.py:457](model/SportGS/utils/loss_utils.py#L457) 写死了 `/data2/fubingshuai/fbs/HOContactopt/data/contact_zones.pkl`，迁移时记得改

---

## 14. 文件索引

```
golf-hand-object/
├── PROJECT_DOC.md                                ← 本文档
├── README.md
├── environment.yml
│
├── run_golf_capture_to_npy.py                   ← §1 顶层入口
├── run_hamer_to_npy.py                           ← §2 HaMER 助手 + 复用工具
├── multiview_hand_init.py                        ← §3 多视角入口
├── generate_masks_sam2.py                        ← §5 SAM2 video mask 标注
│
├── config/
│   ├── baseball_golf.json                        ← §7 接触配置
│   ├── hamer_config.py
│   └── yolo_config.py
│
├── utils/
│   ├── vis_predefined_contacts.py                ← §8 接触点可视化
│   ├── ...                                        其余 utils 与本项目主流程关系不大
│
├── MANO/
│   ├── MANO_RIGHT.pkl  /  MANO_LEFT.pkl
│   └── ...
│
├── model/
│   ├── hamer/                                    ← HaMER 推理
│   ├── Hand_Estimation/                          ← 多视角网络 (子项目)
│   │   ├── visualize_mano.py                     ← multiview 子进程入口
│   │   ├── config/release/GOLF_Inference.yaml
│   │   ├── exp/FT/checkpoints/checkpoint
│   │   └── lib/datasets/{golf,golfInfra_dataset}.py
│   ├── sam2_ckpts/sam2.1_hiera_small.pt
│   └── SportGS/
│       ├── train_contact.py                      ← §6 Contact 阶段入口
│       ├── finetune_force_closure.py             ← §6 力闭合阶段入口
│       ├── gaussian_renderer/render_mesh.py      ← §6.2 loss 装配
│       ├── utils/
│       │   ├── loss_utils.py                     ← compute_contact_loss / FCLoss
│       │   └── contact_config.py                 ← §7 配置 loader
│       ├── models/pose_correction/
│       │   └── pose_correction.py                ← §6.1 优化变量
│       ├── scene/__init__.py                     ← Scene(freeze_pose=...)
│       └── configs/option/iter_arctic_15k.yaml   ← lambda_mask / lambda_contact 等
│
└── out/                                           ← 默认输出目录
    ├── 35_wood_8_01_fbs.npy
    ├── 35_wood_8_01_fbs_post_concat.npy
    ├── 35_wood_8_01_fbs_opt_contact.npy
    └── 35_wood_8_01_fbs_opt_force_closure.npy
```
