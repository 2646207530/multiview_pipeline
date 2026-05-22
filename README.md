# 手物优化pipline (`run_hamer_to_npy.py`)

从 RGB 序列一键跑 **HaMER（手）+ YOLO 检测 + SAR/RootNet**，可选注入物体位姿、手物对齐、Contact/力闭合优化与参考帧重放，输出聚合 `npy`。

---

## 快速开始

```bash
git clone <本仓库> 

# 1. 环境（PyTorch 2.1 + CUDA 11.8 + conda pytorch3d，见 environment.yml 注释）
conda env create -f environment.yml
conda activate hamer_env
# 若使用 --opt_range / --force_closure_range：需本机 CUDA Toolkit，再执行：
# pip install git+https://github.com/graphdeco-inria/simple-knn.git

# 2. 权重（二选一）
#  A) 从 Hugging Face 拉到仓库根目录（与下面路径一致）
pip install -U huggingface_hub && huggingface-cli login
hf download lilfiiiiish/golf-weights --local-dir .

#  B) 按下文「权重目录说明」自行放置文件

# 3. 运行（帧图需数字文件名，如 0.jpg、120.png）
python run_hamer_to_npy.py \
  --input /path/to/rgb_frames \
  --output /path/to/out.npy \
  --seq_name seq_001 \
  --cam_k /path/to/cam_K.txt
```

更全的用法（物体 txt、`--point_r`/`--point_l`、`--opt_range`、`--force_closure_range`、`--ref_frame`）见 **`run_hamer_to_npy.py` 顶部文档字符串**。

---

## 仓库代码结构

| 组件 | 路径 |
|------|------|
| 入口脚本 | `run_hamer_to_npy.py` |
| HaMER 推理 | `model/hamer/` |
| YOLO 手部检测 | `model/yolo/` |
| RootNet / SAR | `model/rootnet/` |
| 配置 | `model/config/` |
| Contact / 力闭合 | `model/SportGS/` |

---

## 权重与数据

因体积与 **MANO/SMPL 许可**，大文件默认不进 Git，可用 **Hugging Face** 同步。

| 资源 | Hugging Face 仓库 | 说明 |
|------|-------------------|------|
| 权重包 | [lilfiiiiish/golf-weights](https://huggingface.co/lilfiiiiish/golf-weights) | HaMER、YOLO、SAR、ONNX（可选）、`hamer/_DATA/data`、根目录 `MANO/` 等 |

### 下载（使用者）

```bash
hf download lilfiiiiish/golf-weights --local-dir /path/to/fbs
```

`--local-dir` 建议设为**本仓库根目录**，这样相对路径与代码默认一致。

### 上传（维护者）

根目录脚本 **`upload_weights_to_hf.sh`**：按 `run_hamer_to_npy` 所需文件逐项/按目录上传到上述 HF 仓库（含 `_DATA/data` 与 `MANO/` 递归）。使用前需 HF Write Token：

```bash
bash upload_weights_to_hf.sh
```

### 权重目录说明（手动放置时）

| 用途 | 路径 |
|------|------|
| HaMER | `model/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt`、`model_config.yaml`（部分环境另有 `hamer_ckpts/model_config.yaml`） |
| HaMER 辅助数据 | `model/hamer/_DATA/data/`（MANO、mean 等，见 [HaMER](https://github.com/geopavlakos/hamer)） |
| YOLOv7 | `model/config/checkpoints/yolov7_best.pt` |
| SAR | `model/rootnet/SAR-resnet34-Root.pth` |
| MANO（smplx / SportGS） | 仓库根下 `MANO/`，内含 `MANO_RIGHT.pkl` 等 |

---

## 环境变量（可选）

| 变量 | 作用 |
|------|------|
| `FBS_MODEL_DIR` | `model/` 的绝对路径（默认：仓库内 `model/`） |
| `MANO_MODEL_PATH` | MANO 父目录（默认：`<仓库根>/MANO`） |
| `SAR_CHECKPOINT_PATH` | SAR 权重文件完整路径 |

---

## 许可说明

**MANO** 等人体模型请遵守官方许可，仅限研究用途。HF 上的权重包仅供已获权用户便捷下载，不替代许可登记。
