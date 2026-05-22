import os

# 仓库内 model/ 目录（与机器无关）
_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    root_dir = os.environ.get("FBS_MODEL_DIR", _MODEL_DIR)
    ckpt_path = os.path.join(root_dir, 'hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt')
    model_cfg = os.path.join(root_dir, 'hamer/_DATA/hamer_ckpts/model_config.yaml')
    onnx_path = os.path.join(root_dir, 'hamer/_DATA/hamer_ckpts/onnx/hamer_inferpy.onnx')
    use_onnx = False

hamer_opt = Config()
