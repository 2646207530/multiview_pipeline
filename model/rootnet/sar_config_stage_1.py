import os
import torch

_ROOTNET_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    pre = 'SAR'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cam_para = [906.96, 906.79, 1920 // 2, 1080 // 2]
    # network
    backbone = 'resnet34'
    in_channels = 512
    num_stage = 1
    num_FMs = 8
    feature_size = 64
    heatmap_size = 32
    num_vert = 778
    num_joints = 21
    # training
    input_img_shape = (256,256)
    # bbox_real = [250, 250, 250] # ????
    bbox_real = (0.3, 0.3)
    depth_box = 0.3
    # -------------
    checkpoint = os.environ.get(
        "SAR_CHECKPOINT_PATH",
        os.path.join(_ROOTNET_DIR, "SAR-resnet34-Root.pth"),
    )
    # checkpoint = '/home/cyc/pycharm/vGesture/checkpoints/SAR-LR-ROOT.pth'


rgb_opt = Config()
