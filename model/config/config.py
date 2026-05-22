
# class Config(object):
#     phase = 'test'
#     rgb_dir = '/home/pt/vGesture/test_img/video_1/rgb/0022.png'
#     depth_dir = '/home/pt/vGesture/test_img/video_1/depth/0022.png'
#     output_root = './output'
#     mano_dir = '/home/pt/vGesture/assets/mano/models/MANO_RIGHT.pkl'
#     save_dir = '/home/pt/vGesture/test_img/output_test'

#     detect_method = 'yolov7'
#     estimate_method = 'sar'
#     seq_len = 15
#     view_num = 2

#     show_body = True
#     save_img = True

import os

class Config(object):
    root_dir = os.path.abspath(os.path.dirname(__file__))
    print(f"Root Directory: {root_dir}")

opt = Config()
