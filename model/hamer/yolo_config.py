class Config:
    weights = '_DATA/yolov7_best.pt'
    imgsz = 640
    augment = True
    conf_thres = 0.25
    iou_thres = 0.35
    classes = [0,1,2]
    agnostic_nms = True
    device = 'cuda'
    save_path = './output'

yolo_opt = Config()