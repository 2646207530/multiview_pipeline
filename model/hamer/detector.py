import argparse
import time
from pathlib import Path

import cv2
import os
import sys
import torch
import torch.backends.cudnn as cudnn
from numpy import random


sys.path.append('./yolov7')

# from yolov7.models.yolo import Detect
# print("Successfully imported Detect from yolo.py")

from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, time_synchronized, TracedModel
from yolov7.utils.datasets import LoadImage

class Detector():
    def __init__(self, config):
        weights, imgsz, self.device = config.weights, config.imgsz, config.device
        self.device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=stride)  # check img_size
        self.model = TracedModel(self.model, self.device, config.imgsz)
        self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))
        self.dataset = LoadImage(img_size=imgsz, stride=stride)
        self.opt = config
        self.detect_savepath = config.save_path

    def detect_webcam_vedio(self, pipe):
        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.stopnum = 10
        count = 0

        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while count < self.stopnum:
                n += 1
                self.cap.grab()
                if n % 15 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        count += 1
                        pred = self.detect(img0)
                        img_bbox = self.plot_bbox(img0, pred)
                        filename = os.path.join(self.detect_savepath, f'img_{count}.png')
                        cv2.imwrite(filename, img_bbox)
                else:
                    continue

        self.cap.release()
        cv2.destroyAllWindows()

        self.save_img_video(count)
        return 0

    def detect_webcam_img(self, pipe):
        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.stopnum = 10
        count = 0

        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 15 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        self.cap.release()
        cv2.destroyAllWindows()

        dets = self.detect(img0)

        return dets, img0

    def detect(self, image):
        set_logging()
        device = self.device
        # device = select_device(self.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            self.model.half()  # to FP16

        model = self.model
        opt = self.opt
        cudnn.benchmark = True  # set True to speed up constant image size inference

        # Run inference
        # Run inference
        imgt0 = time_synchronized()
        img, im0s = self.dataset.process_img(image)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        imgt1 = time_synchronized()

        # Inference
        infert0 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        infert1 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        nmst1 = time_synchronized()
        # Process detections
        dets_list = []
        for i, det in enumerate(pred):  # detections per image
            dets = []
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for i_row, det_row in enumerate(det.tolist()):
                    if det_row[-1] ==1:
                        det_type = 'right'
                    else:
                        det_type = 'left'
                    dets.append([det_type, det_row[:4]])
            dets_list.append(dets)
        endtime = time_synchronized()
        # Print time (inference + NMS)
        # print(f'hand Done.({(1E3 * (imgt1 - imgt0)):.1f}ms) predata, ({(1E3 * (infert1 - infert0)):.1f}ms) Inference, ({(1E3 * (nmst1 - infert1)):.1f}ms) NMS, ({(1E3 * (endtime - imgt0)):.1f}ms) all')
        return pred, dets_list

    def plot_bbox(self, img, bboxs):
        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        for det in bboxs:
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=1)
        return img


    def save_img_video(self, num_frame):
        video = cv2.VideoWriter(
            os.path.join(os.path.join(self.detect_savepath, 'video'), '{}_detect.avi'.format(num_frame)),
            cv2.VideoWriter_fourcc(*'XVID'), 10, (256, 256))

        for i in range(num_frame):
            # 读取图片
            img_path = os.path.join(self.detect_savepath, 'img_{}.png'.format(i))
            img = cv2.imread(img_path)
            video.write(img)
        video.release()
