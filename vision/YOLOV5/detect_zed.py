# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import os
import platform
from re import S
import sys
from pathlib import Path
from numpy.core.defchararray import center

from numpy.core.fromnumeric import shape
from numpy.lib.shape_base import apply_over_axes
# import rospy
# from std_msgs.msg import String

from PIL import Image
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import pyrealsense2 as rs
import time

from utils.augmentations import letterbox
# from sensor_msgs.msg import JointState
import pyzed.sl as sl
fx = 617.0245971679688
fy = 617.330810546875
cx = 320.0
cy = 240.0

# pub = rospy.Publisher('apple_info',JointState,queue_size=10)
# rospy.init_node('apple',anonymous=True)
# apple_XYZ = JointState()
# apple_XYZ.name = {'apple_x','apple_y','apple_z'}
# apple_XYZ.position =[]

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, apply_classifier, check_file, check_img_size, check_imshow, check_requirements,
                           check_suffix, colorstr, increment_path, non_max_suppression, print_args, scale_coords,
                           strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import load_classifier, select_device, time_sync
align_to = rs.stream.color
align = rs.align(align_to)
pipeline = rs.pipeline()
config = rs.config()
colorizer = rs.colorizer()

config.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=0,  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=0.5,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    zed = sl.Camera()
    # 设置相机的分辨率1080和采集帧率30fps
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD1080 video mode  HD720, VGA 672*376
    init_params.camera_fps = 30  # fps可选：15、30、60、100

    err = zed.open(init_params)  # 根据自定义参数打开相机
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)
    runtime_parameters = sl.RuntimeParameters()  # 设置相机获取参数
    runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD
    i = 0
    # 创建sl.Mat对象来存储图像（容器），Mat类可以处理1到4个通道的多种矩阵格式（定义储存图象的类型）
    image = sl.Mat()  # 图像
    disparity = sl.Mat()  # 视差值
    dep = sl.Mat()  # 深度图
    depth = sl.Mat()  # 深度值
    # 获取分辨率
    resolution = zed.get_camera_information().camera_resolution
    w, h = resolution.width, resolution.height
    fx, fy = int(w / 2), int(h / 2)  # 中心点

    device = select_device(device)
    # img = torch.zeros((1, 3, imgsz, imgsz),device = device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    w = str(weights[0] if isinstance(weights, list) else weights)
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        if dnn:
            check_requirements(('opencv-python>=4.1.0',))
            net = cv2.dnn.readNetFromONNX(w)
    try:
        while True:
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:  # 相机成功获取图象
                save_txt = []
                timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # 获取图像被捕获时的时间点
                zed.retrieve_image(image, sl.VIEW.LEFT)  # image：容器，sl.VIEW.LEFT：内容
                img = image.get_data() # 转换成图像数组，便于后续的显示或者储存
                # img = img.conver("RGB")
                color_frame_numpy = np.asanyarray(img)
                color_frame_numpy_trans = color_frame_numpy.transpose((2,1,0))
                input_slice = color_frame_numpy_trans[0:3]
                color_frame_numpy_trans = input_slice.transpose((2, 1, 0))
                # 获取深度
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH, sl.MEM.CPU)  # 深度值
                zed.retrieve_image(dep, sl.VIEW.DEPTH)  # 深度图
                depth_map = depth.get_data()
                dep_map = dep.get_data()
                color_frame_numpy = color_frame_numpy_trans

                # cv2.imshow('zed', image.get_data())
                cv2.imshow('depth', dep.get_data())
                cv2.waitKey(10)
                imgsz = check_img_size(imgsz, s=stride)  # check image size
                source = [source]
                path = source
                imgs = [None]
                imgs[0] = color_frame_numpy
                im0s = imgs.copy()
                img = [letterbox(x, new_shape=imgsz)[0] for x in im0s]
                img = np.stack(img,0)
                img = img[:, :, :, ::-1].transpose(0,3,1,2)
                img = np.ascontiguousarray(img, dtype=np.float16 if half else np.float32)
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                img = torch.from_numpy(img).to(device)
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                pred = model(img, augment =  augment,visualize=visualize)[0]
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms,max_det=max_det)

                for i, det in enumerate(pred):  # detections per image
                    p, s, im0 =path[i],'%g:'%i, im0s[i].copy()
                    s+='%gx%g' %img.shape[2:]
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                    if det is not None and len(det):
                        det[:,:4] = scale_coords(img.shape[2:], det[:,:4], im0.shape).round()
                        for c in det[:,-1].unique():
                            n = (det[:,-1] == c).sum()
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                        for *xyxy, conf, cls in reversed(det):
                            if True:
                                c = int(cls)
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c,True))
                                # 仅用一个深度点的像素表示Z值
                                center_X,center_Y =(xyxy[0]+xyxy[2])/2 , (xyxy[1]+xyxy[3])/2
                                distDZ = depth_map[640,320]
                                center_X, center_Y = int(center_X), int(center_Y)
                                # 使用9个点的深度去除偏差取平均来表示深度 START
                                center_X1, center_Y1 = center_X-1,center_Y+1
                                center_X2, center_Y2 = center_X,center_Y+1
                                center_X3, center_Y3 = center_X+1,center_Y+1
                                center_X4, center_Y4 = center_X-1,center_Y
                                center_X5, center_Y5 = center_X+1,center_Y
                                center_X6, center_Y6 = center_X-1,center_Y-1
                                center_X7, center_Y7 = center_X,center_Y-1
                                center_X8, center_Y8 = center_X+1, center_Y-1
                                centerX = [center_X,center_X1,center_X2,center_X3,center_X4,center_X5,center_X6,center_X7,center_X8]
                                centerY = [center_Y,center_Y1,center_Y2,center_Y3,center_Y4,center_Y5,center_Y6,center_Y7,center_Y8]
                                mu_distDZ = []
                                for i in range (0,len(centerX)):
                                    distDZi = depth.get_value(centerX[i] , centerY[i])
                                    mu_distDZ.append(distDZi)
                                mu_distDZ.sort()
                                new_mu_distDZ = []
                                for i in range(0, len(mu_distDZ) - 3):
                                    new_mu_distDZ.append(mu_distDZ[i][1])
                                Average=sum(new_mu_distDZ)/len(new_mu_distDZ)
                                distDZ = Average
                                # 使用9个点的深度去除偏差取平均来表示深度 END

                                print("distDZ=  ", distDZ)
                                X_ap = (center_X-cx)*distDZ/fx
                                Y_ap = (center_Y-cy)*distDZ/fy

                                print("X_ap:******", X_ap)
                                print("Y_ap:******", Y_ap)
                                cv2.imshow("apple",im0)
                                cv2.waitKey(1)
                                # apple_XYZ.position.append(X_ap)
                                # apple_XYZ.position.append(Y_ap)
                                # apple_XYZ.position.append(distDZ)
                                # pub.publish(apple_XYZ)
                                # apple_XYZ.position.clear()

                key = cv2.waitKey(1)
                if key & 0xFF == ord('q') or key == 27:
                     cv2.destroyAllWindows()
                     break
                time.sleep(0.01)
    finally:
        # Stop streaming
        zed.close()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt



def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))

    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
