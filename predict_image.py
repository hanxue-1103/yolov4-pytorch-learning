import torch
import torchvision.transforms as transforms

import cv2
import numpy as np
import json
# from PIL import Image, ImageDraw, ImageFont
# import matplotlib.pyplot as plt

from models.models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
half = device.type != 'cpu'
classes = ["ship"]

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    # new_unpad = new_shape
    # print("new_unpad:", new_unpad)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    # print("zheli:", nc)
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

def scale_coords_(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def read_img(path="dataset/input/0056.jpg", img_size=416, img=None):
    """
    读取单张图片。
    return 输入到网络的归一化后的图像张量。
    """

    # 图像resize
    if img is None:
        img = cv2.imread(path)  # BGR
    img0 = img
    h0, w0 = img.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 and not True else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)

    img, ratio, pad = letterbox(img, img_size, auto=False, scaleup=True)
    # cv2.imwrite("test2.jpg", img)

    # 图像转成torch
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    # print(img.shape)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = img.unsqueeze(0)
    return img, img0


def init():
    """
    构建验证模式下的模型。
    return 模型
    """

    model = Darknet("cfg/yolov4.cfg", 416)
    model.load_state_dict(torch.load("weights/yolov4-last-416.pth", map_location=device))
    model.to(device).eval()
    return model

def process(model, img, img0):
    """
    res = {
       "objects": [
           {
               "xmin": 1,
               "ymin": 2,
               "xmax": 3,
               "ymax": 4,
               "confidence": 0.8,
               "name": "ship"
           },
           {
               "xmin": 1,
               "ymin": 2,
               "xmax": 3,
               "ymax": 4,
               "confidence": 0.8,
               "name": "ship"
           },
       ]
    }
    """
    pred = model(img, augment=False)[0]
    print("hanxu:", pred.shape)
    pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.5, agnostic=False)  # 一个类别在一个里面

    objects = []
    for i, det in enumerate(pred):
        det[:, :4] = scale_coords_(img.shape[2:], det[:, :4], img0.shape).round()
        det = det.detach().cpu().numpy()
        for bbox in det:
            # objects.append(
            #     {
            #         "xmin": np.float(bbox[0]),
            #         "ymin": np.float(bbox[1]),
            #         "xmax": np.float(bbox[2]),
            #         "ymax": np.float(bbox[3]),
            #         "confidence": np.float(bbox[4]),
            #         "name": classes[int(bbox[-1])]
            #     }
            # )
            cv2.rectangle(img0, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
    
    # 保存画框之后的图片
    cv2.imwrite("1.jpg", img0)

    # 下面两行代码用于保存每张图片的边框信息，如有需要可以返回下面的res变量，但是需要将上面循环里面的备注解开。
    # res = {"objects": objects}
    # return json.dumps(res, indent=4)
    return img0

def detect(img, model):
    """
    视频检测时候调用的函数
    """
    img, img0 = read_img(path="dataset/input/0056.jpg", img_size=416, img=img)
    # model = init()
    res = process(model, img, img0)

    return res

if __name__ == "__main__":
    """
    letterbox() 图片resize后空白区域填充函数
    non_max_suppression() nms
    scale_coords_() 图片尺寸还原

    read_img() 读取单张图片，返回归一化后的张量 和 原图
    init() 初始化模型
    process() 主运行函数。输入参数为model，img 返回每张图得到的bbox结果 json格式
    """
    img, img0 = read_img(path="data/samples/0056.jpg")
    model = init()
    res = process(model, img, img0)
