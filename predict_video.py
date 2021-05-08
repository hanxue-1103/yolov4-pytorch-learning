import time

import cv2
import numpy as np
from PIL import Image

from predict_image import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init():
    """
    构建验证模式下的模型。
    return 模型
    """

    model = Darknet("cfg/yolov4.cfg", 416)
    model.load_state_dict(torch.load("weights/yolov4-last-416.pth", map_location=device))
    model.to(device).eval()
    return model

"""
   capture=cv2.VideoCapture("test.mp4")
   capture=cv2.VideoCapture(0)
"""
capture = cv2.VideoCapture("data/samples/test.mp4")
fps = 0.0

model = init()

while(True):
    t1 = time.time()
    # 读取某一帧
    ref,frame=capture.read()

    frame = detect(frame, model)
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps= %.2f"%(fps))
    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("video",frame)

    c= cv2.waitKey(1) & 0xff 
    if c==27:
        capture.release()
        break
