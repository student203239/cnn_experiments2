import os

import numpy as np
import cv2

from cnnbase2.img_utils import ImgUtlis

filename = "D:\mgr_dir1\\video\MOV_0529.mp4"
print os.path.exists(filename)
cap = cv2.VideoCapture(filename)

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    # print frame.shape
    # print frame.dtype
    h, frame_sq, w = ImgUtlis.make_img_square(frame, move_up=True)
    f2 = ImgUtlis.resize_rgb_image(frame_sq, 128, 128)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', f2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()