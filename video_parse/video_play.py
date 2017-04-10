import os

import numpy as np
import cv2
import time

from cnnbase2.img_utils import ImgUtlis
from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models2 import TinyAlexNet4

filename = "D:\mgr_dir1\\video\MOV_0529.mp4"
print os.path.exists(filename)
cap = cv2.VideoCapture(filename)

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
print length, fps, width, height

model_input = np.zeros((length, 128, 128, 3), dtype='float32')
frame_index = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    # print frame.shape
    # print frame.dtype
    h, frame_sq, w = ImgUtlis.make_img_square(frame, move_up=True)
    f2 = ImgUtlis.resize_rgb_image(frame_sq, 128, 128)
    model_input[frame_index,:,:,:] = f2
    frame_index += 1

    # cv2.imshow('frame', f2)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
print "Done reading video"
config = CnnDirsConfig()
model = TinyAlexNet4(config, None, 'alex_code10')
predicted = model.predict(model_input, verbose=1, batch_size=110)
# print predicted.shape (322L, 1L, 14L, 14L)
frame_time = 1.0 / fps
for i in range(frame_index):
    y = predicted[i,0,:,:]
    img_mul = model.multiply_rgb_img_by_gray_img(y, model_input[i,:,:,:])
    # img = np.kron(img, np.ones((10, 10)))
    cv2.imshow('frame', img_mul)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(frame_time)

cap.release()
cv2.destroyAllWindows()