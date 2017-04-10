import os

import numpy as np
import cv2
import time
import io
from PIL import Image
import matplotlib.pyplot as plt

from skimage import img_as_ubyte

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
model.load_from_file()
predicted = model.predict(model_input, verbose=1, batch_size=110)
# print predicted.shape (322L, 1L, 14L, 14L)
frame_time = 1.0 / fps
# kth = 160
print "Save video"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, fps, (128,128))
for i in range(frame_index):
    start_time = time.time()
    y = predicted[i,0,:,:]
    y[y<=0.25] = 0
    # y_smallest_indx = np.argpartition(y, kth, axis=None)
    # x_i, y_i = np.unravel_index(y_smallest_indx[:kth], y.shape)
    # y[x_i, y_i] = 0
    img_mul = model.multiply_rgb_img_by_gray_img(y, model_input[i,:,:,:], advanced_resize=True)
    # img = np.kron(img, np.ones((10, 10)))

    # plt.figure()
    # plt.hist(y)
    # plt.title("test")
    # buf = io.BytesIO()
    # plt.savefig(buf, format='raw')
    # buf.seek(0)
    # img_np = cv2.imdecode(np.fromstring(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
    # buf.close()
    # plt.clf()
    # plt.cla()
    # img_mul = img_np

    cv2.imshow('frame', img_mul)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    out.write(img_as_ubyte(img_mul))
    print "done ", i
    # to_sleep = frame_time - (time.time() - start_time)
    # if to_sleep > 0:
    #     time.sleep(to_sleep)


cap.release()
out.release()
cv2.destroyAllWindows()

# plt.figure()
# plt.plot([1, 2])
# plt.title("test")
# buf = io.BytesIO()
# plt.savefig(buf, format='png')
# buf.seek(0)
# im = Image.open(buf)
# im.show()
# buf.close()