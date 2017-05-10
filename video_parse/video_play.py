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
cap.release()
print "Done reading video"
config = CnnDirsConfig()
model = TinyAlexNet4(config, None, 'alex_code10.e60.2017-04-08--14-09-55')
model.load_from_file()
predicted = model.predict(model_input, verbose=1, batch_size=110)
# print predicted.shape (322L, 1L, 14L, 14L)
frame_time = 1.0 / fps
# kth = 160
print "Save video"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_resolution = (600, 600)
out = cv2.VideoWriter('output.avi',fourcc, fps, out_resolution)
cap = cv2.VideoCapture(filename)
for i in range(frame_index):
    start_time = time.time()
    y = predicted[i,0,:,:]
    cell_size = float(out_resolution[0]) / y.shape[0]
    y[y<0.1] = 0
    ret, src_frame = cap.read()
    h, src_frame_sq, w = ImgUtlis.make_img_square(src_frame, move_up=True)
    src_frame_sq = ImgUtlis.resize_rgb_image(src_frame_sq, *out_resolution)
    # img_mul = model.multiply_rgb_img_by_gray_img(y, src_frame_sq, advanced_resize=True)
    img_mul = src_frame_sq
    for xi in range(y.shape[0]):
        for yi in range(y.shape[1]):
            if y[yi, xi] > 0:
                p1 = (int(xi * cell_size), int(yi * cell_size))
                p2 = (int((xi + 1) * cell_size), int((yi + 1) * cell_size))
                cv2.rectangle(img_mul, p1, p2, (0, 1, 0), 2)
                txt = ("%.1f" % y[yi, xi])[1:]
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_mul,txt,(p1[0],p2[1]-5), font, 1,(1,0,0),2,cv2.LINE_AA)

    cv2.imshow('frame', img_mul)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    out.write(img_as_ubyte(img_mul))
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

def make_hist():
    pass
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

def remove_smallest_vals():
    pass
    # y_smallest_indx = np.argpartition(y, kth, axis=None)
    # x_i, y_i = np.unravel_index(y_smallest_indx[:kth], y.shape)
    # y[x_i, y_i] = 0