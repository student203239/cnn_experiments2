import cv2

from cnnbase2.img_utils import ImgUtlis
import numpy as np

from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models2 import TinyAlexNet4, TinyAlexNet4Double


def mul_img_by_y(src_frame_sq, y, out_resolution=(640, 480)):
    cell_size = float(out_resolution[0]) / y.shape[0]
    y[y<0.1] = 0
    # h, src_frame_sq, w = ImgUtlis.make_img_square(src_frame, move_up=True)
    # src_frame_sq = ImgUtlis.resize_rgb_image(src_frame_sq, *out_resolution)
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
    return img_mul

cap = cv2.VideoCapture(0)

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
print length, fps, width, height  # -1 0.0 640 480

length = 1
model_input = np.zeros((length, 128, 128, 3), dtype='float32')

config = CnnDirsConfig()
model = TinyAlexNet4(config, None, 'alex_code10.e135.2017-04-09--07-16-27')
# model = TinyAlexNet4Double(config, None, 'june15.experiment4.e40.2017-06-16--02-57-33')
model.load_from_file()

while True:
    ret_val, img = cap.read()

    if ret_val:
        h, frame_sq, w = ImgUtlis.make_img_square(img, move_up=True)
        frame_sq_copy = frame_sq[:,:,:]
        f2 = ImgUtlis.resize_rgb_image(frame_sq, 128, 128)
        frame_index = 0
        model_input[frame_index,:,:,:] = f2

        predicted = model.predict(model_input, verbose=1, batch_size=110)
        predicted_img = predicted[0,0,:,:]

        # ii = mul_img_by_y(frame_sq_copy, predicted_img, frame_sq_copy.shape[:-1])
        # ii = mul_img_by_y(frame_sq_copy, predicted_img, frame_sq_copy.shape[:-1])
        # cv2.imshow('webcam', frame_sq_copy)
        ii = mul_img_by_y(frame_sq_copy, predicted_img, frame_sq_copy.shape[:-1])
        cv2.imshow('Medical.ml image processing by Jacek Skoczylas', ii)


    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()