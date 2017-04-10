import os

import scipy.io
import math
import os
from random import shuffle
import random

import numpy as np
import skimage.io as io
import skimage.transform as tr

from os import fsync, remove

from cnnbase2.cnn_model_base import CnnModelDecorator
from cnnbase2.img_utils import ImgUtlis
from cnnbase2.load_data import GaussianCalc, CnnDirsConfig, Binary, CnnDataLoader
from cnnbase2.masks.small_masks_experiments import SmallMaskGen


class FlicLoader(object):

    def __init__(self):
        self.flic_root = "C:\\Users\\Jacek\\Downloads\\flic\\FLIC"
        self.imgs_dir = os.path.join(self.flic_root, 'images')
        self.matlab = scipy.io.loadmat(os.path.join(self.flic_root, 'examples.mat'))

        gc = GaussianCalc(0, 0, 256, 256, 0.0)
        g = np.vectorize(gc.f)
        self.gaussion_buffer = np.fromfunction(g, (256, 256), dtype='float32')

    def main(self):
        self.build_examples_packet('flic.shuffle.code10', to_hbb_box=self.to_hbb_box_10)

    def main3(self):
        matlab = self.matlab
        l = len(matlab['examples'][0])
        index = 10
        coords = self.coords(index)
        min_x, max_x, min_y, max_y = min(coords[0][:]), max(coords[0][:]), min(coords[1][:]), max(coords[1][:])
        w, im, h = self._load_and_square_im(self.filepath(index), True)
        im = ImgUtlis.resize_rgb_image(im, 128, 128)
        print str(self.istrain(index)) is '1'
        print w, h
        print str(im)
        print str(im.shape)
        print self.torsobox(index)
        x1, y1, x2, y2 = self.torsobox(index)
        io.imshow(im)
        io.show()
        print im[:,:,0].shape
        io.imshow(self.create_y(w, h, *self.torsobox(index)))
        # io.show()
        x1, y1, x2, y2 = self.torsobox(index)
        loader = CnnDataLoader(CnnDirsConfig())
        hbb_box = x1, y1, x2, y2, w, h
        y_train = loader.hbb_box_to_y(np.asarray([hbb_box], 'float32'), (20, 20))[0]
        io.imshow(y_train[:,:,0])
        io.show()
        hbb_box = min_x, min_y, max_x, max_y, w, h
        y_train = loader.hbb_box_to_y(np.asarray([hbb_box], 'float32'), (30, 30))[0]
        y_train = tr.resize(y_train, (128, 128))
        expect_mul_by_img = y_train * im
        # io.imshow(y_train[:,:,0])
        io.imshow(expect_mul_by_img)
        io.show()

    def prepare_data(self, w=128, h=128, test_fac = 0.1, to_hbb_box=None):
        to_hbb_box = to_hbb_box or self.to_hbb_box_torso
        l = sum(self.istrain(i) for i in range(self.mat_len()))
        index_total = 0
        test_size = int(test_fac*l)
        train_size = l - test_size
        # test_size = 20
        # train_size = 200

        x_train = np.zeros((train_size, w, h, 3), dtype='float32')
        hbb_size = len(to_hbb_box(0, 100, 100))
        hbb_box_train = np.zeros((train_size, hbb_size), dtype='int16')
        x_test = np.zeros((test_size, w, h, 3), dtype='float32')
        hbb_box_test = np.zeros((test_size, hbb_size), dtype='int16')

        r = range(self.mat_len())
        # r = range(train_size + test_size)
        shuffle(r)
        for i in r:
            if str(self.istrain(i)) is '0':
                continue
            src_im_w, image, src_im_h = self._load_and_square_im(self.filepath(i), True)
            image = ImgUtlis.resize_rgb_image(image, w, h)
            # io.imshow(image)
            # io.show()
            hbb_box = to_hbb_box(i, src_im_w, src_im_h)

            if index_total >= train_size:
                index = index_total - train_size
                x_test[index,:,:,:] = image
                hbb_box_test[index,:] = hbb_box
            else:
                index = index_total
                x_train[index,:,:,:] = image
                hbb_box_train[index,:] = hbb_box

                # loader = CnnDataLoader(CnnDirsConfig())
                # one_box_array = hbb_box_train[index:index+1, :]
                # y_trains = loader.hbb_box_to_y(one_box_array, (20, 20))
                # y_train = y_trains[0]
                # io.imshow(y_train[:,:,0])
                # io.show()
            index_total += 1
            if index_total % 10 == 1:
                print "{} / {}".format(index_total, l)
        return x_train, hbb_box_train, x_test, hbb_box_test

    def to_hbb_box_torso(self, i, src_im_w, src_im_h):
        x1, y1, x2, y2 = self.torsobox(i)
        return

    def to_hbb_box_bound_box(self, i, src_im_w, src_im_h):
        x1, y1, x2, y2 = self.bound_box(i)
        return

    def to_hbb_box_10(self, i, src_im_w, src_im_h):
        return SmallMaskGen.code_10(self.torsobox(i), self.bound_box(i), src_im_w, src_im_h)

    def build_examples_packet(self, filename, to_hbb_box):
        config = CnnDirsConfig()
        bin = Binary()
        x_train, hbb_box_train, x_test, hbb_box_test = self.prepare_data(test_fac = 0.1, to_hbb_box=to_hbb_box)
        bin.save_pack(config.data_filename(filename), x_train, hbb_box_train, x_test, hbb_box_test)


    def create_y(self, w, h, x1, y1, x2, y2):
        result = np.zeros((h, w), dtype='float32')
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        to_paste = tr.resize(loader.gaussion_buffer, (y2 - y1, x2 - x1))
        result[y1:y2,x1:x2] = 1
        return result

    def create_y_ones(self, w, h, x1, y1, x2, y2):
        result = np.zeros((h, w), dtype='float32')
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        result[y1:y2,x1:x2] = np.ones((y2 - y1, x2 - x1), dtype='float32')
        return result

    def mat_len(self):
        return len(self.matlab['examples'][0])

    def filepath(self, i):
        filepath = self.matlab['examples'][0]['filepath'][i][0]
        return os.path.join(self.imgs_dir, filepath)

    def istrain(self, i):
        return self.matlab['examples'][0]['istrain'][i][0][0]

    def istest(self, i):
        return self.matlab['examples'][0]['istest'][i][0][0]

    def coords(self, i):
        return self.matlab['examples'][0]['coords'][i]

    def torsobox(self, i):
        t = self.matlab['examples'][0]['torsobox'][i][0]
        # return t[0], t[1], t[2], t[3]
        return int(t[0]), int(t[1]), int(t[2]), int(t[3])

    def remove_nans(self, a):
        return [x for x in a if not math.isnan(x)]

    def bound_box(self, i):
        coords = self.coords(i)
        coords_x  = self.remove_nans(coords[0])
        coords_y  = self.remove_nans(coords[1])
        min_x, max_x, min_y, max_y = min(coords_x), max(coords_x), min(coords_y), max(coords_y)
        return min_x, min_y, max_x, max_y

    def _load_and_square_im(self, img_filename, move_up):
        im = io.imread(img_filename)
        return ImgUtlis.make_img_square(im, move_up)


if __name__ == '__main__':
    loader = FlicLoader()
    loader.main()