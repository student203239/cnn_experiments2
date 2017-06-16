import datetime
import os

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import skimage.transform as tr
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float

from cnnbase2.load_data import Binary, CnnDataLoader
from cnnbase2.masks.small_masks_experiments import SmallMaskGen


class DataFeederCnnModelBaseLike(object):

    def __init__(self, config, data_file, load_train=True):
        self.config = config
        self.data_file = data_file
        self.smaller_car = None
        self.is_car_type = None
        self.y_gen_mode = None
        self.load_train = load_train
        self.inited = False

    def init_car_type(self, smaller_car):
        self.smaller_car = smaller_car
        self.is_car_type = True
        self.inited = True
        return self

    def init_human_type(self, y_gen_mode):
        self.y_gen_mode = y_gen_mode
        self.is_car_type = False
        self.inited = True
        return self

    def prepare_data(self, output_shape):
        if not self.inited:
            raise Exception("init type of detection before prepare data")
        return self._prepare_train_data(self.data_file, output_shape)

    def _prepare_train_data(self, data_file, output_shape):
        if self.load_train:
            x_train, hbb_box_train, x_test, hbb_box_test = Binary().load_pack(self.config.data_filename(data_file))
        else:
            x_train, hbb_box_train = None, None
            x_test, hbb_box_test = Binary().load_pack_only_test(self.config.data_filename(data_file))
        X_train = x_train
        X_test = x_test
        y_train, y_test = None, None
        y_train, y_test = self._create_y_train_y_test(hbb_box_train, hbb_box_test, output_shape, y_train, y_test)
        return X_train, X_test, y_train, y_test

    def _create_y_train_y_test(self, hbb_box_train, hbb_box_test, output_shape, y_train=None, y_test=None):
        if self.load_train:
            y_train = self._hbb_box_to_y(hbb_box_train, output_shape, y_train)
        y_test = self._hbb_box_to_y(hbb_box_test, output_shape, y_test)
        return y_train, y_test

    def _hbb_box_to_y(self, data, output_shape, output_y_not_use_here=None):
        if self.is_car_type:
            return self._hbb_box_to_y_cnn_base_like(data, output_shape, output_y_not_use_here)
        else:
            return self._hbb_box_to_y_tiny_alex_like(data, output_shape, output_y_not_use_here)

    def _hbb_box_to_y_cnn_base_like(self, data, output_shape, output_y_not_use_here=None):
        shape = data.shape
        if shape[1] == 6:
            loader = CnnDataLoader(self.config)
            return loader.hbb_box_to_y_car(data, output_shape, self.smaller_car)
        if shape[1] == 10:
            return SmallMaskGen.hbb_box_to_y(data, output_shape)

    def _hbb_box_to_y_tiny_alex_like(self, data, output_shape, output_y=None):
        if self.y_gen_mode is None:
            return self._hbb_box_to_y_cnn_base_like(data, output_shape)
        shape = data.shape
        if shape[1] != 10:
            raise Exception("I accept only code10 coding, you give: " + str(shape))
        if self.y_gen_mode == 'r':
            return SmallMaskGen.hbb_box_to_y(data, output_shape, output_y=output_y)
        elif self.y_gen_mode == 'i':
            return SmallMaskGen.hbb_box_to_y_only_inner(data, output_shape, output_y=output_y)
        elif self.y_gen_mode == 'o':
            return SmallMaskGen.hbb_box_to_y_only_outter(data, output_shape, output_y=output_y)
        else:
            raise Exception("unknown y_gen_mode: " + str(self.y_gen_mode))