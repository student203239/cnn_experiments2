import hashlib

import h5py
from docutils.nodes import subscript
from keras.engine import Layer

from cnnbase2.cnn_model_base import CnnModelDecorator
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, SpatialDropout2D
from keras.utils import np_utils
from keras import backend as K
import numpy as np

from cnnbase2.load_data import CnnDirsConfig, CnnDataLoader
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge

from cnnbase2.masks.small_masks_experiments import SmallMaskGen


class TinyAlexNet(CnnModelDecorator):

    def __init__(self, *args, **kwargs):
        # kwargs['input_shape'] = (128, 128, 3)
        # kwargs['output_shape'] = (11, 11)
        # kwargs['batch_size'] = 128
        super(TinyAlexNet, self).__init__(*args, **kwargs)

    def set_default_filename(self, default_filename):
        self.y_gen_mode = None
        if default_filename.startswith("mayc10"):
            if default_filename.startswith("mayc10r"):
                self.y_gen_mode = 'r'  # regular from 1s inside to gradient out
            elif default_filename.startswith("mayc10i"):
                self.y_gen_mode = 'i'  # only inner ones
            elif default_filename.startswith("mayc10o"):
                self.y_gen_mode = 'o'  # only outter ones
        self.default_filename = default_filename
        if self._prepare_train_data_pack_to_recreate_y[0] != None:
            hbb_box_train, hbb_box_test, output_shape = self._prepare_train_data_pack_to_recreate_y
            self.y_train, self.y_test = self._create_y_train_y_test(hbb_box_train, hbb_box_test, output_shape)
            # print "y_train, y_test = {}, {}".format(self._numpy_sha1(self.y_train), self._numpy_sha1(self.y_test))

    def _numpy_sha1(self, array):
        return hashlib.sha1(array.view(np.uint8)).hexdigest()

    def _new_model(self, input_shape):
        config = CnnDirsConfig()
        weights_path = config.data_filename('alexnet_weights.h5')
        file = h5py.File(weights_path, 'r')
        model = Sequential()
        subsample = self._get_conv1_sub_sample()
        conv_1 = Convolution2D(96, 11, 11, subsample=subsample, activation='relu', name='conv_1', input_shape=(3, 128, 128), dim_ordering='th')
        inputs = Input(shape=(3,128,128))

        # conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
        #                    name='conv_1')(inputs)

        model.add(conv_1)
        # model = Model(input=inputs, output=conv_1)
        model.load_weights_from_hdf5_group_by_name(file)
        self._add_more_layers_to_model(model)
        return model

    def _get_conv1_sub_sample(self):
        return (4, 4)

    def _add_more_layers_to_model(self, model):
        pass

    def learn_now(self, nb_epoch=12, initial_epoch=1):
        history = self.model.fit(self.X_train.transpose((0,3,1,2)), self.y_train.transpose((0,3,1,2)), batch_size=self.batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(self.X_test.transpose((0,3,1,2)), self.y_test.transpose((0,3,1,2))), shuffle=True)
        self.save_history(history)
        return history

    def get_X_train_right_shape(self):
        return self.X_train.transpose((0,3,1,2))

    def evaluate_alex_model(self):
        return self.model.evaluate(self.X_test.transpose((0,3,1,2)), self.y_test.transpose((0,3,1,2)), 12, verbose=1)

    def predict(self, x, batch_size=32, verbose=0):
        return self.model.predict(x.transpose((0,3,1,2)), batch_size=batch_size, verbose=verbose)

    def _hbb_box_to_y(self, data, output_shape, output_y=None):
        if self.y_gen_mode is None:
            return super(TinyAlexNet, self)._hbb_box_to_y(data, output_shape)
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

    def get_predicted_test(self):
        if self.saved_predicted_test is None:
            self.saved_predicted_test = self.model.predict(self.X_test.transpose((0,3,1,2)), batch_size=32)
            # self.saved_predicted_test = self.model.predict(self.X_test, batch_size=32)
        return self.saved_predicted_test

class ConfigurableYDataLoader(CnnDataLoader):

    def __init__(self, *args, **kwargs):
        super(ConfigurableYDataLoader, self).__init__(*args, **kwargs)

    def get_heat_map_loc(self, hbb_box, return_w, return_h):
        inner, outter, w, h = SmallMaskGen.from_code_10(hbb_box)
        x1, y1, x2, y2, w, h = hbb_box
        rr = min(1.0*return_h/h, 1.0*return_w/w)
        y1 = int(y1*rr)
        x1 = int(x1*rr)
        x2 = int(x2*rr+0.5)
        y2 = int(y2*rr+0.5)
        return x1, y1, x2, y2

class TinyAlexNet1(TinyAlexNet):

    def __init__(self, *args, **kwargs):
        kwargs['input_shape'] = (128, 128, 3)
        kwargs['output_shape'] = (11, 11)
        kwargs['batch_size'] = 128
        super(TinyAlexNet1, self).__init__(*args, **kwargs)

    def _add_more_layers_to_model(self, model):
        model.add(Convolution2D(200, 3, 3))
        model.add(Activation('relu'))
        model.add(SpatialDropout2D(0.5, dim_ordering='th'))
        model.add(Convolution2D(250, 3, 3))
        model.add(Activation('relu'))
        model.add(SpatialDropout2D(0.5, dim_ordering='th'))
        model.add(Convolution2D(150, 3, 3))
        model.add(Activation('relu'))
        model.add(Convolution2D(50, 3, 3))
        model.add(SpatialDropout2D(0.5, dim_ordering='th'))
        model.add(Convolution2D(1, 3, 3))

class TinyAlexNet2(TinyAlexNet):

    def __init__(self, *args, **kwargs):
        kwargs['input_shape'] = (128, 128, 3)
        kwargs['output_shape'] = (22, 22)
        kwargs['batch_size'] = 128
        super(TinyAlexNet2, self).__init__(*args, **kwargs)

    def _add_more_layers_to_model(self, model):
        model.add(SpatialDropout2D(0.5, dim_ordering='th'))
        model.add(Convolution2D(120, 3, 3, dim_ordering='th'))
        model.add(Convolution2D(120, 3, 3, dim_ordering='th'))
        model.add(Activation('relu'))
        model.add(SpatialDropout2D(0.5, dim_ordering='th'))
        model.add(Convolution2D(30, 3, 3, dim_ordering='th'))
        model.add(Activation('relu'))
        model.add(Convolution2D(1, 3, 3, dim_ordering='th'))
        pass


class TinyAlexNet3(TinyAlexNet):

    def __init__(self, *args, **kwargs):
        kwargs['input_shape'] = (128, 128, 3)
        kwargs['output_shape'] = (14, 14)
        kwargs['batch_size'] = 64
        super(TinyAlexNet3, self).__init__(*args, **kwargs)

    def _add_more_layers_to_model(self, model):
        model.add(MaxPooling2D((3, 3), strides=(2,2), dim_ordering='th'))
        model.add(Convolution2D(256, 5, 5, dim_ordering='th', border_mode='same', activation='relu'))
        # model.add(Convolution2D(256, 3, 3, dim_ordering='th'))
        # model.add(SpatialDropout2D(0.5, dim_ordering='th'))
        model.add(Convolution2D(356, 3, 3, dim_ordering='th', border_mode='same', activation='relu'))
        model.add(BatchNormalization(axis=1))
        model.add(SpatialDropout2D(0.7, dim_ordering='th'))
        model.add(MaxPooling2D((2, 2), dim_ordering='th'))
        model.add(Convolution2D(200, 5, 5, dim_ordering='th', border_mode='same', activation='relu'))
        model.add(BatchNormalization(axis=1))
            # model.add(Convolution2D(130, 3, 3, dim_ordering='th'))
        model.add(SpatialDropout2D(0.5, dim_ordering='th'))
        model.add(Convolution2D(1, 1, 1, dim_ordering='th', activation='relu'))

    def _get_conv1_sub_sample(self):
        return (2, 2)

class TinyAlexNet4(TinyAlexNet):

    def __init__(self, *args, **kwargs):
        kwargs['input_shape'] = (128, 128, 3)
        kwargs['output_shape'] = (14, 14)
        kwargs['batch_size'] = 64
        if 'output_layers' in kwargs:
            self.output_layers = kwargs['output_layers']
            del kwargs['output_layers']
        else:
            self.output_layers = 1
        super(TinyAlexNet4, self).__init__(*args, **kwargs)

    def _add_more_layers_to_model(self, model):
        model.add(MaxPooling2D((3, 3), strides=(2,2), dim_ordering='th'))
        model.add(Convolution2D(256, 5, 5, dim_ordering='th', border_mode='same', activation='relu'))
        # model.add(Convolution2D(256, 3, 3, dim_ordering='th'))
        # model.add(SpatialDropout2D(0.5, dim_ordering='th'))
        model.add(Convolution2D(356, 3, 3, dim_ordering='th', border_mode='same', activation='relu'))
        model.add(SpatialDropout2D(0.7, dim_ordering='th'))
        model.add(BatchNormalization(axis=1))
        model.add(MaxPooling2D((2, 2), dim_ordering='th'))
        model.add(Convolution2D(200, 5, 5, dim_ordering='th', border_mode='same', activation='relu'))
        model.add(SpatialDropout2D(0.5, dim_ordering='th'))
        model.add(BatchNormalization(axis=1))
            # model.add(Convolution2D(130, 3, 3, dim_ordering='th'))
        model.add(Convolution2D(self.output_layers, 1, 1, dim_ordering='th', activation='relu'))

    def _get_conv1_sub_sample(self):
        return (2, 2)


class TinyAlexNet4Double(TinyAlexNet4):

    def __init__(self, *args, **kwargs):
        kwargs['output_layers'] = 2
        super(TinyAlexNet4Double, self).__init__(*args, **kwargs)

class TinyAlexNet5(TinyAlexNet):

    def __init__(self, *args, **kwargs):
        kwargs['input_shape'] = (128, 128, 3)
        kwargs['output_shape'] = (14, 14)
        kwargs['batch_size'] = 64
        if 'output_layers' in kwargs:
            self.output_layers = kwargs['output_layers']
            del kwargs['output_layers']
        else:
            self.output_layers = 2
        if 'use_sigmoid' in kwargs:
            self.use_sigmoid = kwargs['use_sigmoid']
            del kwargs['use_sigmoid']
        else:
            self.use_sigmoid = True
        super(TinyAlexNet5, self).__init__(*args, **kwargs)

    def _add_more_layers_to_model(self, model):
        trainable = False
        model.add(MaxPooling2D((3, 3), strides=(2,2), dim_ordering='th',
                               name='experiment4_layer_1_MaxPooling2D', trainable=trainable))
        model.add(Convolution2D(256, 5, 5, dim_ordering='th', border_mode='same', activation='relu',
                                name='experiment4_layer_2_Convolution2D', trainable=trainable))
        # model.add(Convolution2D(256, 3, 3, dim_ordering='th'))
        # model.add(SpatialDropout2D(0.5, dim_ordering='th'))
        model.add(Convolution2D(356, 3, 3, dim_ordering='th', border_mode='same', activation='relu',
                                name='experiment4_layer_3_Convolution2D', trainable=trainable))
        model.add(SpatialDropout2D(0.7, dim_ordering='th',
                                   name='experiment4_layer_4_SpatialDropout2D', trainable=trainable))
        model.add(BatchNormalization(axis=1, name='experiment4_layer_5_BatchNormalization', trainable=trainable))
        model.add(MaxPooling2D((2, 2), dim_ordering='th',
                               name='experiment4_layer_6_MaxPooling2D', trainable=trainable))
        model.add(Convolution2D(200, 5, 5, dim_ordering='th', border_mode='same', activation='relu',
                                name='experiment4_layer_7_Convolution2D', trainable=trainable))
        model.add(SpatialDropout2D(0.5, dim_ordering='th',
                                   name='experiment4_layer_8_SpatialDropout2D', trainable=trainable))
        model.add(BatchNormalization(axis=1, name='experiment4_layer_9_BatchNormalization', trainable=trainable))
            # model.add(Convolution2D(130, 3, 3, dim_ordering='th'))
        last_layer_activation = 'sigmoid'
        if not self.use_sigmoid:
            last_layer_activation = 'tanh'
        model.add(Convolution2D(self.output_layers, 1, 1, dim_ordering='th', activation=last_layer_activation,
                                name='experiment4_layer_10_Convolution2D', trainable=True))

    def _get_conv1_sub_sample(self):
        return (2, 2)

if __name__ == '__main__':
    config = CnnDirsConfig()
    # model = TinyAlexNet3(config, 'flic.valid.07', 'second-alex')
    # print model.get_predicted_test().shape

    # m2 = m5_gauss = Model5(config, 'flic.valid.07', 'flic2')
    # models = {'z': model}
    # viewer = ModelsViewer(models, load_models=False)
    # viewer.init_my_canvas()
    from cnnbase2.data_feeders.data_feeder_cnn_model_like import DataFeederCnnModelBaseLike
    from cnnbase2.data_feeders.dumy_data_feeders_merger import DummyFeedersMerge

    load_train = False
    cars_feeder = DataFeederCnnModelBaseLike(config, '5000examples', load_train=load_train).init_car_type(smaller_car=True)
    human_feeder = DataFeederCnnModelBaseLike(config, 'flic.shuffle.code10', load_train=load_train).init_human_type(y_gen_mode='r')

    merger_feeder = DummyFeedersMerge(cars_feeder, human_feeder, load_train=load_train)
    model = TinyAlexNet5(config, prepared_data=merger_feeder)
    model.load_from_file("june15.experiment4.e40.2017-06-16--02-57-33.TinyAlexNet5.model")