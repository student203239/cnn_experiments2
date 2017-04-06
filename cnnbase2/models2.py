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

from cnnbase2.load_data import CnnDirsConfig
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge

from cnnbase2.models import Model5
from cnnbase2.models_viewer import ModelsViewer


class TinyAlexNet(CnnModelDecorator):

    def __init__(self, *args, **kwargs):
        # kwargs['input_shape'] = (128, 128, 3)
        # kwargs['output_shape'] = (11, 11)
        # kwargs['batch_size'] = 128
        super(TinyAlexNet, self).__init__(*args, **kwargs)

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
          verbose=1, validation_data=(self.X_test.transpose((0,3,1,2)), self.y_test.transpose((0,3,1,2))))
        self.save_history(history)
        return history

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

if __name__ == '__main__':
    config = CnnDirsConfig()
    model = TinyAlexNet3(config, 'flic.valid.07', 'second-alex')
    print model.get_predicted_test().shape

    # m2 = m5_gauss = Model5(config, 'flic.valid.07', 'flic2')
    models = {'z': model}
    # viewer = ModelsViewer(models, load_models=False)
    # viewer.init_my_canvas()