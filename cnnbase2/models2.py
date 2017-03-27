import h5py
from docutils.nodes import subscript
from keras.engine import Layer

from cnnbase2.cnn_model_base import CnnModelDecorator
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
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
        kwargs['input_shape'] = (128, 128, 3)
        kwargs['output_shape'] = (11, 11)
        kwargs['batch_size'] = 128
        super(TinyAlexNet, self).__init__(*args, **kwargs)

    def _new_model(self, input_shape):
        weights_path = config.data_filename('alexnet_weights.h5')
        file = h5py.File(weights_path, 'r')
        model = Sequential()
        conv_1 = Convolution2D(96, 11, 11, subsample=(4, 4), activation='relu', name='conv_1', input_shape=(3,128,128), dim_ordering='th')
        inputs = Input(shape=(3,128,128))

        # conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
        #                    name='conv_1')(inputs)

        model.add(conv_1)
        # model = Model(input=inputs, output=conv_1)
        # model.load_weights_from_hdf5_group_by_name(file)
        return model

if __name__ == '__main__':
    config = CnnDirsConfig()
    model = TinyAlexNet(config, 'flic.valid.07', 'first-alex')

    # m2 = m5_gauss = Model5(config, 'flic.valid.07', 'flic2')
    models = {'z': model}
    viewer = ModelsViewer(models, load_models=False)
    viewer.init_my_canvas()