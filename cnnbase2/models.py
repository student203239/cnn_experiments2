from docutils.nodes import subscript
from keras.engine import Layer

from cnnbase2.cnn_model_base import CnnModelDecorator
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import numpy as np

from cnnbase2.load_data import CnnDirsConfig


class Model1(CnnModelDecorator):

    def __init__(self, *args, **kwargs):
        kwargs['input_shape'] = (128, 128, 3)
        kwargs['output_shape'] = (29, 29)
        super(Model1, self).__init__(*args, **kwargs)

    def _new_model(self, input_shape):
        model = Sequential()
        model.add(Convolution2D(10, 5, 5,
                                border_mode='valid',
                                input_shape=input_shape,
                                ))
        model.add(Activation('relu'))
        model.add(Convolution2D(10, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Convolution2D(10, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Convolution2D(1, 1, 1))
        return model

class Model2(CnnModelDecorator):

    def __init__(self, *args, **kwargs):
        kwargs['input_shape'] = (128, 128, 3)
        kwargs['output_shape'] = (22, 22)
        super(Model2, self).__init__(*args, **kwargs)

    def _new_model(self, input_shape):
        model = Sequential()
        model.add(Convolution2D(20, 5, 5,
                                border_mode='valid',
                                input_shape=input_shape,
                                subsample=(2,2)))
        model.add(Convolution2D(25, 3, 3, subsample=(2,2)))
        model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Convolution2D(35, 3, 3))
        model.add(Convolution2D(35, 3, 3))
        model.add(Activation('relu'))
        model.add(Convolution2D(25, 3, 3))
        model.add(Convolution2D(15, 3, 3))
        model.add(Activation('relu'))

        model.add(Convolution2D(1, 1, 1))
        return model

class Model3(CnnModelDecorator):

    def __init__(self, *args, **kwargs):
        kwargs['input_shape'] = (128, 128, 3)
        kwargs['output_shape'] = (12, 12)
        super(Model3, self).__init__(*args, **kwargs)

    def _new_model(self, input_shape):
        model = Sequential()
        model.add(Convolution2D(15, 3, 3,
                                border_mode='valid',
                                input_shape=input_shape,
                                subsample=(1,1)))
        model.add(Convolution2D(15, 3, 3, subsample=(1,1)))
        model.add(Activation('relu'))
        model.add(Convolution2D(20, 3, 3))
        model.add(Convolution2D(20, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3,3)))

        model.add(Convolution2D(25, 3, 3))
        model.add(Convolution2D(25, 3, 3))
        model.add(Activation('relu'))
        model.add(Convolution2D(15, 3, 3))
        model.add(Convolution2D(15, 3, 3))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Convolution2D(15, 3, 3))
        model.add(Convolution2D(15, 3, 3))
        model.add(Activation('relu'))

        model.add(Convolution2D(1, 1, 1))
        return model

class Model4(CnnModelDecorator):

    def __init__(self, *args, **kwargs):
        kwargs['input_shape'] = (128, 128, 3)
        kwargs['output_shape'] = (28, 28)
        super(Model4, self).__init__(*args, **kwargs)

    def _new_model(self, input_shape):
        model = Sequential()
        model.add(Convolution2D(10, 5, 5,
                                border_mode='valid',
                                input_shape=input_shape,
                                ))
        model.add(Activation('relu'))
        model.add(Convolution2D(10, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Convolution2D(10, 3, 3))
        model.add(Activation('relu'))
        model.add(Convolution2D(10, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Convolution2D(1, 1, 1))
        return model

class Model5(CnnModelDecorator):

    def __init__(self, *args, **kwargs):
        kwargs['input_shape'] = (128, 128, 3)
        kwargs['output_shape'] = (24, 24)
        super(Model5, self).__init__(*args, **kwargs)

    def _new_model(self, input_shape):
        model = Sequential()
        model.add(Convolution2D(10, 5, 5,
                                border_mode='valid',
                                input_shape=input_shape,
                                ))
        model.add(Activation('relu'))
        model.add(Convolution2D(10, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Convolution2D(10, 3, 3))
        model.add(Activation('relu'))
        model.add(Convolution2D(10, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Convolution2D(10, 3, 3))
        model.add(Activation('relu'))
        model.add(Convolution2D(10, 3, 3))
        model.add(Activation('relu'))

        model.add(Convolution2D(1, 1, 1))
        return model

class Model6(CnnModelDecorator):

    def __init__(self, *args, **kwargs):
        kwargs['input_shape'] = (128, 128, 3)
        kwargs['output_shape'] = (11, 11)
        kwargs['batch_size'] = 128
        super(Model6, self).__init__(*args, **kwargs)

    def _new_model(self, input_shape):
        model = Sequential()
        model.add(Convolution2D(25, 5, 5,
                                border_mode='valid',
                                input_shape=input_shape,
                                subsample=(2,2)
                                ))
        model.add(Convolution2D(25, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Convolution2D(25, 5, 5, subsample=(2,2)))
        model.add(Convolution2D(25, 3, 3))
        model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2,2)))

        # model.add(Convolution2D(25, 5, 5))
        # model.add(Convolution2D(25, 3, 3))
        # model.add(Convolution2D(25, 3, 3))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Convolution2D(1, 1, 1))
        return model

    def hbb_box_to_y(self, src_y, output_shape, loader):
        w, h = output_shape
        examples = src_y.shape[0]
        y = np.zeros((examples,w,h,1), dtype='float32')
        for i in range(examples):
            y[i,:,:,0] = loader.create_heat_map_model6(src_y[i], w, h)
        return y

def copy_sequential(model):
    layers = model.layers
    new_model = Sequential()
    for layer in layers[:-1]:
        config = layer.get_config()
        name = config['name']
        if name.startswith('convolution2d'):
            conv_layer = Convolution2D.from_config(config)
            print layer.get_weights()
            conv_layer.set_weights(layer.get_weights())
            new_model.add(conv_layer)
        elif name.startswith('maxpooling2d'):
            new_model.add(MaxPooling2D.from_config(config))
        elif name.startswith('activation'):
            new_model.add(Activation.from_config(config))
    return new_model

if __name__ == '__main__':
    model_filename = 'test_model'
    config = CnnDirsConfig()
    model = Model6(config, '100examples', model_filename)
    # model.model = copy_sequential(model.model)
    model.model.layers.pop() # Get rid of the dropout layer
    model.model.layers.pop() # Get rid of the dropout layer
    model.model.layers.pop() # Get rid of the dropout layer
    model.model.outputs = [model.model.layers[-1].output]
    model.model.output_layers = [model.model.layers[-1]]
    model.model.layers[-1].outbound_nodes = []
    model.model.compile(loss='mean_squared_error',
                      optimizer='adadelta',
                      metrics=['accuracy'])
    predicted = model.model.predict_classes(model.X_train)
    print (predicted.shape)