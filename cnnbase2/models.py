from cnnbase2.cnn_model_base import CnnModelDecorator
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

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

if __name__ == '__main__':
    model_filename = 'test_model'
    config = CnnDirsConfig()
    model = Model1(config, '100examples', model_filename)
    predicted = model.model.predict_classes(model.X_train)
    print (predicted.shape)