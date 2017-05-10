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
    from cnnbase2.models2 import TinyAlexNet3
    from cnnbase2.models2 import TinyAlexNet4
    model_filename = 'run2-epoch580' # model6
    model_filename = 'learn-on-8000-before-on-5000-epoch960' #model5
    config = CnnDirsConfig()
    # model = Model5(config, '100examples', model_filename)
    model = TinyAlexNet4(config, 'flic.shuffle.code10', 'alex_code10')
    model.load_from_file()
    layers = model.model.layers
    # predicted = model.model.predict_classes(model.get_X_train_right_shape()[0:2,:,:,:])

    for index,layer in enumerate(layers):
        print "layer {} is {}".format(index, layer)

    print ""
    for index,layer in enumerate(layers):
        if isinstance(layer, Convolution2D):
            W = layer.W.get_value(borrow=True)
            print "layer {} has dimensions {}".format(index, W.shape)
    print ""

    img = model.X_train[0:1,:,:,:]
    # for index,layer in enumerate(layers):
    #     get_3rd_layer_output = K.function([layers[0].input, K.learning_phase()],
    #                           [layer.output])
    #     layer_output = get_3rd_layer_output([img, 0])[0]
    #     print "layer {} has layer_output {}".format(index, layer_output.shape)

    # W = layers[5].W.get_value(borrow=True)

    # print (predicted.shape)
    # print "W:"
    # print (W.shape)

    print ""
    print "LateX:"

    for index,layer in enumerate(layers):
        if isinstance(layer, Convolution2D):
            W = layer.W.get_value(borrow=True)
            kw, kh, _, filters = W.shape
            filters, _, kw, kh = W.shape
            print "\layerconvv{%d}{%d}{%d}" % (kw, kh, filters)
        elif isinstance(layer, Activation):
            print "\layeractivation"
        elif layer.__class__.__name__ == 'MaxPooling2D':
            pool_size = layer.pool_size
            strides = layer.strides
            px, py = pool_size
            sx, sy = strides
            print "\layerMaxPooling{%d}{%d}{%d}{%d}" % (px, py, sx, sy)
        elif layer.__class__.__name__ == 'BatchNormalization':
            print "\layerBatchNormalization"
        elif layer.__class__.__name__ == 'SpatialDropout2D':
            p = layer.p
            print "\layerSpatialDropout{%.1f}" % p
        else:
            print "\layer{%s}" % layer.__class__.__name__

        # get_3rd_layer_output = K.function([layers[0].input, K.learning_phase()],
        #                       [layer.output])
        # layer_output = get_3rd_layer_output([img, 0])[0]
        # _, w, h, ff = layer_output.shape
        _, ff, w, h = layer.output_shape
        print "\layerout{%d}{%d}{%d}" % (w, h, ff)

    # print "\n".join(dir(layers[5]))