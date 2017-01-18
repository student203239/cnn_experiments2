
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


class CnnModelDecorator(object):

    def __init__(self, config, data_file, default_filename='first_cnn_model.model', input_shape=None, output_shape=None, batch_size=128):
        self.default_filename = default_filename
        self.config = config
        self.batch_size = batch_size
        self._prepare_train_data(data_file, output_shape)
        self._create_model(input_shape)

    def _new_model(self, input_shape):
        raise Exception("Please impl this method is child class")

    def _create_model(self, input_shape):
        model = self._new_model(input_shape)
        model.compile(loss='mean_squared_error',
                      optimizer='adadelta',
                      metrics=['accuracy'])
        self.model = model

    def learn_now(self, nb_epoch=12, initial_epoch=0):
        history = self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(self.X_test, self.y_test))
        self.save_history(history)
        return history

    def save_history(self, history):
        filename = self.get_model_filename()
        filename = self.config.model_history_filename(filename)
        columns = ['acc', 'loss', 'val_acc', 'val_loss']
        # hh = history.history
        # history.epoch
        with open(filename + '.txt', 'w+') as f:
            f.write("SRC:\n")
            f.write(str(history.history) + '\n')
            f.write(str(history.epoch) + '\n')
        with open(filename + '.csv', 'w+') as f:
            cols_vals = [history.epoch]
            for c in columns:
                cols_vals.append(history.history[c])
            f.write(";".join(['epoch'] + columns) + '\n')
            for p in zip(*cols_vals):
                f.write(";".join([str(x) for x in p]) + '\n')

    def _prepare_train_data(self, data_file, output_shape):
        x_train, hbb_box_train, x_test, hbb_box_test = Binary().load_pack(self.config.data_filename(data_file))
        self.X_train = x_train
        self.X_test = x_test
        loader = CnnDataLoader(self.config)
        self.y_train = self.hbb_box_to_y(hbb_box_train, output_shape, loader)
        self.y_test = self.hbb_box_to_y(hbb_box_test, output_shape, loader)

    def hbb_box_to_y(self, src_y, output_shape, loader):
        w, h = output_shape
        examples = src_y.shape[0]
        y = np.zeros((examples,w,h,1), dtype='float32')
        for i in range(examples):
            if i == 10:
                print 'ok'
            x1, y1, x2, y2 = loader.get_heat_map_loc(src_y[i], w, h)
            # y[i,y1:y2,x1:x2,0] = loader.get_heat_map_to_paste(x1, y1, x2, y2)
            if y2 > h:
                y2 = h
            if x2 > w:
                x2 = w
            try:
                to_paste = tr.resize(loader.gaussion_buffer, (y2 - y1, x2 - x1))
            except:
                raise
            y[i,y1:y2,x1:x2,0] = to_paste
        return y

    def load_from_file(self, filename=None):
        filename = self.get_model_filename(filename)
        self.model.load_weights(self.config.model_filename(filename))

    def save_model_to_file(self, filename=None):
        filename = self.get_model_filename(filename)
        self.model.save_weights(self.config.model_filename(filename), overwrite=True)

    def get_model_filename(self, filename=None):
        filename = filename or self.default_filename
        if not filename.endswith("." + self.__class__.__name__ + ".model"):
            filename += '.'
            filename += self.__class__.__name__
            filename += '.model'
        return filename

    def show_results(self, interactive=False):
        predicted_test = self.model.predict(self.X_test)

        if interactive:
            index = 0
            io.imshow(self.X_test[index,:,:,:])
            io.show()
            io.imshow(predicted_test[index,:,:,0])
            io.show()

        for index in range(self.X_test.shape[0]):
            x_img = self.X_test[index,:,:,:]
            x_img = exposure.rescale_intensity(x_img, out_range='float')
            x_img = img_as_uint(x_img)

            io.imsave(self.config.model_results_filename('imgs/%d_x_test.png' % index), x_img)

            n = 3
            x_img = np.kron(x_img, np.ones((n,n,1)))
            x_img = exposure.rescale_intensity(x_img, out_range='float')
            x_img = img_as_uint(x_img)
            print 'shape'
            print x_img.shape
            io.imsave(self.config.model_results_filename('imgs_big/%d_x_test_big.png' % index), x_img)

            predicted_img = predicted_test[index,:,:,0]
            predicted_img = exposure.rescale_intensity(predicted_img, out_range='float')
            predicted_img = img_as_uint(predicted_img)
            io.imsave(self.config.model_results_filename('imgs/%d_y_predicted.png' % index), predicted_img)
            self._save_scaled_gray_img(predicted_img, 'imgs_big/%d_y_predicted_big.png' % index, 8)

    def show_more_results(self, interactive=False, prefix=''):
        predicted_test = self.model.predict(self.X_test)

        if interactive:
            index = 0
            io.imshow(self.X_test[index,:,:,:])
            io.show()
            io.imshow(predicted_test[index,:,:,0])
            io.show()

        for index in range(self.X_test.shape[0]):
            x_img = self.X_test[index,:,:,:]
            x_img = exposure.rescale_intensity(x_img, out_range='float')
            self._save_img('imgs/%d_x_test.png' % index, x_img)

            n = 3
            x_img = np.kron(x_img, np.ones((n,n,1)))
            x_img = exposure.rescale_intensity(x_img, out_range='float')
            x_img = img_as_uint(x_img)
            print 'shape'
            print x_img.shape
            io.imsave(self.config.model_results_filename('%simgs_big/%d_x_test_big.png' % (prefix, index)), x_img)

            predicted_img = predicted_test[index,:,:,0]
            predicted_img = exposure.rescale_intensity(predicted_img, out_range='float')
            predicted_img = img_as_uint(predicted_img)
            io.imsave(self.config.model_results_filename('%simgs/%d_y_predicted.png' % (prefix, index)), predicted_img)
            predicted_img = self._save_scaled_gray_img(predicted_img, '%simgs_big/%d_y_predicted_big.png' % (prefix, index), 16)

            x_img2 = x_img.copy()
            x_img2[:,:,0] = x_img[:,:,0] * predicted_img
            x_img2[:,:,1] = x_img[:,:,1] * predicted_img
            x_img2[:,:,2] = x_img[:,:,2] * predicted_img
            self._save_img('%simgs_big/%d_x_by_predicted.png' % (prefix, index), x_img2)

            expected_img = self.y_test[index,:,:,:]
            print "expected_img.shape"
            print expected_img.shape
            expected_img = self._save_scaled_gray_img(expected_img[:,:,0], '%simgs_big/%d_expected_big.png' % (prefix, index), 16)

            x_img2 = x_img.copy()
            x_img2[:,:,0] = x_img[:,:,0] * expected_img
            x_img2[:,:,1] = x_img[:,:,1] * expected_img
            x_img2[:,:,2] = x_img[:,:,2] * expected_img
            self._save_img('%simgs_big/%d_x_by_expected.png' % (prefix, index), x_img2)

    def _save_img(self, img_filename, x_img):
        x_img_unit = img_as_uint(x_img)
        io.imsave(self.config.model_results_filename(img_filename), x_img_unit)

    def _save_scaled_gray_img(self, im, img_filename, n):
        im = np.kron(im, np.ones((n, n)))
        im = exposure.rescale_intensity(im, out_range='float')
        im_uint = img_as_uint(im)
        if im_uint.ndim == 3:
            im_uint = im_uint[:,:,::n]
        io.imsave(self.config.model_results_filename(img_filename), im_uint)
        return im


if __name__ == '__main__':
    model = CnnModelDecorator('100examples64x64', (64, 64, 1), (29, 29), 'test_cnn_model.model')
    predicted = model.model.predict_classes(model.X_train)
    print (predicted.shape)