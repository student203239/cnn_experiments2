# -*- coding: utf-8 -*-

import scipy
from matplotlib import pyplot
import numpy as np

from cnnbase2.data_feeders.data_feeder_cnn_model_like import DataFeederCnnModelBaseLike
from cnnbase2.img_utils import ImgUtlis
from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models import Model5, Model6


class ModelsViewer(object):

    def __init__(self, models_container=None):
        self.filenames = ['mayc10%s.experiment1' % ch for ch in ['r', 'i', 'o']]
        config = CnnDirsConfig()
        if models_container:
            self.models_container = models_container
            print "use predefine models"
        else:
            # self.models_container = ModelsContainerExperiment1(config)
            # self.models_container = ModelsConatiner(self.create_models_cars2_with_data_feeder(config), _is_car_like_predict_shape=False)
            # self.models_container = ModelsContainerExperiment1(CnnDirsConfig(), base_filename='mayc10%s.june12.experiment1')
            from cnnbase2.learns.learn_double_layer import Experiment4ModelContainer
            # self.models_container = Experiment4ModelContainer(config, load_train=False)
            from cnnbase2.learns.learn_sigmoid_experiemnt5 import Experiment5ModelContainer
            self.models_container = Experiment5ModelContainer(config, load_train=False)
        self.models_container.prepare_models_to_view()
        # self.models = self.create_models_cars(config)
        self.model = self.models_container.get_init_model()
        self.key_handlers = {}
        self.key_handlers['1'] = self.src_img
        self.key_handlers['2'] = self.predicted_img
        self.key_handlers['3'] = self.expected_img
        self.key_handlers['4'] = self.show_heatmap
        self.key_handlers['['] = self.next_img
        self.key_handlers[']'] = self.prev_img
        self.key_handlers['m'] = self.change_mul
        self.key_handlers['o'] = self.next_output_layer
        self.key_handlers['i'] = self.save_imgs
        self.now_show_view = '1'
        self.need_update_view = False
        self.mul_by_src_img = False
        self.index = 1
        self.output_layer = 0

    def create_models(self, config):
        # m5_gauss = Model5(config, 'flic.valid.07', 'flic2')
        # m6_gauss = Model6(config, 'flic.valid.07', 'flic2')
        # m5_ones = Model5(config, 'flic.valid.07', 'flic2_ones', prepared_data=m5_gauss.get_prepared_data())
        # m6_ones = Model6(config, 'flic.valid.07', 'flic2_ones', prepared_data=m6_gauss.get_prepared_data())
        from cnnbase2.models2 import TinyAlexNet2
        from cnnbase2.models2 import TinyAlexNet3
        from cnnbase2.models2 import TinyAlexNet4
        # alex_ones = TinyAlexNet2(config, 'flic.valid.07', 'alex1_ones')
        # alex_ones2 = TinyAlexNet2(config, 'flic.valid.07', 'alex1_ones_2')
        # a3 = TinyAlexNet3(config, 'flic.shuffle.code10', 'alex_code10')
        # a4 = TinyAlexNet4(config, 'flic.shuffle.code10', 'alex_code10', prepared_data=a3.get_prepared_data())
        mo = TinyAlexNet4(config, 'flic.shuffle.code10', self.filenames[2])
        mr = TinyAlexNet4(config, 'flic.shuffle.code10', self.filenames[0], prepared_data=mo.get_prepared_data())
        mi = TinyAlexNet4(config, 'flic.shuffle.code10', self.filenames[1], prepared_data=mr.get_prepared_data())
        return {
            # 'z': TinyAlexNet2(config, 'flic.bound', 'alex1_ones'),
            'z': mo,
            'x': mi,
            'c': mr,
            # 'x': TinyAlexNet3(config, 'flic.bound', 'alex_ones_after60.e150.2017-04-01--17-49-23')
            # 'x': m6_gauss,
            # 'c': m5_ones,
            # 'v': m6_ones,
            # 'z': TinyAlexNet3(config, 'flic.100', 'alex2_ones_100'),
            # 'x': TinyAlexNet3(config, 'flic.100', 'alex2_ones_100_epoch_30')
        }

    def create_models_cars(self, config):
        m5 = Model5(config, '1000examples', 'learn-on-8000-before-on-5000-epoch960')
        m6 = Model6(config, '1000examples', 'run2-epoch1120')
        return {'z': m5, 'x': m6}

    def create_models_cars2(self, config):
        from cnnbase2.models2 import TinyAlexNet4
        # m5 = Model5(config, '1000examples', 'learn-on-8000-before-on-5000-epoch960')
        m5 = TinyAlexNet4(config, '1000examples', 'june12.experiment3', smaller_car = False)
        m5.y_gen_mode = None
        m5.default_filename = None
        m6 = TinyAlexNet4(config, '1000examples', "june11.experiment2")
        return {'z': m5, 'x': m6}

    def create_models_cars2_with_data_feeder(self, config):
        from cnnbase2.models2 import TinyAlexNet4
        m5 = TinyAlexNet4(config, default_filename=  'june12.experiment3',
                          prepared_data=DataFeederCnnModelBaseLike(config, '1000examples').init_car_type(smaller_car=False))
        m5.y_gen_mode = None
        m5.default_filename = None
        m6 = TinyAlexNet4(config, default_filename="june11.experiment2",
                          prepared_data=DataFeederCnnModelBaseLike(config, '1000examples').init_car_type(smaller_car=True))
        return {'z': m5, 'x': m6}

    def src_img(self, gca):
        src_img = self.get_Src_img()
        gca.imshow(src_img)
        self.now_show_view = '1'

    def get_Src_img(self):
        print self.model.X_test.shape
        return self.model.X_test[self.index, :, :, :]

    def _get_img_from_y_array(self, index, array):
        print "get img {}".format(array.shape)
        if self.models_container.is_car_like_predict_shape():
            return array[index, :, :, self.output_layer]
        else:
            return array[index, self.output_layer, :, :]

    def predicted_img(self, gca):
        print self.model.get_predicted_test().shape
        predicted_img = self._get_img_from_y_array(self.index, self.model.get_predicted_test())
        predicted_img = np.absolute(predicted_img)
        print "predicted_img:"
        print predicted_img
        src_img = self.get_Src_img()
        if not self.mul_by_src_img:
            src_img = np.ones(shape=src_img.shape, dtype=src_img.dtype)
        predicted_img = self.model.multiply_rgb_img_by_gray_img(predicted_img, src_img, advanced_resize=True)
        # predicted_img = self.model.multiply_rgb_img_by_gray_img(predicted_img, src_img, advanced_resize=False)
        gca.imshow(predicted_img)
        self.now_show_view = '2'

    def expected_img(self, gca):
        expected_img = self.model.y_test[self.index, :, :, self.output_layer]
        src_img = self.get_Src_img()
        if not self.mul_by_src_img:
            src_img = np.ones(shape=src_img.shape, dtype=src_img.dtype)
        expected_img = self.model.multiply_rgb_img_by_gray_img(expected_img, src_img, advanced_resize=True)
        gca.imshow(expected_img)
        self.now_show_view = '3'

    def next_img(self, gca):
        self.index += 1
        self.need_update_view = True

    def prev_img(self, gca):
        self.index -= 1
        self.need_update_view = True

    def change_mul(self, gca):
        self.mul_by_src_img = not self.mul_by_src_img
        self.need_update_view = True

    def show_histogram(self, gca):
        expects = self.model.y_test[:,:,:,0]
        predicts = self.model.get_predicted_test()[:, self.output_layer, :, :]
        expects = ImgUtlis.alfa_cut_image(0.1, expects)
        predicts = ImgUtlis.alfa_cut_image(0.1, predicts)
        errorsStatistics = ImgUtlis.count_advance_errors(expects, predicts)
        import matplotlib.pyplot as plt
        plt.hist(errorsStatistics.type2_list)
        plt.show()
        self.need_update_view = False

    def show_heatmap(self, gca):
        predicted_img = self._get_img_from_y_array(self.index, self.model.get_predicted_test())
        gca.imshow(predicted_img)
        self.now_show_view = '4'

    def next_output_layer(self, gca):
        self.output_layer += 1
        if self.output_layer >= 2:
            self.output_layer = 0
        self.need_update_view = True

    def save_imgs(self, gca):
        self.mul_by_src_img = True
        print dir(gca)
        self.need_update_view = True

    def onkey(self, evt):
        key = str(evt.key)
        gca = evt.canvas.figure.gca()
        if self.models_container.contains_model_key(key):
            self.model = self.models_container.get_model(key, self.model)
            self.key_handlers[self.now_show_view](gca)
            evt.canvas.draw()
        if key in self.key_handlers:
            self.key_handlers[key](gca)
            if self.need_update_view:
                self.key_handlers[self.now_show_view](gca)
                self.need_update_view = False
            evt.canvas.draw()
        else:
            print "unrecognized key: {}".format(key)

    def init_my_canvas(self):
        fig = pyplot.figure()
        # fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', self.onkey)

        # pyplot.imshow(np.zeros((100, 100), dtype='float32'))
        pyplot.imshow(self.get_Src_img())
        pyplot.show()
    # up
    # down
    # left
    # right


class ModelsConatiner(object):
    def __init__(self, models_dict, _is_car_like_predict_shape=False):
        self.models_dict = models_dict
        self._is_car_like_predict_shape = _is_car_like_predict_shape
        self._init_load_models()
        self.init_model_key = 'x'

    def _init_load_models(self):
        for m in self.models_dict.values():
            m.load_from_file()
            m.get_predicted_test()

    def prepare_models_to_view(self):
        pass

    def is_car_like_predict_shape(self):
        return self._is_car_like_predict_shape

    def contains_model_key(self, key):
        return key in self.models_dict

    def get_model(self, model_key, prev_model):
        return self.models_dict[model_key]

    def get_init_model(self):
        if not hasattr(self, 'init_model_key'):
            self.init_model_key = 'x'
        return self.models_dict[self.init_model_key]

    def get_models_keys(self):
        return self.models_dict.keys()


class ModelsContainerExperiment1(ModelsConatiner):
    def __init__(self, config, base_filename='mayc10%s.experiment1'):
        from cnnbase2.models2 import TinyAlexNet4
        self.filenames = [base_filename % ch for ch in ['r', 'i', 'o']]
        mo = TinyAlexNet4(config, 'flic.shuffle.code10', self.filenames[2])
        mr = TinyAlexNet4(config, 'flic.shuffle.code10', self.filenames[0])#, prepared_data=mo.get_prepared_data())
        mi = TinyAlexNet4(config, 'flic.shuffle.code10', self.filenames[1])#, prepared_data=mr.get_prepared_data())
        models_dict = {
            'z': mo,
            'x': mi,
            'c': mr,
        }
        super(ModelsContainerExperiment1, self).__init__(models_dict)

    def get_desc(self, model_key):
        return {'z': "dla metody oznaczania całego zaanotowanego obszaru postaci",
                'x': "dla metody oznaczanie jedynie tułowia",
                "c": "dla podejścia mieszanego"}[model_key]

    def get_short_letter(self, model_key):
        return {"z": 'o', 'x': 'i', 'c': 'r'}[model_key]

    def get_model(self, model_key, prev_model):
        num = {'z':0, 'x':1,'c':2}[model_key]
        model = self.models_dict[model_key]
        #prev_model.set_default_filename(self.filenames[num])
        return model


class ModelsContainerExperiment3(ModelsConatiner):

    def __init__(self, config):
        from cnnbase2.models2 import TinyAlexNet4
        filename = "june11.experiment2"
        small_car_model = TinyAlexNet4(config, '5000examples', filename)

        filename = "june12.experiment3"
        big_car_model = TinyAlexNet4(config, '5000examples', filename, smaller_car = False)
        models_dict = {'z': small_car_model, 'x': big_car_model}
        super(ModelsContainerExperiment3, self).__init__(models_dict)

    def get_desc(self, model_key):
        return {'z': "dla metody oznaczania wycinka obszaru auta",
                'x': "dla metody oznaczania całego obszaru auta"}[model_key]

    def get_short_letter(self, model_key):
        return {"z": 's', 'x': 'b'}[model_key]

DataRange = range(0, 360)
DataRange = map(scipy.deg2rad, DataRange)
Data1 = map(scipy.sin, DataRange)
Data2 = map(scipy.cos, DataRange)

toggle = True

def onclick(event):
    print 'click'
    global toggle

    toggle = not toggle
    event.canvas.figure.clear()

    if toggle:
        print "-"*10
        print '\n'.join(dir(event.canvas.figure.gca()))
        event.canvas.figure.gca().plot(Data1)
    else:
        event.canvas.figure.gca().plot(Data2)

    event.canvas.draw()

def onkey(evt):
    key = evt.key
    print key
    # import pdb
    # pdb.set_trace()

# fig = pyplot.figure()
# fig.canvas.mpl_connect('button_press_event', onclick)
# fig.canvas.mpl_connect('key_press_event', onkey)
#
# pyplot.plot(Data1)
# pyplot.show()
if __name__ == '__main__':
    ModelsViewer().init_my_canvas()