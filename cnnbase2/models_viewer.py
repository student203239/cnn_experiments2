import scipy
from matplotlib import pyplot
import numpy as np

from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models import Model5, Model6


class ModelsViewer(object):

    def __init__(self, models=None, load_models=True):
        config = CnnDirsConfig()
        if models:
            self.models = models
            print "use predefine models"
        else:
            self.models = self.create_models(config)
        # self.models = self.create_models_cars(config)
        self.model = self.models['z']
        # model = Model6(config, '1000distractors', model_filename)
        for m in self.models.values():
            if load_models:
                m.load_from_file()
            m.get_predicted_test()
        self.key_handlers = {}
        self.key_handlers['1'] = self.src_img
        self.key_handlers['2'] = self.predicted_img
        self.key_handlers['3'] = self.expected_img
        self.key_handlers['['] = self.next_img
        self.key_handlers[']'] = self.prev_img
        self.key_handlers['m'] = self.change_mul
        self.now_show_view = '1'
        self.need_update_view = False
        self.mul_by_src_img = False
        self.index = 99

    def create_models(self, config):
        m5_gauss = Model5(config, 'flic.valid.07', 'flic2')
        m6_gauss = Model6(config, 'flic.valid.07', 'flic2')
        m5_ones = Model5(config, 'flic.valid.07', 'flic2_ones', prepared_data=m5_gauss.get_prepared_data())
        m6_ones = Model6(config, 'flic.valid.07', 'flic2_ones', prepared_data=m6_gauss.get_prepared_data())
        return {
            'z': m5_gauss,
            'x': m6_gauss,
            'c': m5_ones,
            'v': m6_ones,
        }

    def create_models_cars(self, config):
        m5 = Model5(config, '1000examples', 'learn-on-8000-before-on-5000-epoch960')
        m6 = Model6(config, '1000examples', 'run2-epoch1120')
        return {'z': m5, 'x': m6}

    def src_img(self, gca):
        src_img = self.get_Src_img()
        gca.imshow(src_img)
        self.now_show_view = '1'

    def get_Src_img(self):
        return self.model.X_test[self.index, :, :, :]

    def predicted_img(self, gca):
        predicted_img = self.model.get_predicted_test()[self.index, :, :, 0]
        if self.mul_by_src_img:
            src_img = self.get_Src_img()
            predicted_img = self.model.multiply_rgb_img_by_gray_img(predicted_img, src_img)
        gca.imshow(predicted_img)
        self.now_show_view = '2'

    def expected_img(self, gca):
        expected_img = self.model.y_test[self.index, :, :, 0]
        if self.mul_by_src_img:
            src_img = self.get_Src_img()
            expected_img = self.model.multiply_rgb_img_by_gray_img(expected_img, src_img)
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

    def onkey(self, evt):
        key = str(evt.key)
        gca = evt.canvas.figure.gca()
        if key in self.models:
            self.model = self.models[key]
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