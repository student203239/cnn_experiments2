import scipy
from matplotlib import pyplot
import numpy as np

from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models import Model5


class ModelsViewer(object):

    def __init__(self):
        model_filename = 'flic2'
        config = CnnDirsConfig()
        self.model = Model5(config, 'flic.valid.07', model_filename)
        # model = Model6(config, '1000distractors', model_filename)
        self.model.load_from_file()
        self.key_handlers = {}
        self.key_handlers['1'] = self.src_img
        self.key_handlers['2'] = self.predicted_img
        self.key_handlers['3'] = self.expected_img
        self.index = 99

    def src_img(self, gca):
        gca.imshow(self.model.X_test[self.index,:,:,:])
        print self.model.y_test.shape

    def predicted_img(self, gca):
        gca.imshow(self.model.X_test[self.index,:,:,:])

    def expected_img(self, gca):
        expected_img = self.model.y_test[self.index, :, :, 0]
        print expected_img.shape
        gca.imshow(expected_img)

    def onkey(self, evt):
        key = str(evt.key)
        if key in self.key_handlers:
            gca = evt.canvas.figure.gca()
            self.key_handlers[key](gca)
            evt.canvas.draw()

    def init_my_canvas(self):
        fig = pyplot.figure()
        # fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', self.onkey)

        pyplot.imshow(np.zeros((100, 100), dtype='float32'))
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

ModelsViewer().init_my_canvas()