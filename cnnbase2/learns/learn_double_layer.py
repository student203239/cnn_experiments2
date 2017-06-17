# -*- coding: utf-8 -*-

import os

import time

from cnnbase2.data_feeders.data_feeder_cnn_model_like import DataFeederCnnModelBaseLike
from cnnbase2.data_feeders.dumy_data_feeders_merger import DummyFeedersMerge
from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models import Model1, Model2, Model3, Model4, Model5, Model6
from cnnbase2.models2 import TinyAlexNet2, TinyAlexNet3, TinyAlexNet4, TinyAlexNet4Double
from cnnbase2.models_viewer import ModelsConatiner


class Experiment4ModelContainer(ModelsConatiner):

    def __init__(self, config, load_train=True):
        self.filename = "june15.experiment4"

        # cars_feeder = DataFeederCnnModelBaseLike(config, '100examples').init_car_type(smaller_car=True)
        # human_feeder = DataFeederCnnModelBaseLike(config, 'flic.small.shuffle.code10').init_human_type(y_gen_mode='r')

        cars_feeder = DataFeederCnnModelBaseLike(config, '5000examples', load_train=load_train).init_car_type(smaller_car=True)
        human_feeder = DataFeederCnnModelBaseLike(config, 'flic.shuffle.code10', load_train=load_train).init_human_type(y_gen_mode='r')

        merger_feeder = DummyFeedersMerge(cars_feeder, human_feeder, load_train=load_train)

        model = TinyAlexNet4Double(config, default_filename=self.filename, prepared_data=merger_feeder)
        model.batch_size = 100
        model.load_from_file("june15.dummy2.experiment4.init")
        models_dict = {'x': model}

        self.models_dict = models_dict
        self._is_car_like_predict_shape = False

    def prepare_models_to_view(self):
        self._init_load_models()

    def get_desc(self, model_key):
        return {'x': "dla dwukanałowego wyjścia"}[model_key]

    def get_short_letter(self, model_key):
        return {'x': 'dl'}[model_key]

def run_experiment4():
    config = CnnDirsConfig()

    container = Experiment4ModelContainer(config)
    filename = container.filename
    model = container.get_init_model()
    model.save_model_to_file(filename + ".init")
    epoches_to_do = 2
    epoch_pack = 20
    init_epoch = 0
    for ei in range(epoches_to_do):
        model.learn_now(epoch_pack)
        epoch = init_epoch + (ei+1) * epoch_pack
        model.save_model_to_file(filename + ('.e%d.' % epoch) + model.last_history_timestamp)
        model.save_model_to_file()
        print "done epoch %d on %s" % (epoch, filename)
    print "LEARN FINISH, GOING TO SYSTEM HIBERNATE"
    time.sleep(50)
    os.system("shutdown.exe /f /h")

if __name__ == '__main__':
    run_experiment4()