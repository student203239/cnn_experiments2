# -*- coding: utf-8 -*-

import os

import time

from cnnbase2.data_feeders.data_feeder_cnn_model_like import DataFeederCnnModelBaseLike
from cnnbase2.data_feeders.dumy_data_feeders_merger import DummyFeedersMerge
from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models2 import TinyAlexNet5
from cnnbase2.models_viewer import ModelsConatiner


class Experiment5ContinueModelContainer(ModelsConatiner):

    def __init__(self, config, load_train=True):
        # cars_feeder = DataFeederCnnModelBaseLike(config, '100examples').init_car_type(smaller_car=True)
        # human_feeder = DataFeederCnnModelBaseLike(config, 'flic.small.shuffle.code10').init_human_type(y_gen_mode='r')

        cars_feeder = DataFeederCnnModelBaseLike(config, '5000examples', load_train=load_train).init_car_type(smaller_car=True)
        human_feeder = DataFeederCnnModelBaseLike(config, 'flic.shuffle.code10', load_train=load_train).init_human_type(y_gen_mode='r')
        merger_feeder = DummyFeedersMerge(cars_feeder, human_feeder, load_train=load_train)

        model_sigmoid = TinyAlexNet5(config, use_sigmoid=True,
                                     default_filename="june17.experiment5.sigmoid.e24.2017-06-18--18-17-03.TinyAlexNet5.model",
                                     prepared_data=merger_feeder)
        model_sigmoid.batch_size = 100
        model_sigmoid.load_from_file()

        model_tanh = TinyAlexNet5(config, use_sigmoid=False, default_filename="june17.experiment5.tanh.e24.2017-06-19--01-41-05.TinyAlexNet5.model",
                                  prepared_data=merger_feeder)
        model_tanh.batch_size = 100
        model_tanh.load_from_file()
        models_dict = {'z':model_tanh, 'x': model_sigmoid}

        self.models_dict = models_dict
        self._is_car_like_predict_shape = False

    # def prepare_models_to_view(self):
    #     self._init_load_models()

    def get_desc(self, model_key):
        return {'z': "dla aktywacji tanh",
            'x': "dla aktywacji sigmoid"}[model_key]

    def get_short_letter(self, model_key):
        return {'z': 'tanh', 'x': 'sigmoid'}[model_key]

def run_experiment5_continue():
    config = CnnDirsConfig()

    container = Experiment5ContinueModelContainer(config)
    filename_base = "june19.experiment5.continue."
    model = container.get_init_model()
    for model_key in container.get_models_keys():
        model = container.get_model(model_key, model)
        filename = filename_base + container.get_short_letter(model_key)
        model.save_model_to_file(filename + ".init")
    epoches_to_do = 3
    epoch_pack = 10
    init_epoch = 24
    for ei in range(epoches_to_do):
        for model_key in container.get_models_keys():
            model = container.get_model(model_key, model)
            filename = filename_base + container.get_short_letter(model_key)
            model.learn_now(epoch_pack)
            epoch = init_epoch + (ei+1) * epoch_pack
            model.save_model_to_file(filename + ('.e%d.' % epoch) + model.last_history_timestamp)
            model.save_model_to_file()
            print "done epoch %d on %s" % (epoch, filename)
    print "LEARN FINISH, GOING TO SYSTEM HIBERNATE"
    time.sleep(50)
    os.system("shutdown.exe /f /h")

if __name__ == '__main__':
    run_experiment5_continue()