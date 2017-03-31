import os

import time

from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models import Model1, Model2, Model3, Model4, Model5, Model6
from cnnbase2.models2 import TinyAlexNet2, TinyAlexNet3

if __name__ == '__main__':
    config = CnnDirsConfig()
    model2 = TinyAlexNet3(config, 'flic.bound', 'alex_ones')
    # model2.load_from_file('alex1_ones')
    model2.learn_now(60)
    model2.save_model_to_file()
    # model2.learn_now(20)
    # model2.save_model_to_file('alex2_ones_100_epoch_30')

    # model2 = TinyAlexNet3(config, 'flic.200', 'alex2_ones_200')
    # model2.learn_now(10)
    # model2.save_model_to_file()
    # model2.learn_now(20)
    # model2.save_model_to_file('alex2_ones_200_epoch_30')
    #
    # model2 = Model6(config, 'flic.valid.07', model_filename)
    # model2.learn_now(120)
    # model2.save_model_to_file()
    #
    # time.sleep(30)
    # os.system("shutdown.exe /f /h")