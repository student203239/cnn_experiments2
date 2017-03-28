import os

import time

from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models import Model1, Model2, Model3, Model4, Model5, Model6
from cnnbase2.models2 import TinyAlexNet2

if __name__ == '__main__':
    config = CnnDirsConfig()
    model_filename = 'alex1_ones_2'
    model2 = TinyAlexNet2(config, 'flic.valid.07', model_filename)
    model2.load_from_file('alex1_ones')
    model2.learn_now(220)
    model2.save_model_to_file()
    #
    # model2 = Model6(config, 'flic.valid.07', model_filename)
    # model2.learn_now(120)
    # model2.save_model_to_file()
    #
    time.sleep(30)
    os.system("shutdown.exe /f /h")
