import os

import time

from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models import Model1, Model2, Model3, Model4, Model5, Model6
from cnnbase2.models2 import TinyAlexNet2, TinyAlexNet3, TinyAlexNet4

if __name__ == '__main__':
    config = CnnDirsConfig()
    filename = "june12.experiment3"
    model = TinyAlexNet4(config, '5000examples', filename, smaller_car = False)
    model.batch_size = 100
    # model = TinyAlexNet4(config, '100examples', filename)
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
    # time.sleep(50)
    # os.system("shutdown.exe /f /h")
