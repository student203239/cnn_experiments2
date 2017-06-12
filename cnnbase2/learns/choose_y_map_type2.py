import os

import time

from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models import Model1, Model2, Model3, Model4, Model5, Model6
from cnnbase2.models2 import TinyAlexNet2, TinyAlexNet3, TinyAlexNet4

if __name__ == '__main__':
    config = CnnDirsConfig()
    # filename = 'mayc10r.experiment1'
    filenames = ['mayc10%s.june12.experiment1' % ch for ch in ['r', 'i', 'o']]
    for filename in filenames:
        init_model = TinyAlexNet4(config, 'flic.small.shuffle.code10', filename)
        init_model.save_model_to_file()
        init_model.save_model_to_file(filename + "_init_model")
    # model = TinyAlexNet4(config, 'flic.small.shuffle.code10', filenames[0])
    model = TinyAlexNet4(config, 'flic.shuffle.code10', filenames[0])
    epoches_to_do = 4
    epoch_pack = 15
    init_epoch = 0
    for ei in range(epoches_to_do):
        for filename in filenames:
            model.set_default_filename(filename)
            model.load_from_file()
            print "Evaluate %s:" % filename
            print model.evaluate_alex_model()
            model.learn_now(epoch_pack)
            epoch = init_epoch + (ei+1) * epoch_pack
            model.save_model_to_file(filename + ('.e%d.' % epoch) + model.last_history_timestamp)
            model.save_model_to_file()
            print "done epoch %d on %s" % (epoch, filename)

    # time.sleep(30)
    # os.system("shutdown.exe /f /h")
