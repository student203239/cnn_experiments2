import os

import time

from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models import Model1, Model2, Model3, Model4, Model5, Model6
from cnnbase2.models2 import TinyAlexNet2, TinyAlexNet3, TinyAlexNet4

if __name__ == '__main__':
    config = CnnDirsConfig()
    # filename = 'mayc10r.experiment1'
    filenames = ['mayc10%s.experiment1' % ch for ch in ['r', 'i', 'o']]
    model = TinyAlexNet4(config, 'flic.small.shuffle.code10', filenames[0])
    # for filename in filenames:
    #     model.save_model_to_file(filename)
    epoches_to_do = 2
    epoch_pack = 2
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
