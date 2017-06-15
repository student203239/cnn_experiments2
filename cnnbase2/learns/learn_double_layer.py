import os

import time

from cnnbase2.data_feeders.data_feeder_cnn_model_like import DataFeederCnnModelBaseLike
from cnnbase2.data_feeders.dumy_data_feeders_merger import DummyFeedersMerge
from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models import Model1, Model2, Model3, Model4, Model5, Model6
from cnnbase2.models2 import TinyAlexNet2, TinyAlexNet3, TinyAlexNet4, TinyAlexNet4Double
from cnnbase2.models_viewer import ModelsConatiner


class Experiment4ModelContainer(ModelsConatiner):

    def __init__(self, config):
        self.filename = "june15.dummy.experiment4"

        # cars_feeder = DataFeederCnnModelBaseLike(config, '5000examples').init_car_type(smaller_car=True)
        cars_feeder = DataFeederCnnModelBaseLike(config, '100examples').init_car_type(smaller_car=True)
        human_feeder = DataFeederCnnModelBaseLike(config, 'flic.small.shuffle.code10').init_human_type(y_gen_mode='r')
        merger_feeder = DummyFeedersMerge(cars_feeder, human_feeder)

        model = TinyAlexNet4Double(config, default_filename=self.filename, prepared_data=merger_feeder)
        model.batch_size = 100
        models_dict = {'x': model}
        super(Experiment4ModelContainer, self).__init__(models_dict, _is_car_like_predict_shape=False)

if __name__ == '__main__':
    config = CnnDirsConfig()

    container = Experiment4ModelContainer(config)
    filename = container.filename
    model = container.get_init_model()
    model.save_model_to_file(filename + ".init")
    epoches_to_do = 2
    epoch_pack = 1
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
