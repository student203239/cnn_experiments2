from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models import Model1, Model2, Model3, Model5, Model6

if __name__ == '__main__':
    model_filename = 'run1'
    model_filename = 'learn-on-5000-before-on-1000'
    model_filename = 'flic1'
    config = CnnDirsConfig()
    model = Model6(config, 'flic.valid.07', model_filename)
    # model = Model6(config, '1000distractors', model_filename)
    model.load_from_file()
    # from keras.utils.visualize_util import  plot
    # plot(model.model, config.model_filename('visualize.png'))
    # print model.model.layers[0].get_config()
    model.show_results(True)
    model.show_more_results()
