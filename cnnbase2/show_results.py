from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models import Model1, Model2, Model3, Model5, Model6

if __name__ == '__main__':
    model_filename = 'run1'
    model_filename = 'learn-on-5000-before-on-1000'
    model_filename = 'run2-epoch580'
    config = CnnDirsConfig()
    model = Model6(config, '1000examples', model_filename)
    # model = Model6(config, '1000distractors', model_filename)
    model.load_from_file()
    # print model.model.layers[0].get_config()
    # model.show_results(True)
    model.show_more_results()
