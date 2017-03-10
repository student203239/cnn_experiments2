from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models import Model1, Model2, Model3, Model4, Model5, Model6

if __name__ == '__main__':
    config = CnnDirsConfig()
    model_filename = 'flic1'
    model2 = Model6(config, 'flic.valid.07', model_filename)
    model2.learn_now(50)
    model2.save_model_to_file()
