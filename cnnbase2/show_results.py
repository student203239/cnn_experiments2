from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models import Model1, Model2, Model3, Model5

if __name__ == '__main__':
    model_filename = 'run1'
    model_filename = 'learn-on-5000-before-on-1000'
    config = CnnDirsConfig()
    model = Model5(config, '1000examples', model_filename)
    model.load_from_file()
    model.show_results(True)
