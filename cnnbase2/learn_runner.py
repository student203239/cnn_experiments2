from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models import Model1

if __name__ == '__main__':
    model_filename = 'run1'
    config = CnnDirsConfig()
    model = Model1(config, '1000examples', model_filename)
    # model.load_from_file()
    history = model.learn_now(50)
    model.save_model_to_file()
