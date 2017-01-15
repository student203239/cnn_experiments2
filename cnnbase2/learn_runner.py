from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models import Model1

if __name__ == '__main__':
    model_filename = 'run1'
    config = CnnDirsConfig()
    model = Model1(config, '5000examples', model_filename)
    model.load_from_file()
    for i in range(2, 10):
        history = model.learn_now(50)
        model.save_model_to_file("%s-epoch%d" % (model_filename, i*50))
        print "Saved: %s-epoch%d" % (model_filename, i*50)
