from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models import Model1, Model2

if __name__ == '__main__':
    model_filename = 'run1'
    config = CnnDirsConfig()
    model = Model2(config, '1000examples', model_filename)
    # model.load_from_file()
    for i in range(7):
        history = model.learn_now(10)
        model.save_model_to_file("%s-epoch%d" % (model_filename, i*10))
        print "Saved: %s-epoch%d" % (model_filename, i*10)
    model.save_model_to_file()

    model = Model2(config, '5000examples', model_filename)
    model.load_from_file()
    for i in range(7,11):
        history = model.learn_now(10)
        model.save_model_to_file("%s-epoch%d" % (model_filename, i*10))
        print "Saved: %s-epoch%d" % (model_filename, i*10)
