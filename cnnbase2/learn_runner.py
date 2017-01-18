from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models import Model1, Model2, Model3, Model4, Model5

if __name__ == '__main__':
    # model_filename = 'run1-epoch70.Model3.model'
    # config = CnnDirsConfig()
    # model = Model3(config, '5000examples', model_filename)
    # model.load_from_file()
    # for i in range(8, 20):
    #     history = model.learn_now(10)
    #     model.save_model_to_file("%s-epoch%d" % (model_filename, i*10))
    #     print "Saved: %s-epoch%d" % (model_filename, i*10)
    # model.save_model_to_file()

    # Error allocating 147456000 bytes of device memory (out of memory). Driver report 1536270336 bytes free and 2147483648 bytes total
    # model = Model3(config, '5000examples', model_filename)
    # model.load_from_file()
    # for i in range(8,32):
    #     history = model.learn_now(10)
    #     model.save_model_to_file("%s-epoch%d" % (model_filename, i*10))
    #     print "Saved: %s-epoch%d" % (model_filename, i*10)

    # model_filename = 'learn-on-1000'
    # config = CnnDirsConfig()
    # model = Model4(config, '1000examples', model_filename)
    # model2 = Model5(config, '1000examples', model_filename)
    # # model.load_from_file()
    # for i in range(10):
    #     history = model.learn_now(20)
    #     model.save_model_to_file("%s-epoch%d" % (model_filename, i*20))
    #     print "Save model4d: %s-epoch%d" % (model_filename, i*20)
    #
    #     history = model2.learn_now(20)
    #     model2.save_model_to_file("%s-epoch%d" % (model_filename, i*20))
    #     print "Saved model5: %s-epoch%d" % (model_filename, i*20)
    # model.save_model_to_file()
    # model2.save_model_to_file()

    config = CnnDirsConfig()
    model_filename = 'learn-on-8000-before-on-5000'
    model2 = Model5(config, '8000examples', model_filename)
    model2.load_from_file('learn-on-5000-before-on-1000-epoch580')
    for i in range(30,70):
        history = model2.learn_now(20)
        model2.save_model_to_file("%s-epoch%d" % (model_filename, i*20))
        print "Saved model5: %s-epoch%d" % (model_filename, i*20)
    model2.save_model_to_file()
