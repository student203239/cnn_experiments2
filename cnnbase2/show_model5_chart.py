import os

from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models import Model1, Model2, Model3, Model5, Model6


def ensure_dir_exist(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)


if __name__ == '__main__':
    print "Main started"
    config = CnnDirsConfig()
    from os import listdir
    from os.path import isfile, join
    mypath = config.model_dir
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and ".Model6." in f]
    model = Model6(config, 'flic.valid.07', "")
    epoch_history = {}
    for f in onlyfiles:
        # print f
        epoch = f[:-len('.Model5.model')]
        epoch = epoch[-3:]
        epoch = int(''.join([x for x in epoch if x >= '0' and x <= '9']))
        if epoch > 600:
            continue
        print epoch
        model.load_from_file(f)
        epoch_history[epoch] = model.evaluate_model()
    import collections
    od = collections.OrderedDict(sorted(epoch_history.items()))
    with open(config.model_results_filename('tmp.csv'), 'w') as f:
        f.write("epoch\ttrain loss\ttest loss\n")
        for epoch,(train_loss, test_loss) in od.iteritems():
            f.write("%d\t%.6f\t%.6f\n" % (epoch, train_loss, test_loss))
            print epoch
