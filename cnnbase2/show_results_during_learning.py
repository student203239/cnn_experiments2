import os

from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models import Model1, Model2, Model3, Model5


def ensure_dir_exist(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)


if __name__ == '__main__':
    config = CnnDirsConfig()
    from os import listdir
    from os.path import isfile, join
    mypath = config.model_dir
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and ".Model5." in f]
    model = Model5(config, '1000examples', "")
    for f in onlyfiles:
        print f
        model.load_from_file(f)
        f += "_up"
        ensure_dir_exist(config.model_results + f)
        ensure_dir_exist(config.model_results + f + "\imgs")
        ensure_dir_exist(config.model_results + f + "\imgs_big")
        model.show_more_results(interactive=False, prefix=f+"/")

    # model_filename = 'run1'
    # model_filename = 'learn-on-5000-before-on-1000'
    # model_filename = 'learn-on-8000-before-on-5000-epoch960'
    # model.load_from_file()
    # model.show_more_results()
