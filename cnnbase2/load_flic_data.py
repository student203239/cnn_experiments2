import os

import scipy.io
import math
import os
from random import shuffle
import random

import numpy as np
import skimage.io as io
import skimage.transform as tr

from os import fsync, remove


class FlicLoader(object):

    def __init__(self):
        self.flic_root = "C:\\Users\\Jacek\\Downloads\\flic\\FLIC"
        self.imgs_dir = os.path.join(self.flic_root, 'images')
        self.matlab = scipy.io.loadmat(os.path.join(self.flic_root, 'examples.mat'))

    def main(self):
        matlab = self.matlab
        l = len(matlab['examples'][0])
        w, im, h = self._load_and_square_im(self.filepath(99), True)
        print str(im)
        print str(im.shape)
        io.imshow(im)
        io.show()

    def mat_len(self):
        return len(self.matlab['examples'][0])

    def filepath(self, i):
        filepath = self.matlab['examples'][0]['filepath'][i][0]
        return os.path.join(self.imgs_dir, filepath)

    def istrain(self, i):
        return self.matlab['examples'][0]['istrain'][i][0][0]

    def istest(self, i):
        return self.matlab['examples'][0]['istest'][i][0][0]

    def torsobox(self, i):
        t = self.matlab['examples'][0]['torsobox'][i][0]
        return t[0], t[1], t[2], t[3]

    def _load_and_square_im(self, img_filename, move_up):
        im = io.imread(img_filename)
        im = im.astype('float32')
        im /= 256
        h = im.shape[0]
        w = im.shape[1]
        if im.ndim == 2:
            im_tmp = np.zeros((h, w, 3), dtype='float32')
            im_tmp[:, :, 0] = im
            im_tmp[:, :, 1] = im
            im_tmp[:, :, 2] = im
            del im
            im = im_tmp
        # im2 = np.random.rand(w,w,3).astype('float32')
        if h > w:
            im2 = np.zeros((h, h, 3), dtype='float32')
            if move_up:
                im2[:, :w, :] = im
            else:
                im2[:, h - w:, :] = im
            del im
        else:
            im2 = np.zeros((w, w, 3), dtype='float32')
            if move_up:
                im2[:h, :, :] = im
            else:
                im2[w - h:, :, :] = im
            del im
        im = im2
        return h, im, w

if __name__ == '__main__':
    loader = FlicLoader()
    loader.main()