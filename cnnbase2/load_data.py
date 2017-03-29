import math
import os
from random import shuffle
import random

import numpy as np
import skimage.io as io
import skimage.transform as tr

from os import fsync, remove

class CnnDirsConfig(object):

    def __init__(self):
        self.metadata_dirname = 'D:/mgr_dir1/metadata/' # bb_box.txt & path.txt inside
        self.img_path_prefix = 'C:/Users/Jacek/Documents/mgr_sem2/mgr/tmp_imgs/'
        self.data_dir = 'D:/mgr_dir1/data/' # x_train, bb_boxes_train
        self.model_dir = 'D:/mgr_dir1/model/' # with leart models
        self.model_history_dir = 'D:/mgr_dir1/model_history/'
        self.model_results = 'D:/mgr_dir1/model_results/'
        self.distractors_dir = 'D:/mgr_dir1/imgs_data/oxc1_100k/distractor/'

    def data_filename(self, filename):
        return self.data_dir + filename

    def model_filename(self, filename):
        return self.model_dir + filename

    def model_history_filename(self, filename):
        return self.model_history_dir + filename

    def model_results_filename(self, filename):
        return self.model_results + filename

    def distractor_filename(self, filename):
        return self.distractors_dir + filename

class CnnDataLoader(object):

    def __init__(self, config):
        self.config = config

        self.bb_box_file = self.config.metadata_dirname+'bb_box.txt'
        self.img_paths_file = self.config.metadata_dirname+'paths.txt'

        #Read the image paths
        self.image_paths = open(self.img_paths_file).read().split('\n')
        self.image_paths = [to_img_path(x) for x in self.image_paths]

        self.bb_boxes = self._load_bb_boxes(self.bb_box_file)

        gc = GaussianCalc(0, 0, 256, 256, 0.0)
        g = np.vectorize(gc.f)
        self.gaussion_buffer = np.fromfunction(g, (256, 256), dtype='float32')
        from os import listdir
        from os.path import isfile, join
        self.distractor_filenames = [f for f in listdir(config.distractors_dir) if isfile(join(config.distractors_dir, f)) and ".jpg" in f]

    def _load_bb_boxes(self, bb_box_file):
        # Read the bounding boxes
        boxes = open(bb_box_file).read().split('\n')
        # Storing the bounding boxes in numpy array of the form x1,y1,x2,y2
        bb_boxes = np.zeros((len(boxes), 4))
        for i in range(len(boxes)):
            bb_boxes[i, :] = [int(x) for x in boxes[i].split(',')]
        return bb_boxes

    def _to_img_path(self, path):
        if path.startswith('car_ims/'):
            path = path[len('car_ims/'):]
        return self.config.img_path_prefix + path

    def create_heat_map(self, hbb_box, return_w, return_h):
        heat_map = np.zeros((return_w, return_h))
        # x1, y1, x2, y2, w, h = hbb_box
        # y1 = int(y1*return_w/w)
        # x1 = int(x1*return_w/w)
        # x2 = int(x2*return_w/w)
        # y2 = int(y2*return_w/w)
        x1, y1, x2, y2 = self.get_heat_map_loc(hbb_box, return_w, return_h)
        heat_map[y1:y2,x1:x2] = tr.resize(self.gaussion_buffer, (y2-y1, x2-x1))
        return heat_map

    def create_heat_map_model6(self, hbb_box, return_w, return_h):
        x1, y1, x2, y2 = self.get_heat_map_loc_float(hbb_box, return_w, return_h)
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            return np.zeros((return_w, return_h), dtype='float32')
        gc = GaussianCalc(x1, y1, x2, y2, 0.0, x2-x1, y2-y1, (2,3))
        g = np.vectorize(gc.f)
        return np.fromfunction(g, (return_w, return_h), dtype='float32')

    def get_heat_map_loc_float(self, hbb_box, return_w, return_h):
        x1, y1, x2, y2, w, h = hbb_box
        rr = min(1.0*return_h/h, 1.0*return_w/w)
        y1 = (y1*rr)
        x1 = (x1*rr)
        x2 = (x2*rr)
        y2 = (y2*rr)
        return x1, y1, x2, y2

    def get_heat_map_loc(self, hbb_box, return_w, return_h):
        x1, y1, x2, y2, w, h = hbb_box
        rr = min(1.0*return_h/h, 1.0*return_w/w)
        y1 = int(y1*rr)
        x1 = int(x1*rr)
        x2 = int(x2*rr)
        y2 = int(y2*rr)
        return x1, y1, x2, y2

    def hbb_box_to_y(self, src_y, output_shape):
        w, h = output_shape
        examples = src_y.shape[0]
        y = np.zeros((examples,w,h,1), dtype='float32')
        for i in range(examples):
            if i == 10:
                print 'ok'
            x1, y1, x2, y2 = self.get_heat_map_loc(src_y[i], w, h)
            if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                continue
            # y[i,y1:y2,x1:x2,0] = loader.get_heat_map_to_paste(x1, y1, x2, y2)
            if y2 > h:
                y2 = h
            if x2 > w:
                x2 = w
            # try:
            #     to_paste = tr.resize(loader.gaussion_buffer, (y2 - y1, x2 - x1))
            # except:
            #     raise
            y[i,y1:y2,x1:x2,0] = 1
        return y

    def get_heat_map_to_paste(self, x1, y1, x2, y2):
        return tr.resize(self.gaussion_buffer, (y2-y1, x2-x1))

    def load_distractor_and_hbb_box(self, return_w=256, return_h=256):
        print "DISTRACTOR IMG"
        index = random.randint(0, len(self.distractor_filenames)-1)
        h, im, w = self._load_and_square_im(self.config.distractor_filename(self.distractor_filenames[index]), True)
        new_im = tr.resize(im, (return_w, return_h))
        return new_im, (0, 0, 0, 0, w, h)

    def load_image_and_hbb_box(self, i, return_w=256, return_h=256, move_up=True):
        # im = io.imread(self.image_paths[i], as_grey=True)
        h, im, w = self._load_and_square_im(self.image_paths[i], move_up)
        x1, y1, x2, y2 = self.bb_boxes[i]
        if not move_up:
            dy = abs(w-h)
            y1 += dy
            y2 += dy
        new_im = tr.resize(im, (return_w, return_h))
        del im
        return new_im, (x1, y1, x2, y2, w, h)

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

    def __load_image_and_heat_map(self, i, return_w=256, return_h=256):
        # im = io.imread(self.image_paths[i], as_grey=True)
        im = io.imread(self.image_paths[i])
        h, w, _ = im.shape
        if h > w:
            raise Exception("Nie zaimplementowano dla pionowych obrazkow, zaimplementuj h > w")
        im = im.astype('float32')
        im /= 256
        im2 = np.zeros((w,w,3), dtype='float32')
        # im2 = np.random.rand(w,w,3).astype('float32')
        im2[:h,:,:] = im
        im = im2
        print np.random.rand(460,460,3).dtype
        x1, y1, x2, y2 = self.bb_boxes[i]

        heat_map = self.create_heat_map((x1, y1, x2, y2, w, w), return_w, return_h)
        # f = 1.0*return_w / w
        # heat_map = self.create_heat_map((int(x1*f), int(y1*f), int(x2*f), int(y2*f), return_w, return_h), return_w, return_h)
        im = tr.resize(im, (return_w, return_h))
        return im, heat_map, (x1, y1, x2, y2, w, h)

    def load_train_and_hbb_box_data(self, test_factor=0.2, w=256, h=256, max_examples=100, move_up=True, distractor_prob=0):
        # examples_count = sum(1 if os.path.isfile(image_paths[i]) else 0 for i in range(0, len(image_paths)))
        # examples_count = min(max_examples, examples_count)
        examples_count = max_examples
        train_size = int(examples_count * (1.0- test_factor))
        test_size = examples_count - train_size

        x_train = np.zeros((train_size, w, h, 3), dtype='float32')
        hbb_box_train = np.zeros((train_size, 6), dtype='int16')
        x_test = np.zeros((test_size, w, h, 3), dtype='float32')
        hbb_box_test = np.zeros((test_size, 6), dtype='int16')
        train_index = 0
        i_range = range(0, len(self.image_paths))
        # shuffle(i_range)
        for i in  i_range:
            # print i
            if os.path.isfile(self.image_paths[i]):
                # print 'file ok'
                print train_index
                print self.image_paths[i]
                if random.random() < distractor_prob:
                    x, hbb_box = self.load_distractor_and_hbb_box(w, h)
                else:
                    x, hbb_box = self.load_image_and_hbb_box(i, w, h, move_up)
                if train_index < train_size:
                    x_train[train_index,:,:,:] = x.astype('float32')
                    hbb_box_train[train_index,:] = hbb_box
                elif train_index < train_size + test_size:
                    i = train_index - train_size
                    x_test[i,:,:,:] = x.astype('float32')
                    hbb_box_test[i,:] = hbb_box
                else:
                    break
                train_index += 1
                del x
                del hbb_box
        return x_train, hbb_box_train, x_test, hbb_box_test


def to_img_path(path):
    if path.startswith('car_ims/'):
        path = path[len('car_ims/'):]
    return img_path_prefix + path

bb_box_file = 'bb_box.txt'
img_paths_file = 'paths.txt'
img_path_prefix = 'C:/Users/Jacek/Documents/mgr_sem2/mgr/tmp_imgs/'

#Read the image paths
# image_paths = open(img_paths_file).read().split('\n')
# image_paths = [to_img_path(x) for x in image_paths]

#Read the bounding boxes
# boxes = open(bb_box_file).read().split('\n')

#Storing the bounding boxes in numpy array of the form x1,y1,x2,y2
# bb_boxes = np.zeros((len(boxes),4))
# for i in range(len(boxes)):
#     bb_boxes[i,:] = [int(x) for x in boxes[i].split(',')]


class GaussianCalc(object):

    def __init__(self, x1, y1, x2, y2, zero_val, sigma_x=None, sigma_y=None,sigma_factors = (6, 8)):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.zero_val = zero_val
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_factors = sigma_factors

    def f(self, u, v):
        x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2
        if x1 < v and v < x2 and y1 < u and u < y2:
            sigma_factors = self.sigma_factors
            sigma_x = ((x2 - x1) / sigma_factors[0])
            sigma_y = ((y2 - y1) / sigma_factors[1])
            _2_sigma_x = 2 * sigma_x * sigma_x
            _2_sigma_y = 2 * sigma_y * sigma_y
            def gauss_at(v_factor, u_factor):
                v1 = v - (x1 + x2) * v_factor
                u1 = u - (y1 + y2) * u_factor
                return math.exp(-((v1 * v1 / _2_sigma_x) + (u1 * u1 / _2_sigma_y)))
            result = gauss_at(0.5, 0.5)
            return min(1, result)
        return self.zero_val

def show_example2_3():
    loader = CnnDataLoader(CnnDirsConfig())
    x_train, hbb_box_train, x_test, hbb_box_test = loader.load_train_and_hbb_box_data(test_factor=0, w=128, h=128, max_examples=3, move_up=True, distractor_prob=1)
    index = 1
    io.imshow(x_train[index,:,:,:])
    io.show()

    heat_map = loader.create_heat_map_model6(hbb_box_train[index,:], 9, 9)

    io.imshow(heat_map)
    io.show()


def sync(fh):
    fh.flush()
    fsync(fh.fileno())

class Binary(object):

    def save_4_dim(self, arr, pth):
        with open(pth, 'wb+') as fh:
            fh.write(b'{0:s} {1:d} {2:d} {3:d} {4:d}\n'.format(arr.dtype, *arr.shape))
            fh.write(arr.data)
            sync(fh)

    def save_2_dim(self, arr, pth):
        with open(pth, 'wb+') as fh:
            fh.write(b'{0:s} {1:d} {2:d}\n'.format(arr.dtype, *arr.shape))
            fh.write(arr.data)
            sync(fh)

    def load_4_dim(self, pth):
        with open(pth, 'rb') as fh:
            dtype, w, h, a, b = str(fh.readline()).split()
            return np.frombuffer(fh.read(), dtype=dtype).reshape((int(w), int(h), int(a), int(b)))

    def load_2_dim(self, pth):
        with open(pth, 'rb') as fh:
            dtype, w, h = str(fh.readline()).split()
            return np.frombuffer(fh.read(), dtype=dtype).reshape((int(w), int(h)))

    def save_pack(self, pth, x_train, hbb_box_train, x_test, hbb_box_test):
        self.save_4_dim(x_train, pth+'.x_train')
        self.save_2_dim(hbb_box_train, pth+'.hbb_train')
        self.save_4_dim(x_test, pth+'.x_test')
        self.save_2_dim(hbb_box_test, pth+'.hbb_test')

    def load_pack(self, pth):
        x_train = self.load_4_dim(pth+'.x_train')
        hbb_box_train = self.load_2_dim(pth+'.hbb_train')
        x_test = self.load_4_dim(pth+'.x_test')
        hbb_box_test = self.load_2_dim(pth+'.hbb_test')
        return x_train, hbb_box_train, x_test, hbb_box_test

def build_examples_packet(examples, distractor_prob, filename):
    config = CnnDirsConfig()
    loader = CnnDataLoader(config)
    bin = Binary()
    x_train, hbb_box_train, x_test, hbb_box_test = loader.load_train_and_hbb_box_data(test_factor=1, w=128, h=128, max_examples=examples, move_up=True, distractor_prob=distractor_prob)
    bin.save_pack(config.data_filename(filename), x_train, hbb_box_train, x_test, hbb_box_test)

if __name__ == '__main__':
    print 'started'
    # show_example2_3()
    # config = CnnDirsConfig()
    # loader = CnnDataLoader(config)
    # bin = Binary()
    build_examples_packet(1000, 0, '1000img')

    # x_train, hbb_box_train, x_test, hbb_box_test = loader.load_train_and_hbb_box_data(test_factor=0.1, w=128, h=128, max_examples=100)
    # bin.save_pack(config.data_filename('100examples'), x_train, hbb_box_train, x_test, hbb_box_test)
    #
    # x_train, hbb_box_train, x_test, hbb_box_test = loader.load_train_and_hbb_box_data(test_factor=0.1, w=128, h=128, max_examples=1000, move_up=False)
    # bin.save_pack(config.data_filename('1000examples_down'), x_train, hbb_box_train, x_test, hbb_box_test)
    #
    # x_train, hbb_box_train, x_test, hbb_box_test = loader.load_train_and_hbb_box_data(test_factor=0.1, w=128, h=128, max_examples=5000)
    # bin.save_pack(config.data_filename('5000examples'), x_train, hbb_box_train, x_test, hbb_box_test)

    # x_train, hbb_box_train, x_test, hbb_box_test = loader.load_train_and_hbb_box_data(test_factor=0.07, w=128, h=128, max_examples=12000)
    # bin.save_pack(config.data_filename('12000examples'), x_train, hbb_box_train, x_test, hbb_box_test)

    # Load test:
    # x_train, hbb_box_train, x_test, hbb_box_test = bin.load_pack(config.data_filename('test_data'))
    # index = 1
    # io.imshow(x_train[index,:,:,:])
    # io.show()
    #
    # heat_map = loader.create_heat_map(hbb_box_train[index,:], 128, 128)
    #
    # io.imshow(heat_map)
    # io.show()

    # bin.save(x_train, 'x_train')
    # bin.save(y_train, 'y_train')
    # bin.save(x_test, 'x_test')
    # bin.save(y_test, 'y_test')
    #
    # xx = bin.load('x_train')
    # print xx.shape