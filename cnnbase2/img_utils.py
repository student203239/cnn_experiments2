
import numpy as np
import skimage.transform as tr


class ImgUtlis(object):

    def __init__(self):
        pass

    @staticmethod
    def resize_rgb_image(im, w, h):
        result = np.zeros((h, w, 3), dtype='float32')
        result[:,:,0] = tr.resize(im[:,:,0], (h,w))
        result[:,:,1] = tr.resize(im[:,:,1], (h,w))
        result[:,:,2] = tr.resize(im[:,:,2], (h,w))
        return result

    @staticmethod
    def make_img_square(im, move_up):
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