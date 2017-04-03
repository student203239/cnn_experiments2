import math
import numpy as np
import skimage.io as io
import skimage.transform as tr

class HorizontalGradientGen(object):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def gen_buffer(self):
        g = np.vectorize(self.f)
        return np.fromfunction(g, (self.h, self.w), dtype='float32')

    def f(self, u, v):
        return float(v*v) / self.w**2

class QuaterRadialGradientGen(object):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def gen_buffer(self):
        g = np.vectorize(self.f)
        return np.fromfunction(g, (self.h, self.w), dtype='float32')

    def f(self, u, v):
        x = u*u + v*v
        return float(x) / (self.w**2 + self.h**2)
        # return float(u+v) / (self.w + self.h)

class QuaterRadialGradientGen2(object):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def gen_buffer(self):
        g = np.vectorize(self.f)
        return np.fromfunction(g, (self.h, self.w), dtype='float32')

    def f(self, u, v):
        result_v = float(v * v) / self.w ** 2
        result_u = float(u * u) / self.h ** 2
        return result_v * result_u


class MarginExperiements1(object):

    def __init__(self, img_w, img_h):
        self.buffer = np.zeros((img_w, img_h), dtype='float32')
        self.radial = QuaterRadialGradientGen2(100, 100).gen_buffer()
        self.horizontal_gradient = HorizontalGradientGen(100, 100).gen_buffer()

    def create_margin(self, inner, outter):
        ox1, oy1, ox2, oy2 = outter
        self.buffer[oy1:oy2, ox1:ox2] = 0.2
        ix1, iy1, ix2, iy2 = inner
        self.buffer[iy1:iy2, ix1:ix2] = 1

        self.buffer[oy1:iy1, ox1:ix1] = tr.resize(self.radial, (iy1-oy1, ix1-ox1))
        self.buffer[iy2:oy2, ox1:ix1] = tr.resize(self.radial, (oy2-iy2, ix1-ox1))[::-1,:]
        self.buffer[iy2:oy2, ix2:ox2] = tr.resize(self.radial, (oy2-iy2, ox2-ix2))[::-1,::-1]
        self.buffer[oy1:iy1, ix2:ox2] = tr.resize(self.radial, (iy1-oy1, ox2-ix2))[:,::-1]

        self.buffer[iy1:iy2, ix2:ox2] = tr.resize(self.horizontal_gradient, (iy2-iy1, ox2-ix2))[:,::-1]
        self.buffer[iy1:iy2, ox1:ix1] = tr.resize(self.horizontal_gradient, (iy2-iy1, ix1-ox1))

        self.buffer[oy1:iy1, ix1:ix2] = tr.resize(self.horizontal_gradient, (ix2-ix1, iy1-oy1)).transpose((1,0))
        self.buffer[iy2:oy2, ix1:ix2] = tr.resize(self.horizontal_gradient, (ix2-ix1, oy2-iy2)).transpose((1,0))[::-1,:]

def main1():
    me = MarginExperiements1(400, 400)
    inner = 150, 150, 200, 250
    outter = 100, 100, 350, 350
    me.create_margin(inner, outter)
    io.imshow(me.buffer)
    io.show()

if __name__ == '__main__':
    main1()
    # print 'run masks experimests'
    # horizontal_gradient_buffer = HorizontalGradientGen(100, 100).gen_buffer()
    # io.imshow(horizontal_gradient_buffer)
    # io.show()

    # vertical_gradient_buffer = horizontal_gradient_buffer.transpose((1,0)) # rot 90 degrees
    # io.imshow(vertical_gradient_buffer)
    # io.show()

    # io.imshow(QuaterRadialGradientGen(100, 100).gen_buffer())
    # io.show()
    # io.imshow(QuaterRadialGradientGen(100, 100).gen_buffer()[:,::-1])
    # io.show()