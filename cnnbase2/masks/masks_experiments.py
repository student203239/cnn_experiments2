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

        self._paste_into(oy1, iy1, ox1, ix1, self.radial)
        self._paste_into(iy2, oy2, ox1, ix1, self.radial, lambda a: a[::-1,:])
        self._paste_into(iy2, oy2, ix2-1, ox2, self.radial, lambda a: a[::-1,::-1])  # needs workaround
        self._paste_into(oy1, iy1, ix2-1, ox2, self.radial, lambda a: a[:,::-1])  # needs workaround

        self._paste_into(iy1, iy2, ix2-1, ox2, self.horizontal_gradient, lambda a: a[:,::-1])  # needs workaround
        self._paste_into(iy1, iy2, ox1, ix1, self.horizontal_gradient)
        self._paste_into(oy1, iy1, ix1, ix2, self.horizontal_gradient, trans=True)
        self._paste_into(iy2, oy2, ix1, ix2, self.horizontal_gradient, lambda a: a[::-1,:], trans=True)

        self.buffer[iy1:iy2+1, ix1:ix2+1] = 1

    def _paste_into(self, y1, y2, x1, x2, img, post_lambda=None, trans=False):
        if y1 == y2 or x1 == x2:
            return
        target_shape = (y2 - y1, x2 - x1)
        if trans:
            target_shape = (x2 - x1, y2 - y1)
        tmp = tr.resize(img, target_shape)
        if trans:
            tmp = tmp.transpose((1,0))
        if post_lambda:
            tmp = post_lambda(tmp)
        self.buffer[y1:y2, x1:x2] = tmp

def main1():
    me = MarginExperiements1(400, 400)
    inner = 150, 150, 200, 250
    outter = 100, 100, 350, 350
    me.create_margin(inner, outter)
    io.imshow(me.buffer)
    io.show()

def main2():
    f=1
    me = MarginExperiements1(14*f, 14*f)
    inner = 5*f, 5*f, 8*f, 7*f
    outter = 5*f, 4*f, 9*f, 9*f
    me.create_margin(inner, outter)
    buffer = me.buffer
    if f > 1:
        buffer = tr.resize(buffer, (14, 14))
    io.imshow(buffer)
    io.show()

if __name__ == '__main__':
    main2()
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