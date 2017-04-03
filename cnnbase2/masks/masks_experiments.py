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

if __name__ == '__main__':
    print 'run masks experimests'
    horizontal_gradient_buffer = HorizontalGradientGen(100, 100).gen_buffer()
    io.imshow(horizontal_gradient_buffer)
    io.show()
    # vertical_gradient_buffer = horizontal_gradient_buffer.transpose((1,0)) # rot 90 degrees
    # io.imshow(vertical_gradient_buffer)
    # io.show()

    # io.imshow(QuaterRadialGradientGen(100, 100).gen_buffer())
    # io.show()
    # io.imshow(QuaterRadialGradientGen(100, 100).gen_buffer()[:,::-1])
    # io.show()