import math
import numpy as np
import skimage.io as io
import skimage.transform as tr
import time


class SmallMaskGen(object):

    def __init__(self, img_w, img_h):
        self.img_w = img_w
        self.img_h = img_h

    def create_margin(self, inner, outter, buffer=None):
        if buffer == None:
            buffer = self.create_buffer()
        ox1, oy1, ox2, oy2 = outter
        ix1, iy1, ix2, iy2 = inner
        # buffer[oy1:oy2, ox1:ox2] = 0.2
        for y in range(int(oy1+0.5), int(oy2+0.5)):
            for x in range(int(ox1+0.5), int(ox2+0.5)):
                val = 1.0
                if y <= iy1 and oy1+1 <= iy1:
                    val *= float(y-oy1)**2 / (iy1-oy1)**2
                if y >= iy2 and oy2 >= iy2+1:
                    val *= float(oy2-y)**2 / (iy2-oy2)**2
                if x <= ix1 and ox1+1 <= ix1:
                    val *= float(x-ox1)**2 / (ix1-ox1)**2
                if x >= ix2 and ox2 >= ix2+1:
                    val *= float(ox2-x)**2 / (ix2-ox2)**2
                buffer[y, x] = val

        # buffer[iy1:iy2, ix1:ix2] = 1

        return buffer

    def create_buffer(self):
        return np.zeros((self.img_w, self.img_h), dtype='float32')

    @staticmethod
    def hbb_box_to_y(src_y, output_shape, bigger_factor=2, output_y=None):
        w, h = output_shape
        examples = src_y.shape[0]
        y = output_y
        if y is None:
            y = np.zeros((examples,w,h,1), dtype='float32')
        gen = SmallMaskGen(w*bigger_factor, h*bigger_factor)
        buffer = gen.create_buffer()
        for i in range(examples):
            inner, outter = SmallMaskGen.get_heat_map_loc(src_y[i], w*bigger_factor, h*bigger_factor)
            buffer[:,:] = 0.0
            gen.create_margin(inner, outter, buffer)
            y[i,:,:,0] = tr.resize(buffer, (h, w))
        return y

    @staticmethod
    def hbb_box_to_y_only_inner(src_y, output_shape, output_y=None):
        w, h = output_shape
        examples = src_y.shape[0]
        y = output_y
        if y is None:
            y = np.zeros((examples,w,h,1), dtype='float32')
        for i in range(examples):
            inner, outter = SmallMaskGen.get_heat_map_loc(src_y[i], w, h)
            x1, y1, x2, y2 = inner
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            y[i,:,:,0] = 0
            y[i,y1:y2,x1:x2,0] = 1
        return y

    @staticmethod
    def hbb_box_to_y_only_outter(src_y, output_shape, output_y=None):
        w, h = output_shape
        examples = src_y.shape[0]
        y = output_y
        if y is None:
            y = np.zeros((examples,w,h,1), dtype='float32')
        for i in range(examples):
            inner, outter = SmallMaskGen.get_heat_map_loc(src_y[i], w, h)
            x1, y1, x2, y2 = outter
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            y[i,:,:,0] = 0
            y[i,y1:y2,x1:x2,0] = 1
        return y

    @staticmethod
    def get_heat_map_loc(hbb_box, return_w, return_h):
        ix1, iy1, ix2, iy2, w, h, ox1, oy1, ox2, oy2 = hbb_box
        rr = min(1.0*return_h/h, 1.0*return_w/w)
        inner = ix1*rr, iy1*rr, ix2*rr, iy2*rr
        outter = ox1*rr, oy1*rr, ox2*rr, oy2*rr
        return inner, outter

    @staticmethod
    def code_10(inner, outter, w, h):
        ox1, oy1, ox2, oy2 = outter
        ix1, iy1, ix2, iy2 = inner
        return ix1, iy1, ix2, iy2, w, h, ox1, oy1, ox2, oy2

    @staticmethod
    def from_code_10(pack_code_10):
        ix1, iy1, ix2, iy2, w, h, ox1, oy1, ox2, oy2 = pack_code_10
        ox1, oy1, ox2, oy2 = outter
        ix1, iy1, ix2, iy2 = inner
        return inner, outter, w, h

if __name__ == '__main__':
    f=20
    start = time.time()
    me = SmallMaskGen(14*f, 14*f)
    inner = 5*f, 5*f, 8*f, 7*f
    outter = 5*f, 4*f, 9*f, 9*f

    # me = SmallMaskGen(100*f, 100*f)
    # inner = 30*f, 40*f, 65*f, 54*f
    # outter = 20*f, 20*f, 80*f, 70*f

    buffer = me.create_margin(inner, outter)
    if f > 1:
        buffer = tr.resize(buffer, (14, 14))
    print "Took: {}" .format(time.time() - start)
    io.imshow(buffer)
    io.show()