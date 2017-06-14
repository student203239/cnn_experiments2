from cnnbase2.data_feeders.data_feeder_cnn_model_like import DataFeederCnnModelBaseLike
from cnnbase2.load_data import CnnDirsConfig
import numpy as np


class DummyFeedersMerge(object):
    
    def __init__(self, feeder1, feeder2):
        self.feeder1 = feeder1
        self.feeder2 = feeder2
        
    def prepare_data(self, output_shape):
        X_train1, X_test1, y_train1, y_test1 = self.feeder1.prepare_data(output_shape)
        X_train2, X_test2, y_train2, y_test2 = self.feeder2.prepare_data(output_shape)
        X_train, y_train = self._merge_data_arrays(X_train1, y_train1, X_train2, y_train2)
        X_test, y_test = self._merge_data_arrays(X_test1, y_test1, X_test2, y_test2)
        return X_train, X_test, y_train, y_test

    def _merge_data_arrays(self, x1, y1, x2, y2):
        print "_merge_data_arrays"
        print "x1 = " + str(x1.shape)  # (900L, 128L, 128L, 3L)
        print "y1 = " + str(y1.shape)  # (900L, 14L, 14L, 1L)
        print "x2 = " + str(x2.shape)  # (200L, 128L, 128L, 3L)
        print "y2 = " + str(y2.shape)  # (200L, 14L, 14L, 1L)
        
        x = np.zeros(self._shapes_add(x1, x2), dtype=x1.dtype)
        x[:x1.shape[0], :,:,:] = x1
        x[x1.shape[0]:, :,:,:] = x2

        y = np.zeros(self._shapes_add(y1, y2, True), dtype=y1.dtype)
        y[:y1.shape[0], :,:,0:0] = y1
        y[y1.shape[0]:, :,:,1:1] = y2

        return x, y
    
    def _shapes_add(self, x1, x2, y=False):
        a1, b1, c1, d1 = x1.shape
        a2, b2, c2, d2 = x2.shape
        assert b1 == b2 and "Nie mozna konkatenowac roznych ksztaltow"
        assert c1 == c2 and "Nie mozna konkatenowac roznych ksztaltow"
        assert d1 == d2 and "Nie mozna konkatenowac roznych ksztaltow"
        if y:
            assert d1 == 1
            d1 = 2
        return a1+a2, b1, c1, d1

if __name__ == '__main__':
    print "TEST DUMMY MERGE DATA"
    config = CnnDirsConfig()
    feeder1 = DataFeederCnnModelBaseLike(config, '1000examples').init_car_type(smaller_car=False)
    feeder2 = DataFeederCnnModelBaseLike(config, 'flic.small.shuffle.code10').init_human_type(y_gen_mode='r')

    merger = DummyFeedersMerge(feeder1, feeder2)

    output_shape = (14, 14)
    X_train, X_test, y_train, y_test = merger.prepare_data(output_shape)
    print "ok"