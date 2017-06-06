
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

    @staticmethod
    def alfa_cut_image(alfa, img):
        result = img > float(alfa)
        return result.astype(int)

    @staticmethod
    def count_advance_error(expected, predicted):
        difss = expected - predicted
        bads = np.count_nonzero(difss)
        goods_and_type1 = np.count_nonzero(difss - 1)
        alls = 1
        for i in range(expected.ndim):
            alls *= expected.shape[i]
        type2 = alls - goods_and_type1
        type1 = bads - type2
        goods = goods_and_type1 - type1
        return type1, type2, goods, alls

    @staticmethod
    def count_advance_errors(expects, predicts):
        samples = expects.shape[0]
        errorsStatistics = ErrorsStatistics(samples)
        for s in range(samples):
            type1, type2, goods, alls = ImgUtlis.count_advance_error(expects[s, :, :], predicts[s, :, :])
            predicted_ones = np.count_nonzero(predicts[s, :, :])
            errorsStatistics._log_result(s, type1, type2, goods, predicted_ones)
        return errorsStatistics


class ErrorsStatistics(object):
    def __init__(self, samples):
        self.samples = samples
        self.type1_list = np.zeros((samples), dtype=int)
        self.type2_list = np.zeros((samples), dtype=int)
        self.goods_list = np.zeros((samples), dtype=int)
        self.precision_list = np.zeros((samples), dtype=int)
        self.recall_list = np.zeros((samples), dtype=int)
        self.f_score_list = np.zeros((samples), dtype=int)
        self.stats = {'type1_list': self.type1_list, 'type2_list': self.type2_list,
                      'goods_list': self.goods_list, 'precision_list': self.precision_list,
                      'recall_list': self.recall_list, 'f_score_list': self.f_score_list}

    def _log_result(self, sample_index, error_type1, error_type2, goods, predicted_ones):
        tp = predicted_ones - error_type1
        precision, recall, f_score = 0, 0, 0
        try:
            precision = tp / predicted_ones
            recall = tp / (tp + error_type2)
            f_score = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            pass
        self.type1_list[sample_index] = error_type1
        self.type2_list[sample_index] = error_type2
        self.goods_list[sample_index] = goods
        self.precision_list[sample_index] = precision
        self.recall_list[sample_index] = recall
        self.f_score_list[sample_index] = f_score

if __name__ == '__main__':
    # ok, type2, ok, type1
    expected =  np.array([1, 0, 1, 0])
    predicted = np.array([1, 1, 0, 0])

    # type1, type2, ok, ok
    expected =  np.array([[0, 1], [1, 0]])
    predicted = np.array([[1, 0], [1, 0]])
    predicted = ImgUtlis.alfa_cut_image(0.5, predicted)
    difss = expected - predicted
    bads = np.count_nonzero(difss)
    goods_and_type1 = np.count_nonzero(difss - 1)
    alls = 1
    for i in range(expected.ndim):
        alls *= expected.shape[i]
    type2 = alls - goods_and_type1
    type1 = bads - type2
    goods = goods_and_type1 - type1
    print type1, type2, goods, alls