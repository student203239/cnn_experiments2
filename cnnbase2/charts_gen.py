# -*- coding: utf-8 -*-

from cnnbase2.img_utils import ImgUtlis
from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models_viewer import ModelsContainerExperiment1
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class ChartsGen():

    def __init__(self):
        self.models = ModelsContainerExperiment1(CnnDirsConfig())
        self.model = self.models.get_init_model()
        self._prepare_pyplot()
        print "constructed"

    def to_filename(self, filename):
        return "../charts/%s" % filename

    def f1_charts(self):
        for key in self.models.get_models_keys():
            self.f1_chart(key)

    def f1_chart(self, model_key):
        self.model = self.models.get_model(model_key, self.model)
        title_sufix = self.models.get_desc(model_key)
        errorsStatistics, diffs = self._create_errorsStatistics(self.model)
        # plt.hist(errorsStatistics.f_score_list, normed=True)

        # plt.hist(errorsStatistics.type2_list, normed=True)
        # plt.xlabel("Liczba błędów typu 2 na pojedynczym przykładzie")
        # plt.ylabel("Częstość w zbiorze walidacyjnym")
        # plt.title("Histogram wystąpień błędów typu 2" + title_sufix)
        # plt.savefig(self.to_filename("error_type_2_o.png"))
        # myarray = errorsStatistics.type1_list
        # weights = np.ones_like(myarray)/float(len(myarray))
        # plt.hist(myarray, weights=weights)
        # plt.xlabel("Wynik error 1 score na pojedynczym przykładzie")
        # plt.ylabel("Częstość w zbiorze walidacyjnym")
        # plt.title("Histogram wartości F1 score\n" + title_sufix)
        # plt.savefig(self.to_filename("e1_hist_%s.png" % self.models.get_short_letter(model_key)))
        # plt.close()
        # plt.close('all')
        model_short_name = self.models.get_short_letter(model_key)

        self._gen_and_save_plot(errorsStatistics.f_score_list, "F1 score", title_sufix, model_short_name)
        self._gen_and_save_plot(errorsStatistics.type1_list, "False Positive", title_sufix, model_short_name)
        self._gen_and_save_plot(errorsStatistics.type2_list, "False Negative", title_sufix, model_short_name)
        self._gen_and_save_plot(errorsStatistics.tp_list, "True Positive", title_sufix, model_short_name)
        self._gen_and_save_plot(errorsStatistics.tn_list, "True Negative", title_sufix, model_short_name)
        self._gen_and_save_plot(errorsStatistics.goods_list, "Corrects predicted", title_sufix, model_short_name)
        self._gen_and_save_plot(errorsStatistics.precision_list, "precision_list", title_sufix, model_short_name)
        self._gen_and_save_plot(errorsStatistics.recall_list, "recall_list", title_sufix, model_short_name)
        self._gen_and_save_plot(diffs, "diffs", title_sufix, model_short_name)
        self._gen_and_save_plot(diffs > 0.1, "diffs_cut0.1", title_sufix, model_short_name)
        self._gen_and_save_plot(diffs, "diffsylog", title_sufix, model_short_name, log_scale_y=True)
        self._gen_and_save_plot(diffs > 0.1, "diffsylog_cut0.1", title_sufix, model_short_name, log_scale_y=True)
        self._gen_and_save_plot(diffs**2, "diffs2", title_sufix, model_short_name)
        self._gen_and_save_plot((diffs**2) > 0.1, "diffs2_cut0.1", title_sufix, model_short_name)
        # self._gen_and_save_plot(diffs**2, "diffs2xlog", title_sufix, model_short_name, log_scale=True)
        # self._gen_and_save_plot(diffs**2, "diffs2xlogylog", title_sufix, model_short_name, log_scale=True, log_scale_y=True)
        self._gen_and_save_plot(diffs**2, "diffs2ylog", title_sufix, model_short_name, log_scale=False, log_scale_y=True)
        self._gen_and_save_plot((diffs**2) > 0.1, "diffs2ylog_cut0.1", title_sufix, model_short_name, log_scale=False, log_scale_y=True)

        # myarray = errorsStatistics.type2_list
        # weights = np.ones_like(myarray)/float(len(myarray))
        # plt.hist(myarray, weights=weights)
        # plt.xlabel("Wynik error 2 score na pojedynczym przykładzie")
        # plt.ylabel("Częstość w zbiorze walidacyjnym")
        # plt.title("Histogram wartości F1 score\n" + title_sufix)
        # plt.savefig(self.to_filename("e2_hist_%s.png" % self.models.get_short_letter(model_key)))
        # plt.close()
        # plt.close('all')

    def _gen_and_save_plot(self, myarray, measure_name, title_sufix, model_short_name, log_scale=False, log_scale_y=False):
        mean = myarray.sum()/float(len(myarray))
        print "%s for %s = %s" % (measure_name, title_sufix, str(mean))
        weights = np.ones_like(myarray)/float(len(myarray))
        if log_scale:
            logspace = np.logspace(0.0, 1.0, 20) / 10.0
            plt.hist(myarray, weights=weights, bins=logspace)
            plt.gca().set_xscale("log")
        else:
            plt.hist(myarray, weights=weights)
        if log_scale_y:
            plt.gca().set_yscale("log")
        plt.xlabel("Wynik %s score na pojedynczym przykładzie" % measure_name)
        plt.ylabel("Częstość w zbiorze walidacyjnym")
        plt.title("Histogram wartości %s\n%s" % (measure_name, title_sufix))
        plt.savefig(self.to_filename("%s_hist_%s.png" % (measure_name.replace(" ", "_"), model_short_name)))

        plt.close()
        plt.close('all')


    def _prepare_pyplot(self):
        import sys
        reload(sys)
        sys.setdefaultencoding('utf8')
        matplotlib.rc('font', family='Arial')

    def _create_errorsStatistics(self, model):
        expects_src = model.y_test[:, :, :, 0]
        predicts_src = model.get_predicted_test()[:, 0, :, :]
        diffs = (expects_src - predicts_src).flatten()
        expects = ImgUtlis.alfa_cut_image(0.05, expects_src)
        predicts = ImgUtlis.alfa_cut_image(0.05, predicts_src)
        errorsStatistics = ImgUtlis.count_advance_errors(expects, predicts)
        return errorsStatistics, diffs


if __name__ == '__main__':
    charts_gen = ChartsGen()
    charts_gen.f1_charts()