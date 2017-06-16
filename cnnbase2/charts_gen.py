# -*- coding: utf-8 -*-

from cnnbase2.img_utils import ImgUtlis
from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models_viewer import ModelsContainerExperiment1, ModelsContainerExperiment3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class ChartsGen():

    def __init__(self, models, filename_pattern="../charts/%s"):
        self.models = models
        self.model = self.models.get_init_model()
        self._prepare_pyplot()
        self.filename_pattern = filename_pattern
        print "constructed"

    def to_filename(self, filename):
        return self.filename_pattern % filename

    def nice_charts(self):
        for key in self.models.get_models_keys():
            self.nice_chart(key)

    def nice_chart(self, model_key):
        self.model = self.models.get_model(model_key, self.model)
        title_sufix = self.models.get_desc(model_key)
        errorsStatistics, diffs = self._create_errorsStatistics(self.model)
        model_short_name = self.models.get_short_letter(model_key)

        naming = {'xlabel': "Wartość błędu kwadratowego na pojedynczym pixelu",
                  'ylabel': "Częstość pixeli w zbiorze walidacyjnym",
                  'title': "Histogram błędów kwadratowych dla wszystkich pixeli w zbiorze walidacyjnym\n" + title_sufix}
        self._gen_and_save_plot_nice(diffs**2, "diffs2ylog", title_sufix, model_short_name, log_scale=False, log_scale_y=True, **naming)

        naming = {'xlabel': "Wartość precyzji na pojedynczym przykładzie",
                  'ylabel': "Częstość przykładu w zbiorze walidacyjnym",
                  'title': "Histogram wartości precyzji w zbiorze walidacyjnym\n" + title_sufix}
        self._gen_and_save_plot_nice(errorsStatistics.precision_list, "precision_list", title_sufix, model_short_name, **naming)

        naming = {'xlabel': "Liczba False Positive pojedynczym przykładzie",
                  'ylabel': "Częstość w zbiorze walidacyjnym",
                  'title': "Histogram False Positive w zbiorze walidacyjnym\n" + title_sufix}
        self._gen_and_save_plot_nice(errorsStatistics.type1_list, "False Positive", title_sufix, model_short_name, **naming)

    def alpha_chart(self):
        def mean_str(myarray):
            return str(myarray.sum()/float(len(myarray)))
        column_names = ['alpha']
        for model_key in self.models.get_models_keys():
            title_sufix = self.models.get_desc(model_key)
            column_names.append("błąd średniokwadratowy " + title_sufix)
            column_names.append("przecyzja " + title_sufix)
            column_names.append("liczba FalsePositive " + title_sufix)
            column_names.append("miara F1 " + title_sufix)
        print ";".join(column_names)

        for alpha_1 in xrange(1, 101):
            alpha = float(alpha_1) / 100.0
            values = [str(alpha)]
            for model_key in self.models.get_models_keys():
                self.model = self.models.get_model(model_key, self.model)
                errorsStatistics, diffs = self._create_errorsStatistics(self.model, alpha)
                values += [mean_str(diffs**2), mean_str(errorsStatistics.precision_list),
                           mean_str(errorsStatistics.type1_list),
                           mean_str(errorsStatistics.f_score_list)]
            print ";".join(values)

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

    def _gen_and_save_plot_nice(self, myarray, measure_name, title_sufix, model_short_name, xlabel, ylabel, title,
                                log_scale=False, log_scale_y=False):
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
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(self.to_filename("%s_hist_%s.png" % (measure_name.replace(" ", "_"), model_short_name)))

        plt.close()
        plt.close('all')


    def _prepare_pyplot(self):
        import sys
        reload(sys)
        sys.setdefaultencoding('utf8')
        matplotlib.rc('font', family='Arial')

    def _create_errorsStatistics(self, model, alpha=0.8):
        expects_src = model.y_test[:, :, :, 0]
        predicts_src = model.get_predicted_test()[:, 0, :, :]
        diffs = (expects_src - predicts_src).flatten()
        expects = ImgUtlis.alfa_cut_image(alpha, expects_src)
        predicts = ImgUtlis.alfa_cut_image(alpha, predicts_src)
        errorsStatistics = ImgUtlis.count_advance_errors(expects, predicts)
        return errorsStatistics, diffs

def charts_gen_experiment1_second_time():
    charts_gen = ChartsGen(ModelsContainerExperiment1(CnnDirsConfig(), base_filename='mayc10%s.june12.experiment1'),
                           "../charts/nices_experiement1_po_raz_2/po_raz2_%s")
    charts_gen.nice_charts()

def charts_gen_experiment3():
    charts_gen = ChartsGen(ModelsContainerExperiment3(CnnDirsConfig()),
                           "../charts/exp3/exp3_%s")
    charts_gen.nice_charts()

def alpha_table_experiement3():
    charts_gen = ChartsGen(ModelsContainerExperiment3(CnnDirsConfig()),
                           "../charts/exp3/exp3_%s")
    charts_gen.alpha_chart()

if __name__ == '__main__':
    # charts_gen_experiment3()
    charts_gen_experiment3()