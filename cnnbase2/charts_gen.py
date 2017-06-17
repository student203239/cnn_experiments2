# -*- coding: utf-8 -*-
from cnnbase2.data_feeders.data_feeder_cnn_model_like import DataFeederCnnModelBaseLike
from cnnbase2.img_utils import ImgUtlis
from cnnbase2.learns.learn_double_layer import Experiment4ModelContainer
from cnnbase2.load_data import CnnDirsConfig
from cnnbase2.models2 import TinyAlexNet4
from cnnbase2.models_viewer import ModelsContainerExperiment1, ModelsContainerExperiment3, ModelsConatiner
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class ChartsGen():

    def __init__(self, models, filename_pattern="../charts/%s"):
        self.models = models
        self.models.prepare_models_to_view()
        self.model = self.models.get_init_model()
        self._prepare_pyplot()
        self.filename_pattern = filename_pattern
        self.output_layer = 0
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

        for alpha_1 in [5, 80]:
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

    def _create_errorsStatistics(self, model, alpha=0.05, select_expect_callable=None):
        expects_src = model.y_test[:, :, :, self.output_layer]
        predicts_src = model.get_predicted_test()[:, self.output_layer, :, :]
        assert expects_src.shape[0] == predicts_src.shape[0]
        assert expects_src.shape[1] == predicts_src.shape[1]
        assert expects_src.shape[2] == predicts_src.shape[2]
        if select_expect_callable is not None:
            expects_src, predicts_src = self._remove_imgs_from_expects(expects_src, predicts_src,
                                                                   keep_callable=select_expect_callable)
        diffs = (expects_src - predicts_src).flatten()
        expects = ImgUtlis.alfa_cut_image(alpha, expects_src)
        predicts = ImgUtlis.alfa_cut_image(alpha, predicts_src)
        errorsStatistics = ImgUtlis.count_advance_errors(expects, predicts)
        return errorsStatistics, diffs

    def _remove_imgs_from_expects(self, expects_src, predicts_src, keep_callable):
        len = expects_src.shape[0]
        nonremoved = 0
        for i in xrange(len):
            if keep_callable(expects_src[i,:,:]):
                nonremoved += 1
        if nonremoved == len:
            return expects_src, predicts_src
        expects_src_new = np.zeros((nonremoved, expects_src.shape[1], expects_src.shape[2]), dtype=expects_src.dtype)
        predicts_src_new = np.zeros((nonremoved, predicts_src.shape[1], predicts_src.shape[2]), dtype=predicts_src.dtype)
        new_index = 0
        for i in xrange(len):
            if keep_callable(expects_src[i,:,:]):
                expects_src_new[new_index,:,:] = expects_src[i,:,:]
                predicts_src_new[new_index,:,:] = predicts_src[i,:,:]
                new_index += 1
        return expects_src_new, predicts_src_new

    def evaluate_double_layer_model_experiement4(self):
        def mean_str(myarray):
            return str(myarray.sum()/float(len(myarray)))
        for output_layer_proccessing in [0, 1]:
            for expect_zeros in [0, 1]:
                self.output_layer = output_layer_proccessing
                if expect_zeros == 1:
                    select_callable = lambda a: a.max() == 0
                else:
                    select_callable = lambda a: a.max() > 0
                for alpha in [0.05, 0.8]:
                    print "Badam DL" + str(output_layer_proccessing) + " przy alfa = " + str(alpha)
                    if expect_zeros == 1:
                        print "Oczekuje nic (zeros) na wyjściu"
                    else:
                        print "Oczekuje zdjęć " + ["aut", "osób"][output_layer_proccessing] + " na wyjściu"
                    errorsStatistics, diffs = self._create_errorsStatistics(self.model, alpha, select_expect_callable=select_callable)
                    print "diff2 = " + mean_str(diffs**2)
                    print "precision = " + mean_str(errorsStatistics.precision_list)
                    print "False Positive = " + mean_str(errorsStatistics.type1_list)
                    print ""

def charts_gen_experiment1_second_time():
    charts_gen = ChartsGen(ModelsContainerExperiment1(CnnDirsConfig(), base_filename='mayc10%s.june12.experiment1'),
                           "../charts/nices_experiement1_po_raz_2/po_raz2_%s")
    charts_gen.nice_charts()

def charts_gen_experiment3():
    charts_gen = ChartsGen(ModelsContainerExperiment3(CnnDirsConfig()),
                           "../charts/exp3/exp3_%s")
    charts_gen.nice_charts()

def charts_gen_experiment4():
    charts_gen = ChartsGen(Experiment4ModelContainer(CnnDirsConfig(), load_train=False),
                           "../charts/exp4/exp4_%s")
    charts_gen.nice_charts()

def alpha_table_experiement3():
    charts_gen = ChartsGen(ModelsContainerExperiment3(CnnDirsConfig()),
                           "../charts/exp3/exp3_%s")
    charts_gen.alpha_chart()

def table_experiement4():
    charts_gen = ChartsGen(Experiment4ModelContainer(CnnDirsConfig(), load_train=False),
                           "../charts/exp4/exp4_%s")
    charts_gen.evaluate_double_layer_model_experiement4()

def table_experiement4_with_prev_models():
    config = CnnDirsConfig()
    base_filename='mayc10%s.june12.experiment1'
    filenames = [base_filename % ch for ch in ['r', 'i', 'o']]
    human_feeder = DataFeederCnnModelBaseLike(config, 'flic.shuffle.code10', load_train=False).init_human_type('r')
    human_model = TinyAlexNet4(config, None, filenames[0],
                               prepared_data=human_feeder)

    filename = "june12.experiment3"
    car_feeder = DataFeederCnnModelBaseLike(config, '5000examples', load_train=False).init_car_type(True)
    car_model = TinyAlexNet4(config, None, filename, smaller_car = True, prepared_data=car_feeder)

    models_dict = {'human_model': human_model, 'car_model': car_model}

    models_conatiner = ModelsConatiner(models_dict)
    models_conatiner.init_model_key = 'human_model'
    models_conatiner.get_desc = lambda model_key:  {'human_model': "dla rozpoznawania tylko ludzi",
                                                    'car_model': "dla rozpoznawania tylko aut"}[model_key]
    charts_gen = ChartsGen(models_conatiner,
                           "../charts/exp4/exp4cmp_%s")
    charts_gen.alpha_chart()

if __name__ == '__main__':
    # charts_gen_experiment3()
    # table_experiement4()
    table_experiement4_with_prev_models()