"""
Module for plotting the data
"""
import logging
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_model import PositionPrediction
from utils import TestResult, get_protein_files_dict

LOG_ENABLED = True
log = print
if LOG_ENABLED:
    # noinspection PyRedeclaration
    log = logging.info


class PlotCreator(object):
    """
    Contains function to plot various aspects of data
    """

    def create_train_plots(self, report_path, test_res):
        """
        Creates all plots for train data
        @param report_path:
        @param test_res:
        @return:
        """
        plots_path = os.path.join(report_path, 'plots_train')
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)
        self.create_correlation_plot(plots_path, test_res)
        self.create_correlation_per_protein_plots(plots_path, test_res)

        # # --- this is very heavy operation --- # #
        # self.create_deltas_versus_plot(plots_path, test_res)
        # self.create_all_deltas_plot(plots_path, test_res)

    def create_valid_plots(self, report_path, test_res):
        """
        Creates all plots for validation data
        @param report_path:
        @param test_res:
        @return:
        """
        plots_path = os.path.join(report_path, 'plots_valid')
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)
        self.create_correlation_plot(plots_path, test_res)
        self.create_deltas_versus_plot(plots_path, test_res)
        self.create_all_deltas_plot(plots_path, test_res)

    def create_all_deltas_plot(self, report_path, test_res):
        """
        Create plot of delta score vs delta rank
        X: Delta (score) = true - pred
        Y: Delta (rank) = true - pred
        @param report_path: path to report folder - where to store the plot files
        @param test_res: test result object
        """
        plt.clf()
        pos_predictions = self._create_position_predictions(test_res)
        self._fill_ranks(pos_predictions)
        xs = [x.get_score_delta() for x in pos_predictions]
        ys = [x.get_rank_delta() for x in pos_predictions]
        assert len(xs) == len(ys)

        df = pd.DataFrame(columns=['xs', 'ys'])
        df['xs'] = xs
        df['ys'] = ys
        sns.scatterplot(data=df, x="xs", y="ys", marker=".")
        plt.xlabel("Delta (score) = true - pred")
        plt.ylabel("Delta (rank) = true - pred")
        plt.grid()
        plot_path = os.path.join(report_path, f'{test_res.model_name}.pos.versus.mae.plot.png')
        plt.savefig(plot_path)
        # plt.show()
        log(f'Created plt: {plot_path}')

    def create_position_deltas_plot(self, report_path, test_res):
        """
        Create plot:
        X: Position
        Y: Delta (rank) = true - pred
        @param report_path: path to report folder - where to store the plot files
        @param test_res: test result object
        """
        plt.clf()
        pos_predictions = self._create_position_predictions(test_res)
        self._fill_ranks(pos_predictions)

        pos_predictions.sort(key=lambda x: x.index)
        xs = [x.index for x in pos_predictions]
        ys = [x.get_rank_delta() for x in pos_predictions]

        assert len(xs) == len(ys)
        plt.plot(xs, ys, '.', color='black')
        plt.xlabel("Position")
        plt.ylabel("Delta (rank) = true - pred")
        plt.grid()
        plot_path = os.path.join(report_path, f'{test_res.model_name}.delta.position.plot.png')
        plt.savefig(plot_path)
        # plt.show()
        log(f'Created plt: {plot_path}')

    @staticmethod
    def create_fine_tune_variants_plot(report_path, file_name, title, mutations_counts, mutations_results):
        """
        Create plot for fine-tuning:
        X - variant count
        Y - MAE
        """
        plt.clf()
        xs = [int(x) for x in mutations_counts]
        ys = [res.test_result.mae for res in mutations_results]
        assert len(xs) == len(ys)
        column_names = ['Count', 'MAE']
        df = pd.DataFrame(columns=column_names)
        df['Count'] = xs
        df['MAE'] = ys
        sns.lineplot(data=df, x="Count", y="MAE", marker='o', markers=True).set_title(f'{title}')
        for x, y in zip(xs, ys):
            plt.text(x=x - 1, y=y + 0.0005, s=f'{y:.4f}')
        plt.grid()

        plot_path = os.path.join(report_path, f'{file_name}.result.plot.png')
        plt.savefig(plot_path)
        # plt.show()
        log(f'Created plt: {plot_path}')

    @staticmethod
    def create_correlation_per_protein_plots(report_path, test_res):
        """
        Creates plots 'Truth vs. Prediction' per protein
        Calculates legend (mae, pearson, spearman, r2) per protein
        @param report_path: folder to store the plots
        @param test_res: test result object
        """

        all_pids = set(list(test_res.pid_values))
        zipped_data = list(zip(test_res.pid_values, test_res.true_values, test_res.pred_values))
        # now data in a format (pid, true val, pred val)
        for pid in all_pids:
            plt.clf()
            per_pid_data = [d for d in zipped_data if d[0] == pid]
            protein_name = get_protein_files_dict()[int(pid)]
            ys = [d[1] for d in per_pid_data]  # all true values
            xs = [d[2] for d in per_pid_data]  # all pred values
            assert len(xs) == len(ys)

            column_names = ['Prediction', 'Truth']
            df = pd.DataFrame(columns=column_names)
            df['Prediction'] = xs
            df['Truth'] = ys

            plt.xlim(0, 1.1)
            plt.ylim(0, 1.1)

            g = sns.scatterplot(data=df, x="Prediction", y="Truth", marker=".")
            # calculate the legend for specific protein
            mse = np.round(mean_squared_error(ys, xs), 4)
            mae = np.round(mean_absolute_error(ys, xs), 4)
            # mape = np.round(mean_absolute_percentage_error(ys, xs), 4)
            pearson = np.round(pearsonr(ys, xs), 4)[0]
            spearman = np.round(spearmanr(ys, xs), 4)[0]
            r2 = np.round(r2_score(ys, xs), 4)
            per_protein_test_res = TestResult()
            per_protein_test_res.mse = mse
            per_protein_test_res.mae = mae
            # per_protein_test_res.mape = mape
            per_protein_test_res.pearson = pearson
            per_protein_test_res.spearman = spearman
            per_protein_test_res.r2 = r2
            legend_str = f'{per_protein_test_res.get_legend()}'
            g.legend([f'{legend_str}'], loc='best')
            plt.grid()
            plt.title(protein_name)

            plot_path = os.path.join(report_path, f'{test_res.model_name}.pid_{int(pid)}.result.plot.png')
            plt.savefig(plot_path)
            # plt.show()
            log(f'Created plt: {plot_path} for {protein_name}')

    @staticmethod
    def create_correlation_plot(report_path, test_res, file_name=None, title=None):
        """
        Creates plot of predicted values vs. true values for scores.
        Stores a plot in a report folder.
        @param title:  plot title
        @param file_name: file to be created
        @param report_path: path to report folder - where to store the plot files
        @param test_res: test result object
        """
        plt.clf()

        xs = test_res.pred_values
        ys = test_res.true_values
        assert len(xs) == len(ys)

        column_names = ['Prediction', 'Truth']
        df = pd.DataFrame(columns=column_names)
        df['Prediction'] = xs
        df['Truth'] = ys

        plt.xlim(0, 1.1)
        plt.ylim(0, 1.1)

        g = sns.scatterplot(data=df, x="Prediction", y="Truth", marker=".")
        if file_name is None:
            plot_path = os.path.join(report_path, f'{test_res.model_name}.result.plot.png')
        else:
            plot_path = os.path.join(report_path, file_name)
        # g.legend([f'{test_res.get_legend()}'], loc='best')
        g.legend([f'{test_res.get_short_legend()}'], loc='best')

        if title is not None:
            plt.title(title)

        plt.grid()
        plt.savefig(plot_path)

        # --- create CSv for DEBUG ---
        # if file_name is None:
        #     csv_path = os.path.join(report_path, f'{test_res.model_name}.result.csv')
        #     df.to_csv(csv_path, sep='\t')

        # plt.show()
        log(f'Created plt: {plot_path}')

    @staticmethod
    def create_deltas_plot(report_path, test_res):
        """
        Creates plot of abs. deltas between true and predicted scores per position in sequence.
        Stores a plot in a report folder.
        @param report_path: path to report folder - where to store the plot files
        @param test_res: test result object
        @param figure: index for plot figure
        """
        plt.clf()
        assert len(test_res.pred_values) == len(test_res.true_values)
        assert len(test_res.true_values) == len(test_res.pos_values)
        ds = np.absolute(test_res.pred_values - test_res.true_values)
        xs = test_res.pos_values
        plt.scatter(xs, ds, marker='.', color='black')
        plt.xlabel("Position")
        plt.ylabel("Delta (score)")
        plt.grid()
        plot_path = os.path.join(report_path, f'{test_res.model_name}.deltas.plot.png')
        plt.savefig(plot_path)
        # plt.show()
        log(f'Created plt: {plot_path}')

    @staticmethod
    def create_deltas_versus_plot(report_path, test_res):
        """
        Create plot
        x axis: position
        y axis: abs. delta score difference
        @param report_path: path to report folder - where to store the plot files
        @param test_res: test result object
        """
        plt.clf()
        assert len(test_res.pred_values) == len(test_res.true_values)
        assert len(test_res.true_values) == len(test_res.pos_values)
        # ds = np.absolute(test_res.pred_values - test_res.true_values)
        ds = np.subtract(test_res.true_values, test_res.pred_values)
        xs = test_res.true_values

        df = pd.DataFrame(columns=['Truth', 'Delta'])
        df['Truth'] = xs
        df['Delta'] = ds
        sns.scatterplot(data=df, x="Truth", y="Delta", marker=".")
        plt.grid()
        plt.xlabel("True score")
        plt.ylabel("Delta (score) = truth - pred")
        plot_path = os.path.join(report_path, f'{test_res.model_name}.deltas.versus.plot.png')
        plt.savefig(plot_path)
        # plt.show()
        print(f'Created plt: {plot_path}')

    @staticmethod
    def create_hoie_vs_deltas_plot(hoie_result, report_path, test_res):
        """
        Create plot
        x axis: "Position"
        y axis: "Delta = Hoie - Ours"
        @param hoie_result: result from Hoie
        @param report_path: path to report folder - where to store the plot files
        @param test_res: test result object
        """
        plt.clf()
        hoie_pred = [x.pred_score for x in hoie_result]
        our_pred = test_res.pred_values
        hoie_pred_arr = np.array(hoie_pred)
        our_pred_arr = np.array(our_pred)
        deltas_array = np.subtract(hoie_pred_arr, our_pred_arr)  # hoie - ours
        deltas = list(deltas_array)
        positions = [x.index for x in hoie_result]
        xs = positions
        ys = deltas
        assert len(xs) == len(ys)
        plt.plot(xs, ys, '.', color='black')
        plt.xlabel("Position")
        plt.ylabel("Delta = Hoie - Ours")
        plt.grid()
        plot_path = os.path.join(report_path, f'{test_res.model_name}.vs.hoie.deltas.plot.png')
        plt.savefig(plot_path)
        # plt.show()
        log(f'Created plt: {plot_path}')

    @staticmethod
    def create_hoie_vs_values_plot(hoie_result, report_path, test_res):
        """
        Create plot
        x axis: Our prediction score
        y axis: Hoie prediction score
        @param hoie_result: result from Hoie
        @param report_path: path to report folder - where to store the plot files
        @param test_res: test result object
        """
        plt.clf()
        hoie_pred = [x.pred_score for x in hoie_result]
        our_pred = test_res.pred_values
        xs = our_pred
        ys = hoie_pred
        assert len(xs) == len(ys)

        df = pd.DataFrame(columns=['Our', 'Hoie'])
        df['Our'] = xs
        df['Hoie'] = ys
        sns.scatterplot(data=df, x="Our", y="Hoie", marker=".")
        plt.xlabel("Our prediction")
        plt.ylabel("Hoie prediction")

        plt.xlim(0, 1.1)
        plt.ylim(0, 1.1)

        plt.grid()
        plot_path = os.path.join(report_path, f'{test_res.model_name}.vs.hoie.values.plot.png')
        plt.savefig(plot_path)
        # plt.show()
        log(f'Created plt: {plot_path}')

    @staticmethod
    def _fill_ranks(pos_predictions):
        pos_predictions.sort(key=lambda x: x.pred_score)
        for pos in pos_predictions:
            pred_rank = pos_predictions.index(pos)
            pos.pred_rank = pred_rank
        pos_predictions.sort(key=lambda x: x.true_score)
        for pos in pos_predictions:
            true_rank = pos_predictions.index(pos)
            pos.true_rank = true_rank

    @staticmethod
    def _create_position_predictions(test_res):
        pos_predictions = []
        for i in range(len(test_res.pred_values)):
            pos = PositionPrediction()
            pos.index = test_res.pos_values[i]
            pos.true_score = test_res.true_values[i]
            pos.pred_score = test_res.pred_values[i]
            pos_predictions.append(pos)
        return pos_predictions

    @staticmethod
    def create_plot_episodes_loss(report_path, losses_list, prot_name, type):
        """
        Plot loss per episode
        @param type: loss type - can be 'train' or 'valid'
        @param report_path: path to report folder
        @param losses_list: list of loss values
        @param prot_name: name of the protein file, for example IF-1.DMS.txt
        """
        if len(losses_list) == 0:
            return
        plt.clf()
        xs = list(range(len(losses_list)))
        ys = losses_list
        x_name = 'Iteration'
        y_name = 'Loss'
        column_names = [x_name, y_name]
        df = pd.DataFrame(columns=column_names)
        df[x_name] = xs
        df[y_name] = ys
        name = prot_name.replace(".txt", "")
        plot_path = os.path.join(report_path, f'{name}.{type}_loss.plot.png')
        title = f'{name} {type} loss'
        sns.lineplot(data=df, x=x_name, y=y_name, marker='.', markers=True).set_title(f'{title}')
        plt.grid()
        # plt.show()
        plt.savefig(plot_path)


if __name__ == '__main__':
    pass
