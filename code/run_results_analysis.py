"""
This module analyses the pickled results and Hoie's result
"""
import os
import pickle
import random

from pathlib import Path
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from utils import HOIE_RESULTS, setup_reports, smape, PRISM_FOLDER, CFG, AA_ALPHABETICAL, ALL_PROTEIN_FILES_DICT
from data_model import Prediction, PositionPrediction, PrismScoreData, Variant
from run_score_histogram import parse_prism_score_file
from scipy.stats import spearmanr
import numpy as np
from sklearn.metrics import mean_absolute_error


def create_plot_mae_vs_hoie_per_variant(report_path, protein_name, result_id):
    """
    Create plot MAE ours vs Hoie's per variant
    @param report_path: path to report folder
    @param protein_name: name of the protein (without prefix), i.e. IF-1_DMS
    @param result_id: result of DLC run id
    """
    print(f'create_plot_mae_vs_hoie_per_variant: {protein_name} : {result_id}')
    plt.clf()
    our_res_path = rf'D:\RUN_RESULTS\{result_id}\eval_result.pkl'
    with open(our_res_path, "rb") as f:
        our_result = pickle.load(f)
    our_maes = []
    for (tv, pv) in zip(our_result.true_values, our_result.pred_values):
        mae = np.round(mean_absolute_error([tv], [pv]), 4)
        our_maes.append(mae)
    hoie_res_path = os.path.join(HOIE_RESULTS, f'{protein_name}.hoie.result.data.pkl')
    with open(hoie_res_path, "rb") as f:
        hoie_result = pickle.load(f)
    hoie_result.sort(key=lambda x: x.true_score)
    hoie_maes = []
    for pr in hoie_result:
        mae = np.round(mean_absolute_error([pr.true_score], [pr.pred_score]), 4)
        hoie_maes.append(mae)
    true_values_hoie = [pr.true_score for pr in hoie_result]
    pred_values_hoie = [pr.pred_score for pr in hoie_result]
    spearman_hoie = np.round(spearmanr(true_values_hoie, pred_values_hoie), 4)[0]
    spearman_mae = np.round(mean_absolute_error(true_values_hoie, pred_values_hoie), 4)
    xs = our_maes
    ys = hoie_maes
    assert len(xs) == len(ys)
    x_name = 'Our_MAE'
    y_name = 'Hoie_MAE'
    column_names = [x_name, y_name]
    df = pd.DataFrame(columns=column_names)
    df[x_name] = xs
    df[y_name] = ys
    g = sns.scatterplot(data=df, x=x_name, y=y_name, marker=".")
    legend = f'Hoie Spearman={spearman_hoie}\nOur Spearman={our_result.spearman}\n--------\n' \
             f'Hoie MAE={spearman_mae}\nOur MAE={our_result.mae}'
    g.legend([legend], loc='best')

    plt.title(f'{protein_name}: MAE ours vs hoies')
    plt.grid()
    # plt.show()
    plt_file = f'{protein_name}.plot.png'
    plot_path = os.path.join(report_path, plt_file)
    plt.savefig(plot_path)


def create_plot_order_vs_hoie_per_variant(report_path, protein_name, result_id):
    """
    Creates scatter plot
    X-axis: predicted rank
    Y-axis: truth rank
    @param report_path: path to report folder
    @param protein_name: name of the protein to analyze
    @param result_id: id of the DLC run to analyze (in a local folder)
    """
    print(f'create_plot_order_vs_hoie_per_variant: {protein_name} : {result_id}')
    plt.clf()
    our_res_path = rf'D:\RUN_RESULTS\{result_id}\eval_result.pkl'
    with open(our_res_path, "rb") as f:
        our_result = pickle.load(f)

    our_data = []
    i = 0
    for (truth, pred) in zip(our_result.true_values, our_result.pred_values):
        prediction = PositionPrediction()
        prediction.true_score = truth
        prediction.pred_score = pred
        prediction.true_rank = i
        our_data.append(prediction)
        i = i + 1
    our_data.sort(key=lambda x: x.pred_score)
    i = 0
    for p in our_data:
        p.pred_rank = i
        i = i + 1

    hoie_res_path = os.path.join(HOIE_RESULTS, f'{protein_name}.hoie.result.data.pkl')
    with open(hoie_res_path, "rb") as f:
        hoie_result = pickle.load(f)
    hoie_result.sort(key=lambda x: x.true_score)
    true_values_hoie = [pr.true_score for pr in hoie_result]
    pred_values_hoie = [pr.pred_score for pr in hoie_result]
    spearman_hoie = np.round(spearmanr(true_values_hoie, pred_values_hoie), 4)[0]

    hoie_data = []
    i = 0
    for p in hoie_result:
        prediction = PositionPrediction()
        prediction.true_score = p.true_score
        prediction.pred_score = p.pred_score
        prediction.true_rank = i
        hoie_data.append(prediction)
        i = i + 1
    hoie_data.sort(key=lambda x: x.pred_score)
    i = 0
    for p in hoie_data:
        p.pred_rank = i
        i = i + 1

    xs = [p.pred_rank for p in our_data]
    ys = [p.true_rank for p in our_data]
    assert len(xs) == len(ys)
    x_name = 'Predicted rank'
    y_name = 'True rank'
    source = 'Source'
    column_names = [x_name, y_name, source]
    df = pd.DataFrame(columns=column_names)

    hxs = [p.pred_rank for p in hoie_data]
    hys = [p.true_rank for p in hoie_data]
    assert len(hxs) == len(hys)

    df[x_name] = xs + hxs
    df[y_name] = ys + hys
    df[source] = ['Ours' for _ in xs] + ['Hoie' for _ in hxs]

    # --- Create CSV ----
    csv_file = f'{protein_name}.ranks.csv'
    csv_path = os.path.join(report_path, csv_file)
    df.to_csv(csv_path, index=False)

    # --- Create plots ----
    g = sns.scatterplot(data=df, x=x_name, y=y_name, s=3, hue=source)
    legend = f'Hoie Sp.={spearman_hoie}\nOur Sp.={our_result.spearman}'
    g.legend(title=legend, loc='best')
    plt.title(f'{protein_name}: Rank pred vs true')
    plt.grid()
    # plt.show()
    plt_file = f'{protein_name}.ranks.plot.png'
    plot_path = os.path.join(report_path, plt_file)
    plt.savefig(plot_path)


def calculate_pred_bins(predictions, num_bins=10):
    """
    Calculate bins for predicted scores in result
    @param predictions:
    @param num_bins:
    @return:
    """
    # for prediction in hoie_result:
    list.sort(predictions, key=lambda x: x.pred_score, reverse=False)
    chunked_list = np.array_split(predictions, num_bins)
    assert len(chunked_list) == num_bins
    bin_id = 0
    for chunk in chunked_list:
        for p in chunk:
            p.pred_bin = bin_id
        bin_id += 1
    for p in predictions:
        assert p.pred_bin is not None


def calculate_true_bins(predictions, num_bins=10):
    """
    Calculate bins for true scores in result
    @param predictions:
    @param num_bins:
    @return:
    """
    # for prediction in hoie_result:
    list.sort(predictions, key=lambda x: x.true_score, reverse=False)
    chunked_list = np.array_split(predictions, num_bins)
    assert len(chunked_list) == num_bins
    bin_id = 0
    for chunk in chunked_list:
        for p in chunk:
            p.true_bin = bin_id
        bin_id += 1
    for p in predictions:
        assert p.true_bin is not None


def calc_bins_spearman(report_path, p_name, result_id):
    """
    Calculates spearman over BINs
    @param report_path:
    @param p_name:
    @param result_id:
    @return:
    """
    our_res_path = rf'D:\RUN_RESULTS\{result_id}\eval_result.pkl'
    with open(our_res_path, "rb") as f:
        our_result = pickle.load(f)
    predictions = []
    for (truth, pred) in zip(our_result.true_values, our_result.pred_values):
        prediction = Prediction()
        prediction.true_score = truth
        prediction.pred_score = pred
        predictions.append(prediction)
    calculate_pred_bins(predictions)
    calculate_true_bins(predictions)
    ys = [p.true_bin for p in predictions]
    xs = [p.pred_bin for p in predictions]
    spearman_bins = np.round(spearmanr(ys, xs), 4)[0]
    print(f'{p_name},{spearman_bins}')


def spearman_test(result_id):
    """
    Gets evaluation result to calculate spearman (and to check if after shuffle spearman is the same)
    @param result_id:
    @return:
    """
    our_res_path = rf'D:\RUN_RESULTS\{result_id}\eval_result.pkl'
    with open(our_res_path, "rb") as f:
        our_result = pickle.load(f)
    predictions = []
    for (truth, pred) in zip(our_result.true_values, our_result.pred_values):
        prediction = Prediction()
        prediction.true_score = truth
        prediction.pred_score = pred
        predictions.append(prediction)

    ys = [p.true_score for p in predictions]
    xs = [p.pred_score for p in predictions]
    spearman1 = np.round(spearmanr(ys, xs), 4)[0]
    mae1 = np.round(mean_absolute_error(ys, xs), 4)

    random.shuffle(predictions)
    ys = [p.true_score for p in predictions]
    xs = [p.pred_score for p in predictions]
    spearman2 = np.round(spearmanr(ys, xs), 4)[0]
    mae2 = np.round(mean_absolute_error(ys, xs), 4)

    assert mae1 == mae2
    assert spearman1 == spearman2


def calc_smape(all_res_folder):
    res_ids = [res_id for res_id in os.listdir(all_res_folder)]
    list_of_tuples = []
    for res_id in res_ids:
        our_res_path = rf'{all_res_folder}\{res_id}\eval_result.pkl'
        with open(our_res_path, "rb") as f:
            eval_result = pickle.load(f)
            mape = np.round(smape(eval_result.pred_values, eval_result.true_values), 4)
            list_of_tuples.append([res_id, str(mape)])
    for t in list_of_tuples:
        print(','.join(t))






if __name__ == '__main__':
    report_path = setup_reports('run_results_analysis')
    # create_plot_mae_vs_hoie_per_variant(report_path, 'PTEN_phosphatase_activity', 76439)
    # create_plot_mae_vs_hoie_per_variant(report_path, 'SUMO1_growth_abundance', 76440)
    # create_plot_mae_vs_hoie_per_variant(report_path, 'P53_abundance_reversed', 76441)
    # create_plot_mae_vs_hoie_per_variant(report_path, 'bla_DMS_a', 76442)
    # create_plot_mae_vs_hoie_per_variant(report_path, 'HAh3n2_DMS', 76443)
    # create_plot_mae_vs_hoie_per_variant(report_path, 'CBS_high_B6_activity', 76096)
    # create_plot_mae_vs_hoie_per_variant(report_path, 'TPK1_growth_abundance', 76097)
    # create_plot_mae_vs_hoie_per_variant(report_path, 'UBI4_E1_binding_limiting_E1', 76098)
    # create_plot_mae_vs_hoie_per_variant(report_path, 'HSP82_DMS', 76099)
    # create_plot_mae_vs_hoie_per_variant(report_path, 'IF-1_DMS', 76100)
    # create_plot_order_vs_hoie_per_variant(report_path, 'PTEN_phosphatase_activity', 76439)
    # create_plot_order_vs_hoie_per_variant(report_path, 'SUMO1_growth_abundance', 76440)
    # create_plot_order_vs_hoie_per_variant(report_path, 'P53_abundance_reversed', 76441)
    # create_plot_order_vs_hoie_per_variant(report_path, 'bla_DMS_a', 76442)
    # create_plot_order_vs_hoie_per_variant(report_path, 'HAh3n2_DMS', 76443)
    # create_plot_order_vs_hoie_per_variant(report_path, 'CBS_high_B6_activity', 76096)
    # create_plot_order_vs_hoie_per_variant(report_path, 'TPK1_growth_abundance', 76097)
    # create_plot_order_vs_hoie_per_variant(report_path, 'UBI4_E1_binding_limiting_E1', 76098)
    # create_plot_order_vs_hoie_per_variant(report_path, 'HSP82_DMS', 76099)
    # create_plot_order_vs_hoie_per_variant(report_path, 'IF-1_DMS', 76100)

    # calc_bins_spearman(report_path, 'CBS_low_B6_activity', 88321)
    # calc_bins_spearman(report_path, 'HMGCR_yeast_complementation_atorvastatin_medium', 88322)
    # calc_bins_spearman(report_path, 'cas9_DMS', 88323)
    # calc_bins_spearman(report_path, 'IF-1_DMS', 88324)

    # calc_bins_spearman(report_path, 'CBS_low_B6_activity', 88362)
    # calc_bins_spearman(report_path, 'HMGCR_yeast_complementation_atorvastatin_medium', 88363)
    # calc_bins_spearman(report_path, 'cas9_DMS', 88364)
    # calc_bins_spearman(report_path, 'IF-1_DMS', 88365)

    # calc_smape(r'D:\Temp')

    print('run_results_analysis: OK')
    # pass
