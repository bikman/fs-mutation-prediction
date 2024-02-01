"""
Create various plot on input data
1. Plot of ddG_vs_Mave and ddE_vs_Mave
2. Plot look-up table for selected protein
"""

import logging
import os
import pickle
import random
import time
import torch
import csv
from pathlib import Path
from scipy.stats import spearmanr
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import random_split
from utils import CFG, DEVICE, PRISM_EVAL_SPLIT, PRISM_TRAIN_SPLIT, PRISM_VALID_SPLIT, setup_reports, \
    get_protein_files_dict, HOIE_RESULTS, PRISM_FOLDER, AA_ALPHABETICAL, AA_DICT
from run_prism_data_creation import log, parse_prism_score_file
from sklearn import preprocessing


def plot_dd_vs_mave():
    """
    Create plot of Spearman correlation
    X axis: ddG vs MAVE score
    Y axis: ddE vs MAVE score
    """
    report_path = setup_reports('plot_dd_vs_mave')
    log(f'Report path:{report_path}')
    log(DEVICE)
    log('=' * 100)

    prism_data_list = parse_prism_files()

    log('Calc Spearman correlations')
    correlations = []
    for prism_data in prism_data_list:
        ddGs = [v.ddG for v in prism_data.variants]
        ddEs = [v.ddE for v in prism_data.variants]
        scores = [v.score for v in prism_data.variants]
        assert len(ddGs) == len(ddEs)
        assert len(ddEs) == len(scores)
        spearman_ddG = abs(np.round(spearmanr(ddGs, scores), 4)[0])
        spearman_ddE = abs(np.round(spearmanr(ddEs, scores), 4)[0])
        correlations.append((prism_data.protein_name, spearman_ddG, spearman_ddE))

    log('=' * 100)
    for corr in correlations:
        print(corr)

    xs = [c[1] for c in correlations]
    ys = [c[2] for c in correlations]
    labels = [c[0] for c in correlations]

    fig, ax = plt.subplots()
    ax.scatter(xs, ys)

    for i, label in enumerate(labels):
        ax.annotate(label, (xs[i], ys[i]))

    plt.xlim([0.0, 0.7])
    plt.ylim([0.0, 0.9])
    plt.grid()
    plt.xlabel("ddG_vs_mave")
    plt.ylabel("ddE_vs_mave")

    plot_path = os.path.join(report_path, 'dd_vs_mave.png')
    plt.savefig(plot_path)
    log(f'Created plt: {plot_path}')


def parse_prism_files():
    log('Create PRISM data')
    prism_data_list = []
    for f in Path(PRISM_FOLDER).rglob('*.txt'):
        file_name = os.path.basename(f)
        if file_name not in get_protein_files_dict().values():
            continue
        prism_data = parse_prism_score_file(f)
        log(prism_data)
        prism_data_list.append(prism_data)
    log('=' * 100)
    return prism_data_list


def plot_heatmaps():
    """
    Creates heatmap plots for each protein
    x-axis: position
    y-axis: mutation dst AA
    color: value
    """
    report_path = setup_reports('plot_heatmap')
    log(f'Report path:{report_path}')
    log(DEVICE)
    log('=' * 100)

    prism_data_list = parse_prism_files()

    row_labels = AA_ALPHABETICAL
    # positions = [str(x) for x in range(1, 72)]

    # for prism_data in [prism_data_list[-1]]:
    for prism_data in prism_data_list:
        positions = set([v.position for v in prism_data.variants])
        positions_array = {k: v for v, k in enumerate(positions)}
        # data_arr = np.zeros((len(row_labels),len(positions)))
        data_arr = np.ones((len(row_labels), len(positions)))
        # data_arr = np.full((len(row_labels),len(positions)), -1)
        for v in prism_data.variants:
            col = positions_array[v.position]
            row = AA_DICT[v.aa_to]
            data_arr[row, col] = v.score
        show_single_protein_heatmap(report_path, prism_data.file_name, data_arr, positions, row_labels)


def show_single_protein_heatmap(report_path, title, data_arr, x_labels, y_labels):
    """
    Creates heatmap plot for single protein
    """
    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(data_arr)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    every_nth = 4
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)

    # --- create text annotations ---
    # for i in range(len(x_labels)):
    #     for j in range(len(y_labels)):
    #         text = ax.text(j, i, "{:.3f}".format(data_arr[i, j]),
    #                        ha="center", va="center", color="w")
    # ax.set_aspect(aspect=0.3)

    ax.set_title(f'{title}')

    # --- add color bar ---
    cbar = ax.figure.colorbar(im, ax=ax, orientation='horizontal')

    fig.tight_layout()

    plot_path = os.path.join(report_path, f'{title}.heatmap.png')
    plt.savefig(plot_path)
    log(f'Created plt: {plot_path}')


def print_seq_lengths():
    """
    Prints to CSV lengths of all protein sequences in database
    """
    report_path = setup_reports('print_seq_lengths')
    log(f'Report path:{report_path}')
    log('Print sequence lengths')
    name_to_len = {}
    for f in Path(PRISM_FOLDER).rglob('*.txt'):
        file_name = os.path.basename(f)
        log(f'Parsing: {file_name}')
        prism_data = parse_prism_score_file(f)
        log(prism_data)
        name_to_len[prism_data.file_name] = len(prism_data.sequence)
    csv_path = os.path.join(report_path, 'seq_lengths.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['File', 'Seq_Length'])
        writer.writeheader()
        for name in name_to_len:
            writer.writerow({'File': name, 'Seq_Length': name_to_len[name]})


def check_normalization():
    """
    TBD
    @return:
    """
    report_path = setup_reports('check_normalization')
    log(f'Report path:{report_path}')
    prism_data_list = parse_prism_files()
    log('checking "inverse_transform"...')
    for pdata in prism_data_list:
        orig_scores = [v.score_orig for v in pdata.variants]
        x = np.array(orig_scores)
        x = x.reshape(-1, 1)
        quantile_transformer = preprocessing.QuantileTransformer(n_quantiles=4)
        x_transformed = quantile_transformer.fit_transform(x)
        norm_scores = x_transformed.flatten().tolist()
        y = np.array(norm_scores)
        y = y.reshape(-1, 1)
        y_transformed = quantile_transformer.inverse_transform(y)  # back to non-normalized
        undo_scores = y_transformed.flatten().tolist()
        for x, y in zip(orig_scores, undo_scores):
            assert abs(x - y) < pow(10, -5)
    log('"inverse_transform" correct!')

    # TODO: add partial fit inversion


if __name__ == '__main__':
    pass
    # plot_dd_vs_mave()
    # plot_heatmaps()
    # print_seq_lengths()
    check_normalization()
