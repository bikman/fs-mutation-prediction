"""
TBD
"""
import logging
import os
import pickle
import time
import seaborn as sns
from pathlib import Path
from statistics import mean

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import random_split

from data_model import Variant, PrismScoreData
from embeddings import EsmMsaEmbedding
from pdb_data import PdbDataParser
from run_prism_data_creation import get_sequence_from_prism_file, get_variants_from_prism_file
from utils import DEVICE, PRISM_FOLDER, CFG, ESM_MODEL, ESM_REGRESSION, TAPE_PRETRAINED, \
    DUMP_ROOT, normalize_scores_only, MAX_SEQUENCE, PRISM_EVAL_SPLIT, PRISM_VALID_SPLIT, PRISM_TRAIN_SPLIT, \
    get_protein_files_dict, AA_ALPHABETICAL, ALL_PROTEIN_FILES_DICT
from utils import setup_reports

LOG_ENABLED = True
log = print
if LOG_ENABLED:
    # noinspection PyRedeclaration
    log = logging.info


def parse_prism_score_file(file_path):
    """
    TBD
    @param file_path: path to PRISM TXT file
    @return: PrismScoreData object
    """
    file_name = os.path.basename(file_path)
    tokens = file_name.split('_')
    protein_name: str = tokens[1]
    if file_name not in get_protein_files_dict().values():
        return None
    sequence = get_sequence_from_prism_file(file_path)
    variants = get_variants_from_prism_file(file_path)
    # normalize_deltas_and_scores(variants)
    data = PrismScoreData()
    data.sequence = sequence
    data.variants = variants
    data.protein_name = protein_name
    data.file_name = file_name
    return data


def create_per_protein_orig_histo(prism_data_list, report_path):
    for prism_data in prism_data_list:
        scores = [v.score_orig for v in prism_data.variants]
        plt.figure()
        plt.hist(scores, bins=100, label=prism_data.protein_name)
        plt.title(prism_data.file_name)
        plt.xlabel("Score")
        plt.ylabel("Count")
        plt.grid()
        plot_path = os.path.join(report_path, f'{prism_data.file_name}.orig.histo.eps')
        plt.savefig(plot_path, format='eps')
        log(f'Created plt: {plot_path}')


def create_per_protein_histo(prism_data_list, report_path):
    for prism_data in prism_data_list:
        scores = [v.score for v in prism_data.variants]
        plt.figure()
        plt.hist(scores, bins=100, label=prism_data.protein_name)
        plt.title(prism_data.file_name)
        plt.xlabel("Truth score (normalized)")
        plt.ylabel("Count")
        plt.grid()
        plot_path = os.path.join(report_path, f'{prism_data.file_name}.histo.png')
        plt.savefig(plot_path)
        log(f'Created plt: {plot_path}')


def create_per_protein_histo_seaborn(prism_data_list, report_path):
    sns.set(style="darkgrid")
    column_names = ['Protein', 'Score']
    for prism_data in prism_data_list:
        plt.clf()
        scores = [v.score for v in prism_data.variants]
        names = [prism_data.protein_name for _ in prism_data.variants]
        df = pd.DataFrame(columns=column_names)
        df['Protein'] = names
        df['Score'] = scores
        tokens = os.path.splitext(prism_data.file_name)[0].split('_')
        title = '_'.join(tokens[1:])
        sns.histplot(data=df, x='Score', bins=100).set_title(title)
        plt.legend([], [], frameon=False)
        # plt.show()
        # --- EPS ---
        # plot_path = os.path.join(report_path, f'{prism_data.file_name}.histo.eps')
        # plt.savefig(plot_path, format='eps')
        # --- SVG ---
        plot_path = os.path.join(report_path, f'{prism_data.file_name}.histo.svg')
        plt.savefig(plot_path, format='svg', transparent=True)
        # -----------
        log(f'Created plt: {plot_path}')


def create_all_scores_histo_seaborn(prism_data_list, report_path, skip_list, is_norm=True):
    plt.clf()
    sns.set(style="darkgrid")
    sns.set_palette("dark")
    column_names = ['Protein', 'Score']
    df = pd.DataFrame(columns=column_names)
    for prism_data in prism_data_list:
        if prism_data.protein_name in skip_list:
            continue
        scores = [v.score for v in prism_data.variants]
        names = [prism_data.file_name for _ in prism_data.variants]
        tmp = pd.DataFrame(columns=column_names)
        tmp['Protein'] = names
        tmp['Score'] = scores
        df = df.append(tmp, ignore_index=True)
    title = 'All Scores'
    if is_norm:
        title += ' (normalized)'
    else:
        title += ' (raw)'
    sns.histplot(data=df, x='Score', hue='Protein', bins=100).set_title(title)
    plt.legend([], [], frameon=False)
    # plt.show()
    plot_path = os.path.join(report_path, f'all_scores.histogram.png')
    plt.savefig(plot_path)
    log(f'Created plt: {plot_path}')


def create_all_scores_histo(prism_data_list, report_path, skip_list):
    for prism_data in prism_data_list:
        if prism_data.protein_name in skip_list:
            continue
        scores = [v.score for v in prism_data.variants]
        plt.figure(1)
        plt.hist(scores, bins=100, label=prism_data.protein_name)
        plt.title('All Scores')
        plt.xlabel("Truth score (normalized)")
        plt.ylabel("Count")
        plt.grid()
    # plt.show()
    plot_path = os.path.join(report_path, f'all_scores.histogram.png')
    plt.savefig(plot_path)
    log(f'Created plt: {plot_path}')


def create_one_out_histo_seaborn(prism_data_list, report_path):
    for prism_data in prism_data_list:
        others_list = [x for x in prism_data_list if x != prism_data]
        others_scores = []
        others_names = []
        for data in others_list:
            others_scores += [v.score for v in data.variants]
            others_names += [data.protein_name for _ in data.variants]
        data_scores = [v.score for v in prism_data.variants]
        data_names = [prism_data.protein_name for _ in prism_data.variants]

        # others_len = len(others_scores) * 1.0
        # data_len = len(data_scores) * 1.0
        # others_scores = [x / others_len for x in others_scores]
        # data_scores = [x / data_len for x in data_scores]

        others_df = pd.DataFrame()
        others_df['Protein'] = others_names
        others_df['Score'] = others_scores

        df = pd.DataFrame()
        df['Protein'] = data_names
        df['Score'] = data_scores

        fig, ax = plt.subplots(1, 2)
        p1 = sns.histplot(data=df, x='Score', bins=100, ax=ax[0])
        p1.set_title(f'{prism_data.protein_name}')
        p1.set_xlabel(None)
        p1.set(xticklabels=[])
        p2 = sns.histplot(data=others_df, x='Score', bins=100, ax=ax[1])
        p2.set_title('others')
        p2.set_xlabel(None)
        p2.set(xticklabels=[])

        # fig.show()
        fig.legend([], [], frameon=False)
        plot_path = os.path.join(report_path, f'{prism_data.file_name}.histo.png')
        plt.savefig(plot_path)
        log(f'Created plt: {plot_path}')


def create_non_normalized_histo(prism_data_list, ft_range_dict, report_path):
    """
    Created histograms of Non-normalized scores
    @param prism_data_list: list of prism data
    @param report_path: path to report folder
    """
    min_max_scores = []
    digits = 5
    for prism_data in prism_data_list:
        scores = [v.score_orig for v in prism_data.variants]
        min_score = round(min(scores), digits)
        max_score = round(max(scores), digits)
        zero_close_ddg_v = min(prism_data.variants, key=lambda v: abs(v.ddG_orig))
        min_ddg_vs = [v for v in prism_data.variants if v.ddG_orig == zero_close_ddg_v.ddG_orig]
        # if len(min_ddg_vs) !=1:
        #     raise Exception(prism_data.file_name)
        min_ddg = round(zero_close_ddg_v.ddG_orig, digits)
        min_max_scores.append(
            (prism_data.file_name, min_score, max_score, round(zero_close_ddg_v.score_orig, digits), min_ddg))

        loop = 1
        for t in ft_range_dict[prism_data.file_name]:
            low_range_lim = float(t[0])
            high_range_lim = float(t[1])

            fig = plt.figure()
            plt.hist(scores, bins=100, label=prism_data.protein_name)
            for v in min_ddg_vs:
                plt.plot(v.score_orig, 0, "or")
            plt.axvline(x=zero_close_ddg_v.score_orig, color='r')

            plt.axvline(x=low_range_lim, color='m')
            plt.axvline(x=high_range_lim, color='m')

            plt.title(prism_data.file_name)
            plt.xlabel("Truth score (original)")
            plt.ylabel("Count")
            plt.grid()
            plot_path = os.path.join(report_path, f'{prism_data.file_name}.orig_score.{loop}.histo.png')
            plt.savefig(plot_path)
            log(f'Created plt: {plot_path}')
            plt.close(fig)
            loop += 1

        low_lim_mean = mean([float(t[0]) for t in ft_range_dict[prism_data.file_name]])
        high_lim_mean = mean([float(t[1]) for t in ft_range_dict[prism_data.file_name]])
        fig = plt.figure()
        plt.hist(scores, bins=100, label=prism_data.protein_name)
        for v in min_ddg_vs:
            plt.plot(v.score_orig, 0, "or")
        plt.axvline(x=zero_close_ddg_v.score_orig, color='r')

        plt.axvline(x=low_lim_mean, color='y')
        plt.axvline(x=high_lim_mean, color='y')

        plt.title(prism_data.file_name)
        plt.xlabel("Truth score (original)")
        plt.ylabel("Count")
        plt.grid()
        plot_path = os.path.join(report_path, f'{prism_data.file_name}.orig_score.ft_mean.histo.png')
        plt.savefig(plot_path)
        plt.close(fig)
        log(f'Created plt: {plot_path}')

    log('-' * 30)
    for t in min_max_scores:
        log(t)


def read_ft_range(file_name):
    """
    @param file_name: CSV file with min and max scores for FT sample
    @return: mave file name -> list((min.max), (min. max), ..))
    """
    res = {}
    file_path = os.path.join(r'D:\Paper_Results\NEW', file_name)
    with open(file_path) as file_handle:
        for line in file_handle:
            tokens = line.strip().split(',')
            mave_file = tokens[1]
            min_score = tokens[3]
            max_score = tokens[4]
            res.setdefault(mave_file, []).append((min_score, max_score))
    return res


def create_mave_vs_pred_histograms(report_path, hoie_res_folder, ft_results_folder):
    """
    Create histogram plot of distributions for MAVE real scores,
    FT non-normalized scores, and Hoie RF non-normalized scores.
    @param report_path: path to create histogram files
    @param hoie_res_folder: folder with non-normalized scores for Hoie RF predictions
    @param ft_results_folder: folder with non-normalized scores for FT predictions
    """
    print('Read FT results')
    prot_to_eval_results = {}
    res_ids = [res_id for res_id in os.listdir(ft_results_folder)]
    protein_index = 1
    for res_id in res_ids:
        our_res_path = rf'{ft_results_folder}\{res_id}\ft_results.pkl'
        with open(our_res_path, "rb") as f:
            ft_results = pickle.load(f)
            protein_file_name = ALL_PROTEIN_FILES_DICT[protein_index].split('.')[0]
            prot_to_eval_results[protein_file_name] = ft_results[0].test_result.nn_pred_values.tolist()
            protein_index += 1

    print('Read RF results')
    for file_path in Path(hoie_res_folder).rglob('*.pkl'):
        print(file_path.name)
        with open(file_path, "rb") as f:
            hoie_nn_pred = pickle.load(f)
            hoie_test_name = file_path.name.split('.')[0]

        print('Read MAVE values')
        CFG['general']['protein_set'] = '1'
        for f in Path(PRISM_FOLDER).rglob('*.txt'):
            ft_test_name = f.name.split('.')[0]
            if ft_test_name != hoie_test_name:
                continue
            print(f.name)
            prism_data = parse_prism_score_file(f)
            if prism_data is None:
                continue
            prism_data.file_name = os.path.basename(f)
            print(prism_data)

            plt.clf()
            sns.set(style="darkgrid")
            # sns.set_palette("dark")
            series_name = 'Type'
            x_ax_name = 'Score'
            column_names = [series_name, x_ax_name]
            df = pd.DataFrame(columns=column_names)

            # --- create frame for MAVE scores ---
            pd_scores = [v.score for v in prism_data.variants]
            names = ['MAVE_score'] * len(pd_scores)
            tmp = pd.DataFrame(columns=column_names)
            tmp[series_name] = names
            tmp[x_ax_name] = pd_scores
            df = df.append(tmp, ignore_index=True)

            # --- create frame for RF Hoie scores ---
            res_scores = hoie_nn_pred
            res_names = ['RF_score'] * len(res_scores)
            tmp = pd.DataFrame(columns=column_names)
            tmp[series_name] = res_names
            tmp[x_ax_name] = res_scores
            df = df.append(tmp, ignore_index=True)

            # assert len(res_scores) == len(pd_scores)

            # --- create frame for NN FT5p scores ---
            ft_scores = prot_to_eval_results[ft_test_name]
            ft_names = ['FT_score'] * len(ft_scores)
            tmp = pd.DataFrame(columns=column_names)
            tmp[series_name] = ft_names
            tmp[x_ax_name] = ft_scores
            df = df.append(tmp, ignore_index=True)

            # bins_count = 100
            bins_count = 20
            hst_plt = sns.histplot(data=df, x=x_ax_name, bins=bins_count, fill=False, hue=series_name)
            # - set title -
            # tokens = os.path.splitext(prism_data.file_name)[0].split('_')
            # title = '_'.join(tokens[1:])
            # hst_plt.set_title(title)
            hst_plt.set(xlabel=None)
            hst_plt.set(ylabel=None)
            plt.xticks(fontsize=18)
            plt.legend([], [], frameon=False)
            # --- PNG ---
            plot_path = os.path.join(report_path, f'{prism_data.file_name}.rf.histo.png')
            plt.savefig(plot_path)
            # --- EPS ---
            plot_path = os.path.join(report_path, f'{prism_data.file_name}.rf.histo.eps')
            plt.savefig(plot_path, format='eps')
            # -----------
            print(f'Created plt: {plot_path}')
    print('OK!')


def print_ft_vs_mave_histograms(report_path, all_res_folder):
    """
    OBSOLETE
    @param report_path:
    @param all_res_folder:
    @return:
    """
    print('Read FT results')
    res_ids = [res_id for res_id in os.listdir(all_res_folder)]
    print(len(res_ids))
    list_of_eval_results = []
    for res_id in res_ids:
        # for res_id in res_ids[:2]:
        our_res_path = rf'{all_res_folder}\{res_id}\ft_results.pkl'
        with open(our_res_path, "rb") as f:
            ft_results = pickle.load(f)
            list_of_eval_results.append(ft_results[0].test_result)
    eval_datas = []
    for res in list_of_eval_results:
        eval_data = PrismScoreData()
        eval_data.variants = []
        for (score, pos, src, dst) in zip(res.nn_pred_values, res.pos_values, res.src_indices, res.dst_indices):
            v = Variant()
            v.score = score
            v.position = pos
            v.aa_from = AA_ALPHABETICAL[int(src)]
            v.aa_to = AA_ALPHABETICAL[int(dst)]
            eval_data.variants.append(v)
        eval_datas.append(eval_data)

    print('Create PRISM score data')
    prism_data_list = []
    CFG['general']['protein_set'] = '1'
    for f in Path(PRISM_FOLDER).rglob('*.txt'):
        # if 'PTEN' not in str(f):
        #     continue
        prism_data = parse_prism_score_file(f)
        if prism_data is None:
            continue
        prism_data.file_name = os.path.basename(f)
        print(prism_data)
        prism_data_list.append(prism_data)

    assert len(eval_datas) == len(prism_data_list)

    for (prism_data, eval_data) in zip(prism_data_list, eval_datas):
        plt.clf()
        sns.set(style="darkgrid")
        # sns.set_palette("dark")
        column_names = ['Protein', 'Score']
        df = pd.DataFrame(columns=column_names)
        pd_scores = [v.score for v in prism_data.variants if v in eval_data.variants]
        assert len(pd_scores) == len(eval_data.variants)
        names = ['MAVE_score'] * len(pd_scores)
        tmp = pd.DataFrame(columns=column_names)
        tmp['Protein'] = names
        tmp['Score'] = pd_scores
        df = df.append(tmp, ignore_index=True)

        res_scores = [v.score for v in eval_data.variants]
        # res_scores += [0] * (len(pd_scores) - len(res_scores))
        res_names = ['FT_score'] * len(res_scores)
        tmp = pd.DataFrame(columns=column_names)
        tmp['Protein'] = res_names
        tmp['Score'] = res_scores
        df = df.append(tmp, ignore_index=True)

        tokens = os.path.splitext(prism_data.file_name)[0].split('_')
        title = '_'.join(tokens[1:])
        # sns.histplot(data=df, x='Score', bins=100, fill=False, hue="Protein").set_title(title)
        sns.histplot(data=df, x='Score', bins=20, fill=False, hue="Protein").set_title(title)
        # --- PNG ---
        plot_path = os.path.join(report_path, f'{prism_data.file_name}.histogram.png')
        plt.savefig(plot_path)
        # --- EPS ---
        plot_path = os.path.join(report_path, f'{prism_data.file_name}.histogram.eps')
        plt.savefig(plot_path, format='eps')
        # -----------
        print(f'Created plt: {plot_path}')
    print('OK!')


def main():
    start_time = time.time()
    report_path = setup_reports('score_histo_creation')
    log(f'Report path:{report_path}')
    log(DEVICE)

    log('Create PRISM score data')
    prism_data_list = []
    for f in Path(PRISM_FOLDER).rglob('*.txt'):
        prism_data = parse_prism_score_file(f)
        if prism_data is None:
            continue
        prism_data.file_name = os.path.basename(f)
        log(prism_data)
        prism_data_list.append(prism_data)

    # --- UNCOMMENT TO USE NORMALIZED SCORES ---
    # normalize_scores_only(prism_data_list)

    # --- uncomment the function you need ---
    # --- histo created with Mathplotlib ---
    # create_all_scores_histo(prism_data_list, report_path, skip_list=[])
    # create_per_protein_histo(prism_data_list, report_path)
    # create_per_protein_orig_histo(prism_data_list, report_path)

    # --- histo created with Seaborn ---
    # create_all_scores_histo_seaborn(prism_data_list, report_path, skip_list=[], is_norm=False)
    # create_all_scores_histo_seaborn(prism_data_list, report_path, skip_list=[], is_norm=True)
    # create_per_protein_histo_seaborn(prism_data_list, report_path)
    # create_one_out_histo_seaborn(prism_data_list, report_path)

    # ft_range_dict = read_ft_range('ft_range_1p.csv')
    # create_non_normalized_histo(prism_data_list, ft_range_dict, report_path)

    create_mave_vs_pred_histograms(report_path, r'hoie_nn_pred', r'D:\Paper_Results\NEW\ft5_results')

    elapsed_time = time.time() - start_time
    log(f'time: {elapsed_time:5.2f} sec')
    print('OK')


if __name__ == '__main__':
    main()

    # tips = sns.load_dataset("tips")
    # sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time")
    # plt.show()
