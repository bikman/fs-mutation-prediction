"""
Take and compare Embeddings
"""

import logging
import os
import pickle
import random
import time
import torch
from pathlib import Path
from scipy.stats import spearmanr
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import random_split
from utils import CFG, DEVICE, PRISM_EVAL_SPLIT, PRISM_TRAIN_SPLIT, PRISM_VALID_SPLIT, setup_reports, \
    get_protein_files_dict, HOIE_RESULTS, PRISM_FOLDER, AA_ALPHABETICAL, AA_DICT, ESM_MODEL, ESM_REGRESSION, DUMP_ROOT
from run_prism_data_creation import log, parse_prism_score_file
from embeddings import EsmMsaEmbedding
from tqdm import tqdm


def check_single_embedding():
    report_path = setup_reports('check_single_embedding')
    log(f'Report path:{report_path}')
    log(DEVICE)
    log('=' * 100)

    prism_data = None
    # protein_file = '999_IF-1_DMS.txt'
    protein_file = '009_SUMO1_growth_abundance.txt'
    for f in Path(PRISM_FOLDER).rglob('*.txt'):
        file_name = os.path.basename(f)
        if file_name != protein_file:
            continue
        prism_data = parse_prism_score_file(f)
        log(prism_data)

    assert prism_data is not None
    wt_sequence = prism_data.sequence

    log(f'Embedding wt protein: {prism_data.protein_name}')

    torch.manual_seed(0)
    sequence_embedder = EsmMsaEmbedding()

    # with torch.no_grad():
    emb = sequence_embedder.embed(wt_sequence)
    seq_emb_wt = torch.squeeze(emb)
    log(f'WT Emb.shape: {seq_emb_wt.shape}')

    position = 42
    log(f'Mutating protein position: {position}')

    mut_sequence = list(wt_sequence)
    mut_sequence[position] = 'W'
    mut_sequence = ''.join(mut_sequence)

    assert wt_sequence[position] != mut_sequence[position]

    log(f'Embedding mut protein: {prism_data.protein_name}')
    torch.manual_seed(0)
    sequence_embedder = EsmMsaEmbedding()
    emb = sequence_embedder.embed(mut_sequence)
    seq_emb_mut = torch.squeeze(emb)
    log(f'WT Emb.shape: {seq_emb_mut.shape}')

    # --- diff between wt and mut ---
    res = torch.sub(seq_emb_wt, seq_emb_mut)

    plt.imshow(res, cmap='hot', interpolation='nearest')
    title = f'{protein_file} mutation position = 42'
    plt.title(title)
    # plt.show()
    plot_path = os.path.join(report_path, f'{prism_data.protein_name}.mutation.diff.png')
    plt.colorbar(orientation='horizontal')
    plt.savefig(plot_path)
    log(f'Created plt: {plot_path}')
    plt.clf()

    plt.imshow(seq_emb_wt, cmap='hot', interpolation='nearest')
    title = f'{protein_file} mutation position = 42'
    plt.title(title)
    # plt.show()
    plot_path = os.path.join(report_path, f'{prism_data.protein_name}.wt.png')
    plt.colorbar(orientation='horizontal')
    plt.savefig(plot_path)
    log(f'Created plt: {plot_path}')
    plt.clf()

    plt.imshow(seq_emb_mut, cmap='hot', interpolation='nearest')
    title = f'{protein_file} WT embedding'
    plt.title(title)
    # plt.show()
    plot_path = os.path.join(report_path, f'{prism_data.protein_name}.mut.png')
    plt.colorbar(orientation='horizontal')
    plt.savefig(plot_path)
    log(f'Created plt: {plot_path}')
    plt.clf()

    diff_mean = torch.mean(res, dim=0)
    xs = [x for x in range(len(diff_mean))]
    fig, ax = plt.subplots()
    ax.plot(xs, diff_mean)
    ax.set_xlabel('tensor index')
    ax.set_ylabel('score')
    title = f'{protein_file} diff mean per dim=0'
    plt.title(title)
    plt.grid()
    # plt.show()
    plot_path = os.path.join(report_path, f'{prism_data.protein_name}.mean.dim0.diff.png')
    plt.savefig(plot_path)
    log(f'Created plt: {plot_path}')
    plt.clf()

    diff_mean = torch.mean(res, dim=1)
    xs = [x for x in range(len(diff_mean))]
    fig, ax = plt.subplots()
    ax.plot(xs, diff_mean)
    ax.set_xlabel('protein position')
    ax.set_ylabel('score')
    title = f'{protein_file} diff mean per dim=1'
    plt.title(title)
    plt.grid()
    # plt.show()
    plot_path = os.path.join(report_path, f'{prism_data.protein_name}.mean.dim1.diff.png')
    plt.savefig(plot_path)
    log(f'Created plt: {plot_path}')
    plt.clf()


def calc_amplitude(array):
    res = abs(max(array)) + abs(min(array))
    return res


def calc_sum(array):
    res = np.sum(np.absolute(array))
    return res


def create_diff_mean_dump(protein_file):
    report_path = setup_reports(f'create_diff_mean_dump:{protein_file}')
    log(f'Report path:{report_path}')
    log(DEVICE)
    log('=' * 100)

    prism_data = None
    for f in Path(PRISM_FOLDER).rglob('*.txt'):
        file_name = os.path.basename(f)
        if file_name != protein_file:
            continue
        log(f'Parsing: {f}')
        prism_data = parse_prism_score_file(f)
        log(prism_data)

    assert prism_data is not None
    wt_sequence = prism_data.sequence

    log(f'Embedding wt protein: {prism_data.protein_name}')
    torch.manual_seed(0)
    sequence_embedder = EsmMsaEmbedding()
    emb = sequence_embedder.embed(wt_sequence)
    seq_emb_wt = torch.squeeze(emb)
    log(f'WT Emb.shape: {seq_emb_wt.shape}')

    diff_mean_data = []

    log('Mutating protein')
    # for v in tqdm(prism_data.variants[:10]):
    for v in tqdm(prism_data.variants):
        # log(f'Mutating position: {v.position}, {v.aa_from} -> {v.aa_to}')
        mutation_ndx = v.position - 1
        assert wt_sequence[mutation_ndx] == v.aa_from
        mut_sequence = list(wt_sequence)
        mut_sequence[mutation_ndx] = v.aa_to
        mut_sequence = ''.join(mut_sequence)
        assert wt_sequence[mutation_ndx] != mut_sequence[mutation_ndx]

        # log(f'Embedding mut protein: {prism_data.protein_name}')
        torch.manual_seed(0)
        sequence_embedder = EsmMsaEmbedding()
        emb = sequence_embedder.embed(mut_sequence)
        seq_emb_mut = torch.squeeze(emb)
        # log(f'WT Emb.shape: {seq_emb_mut.shape}')

        seq_emb_wt_mean = torch.mean(seq_emb_wt, dim=1)
        seq_emb_mut_mean = torch.mean(seq_emb_mut, dim=1)
        diff_mean = torch.sub(seq_emb_wt_mean, seq_emb_mut_mean).numpy()
        assert len(diff_mean) == len(prism_data.sequence)
        diff_mean_data.append((diff_mean, mutation_ndx, v.score))

    assert len(diff_mean_data) == len(prism_data.variants)
    dump_path = os.path.join(DUMP_ROOT, f'{protein_file}.diff.mean.pkl')
    with open(dump_path, "wb") as f:
        pickle.dump(diff_mean_data, f)
    log(f'Created dump: {dump_path}')


def calc_mutations_correlations(protein_file, step):
    report_path = setup_reports(f'calc_mutations_correlations:{protein_file}')
    log(f'Report path:{report_path}')
    log(DEVICE)
    log('=' * 100)

    dump_path = os.path.join(DUMP_ROOT, f'{protein_file}.diff.mean.pkl')
    with open(dump_path, "rb") as f:
        diff_mean_data = pickle.load(f)

    log(f'Loaded data, size={len(diff_mean_data)}')

    amplitudes = []
    integrals = []
    stds = []
    scores = []
    for data in diff_mean_data:
        diff_mean = data[0]
        mutation_ndx = data[1]
        score = data[2]
        start_ndx = mutation_ndx - step
        end_ndx = mutation_ndx + step + 1
        if start_ndx < 0 or end_ndx > len(diff_mean):
            continue
        emb_vector = diff_mean[start_ndx:end_ndx]
        assert len(emb_vector) == 2 * step + 1

        amp = calc_amplitude(emb_vector)
        amplitudes.append(amp)
        integral = calc_sum(emb_vector)
        integrals.append(integral)
        std = np.std(emb_vector)
        stds.append(std)
        scores.append(score)

    log('=' * 100)
    assert len(amplitudes) == len(scores)
    assert len(integrals) == len(scores)
    assert len(stds) == len(scores)
    spearman_amp = abs(np.round(spearmanr(amplitudes, scores), 4)[0])
    spearman_sum = abs(np.round(spearmanr(integrals, scores), 4)[0])
    spearman_std = abs(np.round(spearmanr(stds, scores), 4)[0])
    log(f'Step: +-{step}')
    log(f'Spearman Amplitude: {spearman_amp}')
    log(f'Spearman Integral: {spearman_sum}')
    log(f'Spearman Std Dev: {spearman_std}')


def create_mut_corr_long_vector(protein_file):
    report_path = setup_reports(f'create_mut_corr_long_vector:{protein_file}')
    log(f'Report path:{report_path}')
    log(DEVICE)
    log('=' * 100)

    prism_data = None
    for f in Path(PRISM_FOLDER).rglob('*.txt'):
        file_name = os.path.basename(f)
        if file_name != protein_file:
            continue
        log(f'Parsing: {f}')
        prism_data = parse_prism_score_file(f)
        log(prism_data)

    assert prism_data is not None
    wt_sequence = prism_data.sequence

    log(f'Embedding wt protein: {prism_data.protein_name}')
    torch.manual_seed(0)
    sequence_embedder = EsmMsaEmbedding()
    emb = sequence_embedder.embed(wt_sequence)
    seq_emb_wt = torch.squeeze(emb)
    log(f'WT Emb.shape: {seq_emb_wt.shape}')

    log('Mutating protein')
    diff_data = []
    # for v in tqdm(prism_data.variants[:10]):
    for v in tqdm(prism_data.variants):
        # log(f'Mutating position: {v.position}, {v.aa_from} -> {v.aa_to}')
        mutation_ndx = v.position - 1
        assert wt_sequence[mutation_ndx] == v.aa_from
        mut_sequence = list(wt_sequence)
        mut_sequence[mutation_ndx] = v.aa_to
        mut_sequence = ''.join(mut_sequence)
        assert wt_sequence[mutation_ndx] != mut_sequence[mutation_ndx]

        torch.manual_seed(0)
        sequence_embedder = EsmMsaEmbedding()
        emb = sequence_embedder.embed(mut_sequence)
        seq_emb_mut = torch.squeeze(emb)

        # --- diff between wt and mut ---
        diff = torch.sub(seq_emb_wt, seq_emb_mut)
        diff_vector = diff[mutation_ndx].numpy()

        assert len(diff_vector) == 768
        # assert len(diff_vector) == 1280

        diff_data.append((diff_vector, v.score))

    assert len(diff_data) == len(prism_data.variants)
    dump_path = os.path.join(DUMP_ROOT, f'{protein_file}.diff.vector.pkl')
    with open(dump_path, "wb") as f:
        pickle.dump(diff_data, f)
    log(f'Created dump: {dump_path}')


def calc_mut_corr_long_vector(protein_file):
    report_path = setup_reports(f'calc_mut_corr_long_vector:{protein_file}')
    log(f'Report path:{report_path}')
    log(DEVICE)
    log('=' * 100)

    dump_path = os.path.join(DUMP_ROOT, f'{protein_file}.diff.vector.pkl')
    with open(dump_path, "rb") as f:
        diff_data = pickle.load(f)

    log(f'Loaded data, size={len(diff_data)}')
    amplitudes = []
    integrals = []
    stds = []
    scores = []
    for data in diff_data:
        diff_vector = data[0]
        score = data[1]
        amp = calc_amplitude(diff_vector)
        amplitudes.append(amp)
        integral = calc_sum(diff_vector)
        integrals.append(integral)
        std = np.std(diff_vector)
        stds.append(std)
        scores.append(score)

    log('=' * 100)
    assert len(amplitudes) == len(scores)
    assert len(integrals) == len(scores)
    assert len(stds) == len(scores)
    spearman_amp = abs(np.round(spearmanr(amplitudes, scores), 4)[0])
    spearman_sum = abs(np.round(spearmanr(integrals, scores), 4)[0])
    spearman_std = abs(np.round(spearmanr(stds, scores), 4)[0])
    log(f'{protein_file}')
    log(f'Spearman Amplitude (768): {spearman_amp}')
    log(f'Spearman Integral (768): {spearman_sum}')
    log(f'Spearman Std Dev (768): {spearman_std}')


def print_random_emb_diffs(step=15):
    report_path = setup_reports('print_random_emb_diffs')
    log(f'Report path:{report_path}')
    log(DEVICE)
    log('=' * 100)

    for f in Path(PRISM_FOLDER).rglob('*.txt'):
        file_name = os.path.basename(f)
        if file_name not in get_protein_files_dict().values():
            continue
        log(f'Parsing: {file_name}')
        dump_path = os.path.join(DUMP_ROOT, f'{file_name}.data.step_{step}.pkl')
        if not os.path.isfile(dump_path):
            log(f'Cannot found: {f}')
            continue
        with open(dump_path, "rb") as f:
            prism_data = pickle.load(f)
        assert prism_data is not None

        vs = random.choices(prism_data.variants, k=5)
        for v in vs:
            emb_diff = v.emb_diff
            plt.imshow(emb_diff, cmap='hot', interpolation='nearest')
            title = f'{prism_data.protein_name} mutation position = {v.position}'
            plt.title(title)
            # plt.show()
            plot_path = os.path.join(report_path, f'{prism_data.protein_name}.pos{v.position}.diff.png')
            # plt.colorbar(orientation='horizontal')
            plt.savefig(plot_path)
            log(f'Created plt: {plot_path}')
            plt.clf()


if __name__ == '__main__':
    # === NOTE: all functions here uses ESM MSA embedder ===

    # check_single_embedding()
    # create_diff_mean_dump('009_SUMO1_growth_abundance.txt')
    # create_diff_mean_dump('999_IF-1_DMS.txt')
    # create_diff_mean_dump('021_PAB1_doxycyclin_sensitivity.txt')
    # calc_mutations_correlations('009_SUMO1_growth_abundance.txt', 5)
    # calc_mutations_correlations('009_SUMO1_growth_abundance.txt', 10
    # calc_mutations_correlations('009_SUMO1_growth_abundance.txt', 15)
    # calc_mutations_correlations('999_IF-1_DMS.txt', 5)
    # calc_mutations_correlations('999_IF-1_DMS.txt', 10)
    # calc_mutations_correlations('999_IF-1_DMS.txt', 15)
    # calc_mutations_correlations('021_PAB1_doxycyclin_sensitivity.txt', 5)
    # calc_mutations_correlations('021_PAB1_doxycyclin_sensitivity.txt', 10)
    # calc_mutations_correlations('021_PAB1_doxycyclin_sensitivity.txt', 15)
    # create_mut_corr_long_vector('009_SUMO1_growth_abundance.txt')
    # create_mut_corr_long_vector('999_IF-1_DMS.txt')
    # create_mut_corr_long_vector('021_PAB1_doxycyclin_sensitivity.txt')
    # calc_mut_corr_long_vector('009_SUMO1_growth_abundance.txt')
    # calc_mut_corr_long_vector('999_IF-1_DMS.txt')
    # calc_mut_corr_long_vector('021_PAB1_doxycyclin_sensitivity.txt')
    # print_random_emb_diffs()
    # check_single_embedding()



    pass
