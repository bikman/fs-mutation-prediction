"""
Create pickled files with all variants and diff embeddings per variant
"""
import argparse
import logging
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split
from tqdm import tqdm

from data_model import Variant, PrismScoreData
from dataset import PrismDiffEmbMultisetCreator
from embeddings import EsmEmbeddingFactory
from utils import DEVICE, PRISM_FOLDER, CFG, DUMP_ROOT, MAX_SEQUENCE, get_protein_files_dict, \
    normalize_scores_only, MULTITEST_PROTEINS
from utils import setup_reports, get_embedding_sector, ALL_PROTEIN_FILES_DICT, normalize_deltas_only

LOG_ENABLED = True
log = print
if LOG_ENABLED:
    # noinspection PyRedeclaration
    log = logging.info

USE_SEED = True
SEED = CFG['general']['seed']

if USE_SEED:
    print(f'SEED:{SEED}')
    random.seed(SEED)
    torch.manual_seed(SEED)


def get_sequence_from_prism_file(file_path):
    seq = None
    with open(file_path) as file_handle:
        for line in file_handle:
            if line.startswith('#') and 'sequence:' in line:
                # print(line)
                seq = line.split()[-1].strip()
    assert seq is not None
    if len(seq) > MAX_SEQUENCE:
        seq = seq[:MAX_SEQUENCE]
    return seq


def get_variants_from_prism_file(file_path):
    variants = []
    with open(file_path) as file:
        dataframe = pd.read_csv(file, delim_whitespace=True, comment='#', header=0,
                                keep_default_na=True, na_values=['Na', 'na'])

        for _, row in dataframe.iterrows():
            # noinspection PyBroadException
            try:
                v = Variant()
                v.load_from_series(row)
                if v.position >= MAX_SEQUENCE:
                    continue
                if abs(v.score) > 500:
                    continue
                variants.append(v)
            except Exception:
                continue
    return variants


def parse_prism_score_file(file_path):
    """
    TBD
    @param file_path: path to PRISM TXT file
    @return: PrismScoreData object
    """
    file_name = os.path.basename(file_path)
    tokens = file_name.split('_')
    protein_name: str = tokens[1]
    sequence = get_sequence_from_prism_file(file_path)
    variants = get_variants_from_prism_file(file_path)
    data = PrismScoreData()
    data.sequence = sequence
    data.variants = variants
    data.protein_name = protein_name
    data.file_name = file_name
    return data


def create_seq_embedder():
    log('Creating sequence embeddings')
    sequence_embedder = EsmEmbeddingFactory.get_embedder()
    log(f'Embedder: {sequence_embedder}')
    return sequence_embedder


def create_seq_embedding_dict(prism_data_list, sequence_embedder):
    pname_to_seq_embedding = {}
    for prism_data in prism_data_list:
        if prism_data.file_name not in get_protein_files_dict().values():
            log(f'Omitted protein {prism_data.protein_name}')
            continue
        if prism_data.protein_name in pname_to_seq_embedding:
            log(f'Skipped protein: {prism_data.protein_name}')
            continue
        log(f'Embedding protein: {prism_data.protein_name}')
        emb = sequence_embedder.embed(prism_data.sequence)
        seq_embedding = torch.squeeze(emb)
        log(f'Emb.shape: {seq_embedding.shape}')
        pname_to_seq_embedding[prism_data.protein_name] = seq_embedding
    return pname_to_seq_embedding


def _create_per_protein_splits(dss):
    """
    Take single protein for evaluation,
    others join and randomly split into train and validation sets
    @param dss:
    @return: 3 splits for eval, train, and valid
    """
    eval_prot_number = int(CFG['general']['eval_protein_file_number'])
    eval_prot_filename = get_protein_files_dict()[eval_prot_number]
    log(f'{eval_prot_filename=}')
    eval_ds = [ds for ds in dss if ds.file_name == eval_prot_filename]
    assert len(eval_ds) == 1
    train_val_dss = [ds for ds in dss if ds.file_name != eval_prot_filename]
    assert len(train_val_dss) == len(dss) - len(eval_ds)

    train_split = []
    valid_split = []
    eval_split = eval_ds
    for dataset in train_val_dss:
        train_size = int(0.8 * len(dataset))  # 20 % of train is validation
        validation_size = len(dataset) - train_size
        train_set, validation_set = random_split(dataset, [train_size, validation_size])
        train_split.append(train_set)
        valid_split.append(validation_set)
    return eval_split, train_split, valid_split


def get_seq_emb_wt(prism_data):
    """
    Create WT sequence embedding
    @param prism_data: prism data object
    @return: sequence embedding for wild type
    """
    log('Create embeddings')
    assert prism_data is not None
    wt_sequence = prism_data.sequence
    log(f'Embedding wt protein: {prism_data.protein_name}')
    torch.manual_seed(SEED)
    sequence_embedder = EsmEmbeddingFactory.get_embedder()
    emb = sequence_embedder.embed(wt_sequence)
    seq_emb_wt = torch.squeeze(emb)
    return seq_emb_wt


def get_seq_emb_mut(mutation_ndx, v, wt_sequence):
    """
    Create MUT sequence embedding
    @param mutation_ndx: index of mutation (0-based)
    @param v: mutation variant
    @param wt_sequence: wild type sequence
    @return: sequence embedding for mutated type
    """
    # mutate the sequence
    assert wt_sequence[mutation_ndx] == v.aa_from
    mut_sequence = list(wt_sequence)
    mut_sequence[mutation_ndx] = v.aa_to
    mut_sequence = ''.join(mut_sequence)
    assert wt_sequence[mutation_ndx] != mut_sequence[mutation_ndx]
    torch.manual_seed(SEED)
    # we have to create a new embedder every time
    sequence_embedder = EsmEmbeddingFactory.get_embedder()
    emb = sequence_embedder.embed(mut_sequence)
    seq_emb_mut = torch.squeeze(emb)
    return seq_emb_mut


def _save_diff_embeddings(max_v, step=0):
    """
    Parse MAVE PRISM files
    Use +-step locations to a mutation position to create embedding diff slice.
    All the data is then pickled into DUMP folder
    During training the prism data is loaded from DMP folder
    @param max_v: for debug use. Set as limit of the number of variants for shorter run-time
    @param step: set the number of neighbors to take
    """
    log('Create PRISM datas')
    pid = int(CFG['flow_data_creation']['protein_id'])
    log(f'{pid=}')
    for f in Path(PRISM_FOLDER).rglob('*.txt'):
        file_name = os.path.basename(f)
        if pid != -1 and ALL_PROTEIN_FILES_DICT[pid] != file_name:
            continue
        # ----- use only partial proteins list -------
        if file_name not in get_protein_files_dict().values():
            continue
        # --------------------------------------------
        log(f'Parsing: {file_name}')
        prism_data = parse_prism_score_file(f)
        log(prism_data)

        seq_emb_wt = get_seq_emb_wt(prism_data)  # wild type seq emb

        log(f'WT Emb.shape: {seq_emb_wt.shape}')

        log(f'Mutating protein: {prism_data.protein_name}')
        if max_v > 0:
            log(f'RUNNING SHORT DEBUG DATA CREATION max_v={max_v}')
            prism_data.variants = prism_data.variants[:max_v]  # for debug only
        for v in tqdm(prism_data.variants):
            mutation_ndx = v.position - 1
            seq_emb_mut = get_seq_emb_mut(mutation_ndx, v, prism_data.sequence)  # mutated type seq emb
            emb_diff = torch.sub(seq_emb_wt, seq_emb_mut)  # difference of embeddings

            diff_sector = get_embedding_sector(emb_diff, mutation_ndx, step)
            assert diff_sector.size()[0] == 2 * step + 1
            assert diff_sector.size()[1] == EsmEmbeddingFactory.DIMENSION
            v.emb_diff = diff_sector.cpu().numpy()

        # now we have all the mutations with diff embeddings
        dump_path = os.path.join(CFG['general']['dump_root'], f'{prism_data.file_name}.data.step_{step}.pkl')
        log(f'Saving dump: {dump_path}')
        with open(dump_path, "wb") as f:
            pickle.dump(prism_data, f)
        log(f'Saved {dump_path}')


def _load_diff_embeddings(step=0):
    """
    TBD
    @return:
    """
    emb_dim = EsmEmbeddingFactory.get_emb_dim()
    log('Load PRISM datas')
    prism_datas = []
    protein_dict = get_protein_files_dict()
    log(f'Protein dict length:{len(protein_dict)}')
    log(f'Protein indices: {protein_dict.keys()}')
    log(f'{DUMP_ROOT=}')
    for f in Path(PRISM_FOLDER).rglob('*.txt'):
        file_name = os.path.basename(f)
        if file_name not in protein_dict.values():
            continue
        log(f'Parsing: {file_name}')
        dump_path = os.path.join(DUMP_ROOT, f'{file_name}.data.step_{step}.pkl')
        if not os.path.isfile(dump_path):
            log(f'Cannot found: {f}')
            continue

        with open(dump_path, "rb") as df:
            prism_data = pickle.load(df)
        assert prism_data is not None

        log('Update original scores from MAVE file')
        # update original deltas and scores from MAVE file
        file_prism_data = parse_prism_score_file(str(f))
        for file_v, v in zip(file_prism_data.variants, prism_data.variants):
            v.score = file_v.score
            v.score_orig = file_v.score_orig
            v.ddE = file_v.ddE
            v.ddG = file_v.ddG
            v.ddE_orig = file_v.ddE
            v.ddG_orig = file_v.ddG

        log(f'Loaded dump: {prism_data}')
        prism_datas.append(prism_data)
        for v in prism_data.variants:
            if v.emb_diff.shape != (31, emb_dim):
                raise Exception(f"Illegal shape {v.emb_diff.shape}")
    return prism_datas


def create_prism_score_diff_data(max_v=-1):
    """
    Parse PRISM files and create pickled data for further usage
    Prism files are saved into dumps folder, then they loaded during training
    @param max_v: for debug use. Set as limit of the number of variants for shorter run-time.
    Default value -1 for all variants.
    """
    start_time = time.time()
    report_path = setup_reports('score_data_creation')
    log(f'Report path:{report_path}')
    log('@' * 100)
    log('ver:3.6.23')
    log('@' * 100)
    log(DEVICE)

    log(os.path.basename(__file__))

    log('=' * 100)
    log(f"{CFG['general']['dump_root']=}")
    log(f"{CFG['flow_data_creation']['protein_id']=}")
    log('=' * 100)

    diff_len = int(CFG['general']['diff_len'])
    log(f'{diff_len=}')

    _save_diff_embeddings(max_v=max_v, step=diff_len)

    elapsed_time = time.time() - start_time
    log(f'time: {elapsed_time:5.2f} sec')
    print('OK')


def calculate_bins(prism_data_list):
    """
    Splits list of variants (per protein) to bins
    Assigns bin to every variant
    @param prism_data_list: list of prism data objects
    """
    num_bins = int(CFG['general']['bins'])
    log(f'{num_bins=}')
    log(f'adding bins...')
    for prism_data in prism_data_list:
        list.sort(prism_data.variants, key=lambda x: x.score, reverse=False)
        chunked_list = np.array_split(prism_data.variants, num_bins)
        assert len(chunked_list) == num_bins
        bin_id = 0
        for chunk in chunked_list:
            for v in chunk:
                v.bin = bin_id
            bin_id += 1
        for v in prism_data.variants:
            assert v.bin is not None
    log(f'done!')


def create_diff_emb_splits():
    """
    Load diff embeddings and create 3 splits for train, valid and test
    @return: 3 slits
    """
    log(f'create_diff_emb_splits')
    diff_len = int(CFG['general']['diff_len'])
    log(f'{diff_len=}')

    prism_data_list = _load_diff_embeddings(step=diff_len)

    calculate_bins(prism_data_list)

    normalize_enabled = int(CFG['flow_data_creation']['normalize_scores'])
    log(f'{normalize_enabled=}')
    if normalize_enabled == 1:
        log(f'Performing train scores normalization...')
        normalize_scores_only(prism_data_list)
        log(f'done!')

    normalize_dds_enabled = int(CFG['flow_data_creation']['normalize_deltas'])
    log(f'{normalize_dds_enabled=}')
    if normalize_dds_enabled == 1:
        log(f'Performing train deltas normalization...')
        normalize_deltas_only(prism_data_list)
        log(f'done!')

    total_variants_count = sum([len(d.variants) for d in prism_data_list])
    log(f'{total_variants_count=}')

    log('Create sequence embeddings')
    sequence_embedder = create_seq_embedder()
    pname_to_seq_embedding = create_seq_embedding_dict(prism_data_list, sequence_embedder)
    log(f'{len(pname_to_seq_embedding)=}')

    log('Create datasets')
    set_creator = PrismDiffEmbMultisetCreator(prism_data_list, pname_to_seq_embedding)
    dss = set_creator.create_datasets()

    log('Cutoff mutation variants')
    cutoff_percent = int(CFG['flow_data_creation']['variants_cutoff'])
    if cutoff_percent >= 100 or cutoff_percent <= 0:
        log('Skipped!')
    else:
        log(f'{cutoff_percent=}')
        for ds in dss:
            if ds.file_name == get_protein_files_dict()[int(CFG['general']['eval_protein_file_number'])]:
                log(f'Skipped eval DS {ds}')
                continue
            total = len(ds.data)
            cutoff = int(total * (cutoff_percent / 100.0))
            cut_variants = random.sample(ds.data, cutoff)
            ds.data = cut_variants
            log(f'{total} -> after cut: {len(ds)}')

    # --- filter out datasets for MULTI-TEST proteins ---
    log('Filtering out multi-test proteins')
    log(f'Number of datasets before filtering: {len(dss)}')
    ep = int(CFG['general']['eval_protein_file_number'])
    log(f'{ep=}')
    for sublist in MULTITEST_PROTEINS:
        if ep in sublist:
            victims = [x for x in sublist if x != ep]
            log(f'Found victim ids!!!  {victims}')
            victim_file_names = [get_protein_files_dict()[x] for x in victims]
            dss = [ds for ds in dss if ds.file_name not in victim_file_names]
    log(f'Number of datasets after filtering: {len(dss)}')

    eval_split, train_split, valid_split = _create_per_protein_splits(dss)
    return eval_split, train_split, valid_split, pname_to_seq_embedding, prism_data_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inputs and setup for data creation')
    parser.add_argument('-dump_root', type=str, help='Dump root folder path', required=False)
    parser.add_argument('-pid', type=int, help='Protein to create data for (-1 for "all")', required=False)
    args = parser.parse_args()

    if args.dump_root is not None:
        CFG['general']['dump_root'] = str(args.dump_root)
    if args.pid is not None:
        CFG['flow_data_creation']['protein_id'] = str(args.pid)

    create_prism_score_diff_data()
    # create_prism_score_diff_data(max_v=10)  # will create dumps with only 10 mutations
