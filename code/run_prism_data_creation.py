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
from pdb_data import PdbDataParser
from utils import DEVICE, PRISM_FOLDER, CFG, DUMP_ROOT, \
    MAX_SEQUENCE, PRISM_EVAL_SPLIT, PRISM_VALID_SPLIT, PRISM_TRAIN_SPLIT, ECOD_FOLDER, get_protein_files_dict, \
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
    embedder_type = int(CFG['flow_data_creation']['seq_embedder'])
    log(f'embedder_type={embedder_type}')
    sequence_embedder = EsmEmbeddingFactory.get_embedder(embedder_type)
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


def _create_per_protein_named_splits(dss, valid_names, eval_names):
    """
    Create evaluation and validation dataset splits according to list of names given by user.
    @param eval_names: names of proteins to be used in evaluation set
    @param valid_names: names of proteins to be used in validation set
    @param dss: all datasets
    @return: 3 splits for eval, train, and valid
    """
    assert len(valid_names) > 0
    assert len(eval_names) > 0
    train_split = []
    valid_split = []
    eval_split = []
    for ds in dss:
        if ds.protein_name in eval_names:
            eval_split.append(ds)
        elif ds.protein_name in valid_names:
            valid_split.append(ds)
        else:
            train_split.append(ds)
    assert len(train_split) > 0
    assert len(eval_split) == len(eval_names)
    assert len(valid_split) == len(valid_names)
    return eval_split, train_split, valid_split


def create_prism_diff_emb_data():
    start_time = time.time()
    report_path = setup_reports('create_prism_diff_emb_data')
    log(f'Report path:{report_path}')
    log(DEVICE)

    log(os.path.basename(__file__))
    eval_split, train_split, valid_split, pname_to_seq_embedding = create_diff_emb_splits()

    log(f'{DUMP_ROOT=}')
    if not os.path.exists(DUMP_ROOT):
        os.makedirs(DUMP_ROOT)
        log(f'Created: {DUMP_ROOT}')
    assert os.path.isdir(DUMP_ROOT)
    dump_path = os.path.join(DUMP_ROOT, PRISM_TRAIN_SPLIT)
    with open(dump_path, "wb") as f:
        pickle.dump(train_split, f)
    log(f'Saved: {dump_path}')
    dump_path = os.path.join(DUMP_ROOT, PRISM_VALID_SPLIT)
    with open(dump_path, "wb") as f:
        pickle.dump(valid_split, f)
    log(f'Saved: {dump_path}')
    dump_path = os.path.join(DUMP_ROOT, PRISM_EVAL_SPLIT)
    with open(dump_path, "wb") as f:
        pickle.dump(eval_split, f)
    log(f'Saved: {dump_path}')

    elapsed_time = time.time() - start_time
    log(f'time: {elapsed_time:5.2f} sec')
    print('OK')


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
    embedder_type = int(CFG['flow_data_creation']['seq_embedder'])
    log(f'embedder_type={embedder_type}')
    sequence_embedder = EsmEmbeddingFactory.get_embedder(embedder_type)
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
    embedder_type = int(CFG['flow_data_creation']['seq_embedder'])
    # log(f'embedder_type={embedder_type}')
    sequence_embedder = EsmEmbeddingFactory.get_embedder(embedder_type)
    emb = sequence_embedder.embed(mut_sequence)
    seq_emb_mut = torch.squeeze(emb)
    return seq_emb_mut


def _save_diff_embeddings(max_v=0, step=0):
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


def get_diff_sector_pdb(pdb_datas, emb_diff, v, step):
    """
    Create partial diff embedding for model according to PDB neighbors, if possible
    @param pdb_datas: list of pdb data object per chain
    @param emb_diff: embedding diff (WT - MUT) size (Nres, 768)
    @param v: mutation variant object
    @param step: +- number of mutations to check
    @return: sector of diff embedding of size (step + 1 + step, 768)
    """
    neighbors = []  # default
    # try to find if PDB data has neighbors
    for pdb_data in pdb_datas:
        if v.position in pdb_data.chain.distance_matrix.serial_to_closest:
            neighbors = pdb_data.chain.distance_matrix.serial_to_closest[v.position]
    if len(neighbors) == 0:  # no neighbors found - take by diff sector by indices +-step
        return get_embedding_sector(emb_diff, v.position - 1, step)
    # neighbor positions (1-based) must be converted to indices (0-based)
    neighbors_ndxs = [i - 1 for i in neighbors if 0 <= i - 1 < len(emb_diff)]  # take only legal positions
    assert len(neighbors_ndxs) <= 2 * step + 1
    # --------------------------------------------------------
    # try:
    diff_sector = emb_diff[neighbors_ndxs, :]  # take the sector
    # except:
    #     pass # for the debug
    # --------------------------------------------------------
    # if there were not enough neighbors, there will be not enough columns in sector
    if len(neighbors_ndxs) < 2 * step + 1:
        # pad up with zero rows for missing positions
        pad_count = (2 * step + 1) - len(neighbors_ndxs)
        assert pad_count > 0
        neighbors_ndxs = ([0] * pad_count) + neighbors_ndxs
        padding = torch.nn.ZeroPad2d((0, 0, pad_count, 0))
        diff_sector = padding(diff_sector)
    v.neighbors = neighbors_ndxs
    assert len(neighbors_ndxs) == 2 * step + 1
    assert len(diff_sector) == 2 * step + 1
    return diff_sector


def _save_diff_embeddings_pdb(max_v=0, step=0):
    """
    Parse MAVE PRISM files, parse PDB files,
    Use the list of the closest positions from PDB to a mutation location to create embedding diff slice.
    All the data is then pickled into DUMP folder
    @param max_v: for debug use. Set as limit of the number of variants for shorter run-time
    @param step: set the number of neighbors to take
    """
    log('Create PDB parser')
    num_of_neighbors = step * 2 + 1
    pdb_parser = PdbDataParser(log, num_of_neighbors)

    log('Create PRISM datas from PDB')
    for f in Path(PRISM_FOLDER).rglob('*.txt'):
        file_name = os.path.basename(f)
        if file_name not in get_protein_files_dict().values():
            continue
        log(f'Parsing: {file_name}')
        prism_data = parse_prism_score_file(f)
        log(prism_data)

        seq_emb_wt = get_seq_emb_wt(prism_data)
        log(f'WT Emb.shape: {seq_emb_wt.shape}')

        pdb_datas = None
        for pdb_file in Path(ECOD_FOLDER).rglob('*.pdb'):
            pdb_protein_name = pdb_parser.get_protein_name_from_pdb_file(pdb_file)
            if pdb_protein_name in prism_data.protein_name:
                pdb_datas = pdb_parser.parse_ecod_pdb_file(pdb_file)
        assert pdb_datas is not None
        assert len(pdb_datas) > 0

        log(f'Mutating protein: {prism_data.protein_name}')
        if max_v > 0:
            log(f'RUNNING SHORT DEBUG DATA CREATION max_v={max_v}')
            prism_data.variants = prism_data.variants[:max_v]  # for debug only
        for v in tqdm(prism_data.variants):
            mutation_ndx = v.position - 1
            seq_emb_mut = get_seq_emb_mut(mutation_ndx, v, prism_data.sequence)
            emb_diff = torch.sub(seq_emb_wt, seq_emb_mut)
            diff_sector = get_diff_sector_pdb(pdb_datas, emb_diff, v, step)
            assert diff_sector.size()[0] == 2 * step + 1
            assert diff_sector.size()[1] == EsmEmbeddingFactory.DIMENSION
            v.emb_diff = diff_sector.numpy()

        # now we have all the mutations with diff embeddings
        dump_path = os.path.join(CFG['general']['dump_root'], f'{prism_data.file_name}.data.pdb.step_{step}.pkl')
        log(f'Saving dump: {dump_path}')
        with open(dump_path, "wb") as dump_f:
            pickle.dump(prism_data, dump_f)
        log(f'Saved {dump_path}')


def _load_diff_embeddings(step=0):
    """
    TBD
    @return:
    """
    emb_dim = EsmEmbeddingFactory.get_emb_dim(int(CFG['flow_data_creation']['seq_embedder']))
    log('Load PRISM datas')
    use_pdb = int(CFG['general']['use_pdb'])
    log(f'Use PDB: {use_pdb}')
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
        if use_pdb == 0:
            dump_path = os.path.join(DUMP_ROOT, f'{file_name}.data.step_{step}.pkl')
        elif use_pdb == 1:
            dump_path = os.path.join(DUMP_ROOT, f'{file_name}.data.pdb.step_{step}.pkl')
        else:
            raise Exception(f'Illegal use_pdb={use_pdb}')
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


def create_prism_score_diff_data(max_v=None):
    """
    Parse PRISM files and create pickled data for further usage
    Prism files are saved into dumps folder, then they loaded during training
    @param max_v: for debug use. Set as limit of the number of variants for shorter run-time
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
    log(f"{CFG['flow_data_creation']['seq_embedder']=}")
    log(f"{CFG['general']['dump_root']=}")
    log(f"{CFG['flow_data_creation']['protein_id']=}")
    log(f"{CFG['flow_data_creation']['max_v']=}")
    log('=' * 100)

    diff_len = int(CFG['general']['diff_len'])
    log(f'{diff_len=}')
    if max_v is None:
        max_v = int(CFG['flow_data_creation']['max_v'])

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
    parser.add_argument('-seq_embedder', type=int, help='Sequence embedder type', required=False)
    parser.add_argument('-dump_root', type=str, help='Dump root folder path', required=False)
    parser.add_argument('-pid', type=int, help='Protein to create data for', required=False)
    parser.add_argument('-max_v', type=int, help='Maxi variants in data dump', required=False)
    args = parser.parse_args()

    if args.seq_embedder is not None:
        CFG['flow_data_creation']['seq_embedder'] = str(args.seq_embedder)
    if args.dump_root is not None:
        CFG['general']['dump_root'] = str(args.dump_root)
    if args.pid is not None:
        CFG['flow_data_creation']['protein_id'] = str(args.pid)
    if args.max_v is not None:
        CFG['flow_data_creation']['max_v'] = str(args.max_v)

    create_prism_score_diff_data(max_v=10)
