"""
Author M.Bikman
"""
import random
import logging
import time
import torch
import pickle
from utils import setup_reports, normalize_scores_only
from utils import DEVICE, PRISM_FOLDER, CFG, DUMP_ROOT, PRISM_FINE_TRAIN_SPLIT, \
    PRISM_FINE_EVAL_SPLIT, get_protein_files_dict, normalize_deltas_only, normalize_scores_ds
import os
from pathlib import Path
from run_prism_data_creation import parse_prism_score_file, create_seq_embedder, create_seq_embedding_dict, \
    calculate_bins
from pdb_data import PdbDataParser
from embeddings import EsmMsaEmbedding, TapeEmbedding, NativeEmbedding
from data_model import Variant, PrismScoreData
from dataset import PrismDiffEmbFineTuneDatasetCreator

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


def create_prism_fine_tune_data():
    start_time = time.time()
    report_path = setup_reports('fine_tune_score_train')
    log(f'Report path:{report_path}')
    log(DEVICE)

    log(os.path.basename(__file__))

    # eval_ft_split, train_ft_split = create_fine_tune_splits()
    eval_ft_split, train_ft_split = create_fine_tune_diff_splits()

    log(f'{DUMP_ROOT=}')
    dump_path = os.path.join(DUMP_ROOT, PRISM_FINE_TRAIN_SPLIT)
    with open(dump_path, "wb") as f:
        pickle.dump(train_ft_split, f)
    log(f'Saved: {dump_path}')
    dump_path = os.path.join(DUMP_ROOT, PRISM_FINE_EVAL_SPLIT)
    with open(dump_path, "wb") as f:
        pickle.dump(eval_ft_split, f)
    log(f'Saved: {dump_path}')

    elapsed_time = time.time() - start_time
    log(f'time: {elapsed_time:5.2f} sec')
    print('OK')


def create_fine_tune_diff_splits(pname_to_seq_embedding, min_max_eval_vs=None):
    log(f'create_diff_emb_splits')
    diff_len = int(CFG['general']['diff_len'])
    log(f'{diff_len=}')

    fine_tune_protein_number = int(CFG['general']['eval_protein_file_number'])
    log(f'{fine_tune_protein_number=}')
    if fine_tune_protein_number not in get_protein_files_dict():
        raise Exception(f'Illegal {fine_tune_protein_number=}')
    fine_tune_protein_filename = get_protein_files_dict()[fine_tune_protein_number]

    prism_data_list = []
    for f in Path(PRISM_FOLDER).rglob('*.txt'):
        if fine_tune_protein_filename not in str(f):
            continue
        file_name = os.path.basename(f)
        log(f'Parsing: {file_name}')
        dump_path = os.path.join(DUMP_ROOT, f'{file_name}.data.step_{diff_len}.pkl')
        if not os.path.isfile(dump_path):
            log(f'Cannot found: {f}')
            continue
        with open(dump_path, "rb") as df:
            prism_data = pickle.load(df)
        assert prism_data is not None
        # update original scores from MAVE file
        file_prism_data = parse_prism_score_file(str(f))
        for file_v, v in zip(file_prism_data.variants, prism_data.variants):
            v.score = file_v.score
            v.score_orig = file_v.score_orig
            v.ddE = file_v.ddE
            v.ddG = file_v.ddG
            v.ddE_orig = file_v.ddE
            v.ddG_orig = file_v.ddG
        log(f'Loaded dump: {prism_data}')
        prism_data_list.append(prism_data)
    assert len(prism_data_list) == 1

    calculate_bins(prism_data_list)

    norm_ft_scores = int(CFG['fine_tuning_data_creation']['normalize_scores'])
    if norm_ft_scores == 1:
        raise NotImplementedError("Not supported: norm_ft_scores")
        # log(f'Performing FT all scores normalization...')
        # normalize_scores_only(prism_data_list)
        # log(f'done!')

    norm_ft_dds = int(CFG['fine_tuning_data_creation']['normalize_deltas'])
    if norm_ft_dds == 1:
        raise NotImplementedError("Not supported: norm_ft_dds")
        # log(f'Performing FT deltas normalization...')
        # normalize_deltas_only(prism_data_list)
        # log(f'done!')

    eval_data_type = int(CFG['fine_tuning_data_creation']['eval_data_type'])
    log(f'{eval_data_type=}')
    data_count = int(CFG['fine_tuning_data_creation']['data_count'])
    log(f'{data_count=}')
    destructive_data_only = int(CFG['fine_tuning_data_creation']['destructive_data_only'])
    log(f'{destructive_data_only=}')

    allowed_proteins = [d.protein_name for d in prism_data_list]
    assert len(allowed_proteins) == 1
    fine_tune_name = allowed_proteins[0]
    log(f'{fine_tune_name=}')
    log(f'{fine_tune_protein_filename=}')
    set_creator = PrismDiffEmbFineTuneDatasetCreator()
    set_creator.eval_data_type = eval_data_type
    set_creator.data_count = data_count
    set_creator.destructive_data_only = destructive_data_only
    set_creator.prism_data_list = prism_data_list
    set_creator.name_to_seq_emb = pname_to_seq_embedding
    if min_max_eval_vs is not None:
        set_creator.min_eval_v = min_max_eval_vs[0]
        set_creator.max_eval_v = min_max_eval_vs[1]
    # create datasets
    train_split, eval_split = set_creator.create_datasets()

    log(f'{len(train_split)=}')
    log(f'{len(eval_split)=}')
    assert len(train_split) == 1
    assert len(eval_split) == 1

    # --- create FT normalization per dataset ---
    eval_quantile_transformer = None
    norm_ft_scores = int(CFG['fine_tuning_data_creation']['normalize_scores'])
    if norm_ft_scores == 2:
        log(f'Performing FT ds scores normalization...')
        eval_quantile_transformer = normalize_scores_ds(train_split[0])
        normalize_scores_ds(eval_split[0])  # no need to return transformer here
        log(f'done!')

    return eval_split, train_split, eval_quantile_transformer


if __name__ == '__main__':
    create_prism_fine_tune_data()
    pass
