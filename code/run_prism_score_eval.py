import logging
import os
import pickle
import random
import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
from plots import PlotCreator
from data_model import ModelConfig, Prediction, PositionPrediction
from train import eval_prism_scores_multi_sets
from utils import setup_reports, DEVICE, DUMP_ROOT, TrainParameters, CFG, TestParameters
from utils import PRISM_EVAL_SPLIT, PRISM_TRAIN_SPLIT, PRISM_VALID_SPLIT, PRISM_FINE_EVAL_SPLIT, \
    HOIE_RESULTS, get_protein_files_dict
from engine_struct_attn import PrismScoreEmbDiffSimpleModel, PrismScoreDeltasOnlyModel, PrismScoreDeltasEmbDiffModel, \
    PrismScoreDeltasEmbModel, PrismScoreNoDDGModel, PrismScoreNoDeltasModel, PrismScoreNoDDEModel

LOG_ENABLED = True
log = print
if LOG_ENABLED:
    # noinspection PyRedeclaration
    log = logging.info


def run_eval_on_model(batch_size, ds_list, model):
    test_params = TestParameters()
    test_params.model = model
    test_params.loaders_list = [DataLoader(x, batch_size=batch_size, shuffle=True) for x in ds_list]
    # test_params.loaders_list = [DataLoader(x, batch_size=batch_size, shuffle=False) for x in ds_list]
    log(f'Sum of loaders={sum([len(x) for x in test_params.loaders_list])}')
    test_res = eval_prism_scores_multi_sets(test_params, log)
    test_res.model_name = model.file_name
    log(test_res)
    return test_res


def load_pickled_split(file_to_eval):
    log(f'{DUMP_ROOT=}')
    log(f'Loading data...\n')
    dump_path = os.path.join(DUMP_ROOT, file_to_eval)
    with open(dump_path, "rb") as f:
        eval_split = pickle.load(f)
    log(f'loaded: {dump_path}\n')
    return eval_split


def pickle_test_result(report_path, run_type, test_res):
    """
    Writes test result to pickled file
    @param report_path: folder to save result object there
    @param run_type: result run type
    @param test_res: result object
    """
    file_name = f'{run_type}_result.pkl'
    dump_path = os.path.join(report_path, file_name)
    with open(dump_path, "wb") as f:
        pickle.dump(test_res, f)
    log(f'Saved test result: {dump_path}')


def load_hoie_result():
    """
    Create result for hoie object
    @return: loaded result
    """
    protein_number = int(CFG['general']['eval_protein_file_number'])
    protein_file_name = get_protein_files_dict()[protein_number]
    tokens = (os.path.splitext(protein_file_name)[0]).split('_')
    file_name_no_ext = '_'.join(tokens[1:])
    try:
        dump_path = os.path.join(HOIE_RESULTS, f'{file_name_no_ext}.hoie.result.data.pkl')
        with open(dump_path, "rb") as f:
            hoie_result = pickle.load(f)
        log(f'loaded: {dump_path}')
        return hoie_result
    except Exception as e:
        log(f'error: {e}')
        raise e


if __name__ == '__main__':
    pass
