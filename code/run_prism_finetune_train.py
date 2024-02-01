"""

"""
import glob
import logging
import os

import numpy
import torch
from sklearn.metrics import mean_absolute_error
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from run_prism_finetune_data_creation import create_fine_tune_diff_splits
from prism_score_eval import PlotCreator
from prism_score_train import create_model
from train import train_prism_fine_tune_multi_sets
from utils import TrainParameters, CFG, get_protein_files_dict, TrainResult, TestResult, smape

LOG_ENABLED = True
log = print
if LOG_ENABLED:
    # noinspection PyRedeclaration
    log = logging.info


def run_train_fine_tune(train_dss, eval_dss, train_params):
    """
    TBD
    @param train_dss:
    @param eval_dss:
    @param train_params:
    @return:
    """
    batch_size = int(CFG['flow_train']['batch_size'])
    log(f'Batch size: {batch_size}')
    train_loaders = []
    eval_loaders = []
    for train_set in train_dss:
        assert len(train_set) > 0
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)
    for eval_set in eval_dss:
        assert len(eval_set) > 0
        eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=True)
        eval_loaders.append(eval_loader)
    train_params.train_loaders_list = train_loaders
    train_params.valid_loaders_list = eval_loaders
    log(train_params)
    train_res = train_prism_fine_tune_multi_sets(train_params, log)
    # log(train_res)
    return train_res


def fill_fine_tune_train_params(model, report_path):
    train_params = TrainParameters()
    train_params.model = model
    train_params.loss = torch.nn.MSELoss()
    train_params.loss2 = torch.nn.MSELoss()
    lr = float(CFG['flow_fine_tune']['lr'])
    log(f'FT {lr=}')
    train_params.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if int(CFG['general']['use_scheduler']) > 0:
        gamma = float(CFG['general']['gamma'])
        log(f'{gamma=}')
        step_size = int(CFG['general']['step'])
        log(f'{step_size=}')
        train_params.scheduler = StepLR(train_params.optimizer, step_size=step_size, gamma=gamma)
    train_params.model_path = os.path.join(report_path, model.file_name)
    epochs = int(CFG['flow_fine_tune']['epochs'])
    log(f'Fine tune epochs: {epochs}')
    train_params.epochs = epochs
    train_params.loader_pairs = []
    train_params.bins = int(CFG['general']['bins'])
    train_params.alpha = float(CFG['flow_fine_tune']['alpha'])
    log(f'Fine tune alpha: {train_params.alpha}')
    return train_params


def run_random_positions_fine_tuning(positions_counts, pname_to_seq_embedding, report_path, loops):
    """
    TBD
    @param pname_to_seq_embedding:
    @param positions_counts:
    @param report_path:
    @return:
    """
    position_results = []
    for pos_count in positions_counts:
        CFG['fine_tuning_data_creation']['eval_data_type'] = '2'  # all mutations per position
        CFG['fine_tuning_data_creation']['data_count'] = pos_count
        CFG['fine_tuning_data_creation']['destructive_data_only'] = '0'  # non-destructive only

        tmp_results = []
        for i in range(0, loops):  # repeat the training
            model = create_model()
            model_path = os.path.join(report_path, model.file_name)
            if not os.path.isfile(model_path):
                # model doesn't exist in the report path we have NOT run full flow
                model_path = os.path.join(CFG['flow_fine_tune']['fine_tune_folder'],
                                          CFG['general']['eval_protein_file_number'], model.file_name)
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
            log('=' * 100)
            log('Model loaded for fine tuning...')
            log('=' * 100)

            eval_ft_split, train_ft_split = create_fine_tune_diff_splits(pname_to_seq_embedding)
            log('Created fine tune splits...')
            # this will change name is order not to re-write the main trained model!
            model_file_name = f'{model.file_name}.positions.{pos_count}'
            log(f'{model_file_name=}')
            model.file_name = model_file_name
            log('Updated model name...')
            train_params = fill_fine_tune_train_params(model, report_path)
            train_ft_res = run_train_fine_tune(train_ft_split, eval_ft_split, train_params)
            tmp_results.append(train_ft_res)

        train_average_result = calc_train_ft_average_result(tmp_results)
        position_results.append(train_average_result)
        log(train_average_result)
        log(f'{pos_count=}')
        log(train_average_result.get_test_results())
        log('Finished fine tune pos flow...')
    return position_results


def calculate_ft_nn_mae(train_ft_res, eval_ft_quantile_transformer):
    """
    TBD
    @param train_ft_res:
    @param eval_ft_quantile_transformer:
    """
    x = numpy.array(train_ft_res.test_result.pred_values).reshape(-1, 1)
    nn_pred_values = eval_ft_quantile_transformer.inverse_transform(x).flatten()
    nn_mae = numpy.round(mean_absolute_error(nn_pred_values, train_ft_res.test_result.orig_true_values), 4)
    smape_score = numpy.round(smape(train_ft_res.test_result.orig_true_values, nn_pred_values), 4)
    train_ft_res.test_result.nn_mae = nn_mae
    train_ft_res.test_result.mape = smape_score
    train_ft_res.test_result.nn_pred_values = nn_pred_values


def run_random_mutations_fine_tuning(
        random_mutations_counts, pname_to_seq_embedding, report_path, min_max_eval_vs, is_destructive, loops):
    """
    Fine-tuning on a randomly selected number of mutations
    @param pname_to_seq_embedding:
    @param loops: number of averaging loops
    @param random_mutations_counts: how many variants to check
    @param report_path: where to store results
    @param is_destructive: choose between destructive and non-destructive variants
    """
    random_mutations_results = []
    for mut_count in random_mutations_counts:
        log('=' * 100)
        log(f'Running file tune mutations count: {mut_count}')
        log(f'is_destructive={is_destructive}')
        log('=' * 100)
        CFG['fine_tuning_data_creation']['eval_data_type'] = '1'
        CFG['fine_tuning_data_creation']['data_count'] = mut_count
        CFG['fine_tuning_data_creation']['destructive_data_only'] = is_destructive

        # --- print correlation plot only for non-destructive mutations ---
        should_create_ft_plot = int(is_destructive) == 0
        plots_path = os.path.join(report_path, 'plots_ft')
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)
        protein_name = get_protein_files_dict()[int(CFG['general']['eval_protein_file_number'])]

        tmp_results = []
        for i in range(0, loops):  # repeat the training
            log(f'--- Loop {i} ---')
            model = create_model()
            model_path = os.path.join(report_path, model.file_name)
            if not os.path.isfile(model_path):
                # model doesn't exist in the report path we have NOT run full flow
                model_path = os.path.join(CFG['flow_fine_tune']['fine_tune_folder'],
                                          CFG['general']['eval_protein_file_number'], model.file_name)
                log(f'{model_path=}')

            assert os.path.isfile(model_path)
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
            log('Model loaded for fine tuning...')
            log(f'From: {model_path}')
            eval_ft_split, train_ft_split, eval_ft_quantile_transformer = \
                create_fine_tune_diff_splits(pname_to_seq_embedding, min_max_eval_vs)
            log('Created fine tune splits...')
            # this will change name is order not to re-write the main trained model!
            model_file_name = f'{model.file_name}.mutations.{is_destructive}.{mut_count}'
            log(f'{model_file_name=}')
            model.file_name = model_file_name
            log('Updated model name...')
            train_params = fill_fine_tune_train_params(model, report_path)
            train_ft_res = run_train_fine_tune(train_ft_split, eval_ft_split, train_params)
            calculate_ft_nn_mae(train_ft_res, eval_ft_quantile_transformer)
            tmp_results.append(train_ft_res)

        tran_average_result = calc_train_ft_average_result(tmp_results)

        if should_create_ft_plot:
            plot_title = f'{protein_name}, mut {mut_count}, avg.'
            PlotCreator.create_correlation_plot(plots_path, tran_average_result.test_result, title=plot_title,
                                                file_name=f'ft_corr.mut_count_{mut_count}.average.png')

        tran_average_result.ft_mutation_count = mut_count
        random_mutations_results.append(tran_average_result)
        log(tran_average_result)
        log(f'{mut_count=}')
        log(f'{is_destructive=}')
        log(tran_average_result.get_test_results())
        log('Finished fine tune flow...')
    return random_mutations_results


def calc_train_ft_average_result(results_list):
    """
    Calculate averaged result
    @param results_list: list of training results
    @return: averaged result of all the inputs
    """
    res = TrainResult()
    res.test_result = TestResult()
    maes = []
    nn_maes = []
    mapes = []
    pearsons = []
    spearmans = []
    r2s = []
    for train_res in results_list:
        maes.append(train_res.test_result.mae)
        mapes.append(train_res.test_result.mape)
        nn_maes.append(train_res.test_result.nn_mae)
        pearsons.append(train_res.test_result.pearson)
        spearmans.append(train_res.test_result.spearman)
        r2s.append(train_res.test_result.r2)
    res.test_result.mae = numpy.round(numpy.average(maes), 4)
    res.test_result.mape = numpy.round(numpy.average(mapes), 4)
    res.test_result.nn_mae = numpy.round(numpy.average(nn_maes), 4)
    res.test_result.pearson = numpy.round(numpy.average(pearsons), 4)
    res.test_result.spearman = numpy.round(numpy.average(spearmans), 4)
    res.test_result.r2 = numpy.round(numpy.average(r2s), 4)
    res.test_result.true_values = results_list[0].test_result.true_values
    ave_pred_values = []
    for i in range(len(results_list[0].test_result.true_values)):
        tmp_pred_values = []
        for train_res in results_list:
            tmp_pred_values.append(train_res.test_result.pred_values[i])
        ave_pred_values.append(numpy.average(tmp_pred_values))
    res.test_result.pred_values = numpy.array(ave_pred_values)
    assert len(res.test_result.true_values) == len(res.test_result.pred_values)
    return res


def clean_up_large_files(report_path):
    """
    Remove: *.pkl, *.pt [optional], *.pt.mutations.*, *.pt.positions.*
    """
    log('clean_up_large_files')
    victims = glob.glob(f'{report_path}/*.pt.mutations.*')
    victims += glob.glob(f'{report_path}/*.pt.positions.*')
    log(f'Found {len(victims)} files to clean-up')
    for f in victims:
        try:
            os.remove(f)
            log(f'Removed file: {f}')
        except Exception as e:
            log(f'Error while deleting file : {filePath}, {e}')


if __name__ == '__main__':
    pass
