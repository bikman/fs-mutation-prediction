"""
Runs training and evaluation BOTH
"""
import argparse
import os
import pickle
import time

import numpy
import torch
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error

from data_model import Variant
from prism_score_train import fill_train_parameters, run_train, create_acc_loss_plots, create_model, \
    create_loss_batch_plot
from run_prism_data_creation import create_diff_emb_splits, log
from run_prism_finetune_train import run_random_mutations_fine_tuning, clean_up_large_files
from prism_score_eval import run_eval_on_model, PlotCreator, \
    pickle_test_result
from utils import CFG, DEVICE, PRISM_EVAL_SPLIT, PRISM_TRAIN_SPLIT, PRISM_VALID_SPLIT, setup_reports, \
    get_protein_files_dict, create_result_txt, AA_ALPHABETICAL


def save_splits(report_path, eval_split, train_split, valid_split):
    """
    Save all the data loaders into given directory
    @param report_path: folder to save the data
    @param eval_split: evaluation data
    @param train_split: train data
    @param valid_split: validation data
    """
    assert os.path.isdir(report_path)
    dump_path = os.path.join(report_path, PRISM_TRAIN_SPLIT)
    with open(dump_path, "wb") as f:
        pickle.dump(train_split, f)
    log(f'Saved: {dump_path}')
    dump_path = os.path.join(report_path, PRISM_VALID_SPLIT)
    with open(dump_path, "wb") as f:
        pickle.dump(valid_split, f)
    log(f'Saved: {dump_path}')
    dump_path = os.path.join(report_path, PRISM_EVAL_SPLIT)
    with open(dump_path, "wb") as f:
        pickle.dump(eval_split, f)
    log(f'Saved: {dump_path}')


def create_fine_tuning_mutation_counts(total_eval_mut_count):
    two_perc = int(total_eval_mut_count / 50)
    three_perc = int(total_eval_mut_count / 33.3)
    five_perc = int(total_eval_mut_count / 20)
    return [str(two_perc), str(three_perc), str(five_perc)]


def calculate_nn_mae(test_eval_res):
    """
    TBD
    @param test_eval_res: TBD
    """
    test_eval_res.nn_mae = []
    for perc in [2, 3, 5]:
        total_values = len(test_eval_res.true_values)
        selection_len = int(total_values * perc / 100.0)
        sampled_values = numpy.random.choice(test_eval_res.true_values, selection_len, replace=False)
        x = numpy.array(sampled_values).reshape(-1, 1)
        quantile_transformer = preprocessing.QuantileTransformer(n_quantiles=4)
        quantile_transformer.fit(x)

        y = numpy.array(test_eval_res.pred_values).reshape(-1, 1)
        nn_pred_values = quantile_transformer.inverse_transform(y)
        nn_mae = numpy.round(mean_absolute_error(nn_pred_values, test_eval_res.orig_true_values), 4)
        test_eval_res.nn_mae.append(nn_mae)


def main():
    start_time = time.time()
    report_path = setup_reports('full_flow')
    log('@' * 100)
    log('ver:25.1.24')
    log('@' * 100)
    log(f'Report path:{report_path}')
    log(DEVICE)

    log('=' * 100)
    log(f"{CFG['general']['protein_set']=}")
    log(f"{CFG['general']['eval_protein_file_number']=}")
    log(f"{CFG['general']['model']=}")
    log(f"{CFG['general']['attn_len']=}")
    log(f"{CFG['general']['use_scheduler']=}")
    log(f"{CFG['flow_train']['lr']=}")
    log(f"{CFG['flow_train']['epochs']=}")
    log(f"{CFG['general']['train_set_count']=}")
    log(f"{CFG['flow_data_creation']['variants_cutoff']=}")
    log(f"{CFG['general']['gamma']=}")
    log(f"{CFG['general']['step']=}")
    log(f"{CFG['general']['use_pdb']=}")
    log(f"{CFG['general']['use_deltas_encoder']=}")
    log(f"{CFG['flow_train']['patience']=}")
    log(f"{CFG['general']['bins']=}")
    log(f"{CFG['flow_train']['alpha']=}")
    log(f"{CFG['flow_data_creation']['normalize_scores']=}")
    log(f"{CFG['fine_tuning_data_creation']['normalize_scores']=}")
    log(f"{CFG['flow_data_creation']['normalize_deltas']=}")
    log(f"{CFG['fine_tuning_data_creation']['normalize_deltas']=}")
    log(f"{CFG['flow_train']['batch_norm']=}")
    log(f"{CFG['flow_fine_tune']['batch_norm']=}")
    log('=' * 100)

    log('=' * 100)
    eval_split, train_split, valid_split, pname_to_seq_embedding, prism_data_list = create_diff_emb_splits()

    assert len(eval_split) == 1
    total_eval_mut_count = len(eval_split[0])  # total number of mutations in eval protein
    log(f'{total_eval_mut_count=}')

    log('=' * 100)
    log('Created splits...')
    log('=' * 100)

    # --- save splits to PKL files ---
    # save_splits(report_path, eval_split, train_split, valid_split)

    log('=' * 100)
    log('Saved all big splits...')
    log('=' * 100)

    model = create_model()
    train_params = fill_train_parameters(model, 0, report_path)

    # --------- Train model ---------------
    train_res = run_train(train_split, valid_split, train_params)
    create_acc_loss_plots(report_path, train_res)
    create_loss_batch_plot(report_path, train_res)
    log('=' * 100)
    log('Finished training...')
    log('=' * 100)

    log('=' * 100)
    log('Hoie result loaded...')
    log('=' * 100)

    model = create_model()
    state_dict = torch.load(os.path.join(report_path, model.file_name))
    model.load_state_dict(state_dict)
    log('=' * 100)
    log('Model loaded for evaluation tests...')
    log('=' * 100)

    batch_size = int(CFG['flow_eval']['batch_size'])
    log(f'Batch size: {batch_size}')
    plotter = PlotCreator()

    # --------- Test split evaluation ---------------
    test_eval_res = run_eval_on_model(batch_size, eval_split, model)
    calculate_nn_mae(test_eval_res)
    pickle_test_result(report_path, 'eval', test_eval_res)
    log('=' * 100)
    log('Finished test evaluation...')
    log('=' * 100)

    # find min and max variants from evaluation result
    min_pred_ndx = int(numpy.argmin(test_eval_res.pred_values))
    max_pred_ndx = int(numpy.argmax(test_eval_res.pred_values))
    max_pred_src = int(test_eval_res.src_indices[max_pred_ndx])
    max_pred_dst = int(test_eval_res.dst_indices[max_pred_ndx])
    min_pred_src = int(test_eval_res.src_indices[min_pred_ndx])
    min_pred_dst = int(test_eval_res.dst_indices[min_pred_ndx])
    min_pos = int(test_eval_res.pos_values[min_pred_ndx])
    max_pos = int(test_eval_res.pos_values[max_pred_ndx])
    min_aa_src = AA_ALPHABETICAL[min_pred_src]
    min_aa_dst = AA_ALPHABETICAL[min_pred_dst]
    max_aa_src = AA_ALPHABETICAL[max_pred_src]
    max_aa_dst = AA_ALPHABETICAL[max_pred_dst]
    min_true_val = test_eval_res.true_values[min_pred_ndx]
    max_true_val = test_eval_res.true_values[max_pred_ndx]
    v_pred_min = Variant()
    v_pred_min.position = min_pos
    v_pred_min.score_orig = min_true_val
    v_pred_min.score = min_true_val
    v_pred_min.aa_from = min_aa_src
    v_pred_min.aa_to = min_aa_dst
    v_pred_max = Variant()
    v_pred_max.position = max_pos
    v_pred_max.score_orig = max_true_val
    v_pred_max.score = max_true_val
    v_pred_max.aa_from = max_aa_src
    v_pred_max.aa_to = max_aa_dst
    min_max_eval_v = [v_pred_min, v_pred_max]
    log('Calculated v_pred_min and v_pred_max')
    log(f'{str(v_pred_min)}')
    log(f'{str(v_pred_max)}')

    # --------- Train split evaluation ---------------
    test_res = run_eval_on_model(batch_size, train_split, model)
    plotter.create_train_plots(report_path, test_res)
    pickle_test_result(report_path, 'train', test_res)
    log('=' * 100)
    log('Finished train evaluation...')
    log('=' * 100)

    # --------- Validation split evaluation ---------------
    test_res = run_eval_on_model(batch_size, valid_split, model)
    plotter.create_valid_plots(report_path, test_res)
    pickle_test_result(report_path, 'valid', test_res)
    log('=' * 100)
    log('Finished validation evaluation...')
    log('=' * 100)

    log('/' * 100)
    log('@' * 100)
    log('/' * 100)

    loops = int(CFG['flow_fine_tune']['loops'])
    log(f'{loops=}')

    # --------- Fine tuning part 1 ---------------
    fine_tune_protein_number = int(CFG['general']['eval_protein_file_number'])
    protein_filename = get_protein_files_dict()[fine_tune_protein_number]

    # mutations_counts = list of mutation counts
    mutations_counts = create_fine_tuning_mutation_counts(total_eval_mut_count)
    log(f'Fine tuning mutation counts: {mutations_counts}')
    ft_nd_results = run_random_mutations_fine_tuning(
        mutations_counts, pname_to_seq_embedding, report_path, min_max_eval_v, is_destructive='0', loops=loops)
    title = f'{protein_filename}: fine tuning - random variants'
    plotter.create_fine_tune_variants_plot(report_path, 'random_variants', title, mutations_counts, ft_nd_results)

    create_result_txt(report_path, protein_filename, test_eval_res, ft_nd_results)
    log('Result text file created...')

    log('=' * 100)
    log('Finished fine tune random NON-DESTRUCTIVE mutations training...')
    log('=' * 100)

    try:
        clean_up_large_files(report_path)
        log('Finished cleanup...')
    except Exception as e:
        log(f'Cannot clean-up! Error:{e}')
    log('=' * 100)

    log(f'Report path:{report_path}')

    elapsed_time = time.time() - start_time
    log(f'time: {elapsed_time:5.2f} sec')
    print('OK')


if __name__ == '__main__':
    # -- re-write the configuration if needed --
    parser = argparse.ArgumentParser(description='Inputs and setup for the model')
    parser.add_argument('-prot_set', type=int, help='Select proteins subset for model', required=False)
    parser.add_argument('-ep', type=int, help='Evaluation protein id', required=False)
    parser.add_argument('-model', type=int, help='Model to use for train/eval', required=False)
    parser.add_argument('-lr', type=float, help='Learning rate', required=False)
    parser.add_argument('-epochs', type=int, help='Number of epochs', required=False)
    parser.add_argument('-train_count', type=int, help='Number of protein in train set', required=False)
    parser.add_argument('-v_cutoff', type=int, help='Percentage of variants to be taken from data', required=False)
    parser.add_argument('-attn', type=int, help='Length of attention neighbors', required=False)
    parser.add_argument('-sched', type=int, help='Use scheduler (1 or 0)', required=False)
    parser.add_argument('-gamma', type=float, help='Gamma param for LR scheduler', required=False)
    parser.add_argument('-step', type=int, help='Step param for LR scheduler', required=False)
    parser.add_argument('-use_pdb', type=int, help='Use PDB indexed diffs (1 or 0)', required=False)
    parser.add_argument('-deltas_enc', type=int, help='Use encoder for deltas (1 or 0)', required=False)
    parser.add_argument('-patience', type=int, help='When to stop the training', required=False)
    parser.add_argument('-bins', type=int, help='Number of bins to split the data', required=False)
    parser.add_argument('-alpha', type=float, help='Parameter coefficient for loss', required=False)
    parser.add_argument('-norm_scores', type=int, help='Normalize scores train & eval', required=False)
    parser.add_argument('-norm_scores_ft', type=int, help='Normalize scores FT', required=False)
    parser.add_argument('-norm_deltas', type=int, help='Normalize deltas', required=False)
    parser.add_argument('-batch_norm', type=int, help='Normalize data per batch', required=False)
    args = parser.parse_args()

    if args.prot_set is not None:
        CFG['general']['protein_set'] = str(args.prot_set)
    if args.ep is not None:
        CFG['general']['eval_protein_file_number'] = str(args.ep)
    if args.model is not None:
        CFG['general']['model'] = str(args.model)
    if args.attn is not None:
        CFG['general']['attn_len'] = str(args.attn)
    if args.sched is not None:
        CFG['general']['use_scheduler'] = str(args.sched)
    if args.lr is not None:
        CFG['flow_train']['lr'] = str(args.lr)
    if args.epochs is not None:
        CFG['flow_train']['epochs'] = str(args.epochs)
    if args.train_count is not None:
        CFG['general']['train_set_count'] = str(args.train_count)
    if args.v_cutoff is not None:
        CFG['flow_data_creation']['variants_cutoff'] = str(args.v_cutoff)
    if args.gamma is not None:
        CFG['general']['gamma'] = str(args.gamma)
    if args.step is not None:
        CFG['general']['step'] = str(args.step)
    if args.use_pdb is not None:
        CFG['general']['use_pdb'] = str(args.use_pdb)
    if args.use_pdb is not None:
        CFG['general']['use_deltas_encoder'] = str(args.deltas_enc)
    if args.patience is not None:
        CFG['flow_train']['patience'] = str(args.patience)
    if args.bins is not None:
        CFG['general']['bins'] = str(args.bins)
    if args.alpha is not None:
        CFG['flow_train']['alpha'] = str(args.alpha)
    if args.norm_scores is not None:
        CFG['flow_data_creation']['normalize_scores'] = str(args.norm_scores)
    if args.norm_scores_ft is not None:
        CFG['fine_tuning_data_creation']['normalize_scores'] = str(args.norm_scores_ft)
    if args.norm_deltas is not None:
        CFG['flow_data_creation']['normalize_deltas'] = str(args.norm_deltas)
        CFG['fine_tuning_data_creation']['normalize_deltas'] = str(args.norm_deltas)
    if args.batch_norm is not None:
        CFG['flow_train']['batch_norm'] = str(args.batch_norm)
        CFG['flow_fine_tune']['batch_norm'] = str(args.batch_norm)

    # NOTE: command line args CFG are printed inside main

    main()
