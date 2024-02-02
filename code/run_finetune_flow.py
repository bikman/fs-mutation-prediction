"""
This module run ONLY fine-tuning flow from ready result folders
"""
import argparse
import os
import pickle
import time

import numpy
import torch

from data_model import ModelConfig, Variant
from plots import PlotCreator
from prism_finetune_data_creation import create_fine_tune_diff_splits
from prism_finetune_train import fill_fine_tune_train_params, run_train_fine_tune, calc_train_ft_average_result, \
    calculate_ft_nn_mae
from prism_score_train import create_model
from run_prism_data_creation import create_seq_embedding_dict, create_seq_embedder, _load_diff_embeddings, \
    calculate_bins
from run_prism_data_creation import log
from utils import CFG, DEVICE, setup_reports, get_protein_files_dict, MODELS_FOLDER, AA_ALPHABETICAL


def create_result_txt(report_path, protein_filename, train_res):
    """
    Creates text file with MAE, Sp.
    """
    csv_res = os.path.join(report_path, 'result.txt')
    with open(csv_res, "w") as f:
        f.write(f'name:{protein_filename}\n')
        f.write(f'FT{train_res.ft_mutation_count} mae:{train_res.test_result.mae}\n')
        f.write(f'FT{train_res.ft_mutation_count} mape:{train_res.test_result.mape}\n')
        f.write(f'FT{train_res.ft_mutation_count} nn_mae:{train_res.test_result.nn_mae}\n')
        f.write(f'FT{train_res.ft_mutation_count} spearman:{train_res.test_result.spearman}\n')


def main():
    start_time = time.time()
    report_path = setup_reports('finetune_flow')
    log('@' * 100)
    log('ver:30.1.23')
    log('@' * 100)
    log(f'Report path:{report_path}')
    log(DEVICE)

    log('=' * 100)
    log(f"{CFG['general']['protein_set']=}")
    log(f"{CFG['general']['eval_protein_file_number']=}")
    log(f"{CFG['general']['model']=}")
    log(f"{CFG['flow_train']['lr']=}")
    log(f"{CFG['flow_fine_tune']['epochs']=}")
    log(f"{CFG['flow_fine_tune']['loops']=}")
    log(f"{CFG['general']['bins']=}")
    log(f"{CFG['flow_fine_tune']['alpha']=}")
    log(f"{CFG['fine_tuning_data_creation']['add_max_v']=}")
    log(f"{CFG['fine_tuning_data_creation']['data_count']=}")
    log(f"{CFG['fine_tuning_data_creation']['normalize_scores']=}")
    log(f"{CFG['fine_tuning_data_creation']['normalize_deltas']=}")
    log(f"{CFG['flow_fine_tune']['batch_norm']=}")
    log(f"{CFG['flow_fine_tune']['use_min_max']=}")
    log('=' * 100)

    loops = int(CFG['flow_fine_tune']['loops'])
    log(f'{loops=}')

    diff_len = int(CFG['general']['diff_len'])
    log(f'{diff_len=}')
    prism_data_list = _load_diff_embeddings(step=diff_len)
    calculate_bins(prism_data_list)

    total_variants_count = sum([len(d.variants) for d in prism_data_list])
    log(f'{total_variants_count=}')
    log('=== Create sequence embeddings ===')
    sequence_embedder = create_seq_embedder()
    pname_to_seq_embedding = create_seq_embedding_dict(prism_data_list, sequence_embedder)
    log(f'{len(pname_to_seq_embedding)=}')

    # ------------------------------
    mut_count_str = CFG['fine_tuning_data_creation']['data_count']
    if mut_count_str.endswith('p'):
        perc = int(mut_count_str.strip('p'))
        log(f'calculating {perc} % of mutations...')
        eval_protein_number = int(CFG['general']['eval_protein_file_number'])
        protein_filename = get_protein_files_dict()[eval_protein_number]
        ep_data = [x for x in prism_data_list if x.file_name == protein_filename]
        assert (len(ep_data) == 1)
        total_ep_mut_count = len(ep_data[0].variants)
        perc_mut_count = int(perc / 100.0 * total_ep_mut_count)
        CFG['fine_tuning_data_creation']['data_count'] = f'{perc_mut_count}'
        log(f'{perc_mut_count=}')

    mut_count = int(CFG['fine_tuning_data_creation']['data_count'])
    log(f'{mut_count=}')
    CFG['fine_tuning_data_creation']['eval_data_type'] = '1'  # per mutation, not per-position
    CFG['fine_tuning_data_creation']['destructive_data_only'] = '0'

    plots_path = os.path.join(report_path, 'plots_ft')
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    ep = int(CFG['general']['eval_protein_file_number'])
    protein_name = get_protein_files_dict()[ep]

    tmp_results = []
    for i in range(0, loops):  # repeat the training
        log(f'--- Loop {i} ---')

        # create model config
        log('=== Creating model config ===')
        cfg = ModelConfig()
        cfg.heads = int(CFG['flow_train']['heads'])
        cfg.seq_emb_size = int(CFG['general']['seq_emb_size'])
        cfg.diff_width = int(CFG['general']['diff_len'])
        cfg.attn_len = int(CFG['general']['attn_len'])
        cfg.cz = int(CFG['general']['cz'])
        cfg.deltas_encoder = int(CFG['general']['use_deltas_encoder'])

        log('=== Creating new model ===')
        model = create_model()
        log(model)
        log(f'Num.Model Parameters={sum([p.numel() for p in model.parameters()])}')
        model_num = int(CFG['general']['model'])
        model_path = os.path.join(MODELS_FOLDER, f'model_{model_num}_pretrain', str(ep), model.file_name)
        assert os.path.isfile(model_path)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        log('Model loaded for fine tuning...')
        log(f'From: {model_path}')

        # --- find eval min_max here ---
        min_max_eval_v = None
        use_min_max = int(CFG['flow_fine_tune']['use_min_max'])
        log(f'{use_min_max=}')
        if use_min_max == 1:
            log('$ - using min max in FT samples -$')
            # load eval result
            eval_res_path = os.path.join(MODELS_FOLDER, f'model_{model_num}_pretrain', str(ep), 'eval_result.pkl')
            with open(eval_res_path, "rb") as f:
                test_eval_res = pickle.load(f)
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


        eval_ft_split, train_ft_split, eval_ft_quantile_transformer = \
            create_fine_tune_diff_splits(pname_to_seq_embedding, min_max_eval_v)
        log('Created fine tune splits...')
        model_file_name = f'{model.file_name}.mutations.{mut_count}'
        log(f'{model_file_name=}')
        model.file_name = model_file_name
        log('Updated model name...')
        train_params = fill_fine_tune_train_params(model, report_path)
        train_ft_res = run_train_fine_tune(train_ft_split, eval_ft_split, train_params)
        calculate_ft_nn_mae(train_ft_res, eval_ft_quantile_transformer)
        tmp_results.append(train_ft_res)

    log('=== Pickle FT results ===')
    file_name = f'ft_results.pkl'
    dump_path = os.path.join(report_path, file_name)
    with open(dump_path, "wb") as f:
        pickle.dump(tmp_results, f)
    log(f'Saved FT results: {dump_path}')

    log('=== Calculate average result ===')
    tran_average_result = calc_train_ft_average_result(tmp_results)
    log(tran_average_result.test_result)

    log('=== Create plots ===')
    plot_title = f'{protein_name}, mut {mut_count}, avg.'
    PlotCreator.create_correlation_plot(plots_path, tran_average_result.test_result, title=plot_title,
                                        file_name=f'{protein_name}.ft_corr.mut_count_{mut_count}.average.png')
    tran_average_result.ft_mutation_count = mut_count
    create_result_txt(report_path, protein_name, tran_average_result)
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
    parser.add_argument('-bins', type=int, help='Number of bins to split the data', required=False)
    parser.add_argument('-alpha', type=float, help='Parameter coefficient for loss', required=False)
    parser.add_argument('-data_count', type=str, help='Number of mutations to run fine tuning', required=False)
    parser.add_argument('-norm_scores', type=str, help='Normalize scores for fine-tuning', required=False)
    parser.add_argument('-norm_deltas', type=str, help='Normalize deltas for fine-tuning', required=False)
    parser.add_argument('-batch_norm', type=int, help='Normalize data per batch for fine-tuning', required=False)
    parser.add_argument('-use_min_max', type=int, help='Use min/max replacement in FT sample', required=False)
    args = parser.parse_args()

    if args.prot_set is not None:
        CFG['general']['protein_set'] = str(args.prot_set)
    if args.ep is not None:
        CFG['general']['eval_protein_file_number'] = str(args.ep)
    if args.model is not None:
        CFG['general']['model'] = str(args.model)
    if args.lr is not None:
        CFG['flow_train']['lr'] = str(args.lr)
    if args.epochs is not None:
        CFG['flow_fine_tune']['epochs'] = str(args.epochs)
    if args.bins is not None:
        CFG['general']['bins'] = str(args.bins)
    if args.alpha is not None:
        CFG['flow_fine_tune']['alpha'] = str(args.alpha)
    if args.data_count is not None:
        CFG['fine_tuning_data_creation']['data_count'] = str(args.data_count)
    if args.norm_scores is not None:
        CFG['fine_tuning_data_creation']['normalize_scores'] = str(args.norm_scores)
    if args.norm_deltas is not None:
        CFG['fine_tuning_data_creation']['normalize_deltas'] = str(args.norm_deltas)
    if args.batch_norm is not None:
        CFG['flow_fine_tune']['batch_norm'] = str(args.batch_norm)
    if args.use_min_max is not None:
        CFG['flow_fine_tune']['use_min_max'] = str(args.use_min_max)

    # NOTE: print command line args CFG inside main

    main()
