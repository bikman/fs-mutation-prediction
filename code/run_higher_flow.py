'''
TBD
'''
import argparse
import logging
import os
import pickle
import random
import time
import torch
import glob
import numpy
import torch.optim as optim
from sys import platform
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from utils import CFG, DEVICE, setup_reports, get_protein_files_dict, DATASETS_FOLDER, EPISODES_PRETRAINED, \
    normalize_scores_ds
from run_prism_data_creation import log, _load_diff_embeddings, calculate_bins, create_seq_embedder, \
    create_seq_embedding_dict, PrismDiffEmbMultisetCreator
from run_prism_score_train import create_model_config, create_model
from episodes_dataset import EpisodesCreator, DataContext
from data_model import Prediction
from run_prism_score_eval import PlotCreator, load_hoie_result
from episodes_fn import train_epoch, val_epoch, test_epoch


def create_result_txt(report_path, protein_filename, test_res):
    """
    Creates text file with MAE, Sp.
    """
    csv_res = os.path.join(report_path, 'result.txt')
    with open(csv_res, "w") as f:
        f.write(f'name:{protein_filename}\n')
        f.write(f'mae:{test_res.mae}\n')
        f.write(f'spearman:{test_res.spearman}\n')


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


def main():
    start_time = time.time()
    report_path = setup_reports('run_higher_flow')
    log('@' * 100)
    log('ver:12.12.23')
    log('@' * 100)
    log(f'Report path:{report_path}')
    log(DEVICE)

    log('=' * 100)
    log(f"{CFG['general']['protein_set']=}")
    log(f"{CFG['general']['eval_protein_file_number']=}")
    log(f"{CFG['general']['model']=}")
    log(f"{CFG['flow_train']['lr']=}")
    log(f"{CFG['episodes_train']['episodes']=}")
    log(f"{CFG['general']['bins']=}")
    log(f"{CFG['flow_train']['alpha']=}")
    log(f"{CFG['flow_train']['batch_size']=}")
    log(f"{CFG['episodes_train']['num_ways']=}")
    log(f"{CFG['episodes_train']['support_set_size']=}")
    log(f"{CFG['episodes_train']['query_set_size']=}")
    log(f"{CFG['episodes_train']['inner_epochs']=}")
    log(f"{CFG['episodes_train']['use_pretrain']=}")
    log(f"{CFG['episodes_train']['inner_lr']=}")
    log(f"{CFG['episodes_train']['use_validation']=}")
    log(f"{CFG['episodes_train']['use_scheduler']=}")
    log(f"{ CFG['episodes_train']['norm_episode']=}")
    log('=' * 100)

    SEED = CFG['general']['seed']
    print(f'SEED:{SEED}')
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ===============================================================
    log('=== Load configuration === ')
    num_ways = int(CFG['episodes_train']['num_ways'])  # number of protein to select in train episode
    log(f'{num_ways=}')
    support_set_size = int(CFG['episodes_train']['support_set_size'])  # number of support shots
    log(f'{support_set_size=}')
    query_set_size = int(CFG['episodes_train']['query_set_size'])  # number of query shots
    log(f'{query_set_size=}')
    num_inner_epochs = int(CFG['episodes_train']['inner_epochs'])
    log(f'{num_inner_epochs=}')

    # ===============================================================
    log('=== Create data for ALL proteins === ')

    diff_len = int(CFG['general']['diff_len'])
    log(f'{diff_len=}')

    log('=== Load diff embeddings ===')
    prism_data_list = _load_diff_embeddings(step=diff_len)

    calculate_bins(prism_data_list)
    # normalize_scores_only(prism_data_list)

    # normalize_scores_enabled = int(CFG['flow_data_creation']['normalize_scores'])
    # log(f'{normalize_scores_enabled=}')
    # if normalize_scores_enabled != 1:
    #     log(f'Undoing scores normalization...')
    #     undo_score_normalization(prism_data_list)  # replace normalized scores with original ones
    #     log(f'done!')

    # normalize_dds_enabled = int(CFG['flow_data_creation']['normalize_deltas'])
    # log(f'{normalize_dds_enabled=}')
    # if normalize_dds_enabled == 1:
    #     log(f'Perform deltas normalization...')
    #     normalize_deltas_only(prism_data_list)  # normalize ddG
    #     log(f'done!')

    total_variants_count = sum([len(d.variants) for d in prism_data_list])
    log(f'{total_variants_count=}')

    log('=== Create sequence embeddings ===')
    sequence_embedder = create_seq_embedder()
    pname_to_seq_embedding = create_seq_embedding_dict(prism_data_list, sequence_embedder)
    log(f'{len(pname_to_seq_embedding)=}')

    log('=== Create datasets === ')
    set_creator = PrismDiffEmbMultisetCreator(prism_data_list, pname_to_seq_embedding)
    dss = set_creator.create_datasets()
    assert len(dss) == len(get_protein_files_dict())

    eval_prot_number = int(CFG['general']['eval_protein_file_number'])
    eval_prot_filename = get_protein_files_dict()[eval_prot_number]
    log(f'{eval_prot_filename=}')

    # log('=== Normalize train and valid datasets === ')
    # non_eval_dss = [x for x in dss if x.file_name == eval_prot_filename]
    # for ds in non_eval_dss:
    #     normalize_scores_ds(ds)

    episode_creator = EpisodesCreator(dss, eval_prot_filename)

    if int(CFG['episodes_train']['use_validation']) == 1:
        #     log('=== Create validation dataset === ')
        valid_episode = episode_creator.create_validation_episode(support_set_size)
        log(f'Validation protein: {valid_episode.query_ds.file_name}')

    # create main_model
    use_pretrain = int(CFG['episodes_train']['use_pretrain'])
    log(f'{use_pretrain=}')
    if use_pretrain == 0:
        log('=== Create *NEW* main model === ')
        model = create_model()
    else:
        ep = CFG['general']['eval_protein_file_number']
        log(f'=== Load pre-trained main model: protein {ep} === ')
        model = create_model()
        model_file = os.path.join(EPISODES_PRETRAINED, ep, model.file_name)
        model.load_state_dict(torch.load(model_file))

    # create meta optimizer
    log('=== Optimizer and Scheduler === ')
    meta_lr = float(CFG['flow_train']['lr'])
    log(f'{meta_lr=}')
    meta_opt = optim.Adam(model.parameters(), lr=meta_lr)
    log(meta_opt)
    step_size = int(CFG['general']['step'])
    gamma = float(CFG['general']['gamma'])
    log(f'{step_size=}')
    log(f'{gamma=}')
    meta_scheduler = StepLR(meta_opt, step_size=step_size, gamma=gamma)
    log(meta_scheduler)
    inner_lr = float(CFG['episodes_train']['inner_lr'])
    log(f'{inner_lr=}')
    # ===============================================================
    log('=== Training === ')
    valid_losses = []
    train_losses = []
    epochs = int(CFG['episodes_train']['episodes'])
    for epoch in range(epochs):
        epoch_start_time = time.time()
        log('-' * 30)
        log(f'Iteration: {epoch}')

        # create episode
        episode = episode_creator.create_train_episode(num_ways, support_set_size, query_set_size)
        log(episode)
        context = DataContext()
        context.fill_data_loaders(episode)

        # train the model
        train_loss = train_epoch(context, model, meta_opt, num_inner_epochs, inner_lr)
        train_losses.append(train_loss.item())
        log(f'Train loss=: {train_loss.item()}')

        # using the scheduler
        if int(CFG['episodes_train']['use_scheduler']) == 1:
            meta_scheduler.step()
            inner_lr = meta_opt.param_groups[0]["lr"]
            log(f'LR={inner_lr}')

        # validation (uncomment to run)
        if int(CFG['episodes_train']['use_validation']) == 1:
            val_episode = episode_creator.create_validation_episode(support_set_size)
            log(val_episode)
            context = DataContext()
            context.fill_test_data_loaders(val_episode)
            val_loss = val_epoch(context, model, num_inner_epochs)
            valid_losses.append(val_loss.item())
            log(f'Valid loss=: {val_loss.item()}')

        elapsed_time = time.time() - epoch_start_time
        log(f'time: {elapsed_time:5.2f} sec')

    # save model
    model_path = os.path.join(report_path, model.file_name)
    torch.save(model.state_dict(), model_path)
    # ===============================================================

    log('=== Test === ')
    test_episode = episode_creator.create_test_episode(support_set_size)
    log(test_episode)
    context = DataContext()
    context.fill_test_data_loaders(test_episode)
    test_res = test_epoch(context, model, num_inner_epochs)
    log(test_res)

    log('=' * 100)
    plotter = PlotCreator()
    hoie_result = load_hoie_result()
    log('=' * 100)
    log('Hoie result loaded...')
    test_res.model_name = model.file_name
    plotter.create_eval_plots(report_path, test_res, hoie_result)
    plotter.create_plot_episodes_loss(report_path, valid_losses, episode_creator.valid_prot_filename, 'valid')
    plotter.create_plot_episodes_loss(report_path, train_losses, episode_creator.eval_prot_filename, 'train')

    protein_filename = get_protein_files_dict()[eval_prot_number]
    create_result_txt(report_path, protein_filename, test_res)
    pickle_test_result(report_path, 'eval', test_res)

    # ===============================================================
    log('=' * 20)
    log(f'Report path:{report_path}')

    elapsed_time = time.time() - start_time
    log(f'Total time: {elapsed_time:5.2f} sec')
    print('OK')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inputs and setup for the model')
    parser.add_argument('-prot_set', type=int, help='Select proteins subset for model', required=False)
    parser.add_argument('-ep', type=int, help='Evaluation protein id', required=False)
    parser.add_argument('-model', type=int, help='Model to use for train/eval', required=False)
    parser.add_argument('-lr', type=float, help='Learning rate', required=False)
    parser.add_argument('-episodes', type=int, help='Number of episodes', required=False)
    parser.add_argument('-bins', type=int, help='Number of bins to split the data', required=False)
    parser.add_argument('-alpha', type=float, help='Parameter coefficient for loss', required=False)
    parser.add_argument('-num_ways', type=int, help='Number of ways', required=False)
    parser.add_argument('-s_set_size', type=int, help='Support set size', required=False)
    parser.add_argument('-q_set_size', type=int, help='Query set size', required=False)
    parser.add_argument('-inner_epochs', type=int, help='Number of inner epochs', required=False)
    parser.add_argument('-use_pretrain', type=int, help='Usage of pretrained model', required=False)
    parser.add_argument('-inner_lr', type=float, help='Internal learning rate', required=False)
    parser.add_argument('-use_valid', type=int, help='Use validation', required=False)
    parser.add_argument('-one_prot_episode', type=int, help='Use single protein episode', required=False)
    parser.add_argument('-norm_scores', type=int, help='Use normalized scores', required=False)
    parser.add_argument('-norm_deltas', type=int, help='Use normalized deltas', required=False)
    parser.add_argument('-norm_episode', type=int, help='Use normalized episodes', required=False)
    parser.add_argument('-batch_size', type=int, help='Define batch size', required=False)

    args = parser.parse_args()
    if args.prot_set is not None:
        CFG['general']['protein_set'] = str(args.prot_set)
    if args.ep is not None:
        CFG['general']['eval_protein_file_number'] = str(args.ep)
    if args.model is not None:
        CFG['general']['model'] = str(args.model)
    if args.lr is not None:
        CFG['flow_train']['lr'] = str(args.lr)
    if args.episodes is not None:
        CFG['episodes_train']['episodes'] = str(args.episodes)
    if args.bins is not None:
        CFG['general']['bins'] = str(args.bins)
    if args.alpha is not None:
        CFG['flow_train']['alpha'] = str(args.alpha)
    if args.num_ways is not None:
        CFG['episodes_train']['num_ways'] = str(args.num_ways)
    if args.s_set_size is not None:
        CFG['episodes_train']['support_set_size'] = str(args.s_set_size)
    if args.q_set_size is not None:
        CFG['episodes_train']['query_set_size'] = str(args.q_set_size)
    if args.inner_epochs is not None:
        CFG['episodes_train']['inner_epochs'] = str(args.inner_epochs)
    if args.use_pretrain is not None:
        CFG['episodes_train']['use_pretrain'] = str(args.use_pretrain)
    if args.inner_lr is not None:
        CFG['episodes_train']['inner_lr'] = str(args.inner_lr)
    if args.use_valid is not None:
        CFG['episodes_train']['use_validation'] = str(args.use_valid)
    if args.one_prot_episode is not None:
        CFG['episodes_train']['use_one_protein_episode'] = str(args.one_prot_episode)
    if args.norm_scores is not None:
        CFG['flow_data_creation']['normalize_scores'] = str(args.norm_scores)
    if args.norm_deltas is not None:
        CFG['flow_data_creation']['normalize_deltas'] = str(args.norm_deltas)
    if args.norm_episode is not None:
        CFG['episodes_train']['norm_episode'] = str(args.norm_episode)
    if args.batch_size is not None:
        CFG['flow_train']['batch_size'] = str(args.batch_size)

    # NOTE: print command line args CFG inside main

    main()
