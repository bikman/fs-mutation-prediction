"""
TBD
"""
import logging
import os
import pickle
import random
import time

import torch
from adabelief_pytorch import AdaBelief
from adabound import adabound
from matplotlib import pyplot as plt
from ranger_adabelief import RangerAdaBelief
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from schedulers import GradualWarmupScheduler

from data_model import ModelConfig
from engine_struct_attn import PrismScoreEmbDiffSimpleModel, PrismScoreDeltasOnlyModel, PrismScoreDeltasEmbDiffModel, \
    PrismScoreDeltasEmbModel, PrismScoreNoDDGModel, PrismScoreNoDeltasModel, PrismScoreNoDDEModel
from train import train_prism_scores_multi_sets
from utils import setup_reports, DEVICE, DUMP_ROOT, TrainParameters, CFG, PRISM_TRAIN_SPLIT, PRISM_VALID_SPLIT, \
    PRETRAIN_FOLDER, PRISM_FINE_TRAIN_SPLIT

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


def create_model_config():
    """
    Creates Model Configuration object
    @return: cfg:ModelConfig
    """
    cfg = ModelConfig()
    cfg.heads = int(CFG['flow_train']['heads'])
    cfg.seq_emb_size = int(CFG['general']['seq_emb_size'])
    cfg.diff_width = int(CFG['general']['diff_len'])
    cfg.attn_len = int(CFG['general']['attn_len'])
    cfg.cz = int(CFG['general']['cz'])
    cfg.deltas_encoder = int(CFG['general']['use_deltas_encoder'])
    return cfg


def choose_optimizer(model, pretrained_enabled):
    """
    Choose optimizer according to configuration value
    @param pretrained_enabled: should we use LR for pretrained flow
    @param model: model to train
    @return: optimizer object
    """
    lr = float(CFG['flow_train']['lr'])
    if pretrained_enabled != 0:
        lr = float(CFG['flow_pretrained']['lr'])
    # log(f'LR:{lr}')
    res = torch.optim.Adam(model.parameters(), lr=lr)  # ADAM
    return res


def run_train(train_dss, valid_dss, train_params):
    """
    Run training on given datasets
    @param valid_dss: list of validation datasets
    @param train_dss: list of train datasets
    @param train_params: parameters for training
    """
    batch_size = int(CFG['flow_train']['batch_size'])
    log(f'Batch size: {batch_size}')
    train_loaders = []
    valid_loaders = []
    for train_set in train_dss:
        # ---- here we shuffle batches ----
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
        # ---------------------------------
        train_loaders.append(train_loader)
    for valid_set in valid_dss:
        validation_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
        valid_loaders.append(validation_loader)
    train_params.train_loaders_list = train_loaders
    train_params.valid_loaders_list = valid_loaders
    log(f'Sum of Train loaders={sum([len(x) for x in train_params.train_loaders_list])}')
    log(f'Sum of Valid loaders={sum([len(x) for x in train_params.valid_loaders_list])}')
    log(train_params)
    train_res = train_prism_scores_multi_sets(train_params, log)
    log(train_res)
    return train_res


def load_prism_score_datasets():
    """
    Load datasets from PICKLE files
    @return: lists of train and validation datasets
    """
    log(f'{DUMP_ROOT=}')
    dump_path = os.path.join(DUMP_ROOT, PRISM_TRAIN_SPLIT)
    with open(dump_path, "rb") as f:
        train_split = pickle.load(f)
    log(f'loaded: {dump_path}\n')
    dump_path = os.path.join(DUMP_ROOT, PRISM_VALID_SPLIT)
    with open(dump_path, "rb") as f:
        valid_split = pickle.load(f)
    log(f'loaded: {dump_path}\n')
    use_fine_tune = int(CFG['flow_train']['use_fine_tune_data'])
    log(f'{use_fine_tune=}')
    if use_fine_tune == 1:
        dump_path = os.path.join(DUMP_ROOT, PRISM_FINE_TRAIN_SPLIT)
        with open(dump_path, "rb") as f:
            train_fine_split = pickle.load(f)
        log(f'loaded: {dump_path}\n')
        train_split += train_fine_split
    return train_split, valid_split


def create_model_pretrained():
    model = int(CFG['general']['model'])
    log(f'model={model}')
    cfg = create_model_config()
    log(cfg)
    log('Loading pretrain model')
    log(f'{PRETRAIN_FOLDER=}')
    if model == 1:
        model_obj = PrismScoreEmbDiffSimpleModel(cfg).to(DEVICE)
    elif model == 2:
        model_obj = PrismScoreDeltasOnlyModel(cfg).to(DEVICE)
    elif model == 3:
        model_obj = PrismScoreDeltasEmbDiffModel(cfg).to(DEVICE)
    elif model == 4:
        model_obj = PrismScoreDeltasEmbModel(cfg).to(DEVICE)
    elif model == 5:
        model_obj = PrismScoreNoDDGModel(cfg).to(DEVICE)
    elif model == 6:
        model_obj = PrismScoreNoDDEModel(cfg).to(DEVICE)
    elif model == 7:
        model_obj = PrismScoreNoDeltasModel(cfg).to(DEVICE)
    else:
        raise Exception(f'Not supported model {model}')
    model_file = os.path.join(PRETRAIN_FOLDER, model_obj.file_name)
    model_obj.load_state_dict(torch.load(model_file))
    log(model_obj)
    return model_obj


def create_model():
    """
    Create model object according to config
    @return: model
    """
    model = int(CFG['general']['model'])
    log(f'model={model}')
    cfg = create_model_config()
    log(cfg)
    # create new model
    log('Creating new model')
    if model == 1:
        model_obj = PrismScoreEmbDiffSimpleModel(cfg).to(DEVICE)
    elif model == 2:
        model_obj = PrismScoreDeltasOnlyModel(cfg).to(DEVICE)
    elif model == 3:
        model_obj = PrismScoreDeltasEmbDiffModel(cfg).to(DEVICE)
    elif model == 4:
        model_obj = PrismScoreDeltasEmbModel(cfg).to(DEVICE)
    elif model == 5:
        model_obj = PrismScoreNoDDGModel(cfg).to(DEVICE)
    elif model == 6:
        model_obj = PrismScoreNoDDEModel(cfg).to(DEVICE)
    elif model == 7:
        model_obj = PrismScoreNoDeltasModel(cfg).to(DEVICE)
    else:
        raise Exception(f'Not supported model {model}')
    log(model_obj)
    log(f'Num.Model Parameters={sum([p.numel() for p in model_obj.parameters()])}')
    return model_obj


def create_loss_batch_plot(report_path, train_res):
    """
    Plots model loss per each batch.
    Train function must fill list of batch losses, otherwise nothing wil be printed
    @param report_path: folder to save the plot
    @param train_res: training result object
    """
    ys = train_res.train_loss_per_batch
    if len(ys) == 0:
        return
    xs = range(len(ys))
    plt.clf()

    plt.title("MSE per batch, epoch 0")
    plt.xlabel("Batch")
    plt.ylabel("MSE loss")
    plt.grid()
    # plt.plot(xs, ys)
    plt.scatter(xs, ys, s=0.5)
    plt_file = f'loss_mse_per_batch.plot.png'
    plot_path = os.path.join(report_path, plt_file)
    plt.savefig(plot_path)
    # plt.show()
    print(f'Created {plot_path}')


def create_acc_loss_plots(report_path, train_res):
    """
    Create plot of loss and acc (if possible) per epoch
    @param report_path: folder to save plot
    @param train_res: train result
    """
    train_acc_list = train_res.train_accuracy_per_epoch
    train_loss_list = train_res.train_loss_per_epoch
    valid_acc_list = train_res.validation_accuracy_per_epoch
    valid_loss_list = train_res.validation_loss_per_epoch
    xs = range(len(train_loss_list))
    assert len(train_loss_list) == len(valid_loss_list)
    if len(train_acc_list) > 0:
        assert len(train_acc_list) == len(train_loss_list)
        assert len(valid_acc_list) == len(valid_loss_list)
        assert len(train_acc_list) == len(valid_acc_list)
        fig, axs = plt.subplots(2)
        fig.suptitle("Acc & Loss per epoch")
        axs[0].plot(xs, train_loss_list, 'r', label='Train', linestyle='--', marker='.')
        axs[0].plot(xs, valid_loss_list, 'b', label='Validation', linestyle='--', marker='.')
        axs[0].legend(loc="upper left")
        plt.setp(axs[0], ylabel='Loss')
        axs[0].grid(True)
        axs[0].set_xticks([], minor=True)
        axs[0].set_yticks([], minor=True)
        axs[1].plot(xs, train_acc_list, 'r', label='Train', linestyle='--', marker='.')
        axs[1].plot(xs, valid_acc_list, 'b', label='Validation', linestyle='--', marker='.')
        axs[1].legend(loc="upper left")
        plt.setp(axs[1], ylabel='Accuracy')
        axs[1].grid(True)
        axs[1].set_xticks([], minor=True)
        axs[1].set_yticks([], minor=True)
    else:
        # --- zero first 5 epochs ---
        if len(train_loss_list) > 50:
            train_loss_list[0:4] = [0] * 4
            valid_loss_list[0:4] = [0] * 4
        fig, axs = plt.subplots()
        fig.suptitle("Loss per epoch")
        # --- Plot Train loss ---
        axs.plot(xs, train_loss_list, 'r', label='Train', linestyle='--', marker='.')
        # --- Plot Validation loss ---
        axs.plot(xs, valid_loss_list, 'b', label='Validation', linestyle='--', marker='.')
        # -----------------------------------
        axs.legend(loc="upper left")
        plt.setp(axs, ylabel='Loss')
        axs.grid(True)
        axs.set_xticks([], minor=True)
        axs.set_yticks([], minor=True)

    plt.xlabel("Epoch #")
    plt_file = f'loss_acc_per_epoch.plot.png'
    plot_path = os.path.join(report_path, plt_file)
    plt.savefig(plot_path)
    # plt.show()
    print(f'Created {plot_path}')


def run_multiset_scores_training(report_path):
    """
    Prepares the model and runs the training
    @param report_path: folder to save report
    """
    log(os.path.basename(__file__))
    pretrained_enabled = int(CFG['flow_pretrained']['is_enabled'])
    log(f'{pretrained_enabled=}')

    if pretrained_enabled == 0:
        model = create_model()
    else:
        model = create_model_pretrained()

    # define parameters
    train_params = fill_train_parameters(model, pretrained_enabled, report_path)
    # --- Load datasets ---
    train_dss, valid_dss = load_prism_score_datasets()

    # --- Run training ---
    train_res = run_train(train_dss, valid_dss, train_params)

    # --- Create plot of loss and acc ---
    create_acc_loss_plots(report_path, train_res)

    return model


def fill_train_parameters(model, pretrained_enabled, report_path):
    """
    Create train parameters
    @param model: model to train
    @param pretrained_enabled: is model was pretrained
    @param report_path: folder for reporting
    @return: TrainParameters object
    """
    train_params = TrainParameters()
    train_params.model = model
    train_params.loss = torch.nn.MSELoss()
    train_params.loss2 = torch.nn.MSELoss()
    train_params.optimizer = choose_optimizer(model, pretrained_enabled)
    create_train_scheduler(train_params)
    train_params.model_path = os.path.join(report_path, model.file_name)
    train_params.report_path = report_path
    train_params.epochs = int(CFG['flow_train']['epochs'])
    train_params.patience = int(CFG['flow_train']['patience'])
    train_params.loader_pairs = []
    train_params.bins = int(CFG['general']['bins'])
    train_params.alpha = float(CFG['flow_train']['alpha'])
    return train_params


def create_train_scheduler(train_params):
    """
    Creates scheduler for training (if specified)
    @param train_params: training parameters
    """
    if int(CFG['general']['use_scheduler']) > 0:
        gamma = float(CFG['general']['gamma'])
        log(f'{gamma=}')
        step_size = int(CFG['general']['step'])
        log(f'{step_size=}')
        # ---- Warm up scheduler ----
        # epochs = int(CFG['flow_train']['epochs'])
        # log(f'{epochs=}')
        # scheduler_next = StepLR(train_params.optimizer, step_size=step_size, gamma=gamma)
        # warmup_epochs = int(epochs / 10)
        # log(f'{warmup_epochs=}')
        # train_params.scheduler = GradualWarmupScheduler(train_params.optimizer, 1, warmup_epochs,
        #                                                 after_scheduler=scheduler_next)
        # ---- Regular Lr scheduler ----
        train_params.scheduler = StepLR(train_params.optimizer, step_size=step_size, gamma=gamma)


def main():
    start_time = time.time()
    report_path = setup_reports('score_training')
    log(f'Report path:{report_path}')
    log(DEVICE)

    run_multiset_scores_training(report_path)

    elapsed_time = time.time() - start_time
    log(f'time: {elapsed_time:5.2f} sec')
    print('OK')


if __name__ == '__main__':
    main()
