import math
import os
import random
import time

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from data_model import DebugData
from plots import PlotCreator
from utils import DEVICE, TrainResult, TestResult, CFG, PredictionAccuracy, get_lr, NUM_QUANTILES, smape


def normalize_tensor(target):
    """
    Using quantile transformer 
    @param target:
    @return:
    """
    # --- quantile normalization ---
    x = target.reshape(-1, 1).cpu()
    quantile_transformer = preprocessing.QuantileTransformer(n_quantiles=NUM_QUANTILES)
    transformed = quantile_transformer.fit_transform(x)
    res = torch.from_numpy(transformed.flatten()).to(DEVICE, dtype=torch.float)
    return res


def train_prism_fine_tune_multi_sets(parameters, log):
    result = TrainResult()
    result.test_result = TestResult()

    batch_norm = int(CFG['flow_fine_tune']['batch_norm'])
    log(f'{batch_norm=}')
    # --- Uncomment to pre-save model during Fine-Tuning flow ---
    # if parameters.patience is not None:
    #     torch.save(parameters.model.state_dict(), parameters.model_path)

    for i in range(parameters.epochs):
        log('-' * 30)
        log(f'Epoch FT: {i}')
        train_losses = []
        parameters.model.train()
        true_values = np.array([])
        pred_values = np.array([])
        start_time = time.time()
        # train loop
        log('Train FT')
        for t, train_loader in tqdm(enumerate(parameters.train_loaders_list)):
            for j, (pid, pos, all_deltas, pos_arr, emb_sector, emb_diff, score, bin_id, score_orig, src, dst) in \
                    enumerate(train_loader):
                try:
                    pid = pid.to(DEVICE, dtype=torch.float).unsqueeze(dim=1)
                    pos = pos.to(DEVICE, dtype=torch.float).unsqueeze(dim=1)
                    pos_arr = pos_arr.to(DEVICE, dtype=torch.float)
                    all_deltas = all_deltas.to(DEVICE, dtype=torch.float)
                    emb_sector = emb_sector.to(DEVICE, dtype=torch.float)
                    emb_diff = emb_diff.to(DEVICE, dtype=torch.float)
                    target = score.to(DEVICE, dtype=torch.float)
                    # ---
                    if batch_norm == 1:
                        if len(target) < NUM_QUANTILES:
                            continue
                        target = normalize_tensor(target)
                        if torch.isnan(target).any():
                            log(f'train_prism_fine_tune_multi_sets: Train FT: NaN is found in target: {target.size(dim=0)}')
                            continue
                    # ---
                    target2 = bin_id.to(DEVICE, dtype=torch.float)
                    parameters.optimizer.zero_grad()
                    y_hat = parameters.model(pid, pos, all_deltas, pos_arr, emb_sector, emb_diff)
                    y_hat = y_hat.squeeze(dim=-1)
                    y_hat1 = y_hat[:, 0]
                    y_hat2 = y_hat[:, 1]
                    loss1 = parameters.loss(y_hat1, target)
                    loss2 = parameters.loss2(y_hat2, target2)
                    loss = calc_loss(1.0, parameters.bins, loss1, loss2)
                    loss.backward(retain_graph=True)
                    parameters.optimizer.step()
                    batch_loss = loss.detach()
                    train_losses.append(batch_loss)
                except ValueError as e:
                    raise e
                except Exception as e:
                    log(f'\nFT: Train error: {e}')
                    continue
        epoch_train_loss = torch.mean(torch.tensor(train_losses))

        log(f'ft train loss: {epoch_train_loss}')
        elapsed_time = time.time() - start_time
        log(f'ft time: {elapsed_time:5.2f} sec')

        result.train_loss_per_epoch.append(epoch_train_loss)

    log('Eval FT')
    start_time = time.time()
    epoch_eval_loss = -1
    parameters.model.eval()
    orig_true_values = np.array([])
    src_indices = np.array([])
    dst_indices = np.array([])
    pos_values = np.array([])
    with torch.no_grad():
        for t, valid_loader in tqdm(enumerate(parameters.valid_loaders_list)):
            for j, (
                    pid, pos, all_deltas, pos_arr, emb_sector, emb_diff, score, bin_id, score_orig, src,
                    dst) in enumerate(valid_loader):
                try:
                    pid = pid.to(DEVICE, dtype=torch.float).unsqueeze(dim=1)
                    pos = pos.to(DEVICE, dtype=torch.float).unsqueeze(dim=1)
                    pos_arr = pos_arr.to(DEVICE, dtype=torch.float)
                    all_deltas = all_deltas.to(DEVICE, dtype=torch.float)
                    emb_sector = emb_sector.to(DEVICE, dtype=torch.float)
                    emb_diff = emb_diff.to(DEVICE, dtype=torch.float)
                    target = score.to(DEVICE, dtype=torch.float)
                    # ---
                    if batch_norm == 1:
                        if len(target) < NUM_QUANTILES:
                            continue
                        target = normalize_tensor(target)
                        if torch.isnan(target).any():
                            log(f'train_prism_fine_tune_multi_sets: Eval FT: NaN is found in target: {target.size(dim=0)}')
                            continue
                    # ---
                    target2 = bin_id.to(DEVICE, dtype=torch.float)
                    y_hat = parameters.model(pid, pos, all_deltas, pos_arr, emb_sector, emb_diff)
                    y_hat = y_hat.squeeze(dim=-1)
                    y_hat1 = y_hat[:, 0]
                    y_hat2 = y_hat[:, 1]
                    loss1 = parameters.loss(y_hat1, target)
                    loss2 = parameters.loss2(y_hat2, target2)
                    loss = calc_loss(1.0, parameters.bins, loss1, loss2)
                    true_values = np.append(true_values, target.cpu().numpy())
                    orig_true_values = np.append(orig_true_values, score_orig.cpu().numpy())
                    pred_values = np.append(pred_values, y_hat1.cpu().detach().numpy())
                    pos_values = np.append(pos_values, pos.cpu().numpy())
                    dst_indices = np.append(dst_indices, dst.cpu().numpy())
                    src_indices = np.append(src_indices, src.cpu().numpy())
                    epoch_eval_loss = loss.detach().cpu().numpy()
                except Exception as e:
                    log(f'\nFine Tune: Eval Error: {e}')
                    continue
    mae = np.round(mean_absolute_error(true_values, pred_values), 4)
    mape = np.round(smape(true_values, pred_values), 4)
    pearson = -1
    spearman = -1
    r2 = -1
    if len(true_values) >= 2:
        pearson = np.round(pearsonr(true_values, pred_values), 4)[0]
        spearman = np.round(spearmanr(true_values, pred_values), 4)[0]
        r2 = np.round(r2_score(true_values, pred_values), 4)
    log(f'ft eval loss: {epoch_eval_loss}')
    log(f'ft {mae=}')
    log(f'ft {mape=}')
    log(f'ft {pearson=}')
    log(f'ft {spearman=}')
    log(f'ft {r2=}')

    elapsed_time = time.time() - start_time
    log(f'ft time: {elapsed_time:5.2f} sec')

    best_loss = epoch_eval_loss
    result.best_epoch = -1
    result.test_result.mae = mae
    result.test_result.mape = mape
    result.test_result.pearson = pearson
    result.test_result.spearman = spearman
    result.test_result.r2 = r2
    result.test_result.true_values = true_values
    result.test_result.pred_values = pred_values
    result.test_result.orig_true_values = orig_true_values
    result.test_result.pos_values = pos_values
    result.test_result.src_indices = src_indices
    result.test_result.dst_indices = dst_indices
    # 0.3 = threshold for prediction accuracy
    result.test_result.prediction_accuracy = calc_prediction_accuracy(true_values, pred_values, 0.3)
    log(f'{best_loss=}')

    return result


def calc_loss(alpha, bins, loss1, loss2):
    loss = alpha * loss1 + (1.0 - alpha) * (1.0 / bins) * loss2
    return loss


def train_prism_scores_multi_sets(parameters, log):
    result = TrainResult()
    best_loss = math.inf
    best_train_loss = math.inf
    epochs_since_best = 0
    epochs_since_best_train = 0
    train_patience_threshold = 5

    # save initial checkpoint
    if parameters.patience is not None:
        torch.save(parameters.model.state_dict(), parameters.model_path)

    batch_norm = int(CFG['flow_train']['batch_norm'])
    log(f'{batch_norm=}')

    # --- epochs loop ---
    for i in range(parameters.epochs):
        log('-' * 30)
        log(f'Epoch: {i}')
        start_time = time.time()
        train_losses = []
        parameters.model.train()

        # ---- here we shuffle the data loaders ----
        random.shuffle(parameters.train_loaders_list)
        random.shuffle(parameters.valid_loaders_list)
        # ------------------------------------------

        debug_data = DebugData()
        debug_plots_path = os.path.join(parameters.report_path, 'plots_debug')
        if not os.path.exists(debug_plots_path):
            os.makedirs(debug_plots_path)
        debug_data.report_path = debug_plots_path
        debug_data.log = log
        debug_data.curr_epoch = i

        true_values = np.array([])
        pred_values = np.array([])

        log('Train')
        # --- loaders loop ---
        for t, train_loader in tqdm(enumerate(parameters.train_loaders_list)):
            # --- batch loop ---
            for j, (
                    pid, pos, all_deltas, pos_arr, emb_sector, emb_diff, score, bin_id, score_orig, src,
                    dst) in enumerate(train_loader):
                try:

                    pid = pid.to(DEVICE, dtype=torch.float).unsqueeze(dim=1)
                    pos = pos.to(DEVICE, dtype=torch.float).unsqueeze(dim=1)
                    pos_arr = pos_arr.to(DEVICE, dtype=torch.float)
                    all_deltas = all_deltas.to(DEVICE, dtype=torch.float)
                    emb_sector = emb_sector.to(DEVICE, dtype=torch.float)
                    emb_diff = emb_diff.to(DEVICE, dtype=torch.float)
                    target = score.to(DEVICE, dtype=torch.float)
                    # ---
                    if batch_norm == 1:
                        if len(target) < NUM_QUANTILES:
                            continue
                        target = normalize_tensor(target)
                        if torch.isnan(target).any():
                            log(f'train_prism_scores_multi_sets: Train: NaN is found in target: {target.size(dim=0)}')
                            continue
                    # ---
                    target2 = bin_id.to(DEVICE, dtype=torch.float)
                    parameters.optimizer.zero_grad()
                    y_hat = parameters.model(pid, pos, all_deltas, pos_arr, emb_sector, emb_diff)
                    y_hat = y_hat.squeeze(dim=-1)
                    y_hat1 = y_hat[:, 0]
                    y_hat2 = y_hat[:, 1]
                    loss1 = parameters.loss(y_hat1, target)
                    loss2 = parameters.loss2(y_hat2, target2)
                    loss = calc_loss(parameters.alpha, parameters.bins, loss1, loss2)
                    loss.backward(retain_graph=True)
                    parameters.optimizer.step()
                    batch_loss = loss.detach()
                    if math.isnan(batch_loss.item()):
                        raise ValueError(f'Loss in NaN: batch {j}, loader {t}, epoch {i}')
                    train_losses.append(batch_loss)
                    true_values = np.append(true_values, target.cpu().numpy())
                    pred_values = np.append(pred_values, y_hat1.cpu().detach().numpy())
                except ValueError as e:
                    raise e
                except Exception as e:
                    log(f'\nTrain Error: {e}')
                    continue
        epoch_train_loss = torch.mean(torch.tensor(train_losses))

        # validation loop
        log('Validation')
        validation_losses = []
        parameters.model.eval()
        with torch.no_grad():
            for t, valid_loader in tqdm(enumerate(parameters.valid_loaders_list)):
                for j, (
                        pid, pos, all_deltas, pos_arr, emb_sector, emb_diff, score, bin_id, score_orig, src,
                        dst) in enumerate(valid_loader):
                    try:
                        pid = pid.to(DEVICE, dtype=torch.float).unsqueeze(dim=1)
                        pos = pos.to(DEVICE, dtype=torch.float).unsqueeze(dim=1)
                        pos_arr = pos_arr.to(DEVICE, dtype=torch.float)
                        all_deltas = all_deltas.to(DEVICE, dtype=torch.float)
                        emb_sector = emb_sector.to(DEVICE, dtype=torch.float)
                        emb_diff = emb_diff.to(DEVICE, dtype=torch.float)
                        target = score.to(DEVICE, dtype=torch.float)
                        # ---
                        if batch_norm == 1:
                            if len(target) < NUM_QUANTILES:
                                continue
                            target = normalize_tensor(target)
                            if torch.isnan(target).any():
                                log(f'train_prism_scores_multi_sets: Validation: NaN is found in target: {target.size(dim=0)}')
                                continue
                        # ---
                        target2 = bin_id.to(DEVICE, dtype=torch.float)
                        y_hat = parameters.model(pid, pos, all_deltas, pos_arr, emb_sector, emb_diff)
                        y_hat = y_hat.squeeze(dim=-1)
                        y_hat1 = y_hat[:, 0]
                        y_hat2 = y_hat[:, 1]
                        loss1 = parameters.loss(y_hat1, target)
                        loss2 = parameters.loss2(y_hat2, target2)
                        loss = calc_loss(parameters.alpha, parameters.bins, loss1, loss2)
                        validation_losses.append(loss.detach())
                    except Exception as e:
                        log(f'\nValidation Error: {e}')
                        continue
        epoch_validation_loss = torch.mean(torch.tensor(validation_losses))

        log(f'Epoch {i}')
        log(f'train loss: {epoch_train_loss}')
        log(f'valid. loss: {epoch_validation_loss}')
        elapsed_time = time.time() - start_time
        log(f'time: {elapsed_time:5.2f} sec')

        # Add to result lists
        result.train_loss_per_epoch.append(epoch_train_loss)
        result.validation_loss_per_epoch.append(epoch_validation_loss)

        mae = np.round(mean_absolute_error(true_values, pred_values), 4)
        mape = np.round(smape(true_values, pred_values), 4)
        mse = np.round(mean_squared_error(true_values, pred_values), 4)
        pearson = np.round(pearsonr(true_values, pred_values), 4)[0]
        spearman = np.round(spearmanr(true_values, pred_values), 4)[0]
        r2 = np.round(r2_score(true_values, pred_values), 4)
        test_epoch_res = TestResult()
        test_epoch_res.true_values = true_values
        test_epoch_res.pred_values = pred_values
        test_epoch_res.loss = epoch_train_loss
        test_epoch_res.mae = mae
        test_epoch_res.mape = mape
        test_epoch_res.mse = mse
        test_epoch_res.pearson = pearson
        test_epoch_res.spearman = spearman
        test_epoch_res.r2 = r2

        # --- print train correlation "pred vs. truth" ---
        plots_path = os.path.join(parameters.report_path, 'plots_train')
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)
        if i % 100 == 0:  # print correlation "pred vs. truth"
            PlotCreator.create_correlation_plot(plots_path, test_epoch_res, f'prediction_corr.epoch_{i}.plot.png',
                                                f'Epoch:{i}')
        # --------------------------------------

        # --- uncomment to enable scheduling ---
        if parameters.scheduler is not None:
            parameters.scheduler.step()
            log('Scheduler LR enabled')
        log(f'lr: {get_lr(parameters.optimizer)}')
        # --------------------------------------

        # --- train loss patience ---
        if epoch_train_loss < best_train_loss:
            epochs_since_best_train = 0
            best_train_loss = epoch_train_loss
            log(f'Best train loss: {best_train_loss}')
        else:
            epochs_since_best_train = epochs_since_best_train + 1
            log(f'{best_train_loss=}')
            log(f'{epochs_since_best_train=}')

        # --- uncomment to enable train loss patience threshold ---
        # if epochs_since_best_train > train_patience_threshold:  # sort of patience for train loss
        #     log(f'epochs_since_best_train={epochs_since_best_train} is more than {train_patience_threshold} -exiting')
        #     return result
        # ---------------------------------

        # check patience
        if parameters.patience is not None:
            if epoch_validation_loss < best_loss:
                # we have new best epoch here
                epochs_since_best = 0
                best_loss = epoch_validation_loss
                torch.save(parameters.model.state_dict(), parameters.model_path)
                log(f'Model saved: {parameters.model_path}')
                result.best_epoch = i
                log(f'{best_loss=}')
            else:
                epochs_since_best = epochs_since_best + 1
                log(f'{best_loss=}')
                log(f'{result.best_epoch=}')
            if epochs_since_best > parameters.patience:
                return result

    return result


def calc_prediction_accuracy(true_values, pred_values, t):
    """
    Calculates how accurate our prediction is,
    assuming we have 2 classes - positive (destructive) and negative (non-destructive)
    @param true_values: list of ground truth scores
    @param pred_values: list of prediction scores
    @param t: threshold when the mutation considered positive (destructive)
    @return: (tp + tn) / (tp + tn + fp + fn)
    """
    true_positive = 0  # number of good predictions for destructive mutations
    true_negative = 0  # number of good predictions for non-destructive mutations
    false_positive = 0
    false_negative = 0
    assert len(true_values) == len(pred_values)
    for (true, pred) in zip(true_values, pred_values):
        if true < t and pred < t:
            true_positive += 1
        if true >= t and pred >= t:
            true_negative += 1
        if true < t and pred >= t:
            false_negative += 1
        if true >= t and pred < t:
            false_positive += 1
    assert true_positive + true_negative + false_positive + false_negative == len(true_values)
    res = PredictionAccuracy()
    res.tp = true_positive
    res.tn = true_negative
    res.fp = false_positive
    res.fn = false_negative
    return res


def eval_prism_scores_multi_sets(parameters, log):
    log('Evaluation')
    parameters.model.eval()
    true_values = np.array([])
    orig_true_values = np.array([])
    pred_values = np.array([])
    pos_values = np.array([])
    pid_values = np.array([])
    src_indices = np.array([])
    dst_indices = np.array([])

    batch_norm = int(CFG['flow_train']['batch_norm'])
    log(f'{batch_norm=}')

    with torch.no_grad():
        for t, loader in enumerate(parameters.loaders_list):
            log(f'Loader:{t}')
            for j, (pid, pos, all_deltas, pos_arr, emb_sector, emb_diff, score, bin_id, score_orig, src, dst) in tqdm(
                    enumerate(loader)):
                try:
                    pid = pid.to(DEVICE, dtype=torch.float).unsqueeze(dim=1)
                    pos = pos.to(DEVICE, dtype=torch.float).unsqueeze(dim=1)
                    pos_arr = pos_arr.to(DEVICE, dtype=torch.float)
                    all_deltas = all_deltas.to(DEVICE, dtype=torch.float)
                    emb_sector = emb_sector.to(DEVICE, dtype=torch.float)
                    emb_diff = emb_diff.to(DEVICE, dtype=torch.float)
                    target = score.to(DEVICE, dtype=torch.float)
                    # ---
                    if batch_norm == 1:
                        if len(target) < NUM_QUANTILES:
                            continue
                        target = normalize_tensor(target)
                        if torch.isnan(target).any():
                            log(f'eval_prism_scores_multi_sets: NaN is found in target: {target.size(dim=0)}')
                            continue
                    # ---
                    target2 = bin_id.to(DEVICE, dtype=torch.float)
                    y_hat = parameters.model(pid, pos, all_deltas, pos_arr, emb_sector, emb_diff)
                    y_hat = y_hat.squeeze(dim=-1)
                    y_hat1 = y_hat[:, 0]
                    y_hat2 = y_hat[:, 1]
                    true_values = np.append(true_values, target.cpu().numpy())
                    orig_true_values = np.append(orig_true_values, score_orig.cpu().numpy())
                    pred_values = np.append(pred_values, y_hat1.cpu().numpy())
                    pos_values = np.append(pos_values, pos.cpu().numpy())
                    src_indices = np.append(src_indices, src.cpu().numpy())
                    dst_indices = np.append(dst_indices, dst.cpu().numpy())
                    pid_values = np.append(pid_values, pid.cpu().numpy())
                except Exception as e:
                    log(f'\nEvaluation Error: {e}')
                    continue

    result = TestResult()
    mae = np.round(mean_absolute_error(true_values, pred_values), 4)
    mape = np.round(smape(true_values, pred_values), 4)
    mse = np.round(mean_squared_error(true_values, pred_values), 4)
    pearson = np.round(pearsonr(true_values, pred_values), 4)[0]
    spearman = np.round(spearmanr(true_values, pred_values), 4)[0]
    r2 = np.round(r2_score(true_values, pred_values), 4)
    # 0.3 = threshold for prediction accuracy
    accuracy_obj = calc_prediction_accuracy(true_values, pred_values, 0.3)
    result.prediction_accuracy = accuracy_obj
    result.mae = mae
    result.mape = mape
    result.mse = mse
    result.pearson = pearson
    result.spearman = spearman
    result.r2 = r2
    result.true_values = true_values
    result.pred_values = pred_values
    result.pos_values = pos_values
    result.pid_values = pid_values
    result.src_indices = src_indices
    result.dst_indices = dst_indices
    result.orig_true_values = orig_true_values
    return result
