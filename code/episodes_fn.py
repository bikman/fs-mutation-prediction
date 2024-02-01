'''
TBD
'''
import math
import higher
from tqdm import tqdm
import copy
import torch
import random
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from run_prism_data_creation import log
from utils import CFG, DEVICE, TestResult
from torch.utils.data import DataLoader
from train import calc_loss, calc_prediction_accuracy
from episodes_dataset import DataContext
import torch.nn.functional as F


def unfold_batch(data_batch):
    """
    TBD
    @param data_batch:
    @return:
    """
    (pid, pos, all_deltas, pos_arr, emb_sector, emb_diff, score, bin_id) = data_batch
    pid = pid.to(DEVICE, dtype=torch.float).unsqueeze(dim=1)
    pos = pos.to(DEVICE, dtype=torch.float).unsqueeze(dim=1)
    pos_arr = pos_arr.to(DEVICE, dtype=torch.float)
    all_deltas = all_deltas.to(DEVICE, dtype=torch.float)
    emb_sector = emb_sector.to(DEVICE, dtype=torch.float)
    emb_diff = emb_diff.to(DEVICE, dtype=torch.float)
    target = score.to(DEVICE, dtype=torch.float)
    target2 = bin_id.to(DEVICE, dtype=torch.float)
    return pid, pos, all_deltas, pos_arr, emb_sector, emb_diff, target, target2


def val_epoch(context, model, num_inner_epochs, alpha=0.75, bins=10):
    """
    TBD
    @param context:
    @param model:
    @param num_inner_epochs:
    @param alpha:
    @param bins:
    @return:
    """
    qry_losses = []
    model.train()
    inner_opt = torch.optim.Adam(model.parameters(), lr=float(CFG['flow_train']['lr']))
    for (supp_loader, query_loader) in zip(context.support_loaders_list, context.query_loaders_list):
        for (support_batch, query_batch) in zip(supp_loader, query_loader):
            (pid, pos, all_deltas, pos_arr, emb_sector, emb_diff, target, target2) = unfold_batch(support_batch)
            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (f_model, diff_opt):
                for _ in range(num_inner_epochs):
                    spt_logits = f_model(pid, pos, all_deltas, pos_arr, emb_sector, emb_diff)
                    spt_logits = spt_logits.squeeze(dim=-1)
                    y_hat1 = spt_logits[:, 0]
                    y_hat2 = spt_logits[:, 1]
                    loss1 = F.mse_loss(y_hat1, target)
                    loss2 = F.mse_loss(y_hat2, target2)
                    spt_loss = calc_loss(alpha, bins, loss1, loss2)
                    diff_opt.step(spt_loss)

                # The query loss induced by these parameters.
                (pid, pos, all_deltas, pos_arr, emb_sector, emb_diff, target, target2) = unfold_batch(query_batch)
                qry_logits = f_model(pid, pos, all_deltas, pos_arr, emb_sector, emb_diff)
                qry_logits = qry_logits.squeeze(dim=-1)
                y_hat1 = qry_logits[:, 0]
                y_hat2 = qry_logits[:, 1]
                loss1 = F.mse_loss(y_hat1, target)
                loss2 = F.mse_loss(y_hat2, target2)
                qry_loss = calc_loss(alpha, bins, loss1, loss2)
                qry_losses.append(qry_loss.clone().detach())
    val_loss_value = torch.mean(torch.tensor(qry_losses))
    return val_loss_value


def test_epoch(context, model, num_inner_epochs, alpha=0.75, bins=10):
    """
    TBD
    @param context:
    @param model:
    @param num_inner_epochs:
    @param alpha:
    @param bins:
    @return:
    """
    model.train()
    inner_opt = torch.optim.Adam(model.parameters(), lr=float(CFG['flow_train']['lr']))
    true_values = np.array([])
    pred_values = np.array([])
    pos_values = np.array([])
    pid_values = np.array([])
    for (supp_loader, query_loader) in zip(context.support_loaders_list, context.query_loaders_list):
        for support_batch in supp_loader:
            (pid, pos, all_deltas, pos_arr, emb_sector, emb_diff, target, target2) = unfold_batch(support_batch)
            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (f_model, diff_opt):
                try:
                    for _ in range(num_inner_epochs):
                        spt_logits = f_model(pid, pos, all_deltas, pos_arr, emb_sector, emb_diff)
                        spt_logits = spt_logits.squeeze(dim=-1)
                        y_hat1 = spt_logits[:, 0]
                        y_hat2 = spt_logits[:, 1]
                        loss1 = F.mse_loss(y_hat1, target)
                        loss2 = F.mse_loss(y_hat2, target2)
                        spt_loss = calc_loss(alpha, bins, loss1, loss2)
                        diff_opt.step(spt_loss)

                    for query_batch in query_loader:
                        (pid, pos, all_deltas, pos_arr, emb_sector, emb_diff, target, target2) = unfold_batch(
                            query_batch)
                        qry_logits = f_model(pid, pos, all_deltas, pos_arr, emb_sector, emb_diff)
                        qry_logits = qry_logits.squeeze(dim=-1)
                        y_hat1 = qry_logits[:, 0]
                        y_hat2 = qry_logits[:, 1]
                        true_values = np.append(true_values, target.cpu().numpy())
                        pred_values = np.append(pred_values, y_hat1.detach().cpu().numpy())
                        pos_values = np.append(pos_values, pos.cpu().numpy())
                        pid_values = np.append(pid_values, pid.cpu().numpy())
                except Exception as e:
                    log(f'\nTest Step Error: {e}')
                continue
    result = TestResult()
    mae = np.round(mean_absolute_error(true_values, pred_values), 4)
    mse = np.round(mean_squared_error(true_values, pred_values), 4)
    pearson = np.round(pearsonr(true_values, pred_values), 4)[0]
    spearman = np.round(spearmanr(true_values, pred_values), 4)[0]
    r2 = np.round(r2_score(true_values, pred_values), 4)
    # 0.3 = threshold for prediction accuracy
    accuracy_obj = calc_prediction_accuracy(true_values, pred_values, 0.3)
    result.prediction_accuracy = accuracy_obj
    result.mae = mae
    result.mse = mse
    result.pearson = pearson
    result.spearman = spearman
    result.r2 = r2
    result.true_values = true_values
    result.pred_values = pred_values
    result.pos_values = pos_values
    result.pid_values = pid_values
    return result


def train_epoch(context, model, meta_opt, num_inner_epochs, inner_lr, alpha=0.75, bins=10):
    """
    TBD
    @param inner_lr:
    @param context:
    @param model:
    @param meta_opt:
    @param num_inner_epochs:
    @param alpha:
    @param bins:
    @return:
    """
    qry_losses = []
    model.train()
    inner_opt = torch.optim.Adam(model.parameters(), lr=inner_lr)
    meta_opt.zero_grad()
    for (supp_loader, query_loader) in zip(context.support_loaders_list, context.query_loaders_list):
        for (support_batch, query_batch) in zip(supp_loader, query_loader):
            (pid, pos, all_deltas, pos_arr, emb_sector, emb_diff, target, target2) = unfold_batch(support_batch)
            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (f_model, diff_opt):
                for _ in range(num_inner_epochs):
                    spt_logits = f_model(pid, pos, all_deltas, pos_arr, emb_sector, emb_diff)
                    spt_logits = spt_logits.squeeze(dim=-1)
                    y_hat1 = spt_logits[:, 0]
                    y_hat2 = spt_logits[:, 1]
                    loss1 = F.mse_loss(y_hat1, target)
                    loss2 = F.mse_loss(y_hat2, target2)
                    spt_loss = calc_loss(alpha, bins, loss1, loss2)
                    diff_opt.step(spt_loss)  # note that `step` must take `loss` as an argument!

                # The query loss induced by these parameters.
                (pid, pos, all_deltas, pos_arr, emb_sector, emb_diff, target, target2) = unfold_batch(query_batch)
                qry_logits = f_model(pid, pos, all_deltas, pos_arr, emb_sector, emb_diff)
                qry_logits = qry_logits.squeeze(dim=-1)
                y_hat1 = qry_logits[:, 0]
                y_hat2 = qry_logits[:, 1]
                loss1 = F.mse_loss(y_hat1, target)
                loss2 = F.mse_loss(y_hat2, target2)
                qry_loss = calc_loss(alpha, bins, loss1, loss2)
                qry_loss.backward()
                qry_losses.append(qry_loss.clone().detach())
    meta_opt.step()
    train_loss_value = torch.mean(torch.tensor(qry_losses))
    return train_loss_value
