"""
Used to create various plots for analysis of model training
"""
import os
import sys
import logging
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import pyplot as plt
from results import BINS_RESULTS_DICT
from utils import REPORT_ROOT, CFG, setup_reports
from run_prism_data_creation import _load_diff_embeddings

LOG_ENABLED = True
log = print
if LOG_ENABLED:
    # noinspection PyRedeclaration
    log = logging.info


def create_plot_acc_loss_per_epoch(logfile):
    train_acc_list = []
    train_loss_list = []
    valid_acc_list = []
    valid_loss_list = []
    with open(logfile, "r") as f:
        for line in f.readlines():
            if line.startswith('train acc: '):
                train_acc_list.append(float(line.split(' ')[2].strip()))
            if line.startswith('train loss: '):
                train_loss_list.append(float(line.split(' ')[2].strip()))
            if line.startswith('valid. acc: '):
                valid_acc_list.append(float(line.split(' ')[2].strip()))
            if line.startswith('valid. loss: '):
                valid_loss_list.append(float(line.split(' ')[2].strip()))

    print(train_loss_list)
    print(valid_loss_list)

    if len(train_acc_list) > 0:
        print(train_acc_list)
        print(valid_acc_list)
        assert len(train_acc_list) == len(train_loss_list)
        assert len(valid_acc_list) == len(valid_loss_list)
        assert len(train_acc_list) == len(valid_acc_list)
    else:
        assert len(train_loss_list) == len(valid_loss_list)

    xs = range(len(train_loss_list))
    fig, axs = plt.subplots(2)
    fig.suptitle("Acc & Loss per epoch")
    axs[0].plot(xs, train_loss_list, 'r', label='Train', linestyle='--', marker='.')
    axs[0].plot(xs, valid_loss_list, 'b', label='Validation', linestyle='--', marker='.')
    axs[0].legend(loc="upper left")
    plt.setp(axs[0], ylabel='Loss')
    axs[0].grid(True)
    axs[0].set_xticks([], minor=True)
    axs[0].set_yticks([], minor=True)

    if len(train_acc_list) > 0:
        axs[1].plot(xs, train_acc_list, 'r', label='Train', linestyle='--', marker='.')
        axs[1].plot(xs, valid_acc_list, 'b', label='Validation', linestyle='--', marker='.')
        axs[1].legend(loc="upper left")
        plt.setp(axs[1], ylabel='Accuracy')
        axs[1].grid(True)
        axs[1].set_xticks([], minor=True)
        axs[1].set_yticks([], minor=True)
    plt.xlabel("Epoch #")

    filename = os.path.basename(logfile)
    dir = os.path.dirname(logfile)
    plt_file = f'{filename}.plot.png'
    plot_path = os.path.join(dir, plt_file)
    plt.savefig(plot_path)
    plt.show()
    print(f'Created {plot_path}')


def run_create_plot_acc_loss_per_epoch():
    """
    Gets train log file and creates plots of accuracies and losses per epoch
    """
    sys.argv.append(r'C:\Temp\reports_dump\2022_04_01-23_01_03.score_training.log')
    if len(sys.argv) < 2:
        print('Please specify path to log file')
    else:
        logfile = sys.argv[1]
        create_plot_acc_loss_per_epoch(logfile)


def create_dd_correlation_plot(report_path, all_ddEs, all_truths, x_name):
    plt.clf()

    xs = all_ddEs
    ys = all_truths
    assert len(xs) == len(ys)

    mae = np.round(mean_absolute_error(ys, xs), 4)
    pearson = np.round(pearsonr(ys, xs), 4)[0]
    spearman = np.round(spearmanr(ys, xs), 4)[0]
    r2 = np.round(r2_score(ys, xs), 4)
    legend = f'MAE={mae}\nPearson={pearson}\nSpearman={spearman}\nR2={r2}'

    column_names = [x_name, 'Truth']
    df = pd.DataFrame(columns=column_names)
    df[x_name] = xs
    df['Truth'] = ys

    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)

    plot_path = os.path.join(report_path, f'{x_name}_vs_truth.plot.png')
    g = sns.scatterplot(data=df, x=x_name, y="Truth", marker=".")
    g.legend([legend], loc='best')

    plt.title(f'{x_name} vs Score')
    plt.grid()
    # plt.show()
    plt.savefig(plot_path)


def run_create_correlation_plots_on_data(report_path):
    """
    Create correlations plots:
    ddE vs. Truth Score
    ddG vs. Truth Score
    """
    log(f'create_diff_emb_splits')
    diff_len = int(CFG['general']['diff_len'])
    log(f'{diff_len=}')
    prism_data_list = _load_diff_embeddings(step=15)
    print(f'{len(prism_data_list)=}')

    all_variants = []
    for prism_data in prism_data_list:
        print(prism_data)
        all_variants += prism_data.variants

    print(f'{len(all_variants)=}')

    all_truths = [v.score for v in all_variants]
    all_ddEs = [v.ddE for v in all_variants]
    all_ddGs = [v.ddG for v in all_variants]

    assert len(all_truths) == len(all_ddEs)
    assert len(all_ddGs) == len(all_ddEs)

    create_dd_correlation_plot(report_path, all_ddEs, all_truths, 'ddE')
    create_dd_correlation_plot(report_path, all_ddGs, all_truths, 'ddG')


if __name__ == '__main__':
    report_path = setup_reports('create_plots')

    # run_create_plot_acc_loss_per_epoch()
    # run_create_correlation_plots_on_data(report_path)

    pass
