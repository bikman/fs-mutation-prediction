"""
Module with general utilities.
"""

import configparser
import logging
import os
import sys
from datetime import datetime
from sys import platform

import numpy as np
import torch
from sklearn import preprocessing

# current time
TIMESTAMP = datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%f")
REPORT_ROOT = 'reports'
CONFIG_FILE = 'config.win.ini'
ESM_MODEL = r'C:\MODELS\esm1b_msa\esm_msa1b_t12_100M_UR50S.pt'
ESM_REGRESSION = r'C:\MODELS\esm1b_msa\esm_msa1b_t12_100M_UR50S-contact-regression.pt'
MODELS_FOLDER = r'models'
LOG_FOLDER = r'logs'
RESULTS_PATH = r'D:\RUN_RESULTS'  # windows only
if platform == "linux" or platform == "linux2":
    REPORT_ROOT = '/root/code/reports'
    ESM_MODEL = r'/root/code/esm_pretrain/esm_msa1b_t12_100M_UR50S.pt'
    ESM_REGRESSION = r'/root/code/esm_pretrain/esm_msa1b_t12_100M_UR50S-contact-regression.pt'
    CONFIG_FILE = '/root/code/config.linux.ini'
    MODELS_FOLDER = r'/root/code/models'
    LOG_FOLDER = r'/root/code/logs'
    RESULTS_PATH = r'/root/code/reports'  # windows only


def load_config():
    print('Platform:', platform)
    config = configparser.ConfigParser()
    print('Config found:', os.path.exists(CONFIG_FILE))
    config.read(CONFIG_FILE)
    print(config.sections())
    return config


CFG = load_config()  # global config object
print(CFG)
DUMP_ROOT = str(CFG['general']['dump_root'])  # the folder where all the data will be created
PRISM_TRAIN_SPLIT = 'prism_train_split.pkl'
PRISM_VALID_SPLIT = 'prism_valid_split.pkl'
PRISM_EVAL_SPLIT = 'prism_eval_split.pkl'
PRISM_FINE_TRAIN_SPLIT = 'prism_fine_train_split.pkl'
PRISM_FINE_EVAL_SPLIT = 'prism_fine_eval_split.pkl'

PRISM_FOLDER = r'C:\DATASETS\ECOD_MAVE\mave'
if platform == "linux" or platform == "linux2":
    PRISM_FOLDER = r'/root/code/data/mave'

# device definition
DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')

MAX_SEQUENCE = 1023  # otherwise ESM embedder will fail
NUM_QUANTILES = 4
DIFF_LEN = 15
EMB_SIZE = 768

AMINO_TO_LETTER = {
    'ALA': 'A',
    'CYS': 'C',
    'CME': 'C',  # non-standard
    'CSX': 'C',  # non-standard
    'ASP': 'D',
    'GLU': 'E',
    'PHE': 'F',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LYS': 'K',
    'LEU': 'L',
    'MET': 'M',
    'MSE': 'M',  # non-standard
    'MLU': 'M',  # non-standard
    'ASN': 'N',
    'PRO': 'P',
    'GLN': 'Q',
    'ARG': 'R',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V',
    'CA': '',  # skip illegal token
    'SAH': '',  # skip illegal token
    'PTR': ''  # skip illegal token
}

AA_ALPHABETICAL = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

AA_DICT = {k: v for v, k in enumerate(AA_ALPHABETICAL)}

AA_ALLOWED_MUTATIONS_DICT = {
    'V': 'AILMEFGD',
    'P': 'THSAQRL',
    'M': 'VKTRIL',
    'N': 'KHYTDIS',
    'D': 'NVEAYGH',
    'Q': 'KEPHRL',
    'R': 'SPWCILGKMHTQ',
    'I': 'VRLMKFNTS',
    'K': 'RIMENTQ',
    'H': 'NQPYRDL',
    'S': 'PAWRCILYTFNG',
    '*': 'SWRCLKEYGQ',
    'W': 'GSRCL',
    'E': 'VKAGDQ',
    'F': 'VLYCIS',
    'Y': 'DFHNCS',
    'T': 'PARIKMNS',
    'G': 'VWARCEDS',
    'A': 'VTEPGDS',
    'C': 'SFWYRG',
    'L': 'VSWPRIMFHQ'
}

ALL_PROTEIN_FILES_DICT = {
    1: '002_PTEN_phosphatase_activity.txt',
    # -----------------------------------------------------
    2: '003_PTEN_abundance.txt',
    3: '004_NUDT15_drug_sensitivity_reordered.txt',
    4: '005_NUDT15_abundance_reordered.txt',
    5: '006_CBS_high_B6_activity.txt',
    6: '007_CBS_low_B6_activity.txt',
    7: '008_CALM1_growth_abundance.txt',
    8: '009_SUMO1_growth_abundance.txt',
    9: '010_UBE2I_growth_abundance.txt',
    10: '011_TPK1_growth_abundance.txt',
    11: '012_P53_abundance_reversed.txt',
    12: '013_MAPK1_drug_recovery_reversed.txt',
    13: '014_TPMT_abundance.txt',
    14: '021_PAB1_doxycyclin_sensitivity.txt',
    15: '022_UBI4_dextrose_growth_competition.txt',
    16: '025_BRCA1_BARD1_heterodimer_formation.txt',
    17: '026_BRCA1_E3_ubiquitination_activity.txt',
    18: '027_Src_kinase_activity_catalytic_domain_reversed.txt',
    19: '029_HMGCR_yeast_complementation_atorvastatin_medium.txt',
    20: '030_HMGCR_yeast_complementation_control_medium.txt',
    21: '031_HMGCR_yeast_complementation_rosuvastatin_medium.txt',
    22: '033_LDLRAP1_yeast_2_hybrid_OBFC1_interaction.txt',
    23: '037_UBI4_E1_binding_limiting_E1.txt',
    24: '999_ADRB2_DMS.txt',
    25: '999_bla_DMS_a.txt',
    26: '999_bla_DMS_b.txt',
    27: '999_bla_DMS_c.txt',
    28: '999_bla_DMS_d.txt',
    29: '999_cas9_DMS.txt',
    30: '999_ccdB_DMS.txt',
    31: '999_env_DMS.txt',
    32: '999_GAL4_DMS.txt',
    33: '999_GmR_DMS.txt',
    34: '999_haeIIIM_DMS.txt',
    35: '999_HAh1n1_DMS.txt',
    36: '999_HAh3n2_DMS.txt',
    37: '999_HRas_DMS.txt',
    38: '999_HSP82_DMS.txt',
    # -----------------------------------------------------
    39: '999_IF-1_DMS.txt'
}

ALL_PROTEINS_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                     28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]

"""
All proteins except ones with the strange (almost discrete 27, 30) scores distribution
"""
NORMAL_PROTEINS_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                        28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39]
"""
Proteins checked on 'competitive growth' assay
"""
COMPETITIVE_GROWTH_ASSAY_LIST = [1, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 23, 24, 34, 38, 39]

"""
Proteins checked on 'antibiotics resistance' assay
"""
ANTIBIOTICS_RESISTANCE_LIST = [25, 26, 27, 28, 33]

"""
Used for debug only
"""
# DEBUG_LIST = [1, 2, 39]
DEBUG_LIST = [14, 15, 23, 39]  # Shortest sequences
# DEBUG_LIST = [14, 15, 25, 26, 27, 28, 39]  # Short + multi-test
# DEBUG_LIST = [19, 20, 21, 29, 31]  # Longest sequences


"""
Used to run on smaller protein subsets
"""
PROTEINS_GROUPS_DICT = {
    1: ALL_PROTEINS_LIST,
    2: NORMAL_PROTEINS_LIST,
    3: COMPETITIVE_GROWTH_ASSAY_LIST,
    4: ANTIBIOTICS_RESISTANCE_LIST,
    5: DEBUG_LIST
}

"""
Each sub-list contain protein file indexes for the same protein but different test
"""
MULTITEST_PROTEINS = [
    [1, 2],
    [3, 4],
    [5, 6],
    [15, 23],
    [16, 17],
    [19, 20, 21],
    [25, 26, 27, 28]
]


def get_protein_files_dict():
    """

    @return:
    """
    # get the dictionary according to CFG selection
    protein_set_index = int(CFG['general']['protein_set'])
    try:
        list_of_proteins = PROTEINS_GROUPS_DICT[protein_set_index]
    except Exception as e:
        raise Exception(f'Cannot get list of proteins by index: {protein_set_index}, error: {e}')

    # filter of the ALL_PROTEIN_FILES_DICT according to the list
    proteins_dictionary = {protein_index: ALL_PROTEIN_FILES_DICT[protein_index] for protein_index in list_of_proteins}
    assert len(proteins_dictionary) == len(list_of_proteins)
    eval_protein_file_number = int(CFG['general']['eval_protein_file_number'])
    if eval_protein_file_number not in proteins_dictionary:
        raise Exception(
            f'Illegal eval protein index: {eval_protein_file_number}. \n'
            f'Eval protein must be from protein set (set index: {protein_set_index}) \n'
            f'List of selected proteins: {list_of_proteins}')

    return proteins_dictionary


def normalize_list(values):
    """
    Normalizes list of floats
    @param values: list of floats
    """
    # -- using fitted normalization --
    x = np.array(values)
    x = x.reshape(-1, 1)
    quantile_transformer = preprocessing.QuantileTransformer(n_quantiles=NUM_QUANTILES)
    transformed = quantile_transformer.fit_transform(x)
    res = transformed.flatten().tolist()
    return res, quantile_transformer


def normalize_scores_ds(data_set):
    """
    TBD
    @param data_set:
    @return:
    """
    # data item format
    # [pid, v.position, all_deltas, pos_enc, emb_sector, v.emb_diff, v.score, v.bin, v.score_orig, src, dst]
    score_ndx = 6
    scores = [x[score_ndx] for x in data_set.data]
    norm_scores, quantile_transformer = normalize_list(scores)
    for i, val in enumerate(norm_scores):
        data_set.data[i][score_ndx] = val
    return quantile_transformer


def normalize_deltas_only(prism_data_list):
    """
    Apply normalization to deltas ddG ONLY
    """
    for prism_data in prism_data_list:
        values = [v.ddG_orig for v in prism_data.variants]
        orig_values = list(values)
        norm_values = normalize_list(values)
        assert len(norm_values) == len(prism_data.variants)
        for i, v in enumerate(prism_data.variants):
            v.ddG = norm_values[i]
            v.ddG_orig = orig_values[i]


def normalize_scores_only(prism_data_list):
    """
    Apply normalization to score ONLY
    """
    for prism_data in prism_data_list:
        scores = [v.score for v in prism_data.variants]
        orig_scores = list(scores)
        norm_scores, quantile_transformer = normalize_list(scores)
        prism_data.quantile_transformer = quantile_transformer
        assert len(norm_scores) == len(prism_data.variants)
        for i, v in enumerate(prism_data.variants):
            v.score = norm_scores[i]
            v.score_orig = orig_scores[i]


def get_lr(optimizer):
    return [p['lr'] for p in optimizer.param_groups][0]


def get_embedding_sector(emb_diff, mutation_ndx, step):
    """
    Create the slice of embedding difference tensor according to +-step
    @param emb_diff: emb. diff between WT and MUT type
    @param mutation_ndx: index of mutation
    @param step: +- distance from mutation
    @return: tensor of size (2*step+1, 768)
    """
    start_ndx = mutation_ndx - step
    end_ndx = mutation_ndx + step + 1
    if start_ndx < 0:
        diff_sector = emb_diff[:end_ndx, :]
        pad_count = (2 * step + 1) - diff_sector.size()[0]
        assert pad_count > 0
        # pad left with zeros
        padding = torch.nn.ZeroPad2d((0, 0, pad_count, 0))
        diff_sector = padding(diff_sector)

    elif end_ndx > len(emb_diff):
        diff_sector = emb_diff[start_ndx:, :]
        pad_count = (2 * step + 1) - diff_sector.size()[0]
        assert pad_count > 0
        # pad right with zeros
        padding = torch.nn.ZeroPad2d((0, 0, 0, pad_count))
        diff_sector = padding(diff_sector)
    else:
        diff_sector = emb_diff[start_ndx:end_ndx]
    return diff_sector


class TrainParameters(object):
    """
     Use this class for train run
    """

    def __init__(self):
        self.model = None
        self.epochs = 0
        self.loss = None
        self.loss2 = None
        self.optimizer = None
        self.train_loader = None
        self.valid_loader = None
        self.patience = None  # default
        self.model_path = None  # path to saved PT file
        self.report_path = None  # report path
        self.scheduler = None
        self.loader_pairs = None
        self.train_loaders_list = None
        self.valid_loaders_list = None
        self.bins = None
        self.alpha = None
        self.support_loaders_list = None  # used for episodes training
        self.query_loaders_list = None  # used for episodes training

    def __str__(self):
        return '--------Train params--------\n' \
               f'Epochs: {self.epochs}\n' \
               f'Loss: {self.loss}\n' \
               f'Optimizer:{self.optimizer}\n' \
               f'Scheduler:{self.scheduler}\n' \
               f'Model path:{self.model_path}\n' \
               f'Patience:{self.patience}\n' \
               f'Bins:{self.bins}\n' \
               f'Alpha:{self.alpha}\n' \
               '-----------------------------\n'


class TestParameters(object):

    def __init__(self):
        self.model = None
        self.loader = None
        self.loaders_list = None
        self.loss = None
        self.model_file = None

    def __str__(self):
        return '--------Test params--------\n' \
               f'Loss: {self.loss}\n' \
               f'Model save:{self.model_file}\n' \
               '-----------------------------\n'


class PredictionAccuracy(object):
    """
    Accuracy of prediction for destructive mutations
    """

    def __int__(self):
        self.fp = None
        self.tp = None
        self.fn = None
        self.tn = None

    def get_accuracy(self):
        acc = (self.tn + self.tp) * 1.0 / (self.tn + self.tp + self.fp + self.fn)
        return np.round(acc, 4)

    def __str__(self):
        return f'TP={self.tp}\n' \
               f'TN={self.tn}\n' \
               f'FP={self.fp}\n' \
               f'FN={self.fn}\n' \
               f'{self.get_accuracy()}'


class TestResult(object):
    """
    Used for test evaluation
    """

    def __init__(self):
        self.loss = None  # loss
        self.mae = None  # mean abs error
        self.mse = None  # mean square error
        self.mape = None  # mean square percentage error
        self.pearson = None  # pearson correlation
        self.spearman = None  # spearman correlation
        self.r2 = None  # R^2
        self.true_values = None  # array of truth values for scores
        self.pred_values = None  # array of predicted values for scores
        self.pos_values = None  # array of position values for scores
        self.model_name = None  # name of PT model file
        self.prediction_accuracy = None  # accuracy object
        self.orig_true_values = None  # original true scores before normalization
        self.nn_mae = []  # non-normalized mean square error
        self.src_indices = None  # list of source aa indices in the trained order
        self.dst_indices = None  # list of destination aa indices in the trained order
        self.nn_pred_values = None  # list of non-normalized predicted values

    def get_legend(self):
        res = '--------Test result--------\n'
        if self.loss is not None:
            res += f'Loss: {self.loss}\n'
        if self.mae is not None:
            res += f'MAE: {self.mae}\n'
        if self.mape is not None:
            res += f'MAPE: {self.mape}\n'
        if self.nn_mae is not None:
            res += f'NN MAE: {self.nn_mae}\n'
        if self.mse is not None:
            res += f'MSE: {self.mse}\n'
        if self.pearson is not None:
            res += f'pearson: {self.pearson}\n'
        if self.spearman is not None:
            res += f'spearman: {self.spearman}\n'
        if self.r2 is not None:
            res += f'r2: {self.r2}\n'
        if self.prediction_accuracy is not None:
            res += f'prediction_accuracy:\n{self.prediction_accuracy}\n'
        return res

    def get_short_legend(self):
        res = ''
        if self.mae is not None:
            res += f'MAE: {self.mae}\n'
        if self.spearman is not None:
            res += f'spearman: {self.spearman}\n'
        return res

    def __str__(self):
        return self.get_legend()


class TrainResult(object):
    """
    Result of train + validation run
    """

    def __init__(self):
        self.train_loss_per_batch = []
        self.train_loss_per_epoch = []
        self.train_accuracy_per_epoch = []
        self.validation_loss_per_epoch = []
        self.validation_accuracy_per_epoch = []
        self.best_epoch = -1
        self.test_result = None  # test result object (for fine-tune usage)
        self.ft_mutation_count = -1  # used only for fine-tuning

    def train_accuracy(self):
        if len(self.train_accuracy_per_epoch) > 0:
            return self.train_accuracy_per_epoch[self.best_epoch]
        else:
            return -1

    def validation_accuracy(self):
        if len(self.validation_accuracy_per_epoch) > 0:
            return self.validation_accuracy_per_epoch[self.best_epoch]
        else:
            return -1

    def train_loss_min(self):
        if len(self.train_loss_per_epoch) > 0:
            return self.train_loss_per_epoch[self.best_epoch]
        else:
            return -1

    def valid_loss_min(self):
        if len(self.validation_loss_per_epoch) > 0:
            return self.validation_loss_per_epoch[self.best_epoch]
        else:
            return -1

    def __str__(self):
        return '--------Run result--------\n' \
               f'Train Loss: {self.train_loss_min()}\n' \
               f'Train Acc: {self.train_accuracy()}\n' \
               f'Valid Loss:{self.valid_loss_min()}\n' \
               f'Valid Acc:{self.validation_accuracy()}\n' \
               f'Best Epoch:{self.best_epoch}\n' \
               '-----------------------------\n'

    def get_test_results(self):
        if self.test_result is None:
            return 'Cannot print test result! (Only for fine tune mode)'
        return '--------Test result--------\n' \
               f'MAE: {self.test_result.mae}\n' \
               f'NN MAE: {self.test_result.nn_mae}\n' \
               f'Pearson: {self.test_result.pearson}\n' \
               f'Spearman: {self.test_result.spearman}\n' \
               f'R2: {self.test_result.r2}\n' \
               f'Accuracy:\n{str(self.test_result.prediction_accuracy)}\n' \
               '-----------------------------\n'


def setup_reports(report_type='na'):
    """
    Create setup for logger - both to file and to console
    Also model will be saved in the same folder with logs
    """
    if not os.path.isdir(REPORT_ROOT):
        os.makedirs(REPORT_ROOT)
    report_folder = f'{TIMESTAMP}_report'
    report_path = os.path.join(REPORT_ROOT, report_folder)
    if not os.path.isdir(report_path):
        os.makedirs(report_path)

    logging.getLogger('matplotlib.font_manager').disabled = True
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    log_file = f'{TIMESTAMP}.{report_type}.log'
    output_file_path = os.path.join(report_path, log_file)
    output_file_handler = logging.FileHandler(output_file_path)
    logger.addHandler(output_file_handler)
    # # --- disable double output --- # #
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stdout_handler)
    # ----------------------------------
    print(f'{report_path=}')
    print(f'{output_file_path=}')
    return report_path


def create_result_txt(report_path, protein_filename, test_res, ft_res_arr):
    """
    Creates text file with MAE, Sp., FT4 MAE and Sp.
    """
    if ft_res_arr is not None:  # in case of fine-tuning
        csv_res = os.path.join(report_path, 'result.txt')
        with open(csv_res, "w") as f:
            f.write(f'name:{protein_filename}\n')
            f.write(f'mae:{test_res.mae}\n')
            f.write(f'mape:{test_res.mape}\n')
            f.write(f'nn_mae:{test_res.nn_mae}\n')
            f.write(f'spearman:{test_res.spearman}\n')
            f.write('*' * 30)
            f.write('\n')
            for train_res in ft_res_arr:
                f.write(f'ft{train_res.ft_mutation_count} mae:{train_res.test_result.mae}\n')
                f.write(f'ft{train_res.ft_mutation_count} mape:{train_res.test_result.mape}\n')
                f.write(f'ft{train_res.ft_mutation_count} nn_mae:{train_res.test_result.nn_mae}\n')
                f.write(f'ft{train_res.ft_mutation_count} sp.:{train_res.test_result.spearman}\n')
                f.write('-' * 30)
                f.write('\n')

    else:
        csv_res = os.path.join(report_path, 'result.txt')
        with open(csv_res, "w") as f:
            f.write(f'name:{protein_filename}\n')
            f.write(f'mae:{test_res.mae}\n')
            f.write(f'mape:{test_res.mape}\n')
            f.write(f'nn_mae:{test_res.nn_mae}\n')
            f.write(f'spearman:{test_res.spearman}\n')


def smape(A, F):
    """
    Symmetric mean absolute percentage error (SMAPE or sMAPE)
    is an accuracy measure based on percentage (or relative) errors.
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        tmp = 2.0 * np.abs(F - A) / (np.abs(A) + np.abs(F))
    tmp[np.isnan(tmp)] = 0.0
    return np.sum(tmp) / len(tmp) * 100.0


if __name__ == '__main__':
    pass
