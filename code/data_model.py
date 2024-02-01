"""
Various classes for data representations
"""
import itertools
import math
import re

import numpy as np

from utils import AMINO_TO_LETTER


class Variant(object):
    """
    Class for a single mutation
    """

    def __init__(self):
        self.position = None  # position
        self.aa_from = None  # source
        self.aa_to = None  # destination
        self.score = None  # target
        self.score_orig = None  # target before normalization
        self.ddE = None
        self.ddE_orig = None
        self.ddG = None
        self.ddG_orig = None
        self.emb_diff = None  # embedding diff matrix between WT and mutation
        self.neighbors = None  # indices of positions for emb_diff (when PDB neighbors are used)
        self.pos_encoding = None  # positional encoding
        self.bin = None  # index of bin

    def __eq__(self, other):
        return (self.position, self.aa_to, self.aa_from) == (other.position, other.aa_to, other.aa_from)

    @staticmethod
    def valid_sub(substitution):
        """
        Check if the substitution token is valid
        """
        return bool(re.search(r'^[A-Z]\d{1,4}[A-Z]$', substitution))

    def load_from_series(self, series):
        """
        Init the fields from Pandas Series object
        @param series: pd series
        """
        substitution = series['variant']
        if not self.valid_sub(substitution):
            raise Exception(f'Invalid substitution: {series}')
        score = series['score_00']
        dd_e = series['gemme_score_01']
        dd_g = series['Rosetta_ddg_score_02']
        if math.isnan(score) or math.isnan(dd_e) or math.isnan(dd_g):
            raise Exception(f'Nan value: {series}')
        self.score = score
        self.score_orig = score
        self.ddE = dd_e
        self.ddG = dd_g
        self.ddE_orig = dd_e
        self.ddG_orig = dd_g
        self.position = int(substitution[1:-1])
        self.aa_from = substitution[0]
        self.aa_to = substitution[-1]

    def __str__(self):
        return f'{self.aa_from}->{self.aa_to}@{self.position},{round(self.score, 4)}'


class ModelConfig(object):
    """
    Class used to pass configuration for the model
    """

    def __init__(self):
        self.seq_emb_size = 0  # dimension of the sequence embedding
        self.heads = 0  # number of heads for multi-head attention
        self.attn_len = 0  # number of layers in transformer encoders
        self.diff_width = 0  # with of embedding window diff (plus-minus from the mutation position)
        self.cz = 0  # number of channels in model
        self.deltas_encoder = 0  # use encoder on 'all deltas'

    def __str__(self):
        return f'Model CFG:\n{self.attn_len=}\n{self.heads=}\n{self.seq_emb_size=}\n' \
               f'{self.cz=}\n{self.diff_width}\n'


class DebugData(object):
    """
    Class used to pass debug parameters to the model
    """

    def __init__(self):
        self.curr_epoch = 0
        self.report_path = None
        self.log = None


class PrismScoreData:
    """
    Used for loading sequence and variants from PRISM TXT data files
    For scores regression
    """

    def __init__(self):
        self.protein_name = None
        self.sequence = None
        self.variants = None  # list of mutation variants (including embeddings)
        self.file_name = None
        self.quantile_transformer = None

    def __repr__(self):
        return f'Name:{self.protein_name} Seq:{len(self.sequence)} V:{len(self.variants)} - {self.file_name}'


class Prediction(object):
    """
    Used for parsing results of Hoie
    (to read from Pickled file)
    """

    def __init__(self):
        self.index = None  # position of mutation
        self.aa_from = None  # 'from' amino-acid
        self.aa_to = None  # 'to' amino-acid
        self.true_score = None
        self.pred_score = None
        self.true_bin = None
        self.pred_bin = None


class PositionPrediction(object):
    """
    Used for plot 'Position vs. rank delta'
    """

    def __init__(self):
        self.index = None  # position of mutation
        self.true_score = None
        self.pred_score = None
        self.true_rank = None
        self.pred_rank = None

    def get_rank_delta(self):
        return self.true_rank - self.pred_rank

    def get_score_delta(self):
        return self.true_score - self.pred_score


if __name__ == '__main__':
    pass
