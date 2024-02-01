import torch
import numpy
import math
from torch import nn
from torch.nn import functional as F
from torch import Tensor, AnyType
from typing import Optional

from classifier import Predictor, PredictorLarge, ResidualPredictor
from utils import DEVICE
from matplotlib import pyplot as plt
import time
import os
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class WightedTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(WightedTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(WightedTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> (Tensor, AnyType):
        t_src = torch.transpose(src, 0, 1)
        src2, weights = self.self_attn(t_src, t_src, t_src, attn_mask=src_mask,
                                       key_padding_mask=src_key_padding_mask, need_weights=True)
        src2 = torch.transpose(src2, 0, 1)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, weights


class WeightedTransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(WeightedTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> (Tensor, AnyType):

        output = src
        weights = None
        for mod in self.layers:
            output, weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output, weights


class PrismScoreEmbDiffModel(nn.Module):
    """
    Model that uses sequence embedding,
    diff between WT and mutated embeddings
    And transformer encoders
    --- THIS MODEL USED ONLY AS SUPERCLASS FOR INHERITANCE ---
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.file_name = 'PrismScoreEmbDiffModel.pt'
        self.cz = model_cfg.cz
        self.diff_width = model_cfg.diff_width
        self.seq_emb_size = model_cfg.seq_emb_size
        self.heads = model_cfg.heads
        self.use_deltas_encoder = model_cfg.deltas_encoder

        # create enlarger
        self.enlarger = nn.Sequential(
            nn.Linear(40, 56),
            nn.LeakyReLU(),
            nn.Linear(56, self.cz)
        )

        # create reducers
        reducer_in_size = int(self.seq_emb_size / 2)
        self.seq_emb_reducer = nn.Sequential(
            nn.Linear(self.seq_emb_size, reducer_in_size),
            nn.LeakyReLU(),
            nn.Linear(reducer_in_size, self.cz)
        )
        self.emb_diff_reducer = nn.Sequential(
            nn.Linear(self.seq_emb_size, reducer_in_size),
            nn.LeakyReLU(),
            nn.Linear(reducer_in_size, self.cz)
        )

        # create encoders
        num_layers = model_cfg.attn_len
        emb_encoder_layer = WightedTransformerEncoderLayer(d_model=self.seq_emb_size, nhead=self.heads)
        self.emb_encoder = WeightedTransformerEncoder(emb_encoder_layer, num_layers=num_layers)
        if self.use_deltas_encoder == 1:
            dds_encoder_layer = WightedTransformerEncoderLayer(d_model=self.cz, nhead=self.heads)
            self.dds_encoder = WeightedTransformerEncoder(dds_encoder_layer, num_layers=num_layers)

        # --- create complex MLP score predictor ---
        self.in_size = int(self.cz * (2 * self.diff_width + 1) * 3)
        hid_size1 = int(self.in_size / 2)
        hid_size2 = int(hid_size1 / 2)
        hid_size3 = int(hid_size2 / 2)
        self.mlp = nn.Sequential(  # 4-layers perceptron
            nn.Linear(self.in_size, self.in_size),  # no dimension reducing
            nn.LeakyReLU(),
            nn.Linear(self.in_size, self.in_size),  # no dimension reducing
            nn.LeakyReLU(),
            nn.Linear(self.in_size, hid_size1),
            nn.LeakyReLU(),
            nn.Linear(hid_size1, hid_size3),
            nn.LeakyReLU(),
            nn.Linear(hid_size3, 1)  # score prediction
        )
        # flatten module
        self.flatten = nn.Flatten()

    def forward(self, pid, pos, all_deltas_in, pos_enc, emb_sector, emb_diff_in, debug_data=None):
        all_deltas = self.enlarger(all_deltas_in)

        dds_weights = None

        # --- NOT IN USE ---
        # if self.use_deltas_encoder == 1:
        #     all_deltas, dds_weights = self.dds_encoder(all_deltas)

        emb_sector = self.seq_emb_reducer(emb_sector)
        emb_sector = emb_sector + pos_enc

        emb_diff, emb_weights = self.emb_encoder(emb_diff_in)
        emb_diff = self.emb_diff_reducer(emb_diff)

        # --- debug code ---
        if debug_data is not None:
            epoch = debug_data.curr_epoch
            folder = debug_data.report_path
            log = debug_data.log

            ew_size = emb_weights.size()
            log(f'emb_weights.size={ew_size}')
            plot_tensor(emb_weights[0], folder, f'emb_weights.ep_{epoch}')

            if self.use_deltas_encoder == 1 and dds_weights is not None:
                dds_weights_size = dds_weights.size()
                log(f'dds_weights.size={dds_weights_size}')
                plot_tensor(dds_weights[0], folder, f'dds_weights.ep_{epoch}')

            # emb_diff_in_size = emb_diff_in.size()
            # log(f'emb_diff_in.size={emb_diff_in_size}')
            # plot_tensor(emb_diff_in[0], folder, f'emb_diff_in.ep_{epoch}')

            # all_deltas_in_size = all_deltas_in.size()
            # log(f'all_deltas_in.size={all_deltas_in_size}')
            # plot_tensor(all_deltas_in[0], folder, f'all_deltas_in.ep_{epoch}')

        x = self.flatten(all_deltas)
        y = self.flatten(emb_sector)
        z = self.flatten(emb_diff)

        res = torch.cat((x, y, z), 1)

        # --- debug code ---
        if debug_data is not None:
            epoch = debug_data.curr_epoch
            folder = debug_data.report_path
            plot_tensor(all_deltas[0], folder, f'all_deltas.ep_{epoch}')
            plot_tensor(pos_enc[0], folder, f'pos_enc.ep_{epoch}')
            plot_tensor(emb_sector[0], folder, f'emb_sector.ep_{epoch}')
            plot_tensor(emb_diff[0], folder, f'emb_diff.ep_{epoch}')

        res = self.mlp(res)
        return res


class PrismScoreEmbDiffSimpleModel(PrismScoreEmbDiffModel):
    """
    --- THIS IS THE MAIN FULL MODEL 1 ---
    Same as 'PrismScoreEmbDiffModel'
    but with simpler MLP score predictor
    ------------------------------------
    """

    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self.file_name = 'PrismScoreEmbDiffSimpleModel.pt'

        self.in_size = int(self.cz * (2 * self.diff_width + 1) * 3)
        hid_size1 = int(self.in_size / 2)
        hid_size2 = int(hid_size1 / 4)
        hid_size3 = int(hid_size2 / 4)
        self.mlp = nn.Sequential(  # 4-layers perceptron
            nn.Linear(self.in_size, hid_size1),  # dimension reducing
            nn.LeakyReLU(),
            nn.Linear(hid_size1, hid_size2),  # dimension reducing
            nn.LeakyReLU(),
            nn.Linear(hid_size2, hid_size3),  # dimension reducing
            nn.LeakyReLU(),
            nn.Linear(hid_size3, 2)  # score prediction
        )


class PrismScoreDeltasOnlyModel(PrismScoreEmbDiffSimpleModel):
    """
    --- REDUCED MODEL 2 ---
    Same as 'PrismScoreEmbDiffSimpleModel'
    but with NO EMBEDDINGS used at all
    """

    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self.file_name = 'PrismScoreDeltasOnlyModel.pt'
        self.in_size = int(self.cz * (2 * self.diff_width + 1))
        hid_size1 = int(self.in_size / 2)
        hid_size2 = int(hid_size1 / 4)
        hid_size3 = int(hid_size2 / 4)
        self.mlp = nn.Sequential(  # 4-layers perceptron
            nn.Linear(self.in_size, hid_size1),  # dimension reducing
            nn.LeakyReLU(),
            nn.Linear(hid_size1, hid_size2),  # dimension reducing
            nn.LeakyReLU(),
            nn.Linear(hid_size2, hid_size3),  # dimension reducing
            nn.LeakyReLU(),
            nn.Linear(hid_size3, 2)  # score prediction
        )

    def forward(self, pid, pos, all_deltas_in, pos_enc, emb_sector, emb_diff_in, debug_data=None):
        all_deltas = self.enlarger(all_deltas_in)
        all_deltas = all_deltas + pos_enc
        # ---NO debug code for this model---
        res = self.flatten(all_deltas)
        res = self.mlp(res)
        return res


class PrismScoreDeltasEmbDiffModel(PrismScoreEmbDiffSimpleModel):
    """
    --- REDUCED MODEL 3 ---
    Same as 'PrismScoreEmbDiffSimpleModel'
    but with emb diff and deltas only, NO seq emb
    """

    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self.file_name = 'PrismScoreDeltasEmbDiffModel.pt'
        self.in_size = int(self.cz * (2 * self.diff_width + 1) * 2)
        hid_size1 = int(self.in_size / 2)
        hid_size2 = int(hid_size1 / 4)
        hid_size3 = int(hid_size2 / 4)
        self.mlp = nn.Sequential(  # 4-layers perceptron
            nn.Linear(self.in_size, hid_size1),  # dimension reducing
            nn.LeakyReLU(),
            nn.Linear(hid_size1, hid_size2),  # dimension reducing
            nn.LeakyReLU(),
            nn.Linear(hid_size2, hid_size3),  # dimension reducing
            nn.LeakyReLU(),
            nn.Linear(hid_size3, 2)  # score prediction
        )

    def forward(self, pid, pos, all_deltas_in, pos_enc, emb_sector, emb_diff_in, debug_data=None):
        all_deltas = self.enlarger(all_deltas_in)
        emb_diff, emb_weights = self.emb_encoder(emb_diff_in)
        emb_diff = self.emb_diff_reducer(emb_diff)
        emb_diff = emb_diff + pos_enc
        x = self.flatten(all_deltas)
        y = self.flatten(emb_diff)
        res = torch.cat((x, y), 1)
        res = self.mlp(res)
        return res


class PrismScoreDeltasEmbModel(PrismScoreEmbDiffSimpleModel):
    """
    --- REDUCED MODEL 4 ---
    Same as 'PrismScoreEmbDiffSimpleModel'
    but with emb and deltas only, NO emb diff
    """

    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self.file_name = 'PrismScoreDeltasEmbModel.pt'
        self.in_size = int(self.cz * (2 * self.diff_width + 1) * 2)
        hid_size1 = int(self.in_size / 2)
        hid_size2 = int(hid_size1 / 4)
        hid_size3 = int(hid_size2 / 4)
        self.mlp = nn.Sequential(  # 4-layers perceptron
            nn.Linear(self.in_size, hid_size1),  # dimension reducing
            nn.LeakyReLU(),
            nn.Linear(hid_size1, hid_size2),  # dimension reducing
            nn.LeakyReLU(),
            nn.Linear(hid_size2, hid_size3),  # dimension reducing
            nn.LeakyReLU(),
            nn.Linear(hid_size3, 2)  # score prediction
        )

    def forward(self, pid, pos, all_deltas_in, pos_enc, emb_sector, emb_diff_in, debug_data=None):
        all_deltas = self.enlarger(all_deltas_in)
        emb_sector = self.seq_emb_reducer(emb_sector)
        emb_sector = emb_sector + pos_enc
        x = self.flatten(all_deltas)
        y = self.flatten(emb_sector)
        res = torch.cat((x, y), 1)
        res = self.mlp(res)
        return res


class PrismScoreNoDDGModel(PrismScoreEmbDiffSimpleModel):
    """
    --- REDUCED MODEL 5 ---
    Same as 'PrismScoreEmbDiffSimpleModel'
    but NO DDG data is used in prediction (DDE only)
    """

    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self.file_name = 'PrismScoreNoDDGModel.pt'

        self.enlarger = nn.Sequential(
            nn.Linear(20, 42),
            nn.LeakyReLU(),
            nn.Linear(42, self.cz)
        )

    def forward(self, pid, pos, all_deltas_in, pos_enc, emb_sector, emb_diff_in, debug_data=None):
        deltas_reduced = all_deltas_in[:, :, 20:]
        all_deltas = self.enlarger(deltas_reduced)

        emb_sector = self.seq_emb_reducer(emb_sector)
        emb_sector = emb_sector + pos_enc

        emb_diff, emb_weights = self.emb_encoder(emb_diff_in)
        emb_diff = self.emb_diff_reducer(emb_diff)

        x = self.flatten(all_deltas)
        y = self.flatten(emb_sector)
        z = self.flatten(emb_diff)
        res = torch.cat((x, y, z), 1)
        res = self.mlp(res)
        return res


class PrismScoreNoDDEModel(PrismScoreEmbDiffSimpleModel):
    """
    --- REDUCED MODEL 6 ---
    Same as 'PrismScoreEmbDiffSimpleModel'
    but NO DDE data is used in prediction (DDG only)
    """

    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self.file_name = 'PrismScoreNoDDEModel.pt'

        self.enlarger = nn.Sequential(
            nn.Linear(20, 42),
            nn.LeakyReLU(),
            nn.Linear(42, self.cz)
        )

    def forward(self, pid, pos, all_deltas_in, pos_enc, emb_sector, emb_diff_in, debug_data=None):
        deltas_reduced = all_deltas_in[:, :, :20]
        all_deltas = self.enlarger(deltas_reduced)

        emb_sector = self.seq_emb_reducer(emb_sector)
        emb_sector = emb_sector + pos_enc

        emb_diff, emb_weights = self.emb_encoder(emb_diff_in)
        emb_diff = self.emb_diff_reducer(emb_diff)

        x = self.flatten(all_deltas)
        y = self.flatten(emb_sector)
        z = self.flatten(emb_diff)
        res = torch.cat((x, y, z), 1)
        res = self.mlp(res)
        return res


class PrismScoreNoDeltasModel(PrismScoreEmbDiffSimpleModel):
    """
    --- REDUCED MODEL 7 ---
    Same as 'PrismScoreEmbDiffSimpleModel'
    but NO DDG and DDG data is used in prediction at all!
    """

    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self.file_name = 'PrismScoreNoDeltasModel.pt'
        self.in_size = int(self.cz * (2 * self.diff_width + 1) * 2)
        hid_size1 = int(self.in_size / 2)
        hid_size2 = int(hid_size1 / 4)
        hid_size3 = int(hid_size2 / 4)
        self.mlp = nn.Sequential(  # 4-layers perceptron
            nn.Linear(self.in_size, hid_size1),  # dimension reducing
            nn.LeakyReLU(),
            nn.Linear(hid_size1, hid_size2),  # dimension reducing
            nn.LeakyReLU(),
            nn.Linear(hid_size2, hid_size3),  # dimension reducing
            nn.LeakyReLU(),
            nn.Linear(hid_size3, 2)  # score prediction
        )

    def forward(self, pid, pos, all_deltas_in, pos_enc, emb_sector, emb_diff_in, debug_data=None):
        emb_sector = self.seq_emb_reducer(emb_sector)
        emb_sector = emb_sector + pos_enc

        emb_diff, emb_weights = self.emb_encoder(emb_diff_in)
        emb_diff = self.emb_diff_reducer(emb_diff)

        y = self.flatten(emb_sector)
        z = self.flatten(emb_diff)
        res = torch.cat((y, z), 1)
        res = self.mlp(res)
        return res


def plot_tensor(input_tensor, out_folder, prefix):
    input_tensor = input_tensor.cpu().detach().numpy()
    plt.imshow(input_tensor, cmap='hot', interpolation='nearest')
    title = f'{prefix}'
    plt.title(title)
    # plt.show()
    plot_path = os.path.join(out_folder, f'{prefix}.png')
    plt.colorbar(orientation='horizontal')
    plt.savefig(plot_path)
    plt.clf()


if __name__ == '__main__':
    # seq = torch.FloatTensor([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # res = get_position_encoding(seq, 128)
    pass

    # from data_model import ModelConfig
    # model_cfg = ModelConfig()
    # model_cfg.diff_width = 15
    # model_cfg.heads = 8
    # model_cfg.seq_emb_size = 768
    # model = PrismScoreEmbDiffModel(model_cfg)
    # print(model)
    # all_deltas = torch.rand(8, 31, 41)
    # src_dst = torch.rand(8, 32, 20)
    # emb_diff = torch.rand(8, 31, 768)
    # out = model(all_deltas, src_dst, emb_diff)
    # print(out.size())
