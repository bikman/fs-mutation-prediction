"""
Module for datasets objects creation
"""
import logging
import random

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from embeddings import EsmEmbeddingFactory
from utils import AA_ALPHABETICAL, AA_DICT, get_embedding_sector, CFG, get_protein_files_dict, DIFF_LEN

LOG_ENABLED = True  # put True to enable logging

log = print
if LOG_ENABLED:
    # noinspection PyRedeclaration
    log = logging.info


def validate_aa(protein_data, v):
    """
    AA in the position of mutation must be the same as AA in a full sequence from PRISM file
    (on the relevant position)
    @param protein_data: ProteinData object
    @param v: variant
    @return: None or throws Exception
    """
    seq_from_prism = protein_data.chain.get_string()
    if v.position in protein_data.chain.distance_matrix.serial_to_index:
        aa_index = protein_data.chain.distance_matrix.serial_to_index[v.position]
        if len(seq_from_prism) > aa_index:
            if seq_from_prism[aa_index] == v.aa_from:
                return
    raise IndexError(f'AA mismatch at position {v.position}')


def get_pid(prism_data):
    tmp = [protein_id for protein_id, pname in get_protein_files_dict().items() if pname == prism_data.file_name]
    assert len(tmp) == 1
    pid = tmp[0]
    return pid


class PrismStackedDiffEmbDataset(Dataset):
    """
    Dataset class
    """

    def __init__(self, prism_data, seq_emb):
        self.data = []
        self.protein_name = prism_data.protein_name
        self.file_name = prism_data.file_name
        pid = get_pid(prism_data)
        for v in prism_data.variants:
            # --- filter or NOT-allowed mutations ---
            # allowed_mutations = AA_ALLOWED_MUTATIONS_DICT[v.aa_from]
            # if v.aa_to not in allowed_mutations:
            #     continue
            # ---------------------------------------

            data_item = self.create_data_item(v, prism_data, seq_emb, pid)
            self.data.append(data_item)

    @staticmethod
    def create_data_item(v, prism_data, seq_emb, pid):
        # some constants to use further
        # ------------------------------------------------
        step = DIFF_LEN
        width = 2 * step + 1
        num_of_aa = len(AA_ALPHABETICAL)
        cz = int(CFG['general']['cz'])
        # ------------------------------------------------

        start_pos = v.position - step
        end_pos = v.position + step + 1
        pos_variants = [v for v in prism_data.variants if start_pos <= v.position < end_pos]
        ddGs = np.zeros((width, num_of_aa))
        ddEs = np.zeros((width, num_of_aa))

        # generate positions array
        positions_list = [x for x in range(start_pos, end_pos)]
        if end_pos >= len(prism_data.sequence):
            positions_list = [x if x < len(prism_data.sequence) else 0 for x in positions_list]
        if start_pos < 0:
            positions_list = [x if x >= 0 else 0 for x in positions_list]
        assert len(positions_list) == width
        for x in positions_list:
            assert 0 <= x < len(prism_data.sequence), f'{x=},{len(prism_data.sequence)}'

        pos_enc = PrismStackedDiffEmbDataset._get_position_encoding(positions_list, cz)

        # fill ddG and ddE arrays with all possible deltas
        for curr_v in pos_variants:
            row_ndx = PrismStackedDiffEmbDataset._calc_row_ndx(v.position, curr_v.position, step)
            col_ndx = AA_DICT[curr_v.aa_to]
            assert 0 <= row_ndx < width
            assert 0 <= col_ndx < num_of_aa
            ddGs[row_ndx, col_ndx] = curr_v.ddG
            ddEs[row_ndx, col_ndx] = curr_v.ddE

        # create array of all deltas and position indices
        all_deltas = np.concatenate((ddGs, ddEs), axis=-1)
        assert all_deltas.shape == (width, 2 * num_of_aa)

        # take sector of seq_emb
        mutation_ndx = v.position - 1
        emb_sector = get_embedding_sector(seq_emb, mutation_ndx, step)
        assert emb_sector.size()[0] == 2 * step + 1
        assert emb_sector.size()[1] == EsmEmbeddingFactory.get_emb_dim()

        dst = AA_DICT[v.aa_to]
        src = AA_DICT[v.aa_from]
        data_item = [pid, v.position, all_deltas, pos_enc, emb_sector, v.emb_diff, v.score, v.bin,
                     v.score_orig, src, dst]
        return data_item

    @staticmethod
    def _get_position_encoding(pos_list, d, n=10000):
        """
        From:
        https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
        Calculate sin-cos positional encoding
        @param pos_list: list of positions
        @param d: dimension of the encoder
        @param n: constant normalization
        @return: positional encoding according to value of the position (not index)
        """
        pos_len = len(pos_list)
        pe = np.zeros((pos_len, d))
        for k in range(pos_len):
            for i in np.arange(int(d / 2)):
                denominator = np.power(n, 2 * i / d)
                value = pos_list[k]
                pe[k, 2 * i] = np.sin(value / denominator)
                pe[k, 2 * i + 1] = np.cos(value / denominator)
        return pe

    @staticmethod
    def _calc_row_ndx(v_position, curr_pos, step=15):
        pos_delta = curr_pos - v_position
        row_ndx = step + pos_delta
        return row_ndx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        return data_item

    def __repr__(self):
        return f'{self.protein_name}:{self.__class__}:size={self.__len__()}'


class PrismDiffEmbMultisetCreator(object):
    """
    Creates dataset of
    [protein_id | position | ddE | ddG | src | dst | diff emb | score]
    """

    def __init__(self, prism_data_list, pname_to_seq_emb):
        self.prism_data_list = prism_data_list
        self.name_to_seq_emb = pname_to_seq_emb

    def create_datasets(self):
        res = []
        for prism_data in tqdm(self.prism_data_list):
            seq_emb = self.name_to_seq_emb[prism_data.protein_name]
            ds = PrismStackedDiffEmbDataset(prism_data, seq_emb)
            res.append(ds)
        return res


class PrismDiffEmbFineTuneDataset(Dataset):
    """
    Dataset for FINE TUNING using PDB attention
    Used for a single protein
    """

    def __init__(self, prism_data, seq_emb, variants):
        """
        TBD
        """
        self.data = []
        self.protein_name = prism_data.protein_name
        pid = get_pid(prism_data)
        for v in variants:
            # --- filter or NOT-allowed mutations ---
            # allowed_mutations = AA_ALLOWED_MUTATIONS_DICT[v.aa_from]
            # if v.aa_to not in allowed_mutations:
            #     continue
            # ---------------------------------------

            data_item = PrismStackedDiffEmbDataset.create_data_item(v, prism_data, seq_emb, pid)
            self.data.append(data_item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        return data_item

    def __repr__(self):
        return f'{self.protein_name}:{self.__class__}:size={self.__len__()}'


class PrismDiffEmbFineTuneDatasetCreator(object):
    """
    Creates list of datasets with ONLY ONE dataset for FINE TUNING with single protein
    """

    def __init__(self):
        self.seq_emb = None
        self.prism_data_list = None
        self.name_to_seq_emb = None
        self.pdb_datas = None
        self.eval_data_type = None
        self.data_count = None
        self.destructive_data_only = None
        self.min_sampled_v = None  # minimal sampled variant
        self.max_sampled_v = None  # maximal sampled variant
        self.min_eval_v = None  # minimal evaluated variant
        self.max_eval_v = None  # maximal evaluated variant

    def create_datasets(self):
        """
        TBD
        @return:
        """
        res_train = []
        res_eval = []
        for prism_data in self.prism_data_list:
            seq_emb = self.name_to_seq_emb[prism_data.protein_name]
            # --- create dataset for fine tune training ---
            variants = prism_data.variants  # first take all variants (as default)
            # check if we need only destructive mutations with score close to zero
            if self.destructive_data_only == 1:
                # take only variants close to 0
                variants = [v for v in prism_data.variants if v.score < 0.3]  # TODO: 0.3 can be in CFG
                variants.sort(key=lambda x: x.score)

            if self.eval_data_type == 2:  # take the whole positions!
                positions = list(set([v.position for v in variants]))
                if len(positions) < self.data_count:
                    raise Exception(
                        f'Cannot create fine tune data: number of positions {len(positions)}<{self.data_count}')
                sampled_positions = random.sample(positions, self.data_count)  # choose needed amount of positions
                variants = [v for v in variants if v.position in sampled_positions]
            else:
                if len(variants) < self.data_count:
                    raise Exception(
                        f'Cannot create fine tune data: number of variants {len(variants)}<{self.data_count}')

                # -----------------------------------------------------
                #           enable min-max mechanism here
                # -----------------------------------------------------
                if self.min_eval_v is not None and self.max_eval_v is not None:
                    # find real variant for min_eval_v
                    min_v_candidates = [v for v in variants if
                                        v.position == self.min_eval_v.position
                                        and v.aa_from == self.min_eval_v.aa_from
                                        and v.aa_to == self.min_eval_v.aa_to]
                    assert (len(min_v_candidates) == 1)
                    real_min_v = min_v_candidates[0]
                    log(f'{str(real_min_v)}')

                    max_v_candidates = [v for v in variants if
                                        v.position == self.max_eval_v.position
                                        and v.aa_from == self.max_eval_v.aa_from
                                        and v.aa_to == self.max_eval_v.aa_to]
                    assert (len(max_v_candidates) == 1)
                    real_max_v = max_v_candidates[0]
                    log(f'{str(real_max_v)}')

                    variants = random.sample(variants, self.data_count)  # choose needed amount of mutations
                    log(f'sampled {len(variants)} variants for FT')

                    if real_min_v in variants and real_max_v in variants:
                        log('nothing to replace')
                        pass
                    elif real_min_v not in variants and real_max_v not in variants:
                        potential_victims = [v for v in variants if v != real_min_v and v != real_max_v]
                        potential_victims = potential_victims[2:]
                        variants = potential_victims + [real_min_v, real_max_v]
                        log('replaced both real_min_v and real_max_v')
                    elif real_min_v in variants or real_max_v in variants:
                        potential_victims = [v for v in variants if v != real_min_v and v != real_max_v]
                        potential_victims = potential_victims[1:]
                        variants = potential_victims + [real_min_v, real_max_v]
                        log('replaced one of real_min_v or real_max_v')
                    assert (len(variants) == self.data_count)
                    assert real_min_v in variants
                    assert real_max_v in variants
                    random.shuffle(variants)
                else:
                    variants = random.sample(variants, self.data_count)  # choose needed amount of mutations
                    log(f'sampled {len(variants)} variants for FT')

            # check here what min and max value were sampled for FT
            # variants - are sampled for FT
            self.min_sampled_v = min(variants, key=lambda v: v.score_orig)
            self.max_sampled_v = max(variants, key=lambda v: v.score_orig)

            train_ds = PrismDiffEmbFineTuneDataset(prism_data, seq_emb, variants)
            res_train.append(train_ds)
            # --- create dataset for evaluation ---
            # take all variants that are NOT in trains set
            eval_variants = [v for v in prism_data.variants if v not in variants]

            eval_ds = PrismDiffEmbFineTuneDataset(prism_data, seq_emb, eval_variants)
            res_eval.append(eval_ds)
        # return datasets for training and eval
        return res_train, res_eval


def calculate_bins(prism_data_list):
    num_bins = 10
    for prism_data in prism_data_list:
        list.sort(prism_data.variants, key=lambda x: x.score, reverse=False)
        chunked_list = np.array_split(prism_data.variants, num_bins)
        assert len(chunked_list) == num_bins
        bin_id = 0
        for chunk in chunked_list:
            for v in chunk:
                v.bin = bin_id
            bin_id += 1
        for v in prism_data.variants:
            assert v.bin is not None


if __name__ == '__main__':
    pass
