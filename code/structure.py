"""
TBD
"""

import json
import math

import biotite.structure
from biotite.structure.io import pdbx, pdb
from biotite.structure.residues import get_residues
from biotite.structure import filter_backbone
from biotite.structure import get_chains
from biotite.sequence import ProteinSequence
import numpy as np
from scipy.spatial import transform
from scipy.stats import special_ortho_group
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from typing import Sequence, Tuple, List

import esm
import esm.inverse_folding

from esm.data import BatchConverter


class CoordBatchConverter(BatchConverter):

    def convert(self,  raw_batch, chain_id, device=None):
        self.alphabet.cls_idx = self.alphabet.get_idx("<cath>")
        batch = []
        for coords, confidence, seq in raw_batch:
            if confidence is None:
                confidence = 1.
            if isinstance(confidence, float) or isinstance(confidence, int):
                confidence = [float(confidence)] * len(coords)
            if seq is None:
                seq = 'X' * len(coords)
            batch.append(((coords, confidence), seq))

        # noinspection PyTypeChecker
        coords_and_confidence, strs, tokens = super().__call__(batch)

        # pad beginning and end of each protein due to legacy reasons
        coords = [
            F.pad(torch.tensor(cd[chain_id]), (0, 0, 0, 0, 1, 1), value=np.inf) for cd, _ in coords_and_confidence
        ]
        confidence = [
            F.pad(torch.tensor(cf), (1, 1), value=-1.) for _, cf in coords_and_confidence
        ]
        coords = self.collate_dense_tensors(coords, pad_v=np.nan)
        confidence = self.collate_dense_tensors(confidence, pad_v=-1.)
        if device is not None:
            coords = coords.to(device)
            confidence = confidence.to(device)
            tokens = tokens.to(device)
        padding_mask = torch.isnan(coords[:, :, 0, 0])
        coord_mask = torch.isfinite(coords.sum(-2).sum(-1))
        # confidence = confidence * coord_mask + (-1.) * padding_mask
        confidence = coord_mask + (-1.) * padding_mask
        return coords, confidence, strs, tokens, padding_mask





    @staticmethod
    def collate_dense_tensors(samples, pad_v):
        if len(samples) == 0:
            return torch.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
        max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
        result = torch.empty(
            len(samples), *max_shape, dtype=samples[0].dtype, device=device
        )
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]
            t = samples[i]
            result_i[tuple(slice(0, k) for k in t.shape)] = t
        return result


if __name__ == '__main__':
    # pdb_file = r'C:\DATASETS\ECOD_MAVE\pdbs\IF-1_1AH9_A.pdb'
    # pdb_file = r'C:\DATASETS\ECOD_MAVE\pdbs\BRCA1_1JM7_A.pdb'
    #bad
    # pdb_file = r'C:\DATASETS\ECOD_MAVE\pdbs\TPMT_2H11_A.pdb' #SAH
    # pdb_file = r'C:\DATASETS\ECOD_MAVE\pdbs\Src_2H8H_A.pdb' #PTS
    
    # pdb_file = r'C:\DATASETS\ECOD_MAVE\pdbs_fixed\Src_2H8H_A.pdb'
    pdb_file = r'C:\DATASETS\ECOD_MAVE\pdbs_fixed\TPMT_2H11_A.pdb'

    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()

    structure = esm.inverse_folding.util.load_structure(pdb_file)
    coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)

    target_chain_id = 'A'
    native_seq = native_seqs[target_chain_id]

    print('Native sequence loaded from structure file:')
    print(native_seq)
    print(len(native_seq))
    print('\n')

    batch_converter = CoordBatchConverter(alphabet)
    batch = [(coords, None, None)]
    # noinspection PyTypeChecker
    coords, confidence, strs, tokens, padding_mask = batch_converter.convert(batch, 'A', device='cpu')
    encoder_out = model.encoder.forward(coords, padding_mask, confidence, return_all_hiddens=False)
    # remove beginning and end (bos and eos tokens)
    res = encoder_out['encoder_out'][0][1:-1, 0]
    print(res.shape)
