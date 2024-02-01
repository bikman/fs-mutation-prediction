from sys import platform

import esm
import torch

from utils import ESM_MODEL, ESM_REGRESSION, DEVICE


class EsmEmbedder(object):
    """
    https://github.com/facebookresearch/esm
    ESM2: EsmEmbedder('esm2_t33_650M_UR50D', model, regression, 33)
    ESM: EsmEmbedder('esm_msa1b_t12_100M_UR50S', model, regression, 12)
    """

    def __init__(self, model_name, path_to_model, path_to_regression, layers):
        self.device = DEVICE
        if platform != "linux" and platform != "linux2":
            self.device = 'cpu'
        self.layers = layers
        self.model_name = model_name
        self.model_data = torch.load(path_to_model, map_location=self.device)
        self.regression_data = torch.load(path_to_regression, map_location=self.device)
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_core(model_name, self.model_data,
                                                                                self.regression_data)
        self.model = self.model.to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()

    def embed(self, sequence):
        data = [("protein1", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.layers], return_contacts=True)
        token_representations = results["representations"][self.layers]
        if self.model_name.startswith('esm2'):
            embedding = token_representations.squeeze()[1:-1, :]
        else:
            embedding = token_representations.squeeze()[1:, :]
        return embedding


class EsmMsaEmbedding(object):
    """
    Uses general embedder to create sequence embedding
    https://github.com/facebookresearch/esm
    """

    def __init__(self):
        """
        Creates a model
        @param path_to_model: path to Pt file with model params
        @param path_to_regression: path to Pt file with regression params
        """
        self.embedder = EsmEmbedder('esm_msa1b_t12_100M_UR50S', ESM_MODEL, ESM_REGRESSION, 12)

    def embed(self, sequence):
        """
        Creates an embedding for given string
        @param sequence: string of residues of length Nres
        @return: embedding of size [Nres, 768]
        """
        return self.embedder.embed(sequence)


class EsmEmbeddingFactory(object):
    DIMENSION = -1

    @staticmethod
    def get_emb_dim():
        return 768

    @staticmethod
    def get_embedder():
        EsmEmbeddingFactory.DIMENSION = 768
        sequence_embedder = EsmMsaEmbedding()
        return sequence_embedder


if __name__ == '__main__':
    pass

    print('ESM1b')
    model = r'C:\MODELS\esm1b_msa\esm_msa1b_t12_100M_UR50S.pt'
    regression = r'C:\MODELS\esm1b_msa\esm_msa1b_t12_100M_UR50S-contact-regression.pt'
    embedder = EsmEmbedder('esm_msa1b_t12_100M_UR50S', model, regression, 12)
    seq = 'GTNPLHPNVVSNPVVRLYEQDALRMGKKEQFPYVGTTYRLTEHFHTWTKHALLNAIAQPEQFVEISETLAAAKGINNGDRVTVSSKRGFIRAVAVVTRRL'
    res = embedder.embed(seq)
    print(res.shape)
    print('ok')
