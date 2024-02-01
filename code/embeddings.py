import torch
import esm
from tape.models.modeling_bert import ProteinBertModel
from tape import TAPETokenizer
from torch import nn
from sys import platform

from utils import CFG, ESM_MODEL, ESM_REGRESSION, AMINO_TO_LETTER, ESM2_MODEL, ESM2_REGRESSION, DEVICE


class TapeEmbedding(object):
    """
    https://github.com/songlab-cal/tape
    """

    def __init__(self, path_to_pretrain):
        """
        path_to_pretrain: path to pretrain TAPE model folder
        """
        super().__init__()
        self.model = ProteinBertModel.from_pretrained(path_to_pretrain)
        self.tokenizer = TAPETokenizer(vocab='iupac')

    def embed(self, sequence):
        """
        Creates an embedding for given string
        @param sequence: string of amino-acids, i.e. 'GCTVEDR...AGALITQ'
        @return: embedding of size [Nres, 768]
        """
        tokens = self.tokenizer.encode(sequence)
        token_ids = torch.tensor([tokens])
        seq_output, pooled_output = self.model(token_ids)
        return seq_output[:, 1:-1]  # removing the first [CLS] and last [SEP] BERT indexes


class NativeEmbedding(object):
    """
    Uses the original PyTorch Embedding to encode the aa sequence
    """

    def __init__(self, emb_size):
        """
        Creates dict of amino-acids
        @param emb_size: size of the output embedding (per token)
        """
        aa = list(set(AMINO_TO_LETTER.values()))
        aa.append('X')
        self.aa_dict = {aa[i]: i + 1 for i in range(0, len(aa))}
        self.emb_size = emb_size

    def embed(self, sequence):
        """
        Creates an embedding for given string
        @param sequence: string of residues of length Nres
        @return: embedding of size [Nres, self.emb_size]
        """
        embedder = nn.Embedding(len(sequence), self.emb_size)
        tokens = []
        for c in sequence:
            tokens.append(self.aa_dict[c])
        tokens = torch.tensor(tokens, dtype=torch.int)
        res = embedder(tokens)
        return res


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


class Esm2Embedding(object):
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
        self.embedder = EsmEmbedder('esm2_t33_650M_UR50D', ESM2_MODEL, ESM2_REGRESSION, 33)

    def embed(self, sequence):
        """
        Creates an embedding for given string
        @param sequence: string of residues of length Nres
        @return: embedding of size [Nres, 1280]
        """
        return self.embedder.embed(sequence)


class EsmEmbeddingFactory(object):
    DIMENSION = -1

    @staticmethod
    def get_emb_dim(embedder_type):
        if embedder_type == 4:
            return 1280
        elif embedder_type == 1 or embedder_type == 2:
            return 768
        elif embedder_type == 3:
            return int(CFG['general']['seq_emb_size'])

    @staticmethod
    def get_embedder(embedder_type):
        if embedder_type == 1:
            EsmEmbeddingFactory.DIMENSION = 768
            sequence_embedder = EsmMsaEmbedding()
        elif embedder_type == 2:
            sequence_embedder = TapeEmbedding(TAPE_PRETRAINED)
        elif embedder_type == 3:
            sequence_embedder = NativeEmbedding(int(CFG['general']['seq_emb_size']))
        elif embedder_type == 4:
            EsmEmbeddingFactory.DIMENSION = 1280
            sequence_embedder = Esm2Embedding()
        else:
            raise Exception(f'Not supported embedder_type: {embedder_type}')
        return sequence_embedder


if __name__ == '__main__':
    pass
    # e = NativeEmbedding(128)
    # seq = 'GTNPLHPNVVSNPVVRLYEQDALRMGKKEQFPYVGTTYRLTEHFHTWTKHALLNAIAQPEQFVEISETLAAAKGINNGDRVTVSSKRGFIRAVAVVTRRL'
    # res = e.embed(seq)
    # print(len(seq))
    # print(res.size())
    # pass

    # seq = 'LRXEVKLGQGCFGEVWMGTWNGTTRVAIKTLKPGTMSPEAFLQEAQVMKKLRHEKLVQLYAVVSEEPIYIVTEYMSKGSLLDFLKGETGKYLRLPQLVDMAAQIASGMAYVERMNYVHRDLRAANILVGENLVCKVADFGLARLIEDNEYTARQGAKFPIKWTAPEAALYGRFTIKSDVWSFGILLTELTTKGRVPYPGMVNREXLDQVERGYRMPCPPECPEXLHDLMCQCWRKEPEERPTFEYLQXFL'
    # e = NativeEmbedding(128)
    # e = EsmMsaEmbedding(ESM_MODEL, ESM_REGRESSION)
    # res = e.embed(seq)
    # print(res.size())

    print('ESM2')
    model = r'C:\MODELS\esm_2\esm2_t33_650M_UR50D.pt'
    regression = r'C:\MODELS\esm_2\esm2_t33_650M_UR50D-contact-regression.pt'
    embedder = EsmEmbedder('esm2_t33_650M_UR50D', model, regression, 33)
    seq = 'GTNPLHPNVVSNPVVRLYEQDALRMGKKEQFPYVGTTYRLTEHFHTWTKHALLNAIAQPEQFVEISETLAAAKGINNGDRVTVSSKRGFIRAVAVVTRRL'
    res = embedder.embed(seq)
    print(res.shape)
    print('ok')

    print('ESM1b')
    model = r'C:\MODELS\esm1b_msa\esm_msa1b_t12_100M_UR50S.pt'
    regression = r'C:\MODELS\esm1b_msa\esm_msa1b_t12_100M_UR50S-contact-regression.pt'
    embedder = EsmEmbedder('esm_msa1b_t12_100M_UR50S', model, regression, 12)
    seq = 'GTNPLHPNVVSNPVVRLYEQDALRMGKKEQFPYVGTTYRLTEHFHTWTKHALLNAIAQPEQFVEISETLAAAKGINNGDRVTVSSKRGFIRAVAVVTRRL'
    res = embedder.embed(seq)
    print(res.shape)
    print('ok')

    # seq = 'GTNPLHPNVVSNPVVRLYEQDALRMGKKEQFPYVGTTYRLTEHFHTWTKHALLNAIAQPEQFVEISETLAAAKGINNGDRVTVSSKRGFIRAVAVVTRRL'
    # layers = 20
    # path_to_model = r'C:\MODELS\esm_if1\esm_if1_gvp4_t16_142M_UR50.pt'
    # model_data = torch.load(path_to_model, map_location='cpu')
    # model, alphabet = esm.pretrained.load_model_and_alphabet_core('esm_if1_gvp4_t16_142M_UR50', model_data, None)
    # model = model.eval()
    # print('ok')
