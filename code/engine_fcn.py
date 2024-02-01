"""
Model Fully Connected Network for baseline
"""
import torch
from torch import nn
from classifier import PrismScorePredictor
from data_model import ModelConfig
from utils import CFG


class FcnTableModel(nn.Module):
    """
    This model will be used to get overflow from look-up table
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.file_name = 'FcnTableModel.pt'
        model_size = 2048
        linear_size = int(model_size/4)
        inner_size = int(model_size/2)
        self.pidLinear = nn.Linear(1, linear_size)
        self.posLinear = nn.Linear(1, linear_size)
        self.aaLinear = nn.Linear(1, linear_size)
        self.classifier = nn.Sequential(
            nn.Linear(model_size, inner_size),
            # nn.Dropout(0.1),
            nn.ReLU(),
            nn.BatchNorm1d(inner_size),
            nn.Linear(inner_size, 256),
            # nn.Dropout(0.1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            # nn.Dropout(0.1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1)  # score
        )

    def forward(self, protein_id, position, src, dst):
        pid = self.pidLinear(protein_id)
        pos = self.posLinear(position)
        src = self.aaLinear(src)
        dst = self.aaLinear(dst)
        x = torch.cat((pid, pos, src, dst), 1)  # (bs, 1, 512)
        res = self.classifier(x)
        return res


class FcnModel(nn.Module):
    """
    This model uses only FCN and will be used a baseline
    """

    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.file_name = 'FcnModel.pt'
        self.emb_dim = model_cfg.seq_emb_size
        self.ddLinear = nn.Linear(2, self.emb_dim)
        self.aaLinear = nn.Linear(20, self.emb_dim)
        self.net = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim * 2),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(self.emb_dim * 2, self.emb_dim)
        )
        self.classifier = PrismScorePredictor(self.emb_dim * 4)

    def forward(self, dd, src, dst, attn_ndxs, seq_emb):
        seq_emb = self.net(seq_emb)  # (bs, r, 768)
        seq_emb = torch.mean(seq_emb, dim=1)  # (bs, 768)
        dd = self.ddLinear(dd)
        src = self.aaLinear(src)
        dst = self.aaLinear(dst)
        res = torch.stack((seq_emb, dd, src, dst), 1)  # (bs, 4, 768)
        res = self.classifier(res)
        return res


class FcnModelNoDDG(FcnModel):
    """
    This model based on FCN but without ddG
    """

    def __init__(self, model_cfg: ModelConfig):
        super().__init__(model_cfg)
        self.file_name = 'FcnModelNoDDG.pt'
        self.ddLinear = nn.Linear(1, self.emb_dim)
        self.classifier = PrismScorePredictor(self.emb_dim * 4)

    def forward(self, dd, src, dst, attn_ndxs, seq_emb):
        seq_emb = self.net(seq_emb)  # (bs, r, 768)
        seq_emb = torch.mean(seq_emb, dim=1)  # (bs, 768)
        ddE = dd[:, 0]  # get only ddE, remove ddG: (bs,2) -> (bs)
        ddE = ddE.unsqueeze(1)  # (bs, 1)
        dd = self.ddLinear(ddE)
        src = self.aaLinear(src)
        dst = self.aaLinear(dst)
        res = torch.stack((seq_emb, dd, src, dst), 1)  # (bs, 4, 768)
        res = self.classifier(res)
        return res


if __name__ == '__main__':
    pass
