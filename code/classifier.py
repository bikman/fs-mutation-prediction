"""
Author: Michael Bikman
"""
import torch

from torch import nn


class Predictor(nn.Module):
    """
    Simple prediction module, uses 1 hidden layer and no batch normalisation
    """

    def __init__(self, in_size, hidden_size=512):
        """
        Score regression model from PRISM txt data
        @param in_size: input dimension
        @param hidden_size: inner dimension
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.in_size = in_size
        self.net = nn.Sequential(
            nn.Linear(self.in_size, hidden_size),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)  # score
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.net(x)
        return x


class PredictorLarge(nn.Module):
    """
    Larger prediction module, uses 2 hidden layer and no batch normalisation
    """

    def __init__(self, in_size):
        """
        Score regression model from PRISM txt data
        @param in_size: input dimension
        """
        super().__init__()
        self.in_size = in_size
        hidden_dim1 = int(in_size / 2)
        hidden_dim2 = int(hidden_dim1 / 2)
        hidden_dim3 = int(hidden_dim2 / 2)
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(self.in_size, hidden_dim1),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim3, 1)  # score
        )

    def forward(self, x):
        """
        @param x: size(x) = (1,768)
        @return:
        """
        x = self.flatten(x)
        x = self.net(x)
        return x


class ResidualPredictor(nn.Module):
    """
    MLP with 5 hidden layers and residual connection
    No dropout, no batch normalization
    """

    def __init__(self, in_size):
        super().__init__()
        self.in_size = in_size
        hid_size1 = int(self.in_size / 2)
        hid_size2 = int(hid_size1 / 4)
        hid_size3 = int(hid_size2 / 4)
        self.linear1 = nn.Linear(self.in_size, hid_size1)
        self.linear2 = nn.Linear(hid_size1, hid_size1)
        self.linear3 = nn.Linear(hid_size1, hid_size1)
        self.linear4 = nn.Linear(hid_size1, hid_size2)
        self.linear5 = nn.Linear(hid_size2, hid_size3)
        self.linear6 = nn.Linear(hid_size3, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        y = self.linear2(x)
        y = self.relu(y)
        y = self.linear3(y)
        z = x + y  # residual connection
        z = self.relu(z)
        z = self.linear4(z)
        z = self.relu(z)
        z = self.linear5(z)
        z = self.relu(z)
        z = self.linear6(z)
        return z


if __name__ == '__main__':
    pass
