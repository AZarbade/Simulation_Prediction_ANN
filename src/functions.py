# Imports
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics as tm
from torch.utils.data import Dataset, DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tqdm import tqdm

# * ---------- Creating Functions ---------- * #
class Seq_01(nn.Module):
    def __init__(self, in_dims, out_dims, neurons, dropout) -> None:
        super().__init__()
        self.seq_01 = nn.modules.Sequential(
            nn.Linear(in_dims, neurons),
            nn.Tanh(),
            nn.Dropout(dropout),

            nn.Linear(neurons, neurons),
            nn.Tanh(),
            nn.Dropout(dropout),

            nn.Linear(neurons, neurons),
            nn.Tanh(),

            nn.Linear(neurons, out_dims),
        )
    
    def forward(self, features):
        features = self.seq_01(features)
        return features


def model_train(model, dataloader, loss_fn, optimizer):
    model.train()
    train_loss = 0
    train_score = 0

    for batch, (features, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        features = features.to(device)
        targets = targets.to(device)

        # compute prediction error
        pred = model(features)
        loss = loss_fn(pred, targets)
        # score = tm.functional.r2_score(pred, targets)

        # backpropagation
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # train_score = score.item()

        return train_loss


def model_val(model, dataloader, loss_fn):
    model.eval()
    test_loss = 0
    test_score = 0

    with torch.no_grad():
        for (features, targets) in dataloader:
            features = features.to(device)
            targets = targets.to(device)

            pred = model(features)
            test_loss += loss_fn(pred, targets).item()
            # score = tm.functional.r2_score(pred, targets).item()

            # test_score = score

            return test_loss


class build_dataset(Dataset):
    # Building Custom Dataset
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        current_feature = self.features[idx, :]
        current_label = self.labels[idx]

        features = torch.tensor(current_feature, dtype=torch.float32)
        labels = torch.tensor(current_label, dtype=torch.float32)

        return features, labels