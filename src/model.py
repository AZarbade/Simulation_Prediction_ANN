# Imports
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics as tm
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tqdm import tqdm

print(f'PyTorch version: {torch.__version__}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'PyTorch is using {device}')

# Personal libraries
from dataUtils import impute, normalize

# --------------------

# TODO: Data =>
# ///TODO:     - data imputation
# ///TODO:     - data OneHotEncoding
# ///TODO:     - data normalization
# TODO: Design =>
# TODO:     - K-Fold for training
# TODO:     - Establish benchmarks
# TODO:     - Low capacity network

# PARAMETERS
SEED = 1024
TEST_SPLIT = 0.2
BATCH_SIZE = 16
TRAIN_SET = 'data/train.csv'
# TEST_SET = 'data/test.csv'
NEURONS = 64
LR = 0.003
EPOCHS = 100
DROPOUT = 0.2
FOLDS = 5

# Functions
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


class build_model(nn.Module):
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


def model_train(model, dataloader):
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
        score = tm.functional.r2_score(pred, targets)

        # backpropagation
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_score = score.item()

        return train_loss, train_score


def model_val(model, dataloader):
    model.eval()
    test_loss = 0
    test_score = 0

    with torch.no_grad():
        for (features, targets) in dataloader:
            features = features.to(device)
            targets = targets.to(device)

            pred = model(features)
            test_loss += loss_fn(pred, targets).item()
            score = tm.functional.r2_score(pred, targets).item()

            test_score = score

            return test_loss, test_score


# Data (pre)Processing
df = pd.read_csv(TRAIN_SET, index_col=[0])
df = impute.impute(df, cols='DI')
df = pd.get_dummies(df, columns=['LD'])
df = normalize.normalize(df)
targets = df.pop('DI')
targets = pd.DataFrame(targets)
features = df

print(f'feature shape: {features.shape}')
print(f'target shape: {targets.shape}')

x_trn, x_val, y_trn, y_val = train_test_split(
    features, targets,
    test_size=TEST_SPLIT,
    random_state=SEED
)

trn_dataset = build_dataset(
    features=x_trn,
    labels=y_trn
)
val_dataset = build_dataset(
    features=x_val,
    labels=y_val
)

# Initialized DataLoader
trn_loader = DataLoader(
    dataset=trn_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True,
)


model = build_model(
    in_dims=features.shape[1],
    out_dims=targets.shape[1],
    neurons=NEURONS,
    dropout=DROPOUT,
)
# loss and optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.RMSprop(
    model.parameters(),
    lr=LR,
)
model.to(device)
print(model)


# k fold training
kf = KFold(n_splits=FOLDS)
