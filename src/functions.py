# * ---------- Imports ---------- * #
import wandb
import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
import torchmetrics as tm
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Personal libraries
from dataUtils import impute, normalize

# * ---------- Functions ---------- * #
class myDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


class 


class DNN_seqNet(nn.Module):
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


def fitTraining(model, dataloader, loss_fn, optimizer):
    train_loss = 0
    size = len(dataloader)
    model.train()
    for data, labels in dataloader:
        pred = model(data)

        loss = loss_fn(pred, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
    
    return train_loss / size


def fitValidate(model, dataloader, loss_fn):
    size = len(dataloader)
    valid_loss = 0
    model.eval()
    for data, labels in dataloader:
        pred = model(data)

        loss = loss_fn(pred, labels)

        valid_loss += loss.item()
    
    return valid_loss / size


def meanie(answer):
    mean = sum(answer) / len(answer)
    return mean


def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()
