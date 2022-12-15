# Imports
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

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


# Data (pre)Processing
dataset = 'data/SIDI_Full.csv'

df = impute.impute(dataset, cols='DI')
df = pd.get_dummies(df, columns=['LD'])
df = normalize.normalize(df)

print(df)