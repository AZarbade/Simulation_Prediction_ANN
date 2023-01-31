import numpy as np
import pandas as pd
import wandb
import rtdl
import sklearn
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn.functional as F
import delu
import delu.data

# settings
device = torch.device('cuda')
delu.improve_reproducibility(base_seed=1024)

# wandb config
config = {
    "learning_rate": 0.0003,
    "epochs": 1000,
    "batch_size": 32,
    "neurons": 64,
    "dropout": 0.2,
    "train_split": 0.8,
    "weight_decay": 0.0,
    "patience": 20
}

# data import and split
df = pd.read_csv('data/SIDI_Full.csv')

y_all = df['DI'].astype('float32').to_numpy()
X_all = df.drop('DI', axis=1).astype('float32').to_numpy()

X = {}
y = {}

X['train'], X['test'], y['train'], y['test'] = sklearn.model_selection.train_test_split(
    X_all, y_all, train_size=config['train_split']
)
X['train'], X['val'], y['train'], y['val'] = sklearn.model_selection.train_test_split(
    X['train'], y['train'], train_size=config['train_split']
)

# preprocess features (WIP)
preprocess = sklearn.preprocessing.QuantileTransformer()
preprocess.fit(X['train'])

X = {k: torch.tensor(preprocess.transform(v), device=device) for k, v in X.items()}
y = {k: torch.tensor(v, device=device) for k, v in y.items()}

y_mean = y['train'].mean().item()
y_std = y['train'].std().item()
y = {k: (v - y_mean) / y_std for k, v in y.items()}

# model selection
d_out = 1

import custom_models
from sklearn.ensemble import RandomForestRegressor
# model = custom_models.RandomForest(10, 30, 1)
model = RandomForestRegressor()

