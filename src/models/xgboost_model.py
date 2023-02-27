from utilities import drdo_data

import numpy as np
import pandas as pd
import wandb
from wandb.xgboost import wandb_callback
import xgboost as xgb
import sklearn
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn.functional as F
import delu
import delu.data

# settings
device = torch.device('cpu')
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

df = drdo_data(
    dataset='data/SIDI_Full.csv',
    dep_variable='DI',
    train_split=0.8,
    device=device
)

# Model Selection
model = xgb.XGBRegressor()
model.fit(df.X['train'], df.y['train'])


# wandb init
wandb.init(
    name=f'model_{model.__class__.__name__}',
    project='hvis_rtdl_baseline',
    # project='testing',
    # config=config
)

# Loss metric calc.
from sklearn.metrics import mean_squared_error
import math
def evaluate(part):
    pred = model.predict(df.X[part])
    score = mean_squared_error(df.y[part], pred)
    return score

val_score = evaluate('val')
test_score = evaluate('test')

valid_rmse = math.sqrt(val_score)
test_rmse = math.sqrt(test_score)

# wandb loggers
wandb.log({
    "valid loss": val_score,
    "test loss": test_score,
    "valid rmse": valid_rmse,
    "test rmse": test_rmse,
    # "epoch": epoch,
    # "batch_idx": batch_idx
    })

wandb.finish()
