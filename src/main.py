# * ---------- Imports ---------- * #
import wandb
import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
import torchmetrics as tm
from torch.utils.data import DataLoader, SubsetRandomSampler

from sklearn.model_selection import KFold
from tqdm import tqdm

print(f'PyTorch version: {torch.__version__}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'PyTorch is using {device}')

# Personal libraries
from dataUtils import impute, normalize
import functions as fun

# * ---------- Code ---------- * #
# TODO: Data =>
# ///TODO:     - data imputation
# ///TODO:     - data OneHotEncoding
# ///TODO:     - data normalization
# TODO: Design =>
# ///TODO:     - K-Fold for training
# TODO:     - Establish oneFunction benchmarks
# ///TODO:     - Low capacity network

# * ---------- Project Parameters ---------- * #

TRAIN_SET = 'data/train.csv'
SEED = 1024
VAL_SPLIT = 0.2
MODEL_NAME = 'DNN_seqNet_Sweeps'

# wandb init
config = {
    "learning_rate": 0.0003,
    "epochs": 640,
    "batch_size": 32,
    "neurons": 64,
    "dropout": 0.2,
    "kfolds": 5,
    "group": 1,
}

BATCH_SIZE = config['batch_size']
NEURONS = config['neurons']
LR = config['learning_rate']
EPOCHS = config['epochs']
DROPOUT = config['dropout']
FOLDS = config['kfolds']

# * ---------- Data (pre)Processing ---------- * #
df = pd.read_csv(TRAIN_SET, index_col=[0])
df = impute.impute(df, cols='DI')
df = pd.get_dummies(df, columns=['LD'])
df = normalize.normalize(df)

# seperating features and targets
targets = df.pop('DI').to_numpy(dtype='float32').reshape(-1, 1)
features = df.to_numpy(dtype='float32')


# * ---------- Initialization ---------- * #
dataset = fun.myDataset(features, targets)

model = fun.DNN_seqNet(7, 1, config['neurons'], config['dropout'])

criterion = nn.L1Loss()
optimizer_name = 'RMSprop'
optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=config['learning_rate'])


# * ---------- Cross Validation ---------- * #
train_losses = []
valid_losses = []
kf = KFold(n_splits=config['kfolds'], shuffle=True, random_state=SEED)
f_loop = enumerate(kf.split(dataset))
for fold, (train_ids, valid_ids) in f_loop:
    print(f'\n---------- * ----------')
    print(f'FOLD {fold}')

    MODEL_NAME = MODEL_NAME
    leaf = wandb.init(group=f'{MODEL_NAME}',
                        name=f'fold_{fold}',
                        job_type="training",
                        project="my-fold-project",
                        config=config)

    # Sample elements randomly from a given list of ids, no replacement
    train_sampler = SubsetRandomSampler(train_ids)
    valid_sampler = SubsetRandomSampler(valid_ids)

    # Define data loaders for training and testing data in this fold
    train_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler
    )
    valid_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        sampler=valid_sampler
    )

    # init neural network
    model.apply(fun.reset_weights)

    # training loop
    loop = tqdm(range(config['epochs']))
    for epoch in loop:
        # Starting Training...
        train_loss = fun.fitTraining(model, train_loader, criterion, optimizer)

        # Starting Validation...
        with torch.no_grad():
            valid_loss = fun.fitValidate(model, valid_loader, criterion)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # wandb loggers
        wandb.log({
            "train loss": train_loss,
            "valid loss": valid_loss,
            "epoch": epoch
            })

        # update progress bar
        loop.set_description(f'Current epoch: {epoch}')
        loop.set_postfix(train_loss = train_loss, valid_loss = valid_loss)

    print('Training process has finished. Saving trained model...')
    # saving model
    save_path = f'models/model_fold#{fold}.pth'
    torch.save(model.state_dict(), save_path)

    # Print results
    print(f'Training loss for fold-{fold}: {train_loss}')
    print(f'Validation loss for fold-{fold}: {valid_loss}')

    # Close run for this fold
    leaf.finish()

# Print fold results
print(f'K-FOLD CROSS VALIDATION RESULTS')
print('--------------------------------')
mean_train_loss = fun.meanie(train_losses)
mean_valid_loss = fun.meanie(valid_losses)
print(f'mean_train_loss: {mean_train_loss}')
print(f'mean_valid_loss: {mean_valid_loss}')