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
# TODO:     - Establish benchmarks
# TODO:     - Low capacity network

# * ---------- Project Parameters ---------- * #

TRAIN_SET = 'data/train.csv'
SEED = 1024
VAL_SPLIT = 0.2
BATCH_SIZE = 16
NEURONS = 64
LR = 0.003
EPOCHS = 640
DROPOUT = 0.2
FOLDS = 5
MODEL_NAME = 'DNN_seqNet'

# * ---------- Data (pre)Processing ---------- * #
df = pd.read_csv(TRAIN_SET, index_col=[0])
df = impute.impute(df, cols='DI')
df = pd.get_dummies(df, columns=['LD'])
df = normalize.normalize(df)

# seperating features and targets
targets = df.pop('DI').to_numpy(dtype='float32').reshape(-1, 1)
features = df.to_numpy(dtype='float32')

# * ---------- Functions ---------- * #
class myDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


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


def fitTraining(dataloader, loss_fn, optimizer):
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


def fitValidate(dataloader, loss_fn):
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


# * ---------- Initialization ---------- * #
dataset = myDataset(features, targets)

model = DNN_seqNet(7, 1, NEURONS, DROPOUT)

criterion = nn.L1Loss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=LR)


# * ---------- Cross Validation ---------- * #

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

train_losses = []
valid_losses = []
f_loop = enumerate(kf.split(dataset))
for fold, (train_ids, valid_ids) in f_loop:
    print(f'\n---------- * ----------')
    print(f'FOLD {fold}')

    wandb.init(project="my-fold-project", group=f'{MODEL_NAME}', name=f'fold_{fold}', job_type="training")

    wandb.config = {
        "learning_rate": LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "neurons": NEURONS,
        "dropout": DROPOUT,
        "kfolds": FOLDS
    }

    # Sample elements randomly from a given list of ids, no replacement
    train_sampler = SubsetRandomSampler(train_ids)
    valid_sampler = SubsetRandomSampler(valid_ids)

    # Define data loaders for training and testing data in this fold
    train_loader = DataLoader(
        dataset,
        batch_size=10,
        sampler=train_sampler
    )
    valid_loader = DataLoader(
        dataset,
        batch_size=10,
        sampler=valid_sampler
    )

    # init neural network
    model.apply(reset_weights)

    # training loop
    loop = tqdm(range(EPOCHS))
    for epoch in loop:
        # Starting Training...
        train_loss = fitTraining(train_loader, criterion, optimizer)

        # Starting Validation...
        with torch.no_grad():
            valid_loss = fitValidate(valid_loader, criterion)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # wandb loggers
        wandb.log({
            "train loss": train_loss,
            "valid loss": valid_loss,
            "epoch": epoch
            })

        # update progress bar
        loop.set_description(f'Current epoch: {epoch}/{EPOCHS}')
        loop.set_postfix(train_loss = train_loss, valid_loss = valid_loss)

    print('Training process has finished. Saving trained model...')
    # saving model
    save_path = f'models/model_fold#{fold}.pth'
    torch.save(model.state_dict(), save_path)

    # Print results
    print(f'Training loss for fold-{fold}: {train_loss}')
    print(f'Validation loss for fold-{fold}: {valid_loss}')

    # Close run for this fold
    wandb.join()

# Print fold results
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {FOLDS} FOLDS')
print('--------------------------------')
mean_train_loss = meanie(train_losses)
mean_valid_loss = meanie(valid_losses)
print(f'mean_train_loss: {mean_train_loss}')
print(f'mean_valid_loss: {mean_valid_loss}')
