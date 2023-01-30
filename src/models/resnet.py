from typing import Any, Dict

import numpy as np
import pandas as pd
import wandb
import rtdl
import scipy.special
import sklearn
import sklearn.model_selection
import torch
import torch.nn as nn
import torch.nn.functional as F
import zero

device = torch.device('cpu')
# Docs: https://yura52.github.io/delu/0.0.4/reference/api/zero.improve_reproducibility.html
zero.improve_reproducibility(seed=1024)
task_type = 'regression'

df = pd.read_csv('data/train.csv', index_col=0)

# wandb init
config = {
    # "learning_rate": 0.0003,
    "epochs": 1000,
    "batch_size": 32,
    # "neurons": 64,
    # "dropout": 0.2,
    "train_split": 0.8
}

assert task_type in ['binclass', 'multiclass', 'regression']

y_all = df['DI'].astype('float32' if task_type == 'regression' else 'int64').to_numpy()
X_all = df.drop('DI', axis=1).astype('float32').to_numpy()

X = {}
y = {}

X['train'], X['test'], y['train'], y['test'] = sklearn.model_selection.train_test_split(
    X_all, y_all, train_size=config['train_split']
)
X['train'], X['val'], y['train'], y['val'] = sklearn.model_selection.train_test_split(
    X['train'], y['train'], train_size=config['train_split']
)

# not the best way to preprocess features, but enough for the demonstration
preprocess = sklearn.preprocessing.StandardScaler()
preprocess.fit(X['train'])

X = {
    k: torch.tensor(preprocess.transform(v), device=device)
    for k, v in X.items()
}
y = {k: torch.tensor(v, device=device) for k, v in y.items()}


# !!! CRUCIAL for neural networks when solving regression problems !!!
if task_type == 'regression':
    y_mean = y['train'].mean().item()
    y_std = y['train'].std().item()
    y = {k: (v - y_mean) / y_std for k, v in y.items()}
else:
    y_std = y_mean = None

if task_type != 'multiclass':
    y = {k: v.float() for k, v in y.items()}


# model selection
d_out = 1

model = rtdl.ResNet.make_baseline(
    d_in=X_all.shape[1],
    d_main=128,
    d_hidden=256, 
    dropout_first=0.2,
    dropout_second=0.0,
    n_blocks=2,
    d_out=d_out,
)
lr = 0.001
weight_decay = 0.0

model.to(device)
optimizer = (
    model.make_default_optimizer()
    if isinstance(model, rtdl.FTTransformer)
    else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
)
loss_fn = (
    F.binary_cross_entropy_with_logits
    if task_type == 'binclass'
    else F.cross_entropy
    if task_type == 'multiclass'
    else F.mse_loss
)

def apply_model(x_num, x_cat=None):
    if isinstance(model, rtdl.FTTransformer):
        return model(x_num, x_cat)
    elif isinstance(model, (rtdl.MLP, rtdl.ResNet)):
        assert x_cat is None
        return model(x_num)
    else:
        raise NotImplementedError(
            f'Looks like you are using a custom model: {type(model)}.'
            ' Then you have to implement this branch first.'
        )


@torch.no_grad()
def evaluate(part):
    model.eval()
    prediction = []
    for batch in zero.iter_batches(X[part], 1024):
        prediction.append(apply_model(batch))
    prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
    target = y[part].cpu().numpy()

    if task_type == 'binclass':
        prediction = np.round(scipy.special.expit(prediction))
        score = sklearn.metrics.accuracy_score(target, prediction)
    elif task_type == 'multiclass':
        prediction = prediction.argmax(1)
        score = sklearn.metrics.accuracy_score(target, prediction)
    else:
        assert task_type == 'regression'
        score = sklearn.metrics.mean_squared_error(target, prediction) ** 0.5 * y_std
    return score


# Create a dataloader for batches of indices
# Docs: https://yura52.github.io/delu/reference/api/zero.data.IndexLoader.html
batch_size = 32
train_loader = zero.data.IndexLoader(len(X['train']), batch_size, device=device)

# Create a progress tracker for early stopping
# Docs: https://yura52.github.io/delu/reference/api/zero.ProgressTracker.html
progress = zero.ProgressTracker(patience=100)

print(f'Test score before training: {evaluate("test"):.4f}')


wandb.init(
    # group=f'impute&OHC&normalization',
    name=f'model_{model.__class__.__name__}',
    project='hvis_rtdl_baseline',
    config=config)

n_epochs = config['epochs']
report_frequency = len(X['train']) // batch_size // 5
val_scores = []
test_scores = []
for epoch in range(1, n_epochs + 1):
    for iteration, batch_idx in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        x_batch = X['train'][batch_idx]
        y_batch = y['train'][batch_idx]
        loss = loss_fn(apply_model(x_batch).squeeze(1), y_batch)
        loss.backward()
        optimizer.step()
        if iteration % report_frequency == 0:
            print(f'(epoch) {epoch} (batch) {iteration} (loss) {loss.item():.4f}')

    val_score = evaluate('val')
    test_score = evaluate('test')

    # val_scores.append(val_score)
    # test_scores.append(test_score)

    # wandb loggers
    wandb.log({
        # "train loss": test_score,
        "valid loss": val_score,
        "test loss": test_score,
        # "train rmse": train_rmse,
        # "valid rmse": valid_rmse,
        # "running seed": running_seed,
        "epoch": epoch,
        "batch_idx": batch_idx
        })

    print(f'Epoch {epoch:03d} | Validation score: {val_score:.4f} | Test score: {test_score:.4f}', end='')
    progress.update((-1 if task_type == 'regression' else 1) * val_score)
    if progress.success:
        print(' <<< BEST VALIDATION EPOCH', end='')
    print()
    if progress.fail:
        break

# Close run
wandb.finish()