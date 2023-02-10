from utilities import drdo_data

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

df = drdo_data(
    dataset='data/SIDI_Full.csv',
    dep_variable='DI',
    train_split=0.8,
    device=device
)
# model selection
d_out = 1

model = rtdl.FTTransformer.make_default(
    n_num_features=df.X['train'].shape[1],
    cat_cardinalities=None,
    last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
    d_out=d_out,
)

model.to(device)
optimizer = (
    torch.optim.AdamW(model.parameters(),
    lr=config['learning_rate'],
    weight_decay=config['weight_decay'])
)
loss_fn = torch.nn.MSELoss()

@torch.no_grad()
def evaluate(part):
    model.eval()
    pred = model(df.X[part], x_cat=None).squeeze(1)
    target = df.y[part]
    score = loss_fn(pred, target)
    return score

# Create a dataloader for batches of indices
batch_size = config['batch_size']
train_loader = delu.data.make_index_dataloader(len(df.X['train']), config['batch_size'])

# Create a progress tracker for early stopping
progress = delu.ProgressTracker(config['patience'])
print(f'Test score before training: {evaluate("test"):.4f}')

# wandb init
wandb.init(
    name=f'model_{model.__class__.__name__}',
    project='hvis_rtdl_baseline',
    config=config)

n_epochs = config['epochs']
for epoch in range(1, n_epochs + 1):
    for iteration, batch_idx in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        x_batch = df.X['train'][batch_idx]
        y_batch = df.y['train'][batch_idx]
        loss = loss_fn(model(x_batch, x_cat=None).squeeze(1), y_batch)
        loss.backward()
        optimizer.step()

    val_score = evaluate('val')
    test_score = evaluate('test')

    valid_rmse = np.sqrt(val_score.cpu().numpy())
    test_rmse = np.sqrt(test_score.cpu().numpy())

    # wandb loggers
    wandb.log({
        "valid loss": val_score,
        "test loss": test_score,
        "valid rmse": valid_rmse,
        "test rmse": test_rmse,
        "epoch": epoch,
        "batch_idx": batch_idx
        })

    print(f'Epoch {epoch:03d} | Validation score: {val_score:.4f} | Test score: {test_score:.4f}', end='')
    progress.update((-1) * val_score)
    if progress.success:
        print(' <<< BEST VALIDATION EPOCH', end='')
    print()
    if progress.fail:
        break

# Close run
wandb.finish()