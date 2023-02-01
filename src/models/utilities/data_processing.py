import pandas as pd
import sklearn.model_selection
import torch
from torch.utils.data import Dataset


class drdo_data(Dataset):
    def __init__(self, dataset, dep_variable, train_split, device) -> None:
        self.train_split = train_split
        self.device = device
        self.dataset = dataset
        self.dep_variable = dep_variable
        self.df = pd.read_csv(dataset)
        self.X = {}
        self.y = {}

        y_all = self.df[dep_variable].astype('float32').to_numpy()
        X_all = self.df.drop(dep_variable, axis=1).astype('float32').to_numpy()

        self.X['train'], self.X['test'], self.y['train'], self.y['test'] = sklearn.model_selection.train_test_split(
            X_all, y_all, train_size=train_split)

        self.X['train'], self.X['val'], self.y['train'], self.y['val'] = sklearn.model_selection.train_test_split(
            self.X['train'], self.y['train'], train_size=train_split)

        preprocess = sklearn.preprocessing.QuantileTransformer()
        preprocess.fit(self.X['train'])

        self.X = {k: torch.tensor(preprocess.transform(v), device=device) for k, v in self.X.items()}
        self.y = {k: torch.tensor(v, device=device) for k, v in self.y.items()}

        self.y_mean = self.y['train'].mean().item()
        self.y_std = self.y['train'].std().item()
        self.y = {k: (v - self.y_mean) / self.y_std for k, v in self.y.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.X['train'][index], self.y['train'][index]
