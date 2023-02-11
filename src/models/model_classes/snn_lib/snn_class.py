import torch
import torch.nn as nn
from collections import OrderedDict

class SNN_model(nn.Module):
    def __init__(self, d_in, d_out, d_hidden, n_layers: int, dropout: float) -> None:
        super().__init__()
        layers = OrderedDict()
        for i in range(n_layers):
            if i == 0:
                layers[f'fc_{i}'] = nn.Linear(d_in, d_hidden)
            else:
                layers[f'fc_{i}'] = nn.Linear(d_hidden, d_hidden)
            
            layers[f'selu_{i}'] = nn.SELU()
            layers[f'dropout_{i}'] = nn.AlphaDropout(p=dropout)
        layers[f'fc_{i+1}'] = nn.Linear(d_hidden, d_out, bias=True)
        self.net = nn.Sequential(layers)

    def forward(self, x):
        x = self.net(x)
        return x
    
'''
Credits for the model:
    https://github.com/tonyduan/snn/tree/965e9f0719b2e6702d2e6a8461792a781b48ed46
'''