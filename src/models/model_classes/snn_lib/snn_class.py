import torch.nn as nn
import math
from collections import OrderedDict


class SNN_model(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout, n_layers, use_selu: bool=True) -> None:
        super(SNN_model, self).__init__()
        layers = OrderedDict()
        for i in range(n_layers - 1):
            if i == 0:
                layers[f'fc_{i}'] = nn.Linear(d_in, d_hidden, bias=False)
            else:
                layers[f'fc_{i}'] = nn.Linear(d_hidden, d_hidden, bias=False)
            layers[f'selu_{i}'] = nn.SELU()
            layers[f'dropout_{i}'] = nn.AlphaDropout(p=dropout)
        layers[f'fc_{i + 1}'] = nn.Linear(d_hidden, d_out, bias=True)

        self.net = nn.Sequential(layers)

        if use_selu:
            for param in self.net.parameters():
                # biases zero
                if len(param.shape) == 1:
                    nn.init.constant_(param, 0)
                # others using lecun-normal initialization
                else:
                    nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        x = self.net(x)
        return x

'''
Credits:
https://github.com/bioinf-jku/SNNs
'''