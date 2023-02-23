import torch.nn as nn
import math
from collections import OrderedDict


class SNN_model(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout, n_layers) -> None:
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
        self.reset_parameters()

    def forward(self, x):
        x = self.net(x)
        return x

    def reset_parameters(self):
        for layer in self.net:
            if not isinstance(layer, nn.Linear):
                continue
            nn.init.normal_(layer.weight, std=1 / math.sqrt(layer.out_features))
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(layer.bias, -bound, bound)

'''
Credits:
https://github.com/tonyduan/snn
'''