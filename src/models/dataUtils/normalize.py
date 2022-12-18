import pandas as pd
import numpy as np
import torch

def normalize(dataframe):
    # reading data and converting to pandas DataFrame
    dataframe = pd.DataFrame(dataframe)
    # getting header for later use
    names = dataframe.columns.values

    # converting data to (numpy -> torch.tensor)
    n_data = dataframe.to_numpy()
    t_data = torch.tensor(n_data)

    # calculating data mean and variance
    data_mean = torch.mean(t_data, dim=0)
    data_var = torch.var(t_data, dim=0)

    # normalizing data
    data_norm = (t_data - data_mean) / torch.sqrt(data_var)

    # converting data to pandas DataFrame
    data_norm = pd.DataFrame(data_norm, columns=names)

    return data_norm

