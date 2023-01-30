import pandas as pd

from dataUtils import impute, normalize

df = pd.read_csv('data/SIDI_Full.csv')
df = impute.impute(df, cols='DI')
df = pd.get_dummies(df, columns=['LD'])
df = normalize.normalize(df)

# ---
pp_train = df.to_csv('data/pp_full.csv')
