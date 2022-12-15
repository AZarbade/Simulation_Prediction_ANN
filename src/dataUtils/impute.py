import pandas as pd
import numpy as np

def impute(dataframe, cols):
    # reading data
    df = dataframe

    # checking for "nulls" and "zeros"
    print(f'Checking NaNs:\n{df.isna().sum()}')
    # print(f'Checking zeros:\n   {df[cols]} has: {df[cols].value_counts()[0]} zeros')

    # replacing zeros with nulls for easier processing
    df[cols] = df[cols].replace({
        '0': np.nan, # string "0"
        0:np.nan # value "0"
    })

    # importing imputing library
    from sklearn.impute import SimpleImputer
    imp_median = SimpleImputer(
        missing_values = np.nan,
        strategy = 'median',
        add_indicator=True
    )

    # getting column names for later
    names = np.array(df.columns.values)
    meh = np.array(['Imputed'])
    new_names = np.append(names, meh)

    imp = imp_median.fit_transform(df)

    # creating imputed DataFrame and adding headers
    dataframe = pd.DataFrame(imp, columns=new_names)

    return dataframe

