#%%
import pandas as pd
import numpy as np
import pickle

#%%
def save_pickle(filename, dataframe):
    with open(filename, 'wb') as file:
        pickle.dump(dataframe, file)

#%%
filepath_output = '/Users/lorenzo/Desktop/PPG/db_ppg_peaks_filtered.pickle'

with open('/Users/lorenzo/Desktop/PPG/db_ppg_peaks.pickle', 'rb') as file:
    df = pickle.load(file)

# %%
quantile1 = np.quantile(df['age'], 0.33)
quantile2 = np.quantile(df['age'], 0.66)
# %%
df = df[ (df['age'] < quantile1) | (df['age'] > quantile2)]
# %%
save_pickle(filepath_output, df)
# %%
