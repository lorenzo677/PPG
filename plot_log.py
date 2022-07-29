#%%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import config_context
import yaml

input_file = '/Users/lorenzo/Desktop/PPG/logging/training.log'
config_file = '/Users/lorenzo/Desktop/PPG/config.yaml'

with open(config_file, 'r') as file:
    dict = yaml.safe_load(file)

EPOCHS = dict['EPOCHS']
BATCH_SIZE = dict['BATCH_SIZE']
BETA = dict['BETA']
LEARNING_RATE = dict['LEARNING_RATE']

df = pd.read_csv(input_file)

plt.plot(df['epoch'], df['loss'], label = 'loss')
plt.plot(df['epoch'], df['val_loss'], label = 'val loss')

plt.legend()

output_file = f'/Users/lorenzo/Desktop/PPG/loss/{EPOCHS}_{BATCH_SIZE}_{BETA}_{LEARNING_RATE}.png'

#plt.savefig(output_file, dpi = 1024)

# %%
