#%%
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from shutil import copy2

input_file = '/Users/lorenzo/Desktop/PPG/logging/training.log'
config_file = '/Users/lorenzo/Desktop/PPG/config.yaml'

with open(config_file, 'r') as file:
    dict = yaml.safe_load(file)
    
EPOCHS = dict['EPOCHS']
BATCH_SIZE = dict['BATCH_SIZE']
BETA = dict['BETA']
LEARNING_RATE = dict['LEARNING_RATE']

output_file = f'/Users/lorenzo/Desktop/PPG/loss/{EPOCHS}_{BATCH_SIZE}_{BETA}_{LEARNING_RATE}'

df = pd.read_csv(input_file)

copy2(input_file, output_file + '.txt')

plt.plot(df['epoch'], df['loss'], label = 'loss')
plt.plot(df['epoch'], df['val_loss'], label = 'val loss')
plt.legend()

plt.savefig(output_file + '.png', facecolor = 'white', dpi = 1024)

# %%
