import matplotlib.pyplot as plt
import pandas as pd

input_file = '/Users/lorenzo/Desktop/PPG/logging/training.log'

df = pd.read_csv(input_file)

plt.plot(df['epoch'], df['loss'], label = 'loss')
plt.plot(df['epoch'], df['val_loss'], label = 'val loss')

plt.legend()
plt.show()