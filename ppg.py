#%%
import tensorflow as tf
import numpy as np
import keras.backend as K
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import yaml
import os
from keras import initializers
#%%
# =============================================================================
# CUSTOM SAMPLING LAYER
# =============================================================================

class Sampling(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name)
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean
#%%
# =============================================================================
# Load pre-saved model
# =============================================================================

config_file = '/Users/lorenzo/Desktop/PPG/config.yaml'
best_model_path = '/Users/lorenzo/Desktop/PPG/callbacks/'

with open(config_file, 'r') as file:
    dict = yaml.safe_load(file)

EPOCHS = dict['EPOCHS']
BATCH_SIZE = dict['BATCH_SIZE']
BETA = dict['BETA']
LEARNING_RATE = float(dict['LEARNING_RATE'])

# set the dimensionality of the latent space to a plane for visualization later
LATENT_DIM = 2

#%%
with open('/Users/lorenzo/Desktop/PPG/db_ppg_one_peak_filtered.pickle', 'rb') as file:
    df = pickle.load(file)
#print(df)

age_labels = df['age']
bpm_labels = df['bpm']
data = np.asarray([ d/np.max(np.abs(d)) for d in df['signal']])

train, test, train_age_labels, test_age_labels, train_bpm_labels, test_bpm_labels = train_test_split(data, age_labels, bpm_labels,
                                                          test_size=0.20,
                                                          random_state=42)
validation, test, validation_age_labels, test_age_labels, validation_bpm_labels, test_bpm_labels = train_test_split(test, test_age_labels, test_bpm_labels, 
                                                          test_size=0.50,
                                                          random_state=42)                                                           
train = np.expand_dims(train, axis=-1)
validation = np.expand_dims(validation, axis=-1)
test = np.expand_dims(test, axis=-1)

# the data span is [-1, 1] with filler character '-' at -1
# 80% train, 10% validation, 10% test

#%%

# =============================================================================
# CONVOLUTIONAL NETWORK
# =============================================================================


opt = tf.keras.optimizers.Nadam(LEARNING_RATE)

different_values_per_sample = np.prod(data.shape[1:])
#dilat_rates = [2, 2, 4, 8, 16]
dilat_rates = [1]*11 #18

#%%
# =============================================================================
# CALLBACKS
# =============================================================================

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                      verbose=1, patience=16)

# checkp = tf.keras.callbacks.ModelCheckpoint(best_model_path + f'''PPG_best_{LATENT_DIM}D_BETA{BETA}mix_series_1peak.hdf5''',
#                                             monitor='val_loss', verbose=1, save_best_only=True,
#                                             save_weights_only=False)
checkp = tf.keras.callbacks.ModelCheckpoint(best_model_path + f'''LASTBEST_{LATENT_DIM}D_BETA_{BETA}_EPOCHS_50_series_1peak_tutto_elu_tranne1_random_normal.hdf5''',
                                            monitor='val_loss', verbose=1, save_best_only=True,
                                            save_weights_only=False)

logger = tf.keras.callbacks.CSVLogger('/Users/lorenzo/Desktop/PPG/logging/training.log')

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=50, min_lr=0.000001)


# =============================================================================
# WaveNet-like convolutional ENCODER
# =============================================================================

# NOTES: using "valid" convolution instead of "causal" to implement a wave
# architecture in both direction

#%%
inp = tf.keras.layers.Input(shape=train.shape[1:], name="Encoder_begin")  # use None for sequences of variable lentgh
k_reg = tf.keras.regularizers.l1_l2(l1=0., l2=0.)                
b_reg = tf.keras.regularizers.l1_l2(l1=0., l2=0.) 
wave = tf.keras.layers.Conv1D(filters=1, kernel_size=5,
                              padding='valid', 
                              #padding='same',
                              dilation_rate=1,
                              activation="elu",
                              #kernel_initializer="lecun_uniform",
                              #kernel_initializer="glorot_uniform",
                              kernel_initializer=initializers.random_normal(seed=42),
                              kernel_regularizer=k_reg,
                              bias_initializer='zeros',
                              bias_regularizer=b_reg)(inp)
# waves_list = [wave]

for i in dilat_rates:
    wave = tf.keras.layers.Conv1D(filters=1, kernel_size=5,
                                  padding='valid', 
                                  #padding='same',
                                  dilation_rate=i,
                                  activation="elu",
                                  #kernel_initializer="lecun_uniform",
                                  #kernel_initializer="glorot_uniform",
                                  kernel_initializer=initializers.random_normal(seed=42),
                                  kernel_regularizer=k_reg,
                                  bias_initializer='zeros',
                                  bias_regularizer=b_reg)(wave) #glorot_uniform
    # waves_list += [wave]
                             
# waves = tf.keras.layers.concatenate(waves_list, axis=1)
# waves = tf.keras.layers.add(waves_list)
wave = tf.keras.layers.Flatten()(wave)
latent_means = tf.keras.layers.Dense(LATENT_DIM, name="Encoder_end_means")(wave)
latent_logvars = tf.keras.layers.Dense(LATENT_DIM, kernel_initializer = 'zeros', name="Encoder_end_logvars")(wave)

latent_space = Sampling(name="Encoder_end_space")([latent_means, latent_logvars])


encoder = tf.keras.Model(inputs=[inp], outputs=[latent_means, latent_logvars, latent_space])

#%%
# =============================================================================
# Decoder
# =============================================================================


decoder_input = tf.keras.layers.Input(shape=[LATENT_DIM], name="Decoder_begin")
mid = tf.keras.layers.Dense(units=np.prod(train.shape[1:]), activation="relu")(decoder_input)
mid = tf.keras.layers.Reshape(target_shape=train.shape[1:])(mid)
mid = tf.keras.layers.Conv1DTranspose(filters=1, 
                                      kernel_size=5,
                                      padding='same',
                                      activation='elu')(mid)

for i in dilat_rates[::-1]:
     mid = tf.keras.layers.Conv1DTranspose(filters=1, 
                                           kernel_size=5, 
                                           dilation_rate=i,
                                           padding='same',
                                           activation='elu')(mid)
reconstruction = tf.keras.layers.Conv1DTranspose(
                                                filters=1,
                                                kernel_size=5,
                                                padding='same',
                                                name="Decoder_end")(mid)


decoder = tf.keras.Model(inputs=[decoder_input], outputs=[reconstruction])

#%%
# =============================================================================
# Wave-BETA-VAE
# =============================================================================

latent_m, latent_lv, latent_coordinates = encoder(inp)
out = decoder(latent_coordinates)
wave_vae = tf.keras.Model(inputs=[inp], outputs=[out])

# KL divergence assuming 2 Gaussians and using G1 = N(0, 1) and precomputed log_var
latent_loss = -0.5 * K.sum(1 + latent_lv - K.exp(latent_lv) -\
                           K.square(latent_m), axis=-1)

wave_vae.add_loss(K.mean(latent_loss)/different_values_per_sample*BETA)
wave_vae.compile(loss="MSE", optimizer=opt)  # adding loss for reconstruction error

#%%
name = 'LASTBEST_2D_BETA_0.2_EPOCHS_50_mix_0_EPOCHS_50_series_1peak_tutto_elu_tranne1_random_normal'
try:
    os.mkdir('/Users/lorenzo/Desktop/PPG/best_separation/'+ name)
except:
    pass 
name_save = '/Users/lorenzo/Desktop/PPG/best_separation/'+ name + '/' + name
wave_vae.load_weights(best_model_path + name + '.hdf5')
#%%
wave_vae.load_weights(best_model_path + 'LASTBEST_2D_BETA_0.2_EPOCHS_50_mix_0_EPOCHS_50_series_1peak_tutto_elu_tranne1_random_normal.hdf5')
#%%
history = wave_vae.fit(train, train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                       validation_data=(validation, validation),
                       callbacks=[logger, checkp])#, reduce_lr])
#%%
def plot_clusters_3D(data, labels, plot_type):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(data)
    #plt.figure(figsize=(12, 10))
    plt.title(f'0 vs 1 {plot_type}')
    plt.scatter(z_mean[:, 0], z_mean[:, 1], s=2, alpha=0.7, c=labels, cmap='rainbow')
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(f'/Users/lorenzo/Desktop/PPG/images/0_vs_1_{plot_type}.png')
    plt.show()
    plt.scatter(z_mean[:, 1], z_mean[:, 2], s=2, alpha=0.7, c=labels, cmap='rainbow')
    plt.colorbar()
    plt.xlabel("z[1]")
    plt.ylabel("z[2]")
    plt.title(f'1 vs 2 {plot_type}')
    plt.savefig(f'/Users/lorenzo/Desktop/PPG/images/1_vs_2_{plot_type}.png')
    plt.show()
    plt.scatter(z_mean[:, 0], z_mean[:, 2], s=2, alpha=0.7, c=labels, cmap='rainbow')
    plt.colorbar()
    plt.title(f'0 vs 2 {plot_type}')
    plt.xlabel("z[0]")
    plt.ylabel("z[2]")
    #plt.savefig(f'/Users/lorenzo/Desktop/PPG/images/3D_0_vs_2_{plot_type}.png')
    plt.show()

def plot_clusters_2D(data, labels, plot_type):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(data)
    #plt.figure(figsize=(12, 10))
    plt.title(f'0 vs 1 {plot_type}')
    plt.scatter(z_mean[:, 0], z_mean[:, 1], s=0.5, alpha=0.7, c=labels, cmap='rainbow', facecolor='w')
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig( '/Users/lorenzo/Desktop/validation_clusters.png', dpi=300, facecolor='w')
    plt.show()
import itertools
def plot_clusters_with_points_2D(data, labels, plot_type):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(data)
    #plt.figure(figsize=(12, 10))
    #plt.title(f'0 vs 1 {plot_type}')
    grid_x = np.linspace(-1.7, 1, 11)
    grid_y = np.linspace(1, -1, 11)
    plt.scatter(z_mean[:, 0], z_mean[:, 1], s=0.5, alpha=0.4, c=labels, cmap='rainbow', facecolor='w')
    plt.colorbar()
    for i in itertools.product(grid_x, grid_y):
        plt.scatter(i[0], i[1], s=2, alpha=1, c='black')
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig( '/Users/lorenzo/Desktop/validation_clusters_points.png', dpi=300, facecolor='w')
    plt.show()

def plot_young_old_2D(data, labels, threshold):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(data)
    #plt.figure(figsize=(12, 10))
    plt.title(f'young vs old')
    
    x = z_mean[:, 0]
    y = z_mean[:, 1]
    plt.scatter(x, y, s=0.7, alpha=0.7, c='red')
    above_threshold = [i for i, val in enumerate(labels) if val>threshold]
    x_above_threshold = x[above_threshold]
    y_above_threshold = y[above_threshold]
    plt.scatter(x_above_threshold, y_above_threshold, s=0.2, alpha=0.7, c='blue')
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.figsize=(10,12)
    plt.savefig('/Users/lorenzo/Desktop/validation_young_old.png', dpi=300, facecolor='w')
    plt.show()


def plot_separately(data, labels, threshold):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(data)
    x = z_mean[:, 0]
    y = z_mean[:, 1]
    below_threshold = [i for i, val in enumerate(labels) if val<threshold]
    x_below_threshold = x[below_threshold]
    y_below_threshold = y[below_threshold]
    above_threshold = [i for i, val in enumerate(labels) if val>=threshold]
    x_above_threshold = x[above_threshold]
    y_above_threshold = y[above_threshold]

    plt.figsize=(8,11)
    #ax1 = plt.subplot(121)
    plt.title(f'young')
    plt.scatter(x_below_threshold, y_below_threshold, s=0.7, alpha=0.7, c='blue')
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig('/Users/lorenzo/Desktop/validation_young.png', dpi=300, facecolor='w')
    plt.show()
    #ax2 = plt.subplot(122, sharey=ax1, sharex=ax1)
    plt.title(f'old')
    plt.scatter(x_above_threshold, y_above_threshold, s=.7, alpha=0.7, c='red')
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    
    plt.savefig('/Users/lorenzo/Desktop/validation_old.png', dpi=300, facecolor='w')
    plt.show()

def plots_3_separately(data, labels, threshold):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(data)
    x = z_mean[:, 0]
    y = z_mean[:, 1]
    below_threshold = [i for i, val in enumerate(labels) if val<threshold]
    x_below_threshold = x[below_threshold]
    y_below_threshold = y[below_threshold]
    above_threshold = [i for i, val in enumerate(labels) if val>=threshold]
    x_above_threshold = x[above_threshold]
    y_above_threshold = y[above_threshold]

    fig, ax = plt.subplots(1, 3, figsize=(20,7), sharey=True)
    ax[0].set_title('young', size=18)
    ax[1].set_title('young and old', size=18)
    ax[2].set_title('old', size=18)
    ax[0].set_xlabel("z[0]", size=18)
    ax[1].set_ylabel("z[1]", size=18)
    ax[2].set_xlabel("z[0]", size=18)
    ax[0].set_ylabel("z[1]", size=18)
    ax[1].set_xlabel("z[0]", size=18)
    ax[2].set_ylabel("z[1]", size=18)
    ax[0].scatter(x_below_threshold, y_below_threshold, s=.9, alpha=0.7, c='blue')
    ax[1].scatter(x_below_threshold, y_below_threshold, s=.9, alpha=0.5, c='blue', label='young')
    ax[1].scatter(x_above_threshold, y_above_threshold, s=.9, alpha=0.5, c='red', label='old')
    ax[2].scatter(x_above_threshold, y_above_threshold, s=.9, alpha=0.7, c='red')
    ax[1].legend()
    plt.savefig('/Users/lorenzo/Desktop/test_young_old.png', facecolor='w', dpi=300)
    plt.show()
plots_3_separately(test, test_age_labels, 50)

#%%
plot_separately(validation, validation_age_labels, 50)
plot_clusters_2D(validation, validation_age_labels, 'age')
plot_clusters_with_points_2D(validation, validation_age_labels, 'age') 
plot_young_old_2D(validation, validation_age_labels, 50)
#%%
if LATENT_DIM == 3:
    plot_clusters_3D(test, test_age_labels, 'age')
else:
    plot_clusters_2D(test, test_age_labels, 'age', BETA)                                     

#%%
plot_young_old_2D(test, test_age_labels, 50)
# %%
if LATENT_DIM == 3:
    plot_clusters_3D(test, test_bpm_labels, 'bpm')
else:
    plot_clusters_2D(test, test_bpm_labels, 'bpm')


#%%

def plot_latent_images(n=10):
    """Plots n x n digit images decoded from the latent space."""

    grid_x = np.linspace(-1.7, 1, n)
    grid_y = np.linspace(1, -1, n)
    fig, ax = plt.subplots(nrows=n, ncols=n, figsize=(12,12))
    c=0
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            c+=1
            z = np.array([[xi, yi]])
            x_decoded =decoder.predict(z)
            digit = x_decoded[0].reshape(64)
            ax[i][j].set_title(f'({round(xi,2)},{round(yi,2)})', fontsize=10)
            ax[i][j].axis('off')
            ax[i][j].set_facecolor('white')
            ax[i][j].plot(digit)
    plt.savefig('/Users/lorenzo/Desktop/_latent_space.png', dpi=300, facecolor='w')

plot_latent_images(11)

# %%
