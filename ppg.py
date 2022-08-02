#%%
import tensorflow as tf
import numpy as np
import keras.backend as K
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import datetime
import yaml

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

with open(config_file, 'r') as file:
    dict = yaml.safe_load(file)

EPOCHS = dict['EPOCHS']
BATCH_SIZE = dict['BATCH_SIZE']
BETA = dict['BETA']
LEARNING_RATE = float(dict['LEARNING_RATE'])

# set the dimensionality of the latent space to a plane for visualization later
LATENT_DIM = 2


# different_values_per_sample = np.prod(data.shape[1:])
# new_model = tf.keras.models.load_model(f'''/Users/lorenzo/Desktop/PPG/Cells_vae_best\
#                                         _{LATENT_DIM}D_BETA{BETA}.hdf5''',
#                                         custom_objects={'Sampling': Sampling})

# new_encoder = new_model.get_layer(index=1)
# new_decoder = new_model.get_layer(index=2)


# =============================================================================
# redict with the loaded model
# =============================================================================


# use = data  # select between, det (whole data), train, validation, or test
# m, lv, c = new_encoder.predict(use)  # means, logvariances, coordinates
# recon = new_decoder.predict(c)       # reconstruced sequences
# data = np.asarray(db).copy(order='C')

# fnn = FaissKNeighbors(k=5)
# ips = np.arange(len(data), dtype=int).copy(order='C')
# fnn.fit(data, ips)
# nns = fnn.predict(data)

#%%
with open('/Users/lorenzo/Desktop/PPG/db_ppg_peaks_filtered.pickle', 'rb') as file:
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
dilat_rates = [2, 4, 8, 16]
x = datetime.datetime.now()
#%%
# =============================================================================
# CALLBACKS
# =============================================================================

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                      verbose=1, patience=16)

checkp = tf.keras.callbacks.ModelCheckpoint(
                                            f'''/Users/lorenzo/Desktop/PPG/callbacks/PPG_vae_best_{LATENT_DIM}D_BETA{BETA}.hdf5''',
                                            monitor='val_loss', verbose=1, save_best_only=True,
                                            save_weights_only=False)

logger = tf.keras.callbacks.CSVLogger('/Users/lorenzo/Desktop/PPG/logging/training.log')
#{x.year}-{x.month}-{x.day}-{x.hour}-{x.minute}.log')


# =============================================================================
# WaveNet-like convolutional ENCODER
# =============================================================================

# NOTES: using "valid" convolution instead of "causal" to implement a wave
# architecture in both direction

#%%
inp = tf.keras.layers.Input(shape=train.shape[1:], name="Encoder_begin")  # use None for sequences of variable lentgh

wave = tf.keras.layers.Conv1D(filters=32, kernel_size=3,
                              padding='valid', dilation_rate=1,
                              activation="elu",
                              kernel_initializer="glorot_uniform")(inp)
waves_list = [wave]

for i in dilat_rates:
    wave = tf.keras.layers.Conv1D(filters=32, kernel_size=3,
                                  padding='valid', dilation_rate=i,
                                  activation="elu",
                                  kernel_initializer="glorot_uniform")(wave)
    waves_list += [wave]

waves = tf.keras.layers.concatenate(waves_list, axis=1)

waves = tf.keras.layers.Flatten()(waves)
latent_means = tf.keras.layers.Dense(LATENT_DIM, name="Encoder_end_means")(waves)
latent_logvars = tf.keras.layers.Dense(LATENT_DIM, kernel_initializer = 'zeros', name="Encoder_end_logvars")(waves)

latent_space = Sampling(name="Encoder_end_space")([latent_means, latent_logvars])


encoder = tf.keras.Model(inputs=[inp], outputs=[latent_means, latent_logvars, latent_space])

#%%
# =============================================================================
# Decoder
# =============================================================================


decoder_input = tf.keras.layers.Input(shape=[LATENT_DIM], name="Decoder_begin")
mid = tf.keras.layers.Dense(units=np.prod(train.shape[1:]), activation="elu")(decoder_input)
mid = tf.keras.layers.Reshape(target_shape=train.shape[1:])(mid)
mid = tf.keras.layers.Conv1DTranspose(filters=16, kernel_size=3,
                                      padding='same', activation='elu')(mid)

for i in dilat_rates[::-1]:
     mid = tf.keras.layers.Conv1DTranspose(filters=4, kernel_size=5, dilation_rate=1,
                                           padding='same', activation='elu')(mid)
# for i in dilat_rates[::-1]:
#     mid = tf.keras.layers.Conv1DTranspose(filters=16, kernel_size=3, dilation_rate=i,
#                                           padding='same', activation='elu')(mid)
mid = tf.keras.layers.Conv1DTranspose(filters=8, kernel_size=3,
                                      padding='same', activation='elu')(mid)

reconstruction = tf.keras.layers.Conv1DTranspose(
                                                filters=1,
                                                kernel_size=3, padding='same', name="Decoder_end")(mid)


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

history = wave_vae.fit(train, train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                       validation_data=(validation, validation),
                       callbacks=[checkp, logger, es])

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
    plt.savefig(f'/Users/lorenzo/Desktop/PPG/images/3D_0_vs_2_{plot_type}.png')
    plt.show()

def plot_clusters_2D(data, labels, plot_type):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(data)
    #plt.figure(figsize=(12, 10))
    plt.title(f'0 vs 1 {plot_type}')
    plt.scatter(z_mean[:, 0], z_mean[:, 1], s=2, alpha=0.7, c=labels, cmap='rainbow')
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(f'/Users/lorenzo/Desktop/PPG/images/2D_0_vs_1_{plot_type}.png')
    plt.show()
# %%
wave_vae.load_weights( f'/Users/lorenzo/Desktop/PPG/callbacks/PPG_vae_best_{LATENT_DIM}D_BETA{BETA}.hdf5')
# %%
if LATENT_DIM == 3:
    plot_clusters_3D(test, test_bpm_labels, 'bpm')
else:
    plot_clusters_2D(test, test_bpm_labels, 'bpm')

#%%
if LATENT_DIM == 3:
    plot_clusters_3D(test, test_age_labels, 'age')
else:
    plot_clusters_2D(test, test_age_labels, 'age')
# %%

quantile1 = np.quantile(test_age_labels, 1/3)
quantile2 = np.quantile(test_age_labels, 2/3)

idx =[]
test_age_labels = test_age_labels.reset_index(drop=True)
for i in range(len(test_age_labels)):
    if test_age_labels[i]<quantile1 or test_age_labels[i]>quantile2:
        idx.append(i)

new_test_data = []
new_test_label = []
for el in idx:
    new_test_data.append(test[el])
    new_test_label.append(test_age_labels[el])
new_test_data=np.array(new_test_data)
if LATENT_DIM == 3:
    plot_clusters_3D(new_test_data, new_test_label, 'age')
else:
    plot_clusters_2D(new_test_data, new_test_label, 'age')

# %%
