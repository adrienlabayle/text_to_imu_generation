import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.utils import shuffle

# ---------------------------
# 1. Chargement et Preprocessing
# ---------------------------
path = "D:/IMDS/MALMO/IMU_LM_Data/data/merged_dataset/filtered_activities_dataset.parquet"
df = pd.read_parquet(path)

sensor_cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

# Nettoyage des données numériques
df[sensor_cols] = df[sensor_cols].interpolate(method='linear')
df = df.bfill().ffill()

# ---------------------------
# 2. Création des fenêtres (2 secondes à 50Hz)
# ---------------------------
WINDOW_SIZE = 100
STRIDE = 50
features = sensor_cols

X, labels = [], []
group_cols = ["dataset", "subject_id", "session_id"]

for _, group in df.groupby(group_cols):
    data = group.sort_values("timestamp_ns")
    signals = data[features].values
    acts = data["global_activity_id"].values

    for i in range(0, len(signals) - WINDOW_SIZE, STRIDE):
        window = signals[i:i + WINDOW_SIZE]
        label = np.bincount(acts[i:i + WINDOW_SIZE]).argmax()
        X.append(window)
        labels.append(label)

X = np.array(X, dtype=np.float32)
labels = np.array(labels)
X, labels = shuffle(X, labels, random_state=42)

# Normalisation
X_mean = np.mean(X, axis=(0, 1))
X_std = np.std(X, axis=(0, 1))
X = (X - X_mean) / (X_std + 1e-7)

# ---------------------------
# 3. Paramètres
# ---------------------------
timesteps = X.shape[1]
n_features = X.shape[2]
latent_dim = 32

# ---------------------------
# 4. Encoder (Amélioré avec Flatten)
# ---------------------------
encoder_inputs = layers.Input(shape=(timesteps, n_features))
x = layers.Conv1D(64, 5, activation='relu', padding='same')(encoder_inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(2)(x)  # 100 -> 50

x = layers.Conv1D(128, 5, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(2)(x)  # 50 -> 25

x = layers.Flatten()(x)  # Conserve plus de détails que GlobalAveragePooling
x = layers.Dense(256, activation='relu')(x)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])
encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# ---------------------------
# 5. Decoder (Symétrique à l'Encoder)
# ---------------------------
latent_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(25 * 128, activation='relu')(latent_inputs)
x = layers.Reshape((25, 128))(x)

x = layers.UpSampling1D(2)(x)  # 25 -> 50
x = layers.Conv1D(128, 5, activation='relu', padding='same')(x)

x = layers.UpSampling1D(2)(x)  # 50 -> 100
x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)

decoder_outputs = layers.Conv1D(n_features, 5, activation='linear', padding='same')(x)
decoder = Model(latent_inputs, decoder_outputs, name="decoder")


# ---------------------------
# 6. Classe VAE avec perte KL pondérée (Beta-VAE)
# ---------------------------
class VAE(Model):
    def __init__(self, encoder, decoder, beta=0.01, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta  # Facteur de pondération pour la perte KL
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            # Somme des erreurs pour donner plus de poids à la reconstruction
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mse(data, reconstruction), axis=-1))
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}


# ---------------------------
# 7. Entraînement
# ---------------------------
vae = VAE(encoder, decoder, beta=0.01)
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

early_stop = tf.keras.callbacks.EarlyStopping(monitor='recon_loss', patience=10, restore_best_weights=True, mode='min')

vae.fit(X, epochs=3, batch_size=256, callbacks=[early_stop])  # Augmenté à 50 époques



# ---------------------------
# RECONSTRUCTION COURTE (2s)
# ---------------------------
i = 0
orig = X[i]

_, _, z = vae.encoder.predict(X[i:i+1])
recon = vae.decoder.predict(z)[0]

plt.figure(figsize=(10,5))

for j in range(n_features):
    plt.subplot(n_features, 1, j+1)
    plt.plot(orig[:, j], label="orig")
    plt.plot(recon[:, j], label="recon", alpha=0.7)

    if j == 0:
        plt.legend()

plt.suptitle("Reconstruction 2 secondes")
plt.tight_layout()
plt.show()



# ---------------------------
# 8. Visualisation (1 minute / 30 fenêtres)
# ---------------------------
n_windows = 30
segment_orig = X[:n_windows]
_, _, z_vals = vae.encoder.predict(segment_orig)
segment_recon = vae.decoder.predict(z_vals)

# On aplatit pour voir la minute en continu
full_orig = segment_orig[:, :, 0].flatten()
full_recon = segment_recon[:, :, 0].flatten()

plt.figure(figsize=(15, 5))
plt.plot(full_orig, label="Original acc_x", alpha=0.7)
plt.plot(full_recon, label="Reconstruit acc_x", linestyle='--')
plt.legend()
plt.title("Reconstruction sur 60 secondes (acc_x)")
plt.show()



# ---------------------------
# UMAP LATENT
# ---------------------------
z_mean, _, _ = vae.encoder.predict(X[:5000], batch_size=256)

reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(z_mean)

plt.figure(figsize=(8,6))
plt.scatter(
    embedding[:,0],
    embedding[:,1],
    c=labels[:5000],
    cmap='tab20',
    s=5
)
plt.colorbar()
plt.title("UMAP du latent space")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.show()