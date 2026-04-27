"""
=============================================================================
IMU SIGNAL GENERATION — Variational Autoencoder (VAE) + Expert Validation
=============================================================================

PROJECT CONTEXT
---------------
This script is part of the text_to_imu_generation project.
The goal is to generate realistic synthetic IMU signals (accelerometer) for
specific activities (walking, running, etc.) using a Variational Autoencoder.

The generated signals are validated using:
  1. Three independent classifiers ("judges") trained on real data
  2. Gait analysis (autocorrelation-based step detection)
  3. Spectrogram visual inspection (time-frequency representation)

DATASET
-------
The input dataset (unified_dataset.parquet) was built from multiple public
wearable activity recognition datasets:
    - RecoFit, PAMAP2, Opportunity++, Samosa, Wear, UT_Watch

Each was individually preprocessed and merged via the IMU_LM_Data pipeline:
    https://github.com/Abradshaw1/IMU_LM_Data

Key properties of the unified dataset:
  - Wrist-worn sensor only
  - 50 Hz sampling rate, FLU coordinate frame
  - Acceleration in m/s² (gravity included)
  - Harmonized activity labels via a global ontology (global_activity_id)

SIGNAL PREPROCESSING (done in this script)
-------------------------------------------
  - Downsampled 50 Hz → 25 Hz (iloc[::2]) to reduce model complexity
  - Windowing: 2-second windows (50 samples @ 25 Hz)
  - Stride: 25 samples (50% overlap) — intentional, used to increase dataset
    size since the original data was limited
  - Normalization: zero-mean, unit-variance per axis (fit on train set only)

AUTHORS
-------
Adrien, Théo Lassale — Stage Malmö
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras import layers, Model, callbacks
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split
from collections import Counter
import gc

# Keras 3 requires eager execution to avoid graph-mode issues with custom
# train_step / test_step overrides
tf.config.run_functions_eagerly(True)

# Fix seeds for reproducibility across numpy and tensorflow
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =============================================================================
# 1. DATA LOADING, CLEANING AND LABEL REMAPPING
# =============================================================================
gc.collect()

# Path to the unified parquet file produced by the IMU_LM_Data merge pipeline
path = "C:/Users/Théo Lassale/Desktop/Perso/Stage/Stage_Malmö/IMU_LM_Data/data/merged_dataset/unified_dataset.parquet"

# Load only the columns we need to save RAM
df = pd.read_parquet(path, columns=["acc_x", "acc_y", "acc_z", "global_activity_id",
                                     "dataset", "subject_id", "session_id"])

# Cast to float32 to halve memory usage vs float64
for col in ["acc_x", "acc_y", "acc_z"]:
    df[col] = df[col].astype(np.float32)

gc.collect()

# Remove outlier/ambiguous label IDs (global_activity_id >= 100 means
# "unknown_activity" or dataset-specific edge cases in the ontology)
df = df[df["global_activity_id"] < 100].copy()

# Remap original label IDs to consecutive integers starting at 0
# (required by sparse_categorical_crossentropy in the judges)
# Example: if the dataset has IDs [1, 3, 7], they become [0, 1, 2]
unique_labels = sorted(df["global_activity_id"].unique())
label_mapping = {old: new for new, old in enumerate(unique_labels)}
df["global_activity_id"] = df["global_activity_id"].map(label_mapping)

print(f"Number of activity classes detected: {len(unique_labels)}")

features = ["acc_x", "acc_y", "acc_z"]

# Downsample from 50 Hz to 25 Hz by keeping every other sample
# This halves model complexity while preserving all relevant gait frequencies
# (gait fundamentals are at 1.5–3 Hz, well below the 12.5 Hz Nyquist limit)
df = df.iloc[::2].copy()

# Interpolate any remaining NaN gaps (sensor dropouts), then fill edges
df[features] = df[features].interpolate().bfill().ffill()

# ── Windowing ────────────────────────────────────────────────────────────────
# WINDOW_SIZE = 50 samples = 2 seconds @ 25 Hz
# STRIDE      = 25 samples = 50% overlap between consecutive windows
# The overlap is intentional: it multiplies the number of training samples,
# which was necessary because the raw dataset was relatively small.
WINDOW_SIZE, STRIDE = 50, 25
X, labels = [], []

# Segment each recording session independently to avoid windows that span
# two different sessions or subjects
for _, group in df.groupby(["dataset", "subject_id", "session_id"]):
    sig = group[features].values
    acts = group["global_activity_id"].values
    for i in range(0, len(sig) - WINDOW_SIZE, STRIDE):
        X.append(sig[i:i + WINDOW_SIZE])
        # Assign the most frequent label within the window (majority vote)
        most_common = Counter(acts[i:i + WINDOW_SIZE]).most_common(1)[0][0]
        labels.append(most_common)

X = np.array(X, dtype=np.float32)   # shape: (n_windows, 50, 3)
labels = np.array(labels)

# Free the raw dataframe — no longer needed
del df
gc.collect()

# ── Train / Val / Test split (70 / 20 / 10) ──────────────────────────────────
X_temp, X_test, y_temp, y_test = train_test_split(X, labels, test_size=0.10, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2222, random_state=SEED)

del X_temp, y_temp
gc.collect()

# ── Normalization ─────────────────────────────────────────────────────────────
# Zero-mean, unit-variance normalization per axis.
# Statistics are computed on the TRAINING set only to avoid data leakage.
mean, std = np.mean(X_train, axis=(0, 1)), np.std(X_train, axis=(0, 1))
X_train = (X_train - mean) / (std + 1e-7)
X_val   = (X_val   - mean) / (std + 1e-7)
X_test  = (X_test  - mean) / (std + 1e-7)

# Model hyperparameters
timesteps   = 50   # window length in samples
n_features  = 3    # acc_x, acc_y, acc_z
latent_dim  = 32   # size of the VAE latent space z

# =============================================================================
# 2. VAE MODEL DEFINITION
# =============================================================================
#
# Architecture overview:
#
#   Input (50, 3)
#       │
#   ┌───▼──────────────────┐
#   │  ENCODER             │   Conv1D stack → Dense → (z_mean, z_log_var)
#   └───┬──────────────────┘
#       │  Reparameterization trick: z = z_mean + exp(0.5 * z_log_var) * ε
#   ┌───▼──────────────────┐
#   │  DECODER             │   Dense → Reshape → UpSampling + Conv1D → (50, 3)
#   └───┬──────────────────┘
#       │
#   Reconstructed signal (50, 3)
#
# Loss = Reconstruction loss (MSE) + β * KL divergence
# β = 0.01 (small β keeps the latent space useful for generation)
#
# The reparameterization trick makes the sampling step differentiable:
#   instead of sampling z ~ N(μ, σ²) directly (not differentiable),
#   we compute z = μ + σ * ε  where ε ~ N(0, 1)
#   This allows gradients to flow back through μ and σ.

def get_encoder():
    """
    Encoder: maps an input window (50, 3) to a distribution in latent space.
    Returns a Model with two outputs: z_mean and z_log_var (both shape: latent_dim).
    """
    inputs = layers.Input(shape=(timesteps, n_features))
    x = layers.Conv1D(32, 5, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)                                       # (25, 32)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)                                       # (12, 64) — note: floor(25/2)=12
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    z_mean    = layers.Dense(latent_dim)(x)   # no activation — unbounded output
    z_log_var = layers.Dense(latent_dim)(x)   # log(σ²) for numerical stability
    return Model(inputs, [z_mean, z_log_var])


def get_decoder():
    """
    Decoder: maps a latent vector z (latent_dim,) back to a signal window (50, 3).
    Mirror architecture of the encoder: Dense → Reshape → UpSampling + Conv1D.
    Cropping1D is used to fix the output length back to exactly 50 after upsampling.
    """
    inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(13 * 128, activation='relu')(inputs)
    x = layers.Reshape((13, 128))(x)          # match encoder bottleneck shape
    x = layers.UpSampling1D(2)(x)             # (26, 128)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)             # (52, 64)
    x = layers.Cropping1D(cropping=(1, 1))(x) # (50, 64) — trim back to 50 samples
    x = layers.Conv1D(32, 5, activation='relu', padding='same')(x)
    outputs = layers.Conv1D(n_features, 5, activation='linear', padding='same')(x)
    return Model(inputs, outputs)


class VAE(Model):
    """
    Variational Autoencoder with custom train_step and test_step.

    Custom steps are needed because:
      - The VAE loss combines reconstruction loss + KL divergence
      - Keras 3 requires explicit metric tracking via update_state/result()
      - We feed raw signals (not (input, target) pairs) so the default
        train_step would break — tf.data.Dataset.from_tensor_slices(X)
        passes only x, no y
    """

    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec
        # Keras 3: metrics must be declared here for .fit() to track them
        self.loss_tracker  = keras.metrics.Mean(name="loss")
        self.recon_tracker = keras.metrics.Mean(name="recon")

    @property
    def metrics(self):
        return [self.loss_tracker, self.recon_tracker]

    def _compute_loss(self, x, training=False):
        """
        Computes total VAE loss for a batch x.

        Reconstruction loss: mean MSE summed over timesteps
            → measures how well the decoder reconstructs the input

        KL divergence: -0.5 * mean(1 + log_var - mean² - exp(log_var))
            → regularizes the latent space toward N(0, 1)
            → ensures the latent space is smooth and continuous,
               which is what makes interpolation / sampling meaningful

        β = 0.01: small weight on KL to prioritize reconstruction quality
        """
        zm, zv = self.enc(x, training=training)
        # Reparameterization trick
        z = zm + tf.exp(0.5 * zv) * tf.random.normal(tf.shape(zm))
        recon = self.dec(z, training=training)

        r_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mse(x, recon), axis=-1))
        kl_loss = -0.5 * tf.reduce_mean(1 + zv - tf.square(zm) - tf.exp(zv))
        total = r_loss + 0.01 * kl_loss
        return total, r_loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            total, r_loss = self._compute_loss(data, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(total, self.trainable_weights), self.trainable_weights)
        )
        self.loss_tracker.update_state(total)
        self.recon_tracker.update_state(r_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        total, r_loss = self._compute_loss(data, training=False)
        self.loss_tracker.update_state(total)
        self.recon_tracker.update_state(r_loss)
        return {m.name: m.result() for m in self.metrics}


# Build and compile the VAE
vae_enc, vae_dec = get_encoder(), get_decoder()
vae_model = VAE(vae_enc, vae_dec)
vae_model.compile(optimizer='adam')

# Use tf.data.Dataset to feed raw signals (no labels needed for the VAE)
# prefetch() overlaps CPU data prep with GPU training for speed
train_ds = tf.data.Dataset.from_tensor_slices(X_train).batch(256).prefetch(tf.data.AUTOTUNE)
val_ds   = tf.data.Dataset.from_tensor_slices(X_val).batch(256).prefetch(tf.data.AUTOTUNE)

early_stop_vae = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

print("\n--- Training VAE ---")
vae_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[early_stop_vae],
    verbose=1
)

# =============================================================================
# 3. THE 3 JUDGES (CLASSIFIERS FOR EXPERT VALIDATION)
# =============================================================================
#
# "Train on real, evaluate on generated"
#
# The idea: train 3 diverse classifiers on real IMU data, then feed them
# the VAE-generated signals and check if they still predict the correct label.
# If all 3 agree on the right label with high confidence → the generated
# signal is considered realistic / credible.
#
# Using 3 different architectures (ConvLSTM, CNN, MLP) reduces the risk of
# one architecture being fooled by artifacts specific to that model.

def build_judges(n_classes):
    """
    Returns a list of 3 compiled Keras classifiers, all trained to predict
    activity labels from (50, 3) IMU windows.
    """
    i = layers.Input(shape=(50, 3))

    # --- Judge 1: DeepConvLSTM ---
    # Strong at capturing both local patterns (Conv) and temporal dynamics (LSTM)
    # Reference architecture from Ordóñez & Roggen (2016)
    x1 = layers.Conv1D(64, 5, activation='relu', padding='same')(i)
    x1 = layers.Conv1D(64, 5, activation='relu', padding='same')(x1)
    x1 = layers.LSTM(128, return_sequences=True)(x1)
    x1 = layers.LSTM(128)(x1)
    x1 = layers.Dense(128, activation='relu')(x1)
    x1 = layers.Dropout(0.3)(x1)
    m1 = Model(i, layers.Dense(n_classes, activation='softmax')(x1), name="DeepConvLSTM")

    # --- Judge 2: Simple CNN ---
    # Lightweight, good at detecting local frequency patterns
    x2 = layers.Conv1D(64, 3, activation='relu', padding='same')(i)
    x2 = layers.Conv1D(64, 3, activation='relu', padding='same')(x2)
    x2 = layers.GlobalAveragePooling1D()(x2)
    x2 = layers.Dense(128, activation='relu')(x2)
    x2 = layers.Dropout(0.3)(x2)
    m2 = Model(i, layers.Dense(n_classes, activation='softmax')(x2), name="CNN_Simple")

    # --- Judge 3: Simple MLP ---
    # Treats the window as a flat feature vector — simple baseline
    x3 = layers.Flatten()(i)
    x3 = layers.Dense(256, activation='relu')(x3)
    x3 = layers.Dropout(0.3)(x3)
    x3 = layers.Dense(128, activation='relu')(x3)
    x3 = layers.Dropout(0.3)(x3)
    m3 = Model(i, layers.Dense(n_classes, activation='softmax')(x3), name="MLP_Simple")

    return [m1, m2, m3]


judges = build_judges(len(unique_labels))

for m in judges:
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(f"\nTraining judge: {m.name}...")

    # Each judge gets its own EarlyStopping instance — they must be independent
    # (a shared callback would have shared internal state across judges)
    early_stop_judge = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        mode='max',
        restore_best_weights=True
    )

    m.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=256,
        callbacks=[early_stop_judge],
        verbose=1
    )

# =============================================================================
# 4. EXPERT VALIDATION: GAIT ANALYSIS + JUDGES + SPECTROGRAM
# =============================================================================

def analyse_gait(sig, fs=25):
    """
    Estimates step regularity from a raw IMU signal using autocorrelation.

    How it works:
      1. Compute the acceleration magnitude (norm of all 3 axes)
      2. Remove DC offset (mean subtraction) to center the signal
      3. Compute normalized autocorrelation
      4. Look for the first peak in the lag range [0.4s, 1.2s]
         (corresponds to realistic step cadence: ~50–150 steps/min)

    Returns:
      - score  : autocorrelation peak height [0, 1]
                 → 1.0 = perfectly periodic, 0.0 = no periodicity
                 → real walking typically scores > 0.5
      - step_t : estimated step period in seconds
                 → used to compute cadence (steps/min = 60 / step_t)
    """
    mag = np.sqrt(np.sum(np.square(sig), axis=1))
    mag = mag - np.mean(mag)
    corr = np.correlate(mag, mag, mode='full')[len(mag) - 1:]
    corr /= np.max(corr)
    min_l, max_l = int(0.4 * fs), int(1.2 * fs)
    if len(corr) > max_l:
        p_idx = np.argmax(corr[min_l:max_l]) + min_l
        return corr[p_idx], p_idx / fs
    return 0, 1e-7


def full_expert_validation(old_label_id, nom_activite):
    """
    Full validation pipeline for one activity class.

    Steps:
      1. Encode real training windows for this class → get z_mean distribution
      2. Sample new z vectors from that distribution (with slight extra variance
         to encourage diversity in the generated signals)
      3. Decode z → synthetic signal windows
      4. Concatenate windows into a ~1 minute signal
      5. Run gait analysis (autocorrelation)
      6. Compute spectrogram (time-frequency representation)
      7. Ask all 3 judges to classify the generated windows

    Parameters:
      old_label_id  : original global_activity_id BEFORE remapping
                      (use this to identify the activity in the original ontology)
      nom_activite  : human-readable activity name for plot titles
    """
    # Convert original label ID to the remapped (consecutive) ID used internally
    label_cible = label_mapping[old_label_id]
    indices = np.where(y_train == label_cible)[0]

    # Encode up to 200 real windows to estimate the latent distribution
    zm_label, _ = vae_enc.predict(X_train[indices[:200]], verbose=0)
    z_centre = np.mean(zm_label, axis=0)   # centroid of the latent cluster
    z_std    = np.std(zm_label, axis=0)    # spread of the latent cluster

    # Sample 30 new latent vectors from N(z_centre, 1.5 * z_std)
    # The 1.5 multiplier adds a bit more variance to cover intra-class diversity
    z_gen = np.random.normal(loc=z_centre, scale=z_std * 1.5, size=(30, latent_dim))

    # Decode to get 30 synthetic windows of shape (50, 3)
    gen_windows = vae_dec.predict(z_gen, verbose=0)

    # Flatten the 30 windows into a single continuous signal (~60 seconds @ 25 Hz)
    full_sig = gen_windows.reshape(-1, 3)

    # ── Gait analysis ────────────────────────────────────────────────────────
    score_gait, step_t = analyse_gait(full_sig)

    # ── Spectrogram (time-frequency representation) ───────────────────────────
    # Uses acc_x axis. nperseg=32 → frequency resolution of 25/32 ≈ 0.78 Hz
    # shading='gouraud' → smooth color interpolation between frequency bins
    f, t_spec, Sxx = signal.spectrogram(full_sig[:, 0], fs=25,
                                         nperseg=32, noverlap=16)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2)

    # Row 0: full generated signal (~60s)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(full_sig[:, 0], color='orange', lw=1)
    ax1.set_title(f"Generated ~1 min signal: {nom_activite}")
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("acc_x (normalized)")

    # Row 1 left: zoom on 5 seconds to check periodicity visually
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(full_sig[250:375, 0], marker='.', color='red', lw=1)
    ax2.set_title("5s zoom — periodicity check")
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("acc_x")

    # Row 1 right: spectrogram (X=time, Y=frequency, color=power in dB)
    ax3 = fig.add_subplot(gs[1, 1])
    im = ax3.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10),
                        shading='gouraud', cmap='magma')
    plt.colorbar(im, ax=ax3, label='Power (dB)')
    ax3.set_title("Spectrogram (acc_x)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Frequency (Hz)")

    # Row 2: judges verdicts text panel
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    # Reverse the label remapping to compare against the original ontology IDs
    inv_map = {v: k for k, v in label_mapping.items()}
    txt = "JUDGE VERDICTS:\n"
    for m in judges:
        p = m.predict(gen_windows, verbose=0)
        # Most frequent predicted class across all 30 generated windows
        c_new = Counter(np.argmax(p, axis=1)).most_common(1)[0][0]
        c_old = inv_map[c_new]
        confiance = np.mean(np.max(p, axis=1)) * 100
        verdict = '✅ CREDIBLE' if c_old == old_label_id else '❌ INCOHERENT'
        txt += f"- {m.name}: Predicted label {c_old} | Confidence {confiance:.1f}% | {verdict}\n"

    txt += f"\nGAIT SCORE: {score_gait:.2f}/1.0 | Cadence: {60 / step_t:.1f} steps/min"
    txt += "\n(Gait score > 0.5 indicates realistic periodicity for walking/running)"
    ax4.text(0.05, 0.1, txt, fontsize=12, family='monospace',
             bbox=dict(facecolor='white', alpha=0.8))

    plt.suptitle(f"Expert Validation — {nom_activite}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# Run full validation for walking (original global_activity_id = 1 before remapping)
# To validate other activities, call full_expert_validation with their original ID
# e.g. full_expert_validation(old_label_id=3, nom_activite="Running")
full_expert_validation(old_label_id=1, nom_activite="Walking")
