# Synthetic IMU Data Generation — Text-to-IMU Pipeline

> Generative AI project for synthetic accelerometer signal generation, conditioned on physical activity labels, with an end-to-end text-to-IMU pipeline.

---

## Overview

This project explores the generation of **synthetic IMU (Inertial Measurement Unit) signals** using three generative deep learning architectures : a **Variational Autoencoder (VAE)**, a **Diffusion model**, and a **Generative Adversarial Network (GAN)**. The generated signals are conditioned on activity labels (rest, walk, run/jog, stairs) and evaluated through a multi-level validation framework.

The long-term vision is to enable **data augmentation for medical and health monitoring applications** — such as gait analysis, fall detection, and rehabilitation monitoring — where collecting labeled sensor data from specific populations is costly, constrained, or subject to privacy regulations.

---

## Project Structure

```
.
├── 01_train_and_save_diffusion.ipynb       # Diffusion model training + model saving
├── 02_train_and_save_prompt_classifier.ipynb # USE + Logistic Regression prompt classifier
├── 03_text_to_imu_pipeline.py              # Interactive text-to-IMU inference script
├── comparaison_vae_diffusion_gan.ipynb     # VAE / Diffusion / GAN training + evaluation
├── evaluation_juges.ipynb                  # Judge classifier evaluation + confusion matrices
├── prompting_part.ipynb                    # USE embedding + classifier experiments
├── training_phrases.csv                    # Labeled prompt dataset (20+ phrases/class)
├── saved_models/
│   ├── diffusion_model.pth                 # Trained diffusion model weights
│   ├── metadata.json                       # Normalization stats + model hyperparameters
│   ├── prompt_classifier.pkl               # Trained logistic regression classifier
│   └── prompt_label_mapping.json           # Label mapping for the prompt classifier
└── README.md
```

---

## Dataset

### Sources

The unified dataset was built by aggregating **eight publicly available IMU datasets** :

| Dataset | Description |
|---|---|
| **Capture24** | 24-hour wrist accelerometry, Oxford |
| **Opportunity++** | Daily life activities, wrist + body |
| **PAMAP2** | Physical activity monitoring |
| **RecoFit** | Gym exercise recognition |
| **UT-Watch** | Wrist-based activity recognition |
| **Wear** | Outdoor activity recognition |
| **WISDM** | Smartphone/smartwatch activity dataset |
| **Samosa** | Smartwatch activity and motion dataset |

### Unification Pipeline

All datasets were preprocessed through a unified pipeline ([IMU_LM_Data](https://github.com/Abradshaw1/IMU_LM_Data)) that standardizes :

- **Sampling rate** : all signals resampled to **50 Hz** using FIR anti-aliasing filtering
- **Coordinate frame** : unified **FLU (Forward–Left–Up)** orientation across all datasets
- **Sensor placement** : **wrist-only** sensor stream retained
- **Units** : acceleration in m/s² (gravity included), gyroscope in rad/s
- **Activity labels** : harmonized into a shared ontology (e.g. `nordic_walking` → `walk`)

The result is a unified Parquet file of approximately **730 million rows** at 50 Hz, covering 22 activity classes.

### Label Selection

After analyzing the class distribution, we selected **four activity classes** for our generative models, chosen for their data volume and biomechanical distinctiveness :

| ID | Label | Description |
|---|---|---|
| 0 | `rest_inactive` | Sedentary / resting state |
| 2 | `walk` | Normal walking |
| 3 | `run_jog` | Running / jogging |
| 4 | `stairs` | Stair climbing |

> **Note :** Early experiments used a 3-class setup (rest_inactive, walk, posture_stationary). The 4-class setup (including run_jog and stairs) was adopted later in the project after further dataset analysis. Both configurations appear in the results.

### Preprocessing

To address class imbalance, **session-level subsampling** was applied — keeping entire recording sessions rather than individual samples, to preserve the temporal continuity of signals :

- `rest_inactive` : 1/5 of sessions kept
- `walk` : 1/2 of sessions kept
- `run_jog`, `stairs` : all sessions kept

Each class was then normalized independently using a `StandardScaler` to avoid leakage between activity types.

---

## Generative Models

All three models are **conditionally trained** on the activity label, meaning the generation can be controlled by specifying the desired activity class.

### VAE — Variational Autoencoder

- Encoder : 4 Conv1D blocks with BatchNorm + ReLU → Dense → z_mean / z_log_var
- Decoder : Dense → 4 Conv1DTranspose blocks with BatchNorm + ReLU
- Latent dim : 64
- Training : β-VAE with **KL annealing** (β increases from 0 to 1 over 40 epochs)
- Input : magnitude of acceleration (1 channel, 50 timesteps)

### Diffusion Model

- Architecture : conditional U-Net-style Conv1D with time embedding + label embedding
- Noise schedule : linear β schedule, 200 diffusion steps
- Sampling : DDPM reverse process
- Input : 3-axis accelerometer signal (3 channels, 50 timesteps)

### GAN — Generative Adversarial Network

- Generator : latent vector + label embedding → ConvTranspose1D decoder
- Discriminator : signal + label embedding → real/fake score
- Training : **WGAN-GP** (Wasserstein loss + gradient penalty, λ=10, N_critic=5)
- Latent dim : 32

---

## Evaluation Framework

Generated signals are evaluated through a **three-level validation framework** :

### 1. Spectral Analysis (Welch PSD)

Power Spectral Density comparison between real and generated signals using Welch's method. Evaluates whether the frequency content (e.g. dominant gait frequency ~1-2 Hz) is preserved.

### 2. Judge Classifiers

Three classifiers trained **exclusively on real data** assess the realism of generated signals :

| Judge | Architecture | Test Accuracy |
|---|---|---|
| DeepConvLSTM | Conv1D × 2 + LSTM × 2 | ~76% |
| CNNSimple | Conv1D × 2 + AvgPool + Dense | ~72% |
| MLPSimple | Flatten + Dense × 3 + Dropout | ~79% |

A generated signal is considered realistic if the judge correctly classifies it into the intended activity class.

### 3. Statistical Testing

Over **100 independent inference runs**, pairwise comparisons between models using **Wilcoxon signed-rank tests** (non-parametric, appropriate for small samples). Results include confidence intervals via bootstrap resampling.

---

## Text-to-IMU Pipeline

The end-to-end pipeline maps a natural language prompt to a synthetic IMU signal in three steps :

```
User prompt  →  USE Encoder  →  Prompt Classifier  →  Diffusion Model  →  IMU Signal
"I went for     (512-dim          (Logistic              (Conditional         (3 axes,
 a jog"          embedding)         Regression)            DDPM)               n seconds)
```

### Components

**Text Encoder** : Universal Sentence Encoder (USE v4) from TensorFlow Hub — produces a 512-dimensional semantic embedding for the input prompt.

**Prompt Classifier** : Logistic Regression trained on ~20 labeled phrases per class. Supports 3 generative classes : `rest`, `walk`, `run_jog` (stairs excluded from the prompting pipeline).

**Generative Model** : Conditional Diffusion model, generating 3-axis accelerometer windows of 50 timesteps (1 second at 50 Hz), concatenated to produce signals of arbitrary duration.

### Usage

```bash
python 03_text_to_imu_pipeline.py
```

```
📝 Enter your prompt : I went jogging this morning

Prompt        : 'I went jogging this morning'
Label détecté : run_jog  (confiance : 0.924)
Durée cible   : 30s → 1500 fenêtres → 75000 points
✓ Signal généré — shape (3, 75000)
```

The script generates the signal, displays the full signal + 5-second zoom + spectrogram, and optionally saves the output as CSV.

### Prerequisites

```
saved_models/
├── diffusion_model.pth
├── metadata.json
├── prompt_classifier.pkl
└── prompt_label_mapping.json
```

---

## Installation

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
```

**Main dependencies :**

```
torch
tensorflow
tensorflow-hub
numpy
pandas
pyarrow
scikit-learn
scipy
matplotlib
joblib
```

---

## Results Summary

| Model | JudgeAcc (avg) | GaitScore (avg) | Best class |
|---|---|---|---|
| VAE | — | — | walk |
| Diffusion | — | — | run_jog |
| GAN | — | — | run_jog |

> Fill in your actual numbers once final results are available.

All three models are capable of producing plausible IMU signals. The Diffusion model showed particularly strong performance in terms of judge accuracy, especially on the run_jog class. The VAE produces smoother signals but sometimes lacks the sharpness of real activity patterns. The GAN (WGAN-GP) is the most sensitive to training stability but competitive on well-represented classes.

---
 
## Limitations & Future Work

- **Data volume and quality**: despite aggregating 8 public datasets (~730M rows), some classes remain underrepresented after harmonization and per-session subsampling. Future work should focus on gathering more exploitable data (additional public datasets, more subject/sensor diversity) to strengthen the generative models, especially on minority classes.

- **Preprocessing**: the current unification pipeline (50Hz resampling, FLU frame, wrist-only sensor) would benefit from a closer review — in particular, the impact of per-session subsampling on intra-class diversity, and whether keeping only the wrist sensor is optimal for every activity (e.g. `stairs`, where other sensor placements might be more informative).

- **Temporal continuity of the generative model**: the diffusion model currently generates fixed-size windows independently (each sampled from i.i.d. Gaussian noise) and concatenates them to produce signals of arbitrary duration. This introduces discontinuities at window boundaries, which can hurt the spectral realism of longer generated signals. Several improvements are being considered, from simplest to most involved: overlapping windows with crossfade (overlap-add), conditioning the sampling process on the previous window's context (RePaint-style inpainting), or moving to a natively sequential architecture (VRNN, or diffusion combined with a recurrent/state-space backbone) to guarantee end-to-end temporal coherence.

- **Prompting**: the text-to-IMU pipeline relies on a simple classifier (logistic regression on USE embeddings), and yet already performs quite well on the 3 covered classes. This is currently the most promising part of the project: a relatively small amount of additional effort (fine-tuning a small language model, expanding the training phrases dataset, handling unseen or ambiguous phrasings) could meaningfully improve robustness and extend it to more classes (e.g. bringing back `stairs`).

- **Gyroscope**: the dataset contains gyroscope channels that are not used by the generative models — a natural extension of the project.

- **Downstream validation**: the generated data has not yet been tested as augmentation for a downstream medical classification task (e.g. gait disorder detection).
 
---
 
## Authors
 
Project developed as part of a research internship at Malmö University.