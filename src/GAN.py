# =========================================================
# GAN CONDITIONNEL — Génération de signaux IMU (magnitude)
# WGAN-GP Conditionnel + Juges + ElderNet Gait Score
#
# Basé sur : comparaison_vae_diffusion_gan_v3_corrige.ipynb
# Version GAN uniquement — mêmes bases de données, juges et métriques
# =========================================================

import gc
import sys
import os
import copy
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy import signal as scipy_signal
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

# =========================================================
# 1. CONFIGURATION
# =========================================================

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print("===== DEBUG PYTHON / CUDA =====")
print("Python executable:", sys.executable)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This script requires a GPU.")

device = torch.device("cuda:0")
print("Device utilisé :", device)
print("GPU utilisée :", torch.cuda.get_device_name(0))
print("================================")

# ── Chemin du fichier parquet ────────────────────────────────────────────────
path = r"C:\Users\adril\Downloads\unified_dataset_filtered_4_balanced.parquet"

# ── Hyper-paramètres fenêtrage ───────────────────────────────────────────────
WINDOW_SIZE = 50
STRIDE      = 25
BATCH_SIZE  = 4096
FS          = 50          # fréquence d'échantillonnage (Hz)

# ── Juges ────────────────────────────────────────────────────────────────────
JUDGE_EPOCHS = 50

# ── GAN (WGAN-GP) ────────────────────────────────────────────────────────────
GAN_LATENT_DIM = 32
GAN_LABEL_EMB  = 16
GAN_EPOCHS     = 50
GAN_LR         = 1e-4
N_CRITIC       = 5
LAMBDA_GP      = 10.0

# ── Évaluation statistique ───────────────────────────────────────────────────
N_INFERENCE_RUNS = 100   # répétitions d'inférence pour les stats
N_GEN_PER_RUN    = 30    # fenêtres générées par run et par classe


# =========================================================
# 2. CHARGEMENT — MAGNITUDE (entraînement) + 3 AXES (évaluation)
# =========================================================
# Dataset : unified_dataset_filtered_4_balanced.parquet
#   Labels : 0=rest_inactive  2=walk  3=run_jog  4=stairs
#   Données déjà normalisées par label (StandardScaler appliqué en amont)
#   Fréquence : 50 Hz
#
# On calcule la magnitude sur les 3 axes normalisés.
# X_mag  (1 canal)  → entraînement GAN / Juges
# X_3ax  (3 canaux) → analyse physique / ElderNet gait score 3D

X_mag  = []
X_3ax  = []
labels = []
groups = []

parquet_file = pq.ParquetFile(path)
print(f"Nombre de paquets à traiter : {parquet_file.num_row_groups}")

for i in range(parquet_file.num_row_groups):
    chunk = parquet_file.read_row_group(
        i,
        columns=[
            "acc_x", "acc_y", "acc_z",
            "global_activity_id", "dataset", "subject_id", "session_id",
        ],
    ).to_pandas()

    acc_mag = np.sqrt(
        chunk["acc_x"].values**2
        + chunk["acc_y"].values**2
        + chunk["acc_z"].values**2
    ).astype(np.float32)

    chunk["acc_mag"] = acc_mag

    for gk, gdf in chunk.groupby(["dataset", "subject_id", "session_id"]):
        mag_sig = gdf["acc_mag"].values.astype(np.float32)
        ax_sig  = gdf[["acc_x", "acc_y", "acc_z"]].values.astype(np.float32)
        acts    = gdf["global_activity_id"].values

        if len(mag_sig) < WINDOW_SIZE:
            continue

        for j in range(0, len(mag_sig) - WINDOW_SIZE + 1, STRIDE):
            X_mag.append(mag_sig[j: j + WINDOW_SIZE])
            X_3ax.append(ax_sig[j: j + WINDOW_SIZE])
            lbl = Counter(acts[j: j + WINDOW_SIZE]).most_common(1)[0][0]
            labels.append(lbl)
            groups.append(str(gk))

    del chunk
    gc.collect()

    if (i + 1) % 5 == 0:
        print(f"Paquet {i + 1}/{parquet_file.num_row_groups} terminé...")

X_mag  = np.asarray(X_mag,  dtype=np.float32)   # (N, 50)
X_3ax  = np.asarray(X_3ax,  dtype=np.float32)   # (N, 50, 3)
labels = np.asarray(labels)
groups = np.asarray(groups)

print(f"Chargement terminé. Fenêtres : {len(X_mag)}")
print("Distribution des labels :")
for lbl, name in {0: "rest_inactive", 2: "walk", 3: "run_jog", 4: "stairs"}.items():
    n = np.sum(labels == lbl)
    if n > 0:
        print(f"  [{lbl}] {name} : {n:,} fenêtres")


# =========================================================
# 3. MAPPING DES LABELS
# =========================================================

unique_labels     = sorted(np.unique(labels))
label_mapping     = {old: new for new, old in enumerate(unique_labels)}
inv_label_mapping = {new: old for old, new in label_mapping.items()}
labels_mapped     = np.asarray([label_mapping[l] for l in labels], dtype=np.int64)
n_classes         = len(unique_labels)

print("Mapping des labels :")
for old, new in label_mapping.items():
    print(f"  {old} → {new}")
print(f"Nombre de classes : {n_classes}")


# =========================================================
# 4. SPLIT TRAIN / VAL / TEST PAR SESSION (stratifié groupe)
# =========================================================

gss_test = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
train_val_idx, test_idx = next(gss_test.split(X_mag, labels_mapped, groups=groups))

X_mag_tv = X_mag[train_val_idx];  y_tv = labels_mapped[train_val_idx]
g_tv     = groups[train_val_idx]
X_3ax_tv = X_3ax[train_val_idx]

X_mag_test = X_mag[test_idx];    y_test = labels_mapped[test_idx]
X_3ax_test = X_3ax[test_idx]

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2222, random_state=SEED)
tr_idx, va_idx = next(gss_val.split(X_mag_tv, y_tv, groups=g_tv))

X_mag_train = X_mag_tv[tr_idx];  y_train = y_tv[tr_idx]
X_3ax_train = X_3ax_tv[tr_idx]

X_mag_val   = X_mag_tv[va_idx];  y_val   = y_tv[va_idx]
X_3ax_val   = X_3ax_tv[va_idx]

print(f"Train : {len(X_mag_train)}   Val : {len(X_mag_val)}   Test : {len(X_mag_test)}")


# =========================================================
# 5. NORMALISATION (z-score calculé sur train uniquement)
# =========================================================

mag_mean = np.mean(X_mag_train)
mag_std  = np.std(X_mag_train)
print(f"Magnitude — Mean: {mag_mean:.6f}  Std: {mag_std:.6f}")

def norm_mag(x):
    return (x - mag_mean) / (mag_std + 1e-8)

def denorm_mag(x):
    return x * (mag_std + 1e-8) + mag_mean

# Normalisation magnitude
Xtn = norm_mag(X_mag_train)
Xvn = norm_mag(X_mag_val)
Xen = norm_mag(X_mag_test)

# Normalisation 3 axes (par axe, stats calculées sur train)
ax_mean = X_3ax_train.mean(axis=(0, 1), keepdims=True)   # (1, 1, 3)
ax_std  = X_3ax_train.std(axis=(0, 1), keepdims=True)

def norm_3ax(x):
    return (x - ax_mean) / (ax_std + 1e-8)

X_3ax_train_n = norm_3ax(X_3ax_train)
X_3ax_val_n   = norm_3ax(X_3ax_val)
X_3ax_test_n  = norm_3ax(X_3ax_test)

# ── Tenseurs PyTorch ─────────────────────────────────────────────────────────

def to_t(a, dtype=torch.float32):
    return torch.tensor(a, dtype=dtype)

# Magnitude : (N, 1, 50)
Xtn_t = to_t(Xtn).unsqueeze(1)
Xvn_t = to_t(Xvn).unsqueeze(1)
Xen_t = to_t(Xen).unsqueeze(1)

ytr_t = to_t(y_train, torch.long)
yva_t = to_t(y_val,   torch.long)
yte_t = to_t(y_test,  torch.long)

train_ds = TensorDataset(Xtn_t, ytr_t)
val_ds   = TensorDataset(Xvn_t, yva_t)
test_ds  = TensorDataset(Xen_t, yte_t)

def make_loader(ds, shuffle=False):
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, pin_memory=True)

train_loader = make_loader(train_ds, shuffle=True)
val_loader   = make_loader(val_ds)
test_loader  = make_loader(test_ds)

print("Tenseurs prêts.")


# =========================================================
# 6. ARCHITECTURE GAN CONDITIONNEL (WGAN-GP)
# =========================================================

class ConditionalGenerator(nn.Module):
    def __init__(self, n_classes, latent_dim=32, label_emb_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_emb  = nn.Embedding(n_classes, label_emb_dim)
        self.dec_input  = nn.Linear(latent_dim + label_emb_dim, 128 * 13)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 3, stride=2, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 1, 5, padding=2),
            nn.Tanh(),
        )

    def forward(self, z, y):
        ye = self.label_emb(y)
        h  = self.dec_input(torch.cat([z, ye], 1)).view(-1, 128, 13)
        return self.decoder(h)

    @torch.no_grad()
    def sample(self, n_samples, class_id, device):
        self.eval()
        z = torch.randn(n_samples, self.latent_dim, device=device)
        y = torch.full((n_samples,), class_id, dtype=torch.long, device=device)
        return self.forward(z, y)


class ConditionalDiscriminator(nn.Module):
    def __init__(self, n_classes, label_emb_dim=16, signal_len=50):
        super().__init__()
        self.label_emb  = nn.Embedding(n_classes, label_emb_dim)
        self.label_proj = nn.Linear(label_emb_dim, signal_len)
        self.conv = nn.Sequential(
            nn.Conv1d(2, 64,  5, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x, y):
        ye = self.label_emb(y)
        yp = self.label_proj(ye).unsqueeze(1)
        return self.fc(self.conv(torch.cat([x, yp], 1)))


def gradient_penalty(disc, real, fake, y, device):
    N   = real.size(0)
    eps = torch.rand(N, 1, 1, device=device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_out  = disc(interp, y)
    grads  = torch.autograd.grad(d_out, interp,
                                  grad_outputs=torch.ones_like(d_out),
                                  create_graph=True, retain_graph=True)[0]
    return ((grads.norm(2, dim=[1, 2]) - 1) ** 2).mean()


def train_gan(generator, discriminator, loader, val_loader=None, epochs=GAN_EPOCHS):
    opt_G = optim.Adam(generator.parameters(),     lr=GAN_LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=GAN_LR, betas=(0.5, 0.999))

    for ep in range(epochs):
        generator.train(); discriminator.train()
        g_losses, d_losses = [], []

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            N = xb.size(0)

            # ── Discriminateur (N_CRITIC steps) ──
            for _ in range(N_CRITIC):
                z    = torch.randn(N, GAN_LATENT_DIM, device=device)
                fake = generator(z, yb).detach()
                gp   = gradient_penalty(discriminator, xb, fake, yb, device)
                loss_D = (discriminator(fake, yb).mean()
                          - discriminator(xb, yb).mean()
                          + LAMBDA_GP * gp)
                opt_D.zero_grad(); loss_D.backward(); opt_D.step()
                d_losses.append(loss_D.item())

            # ── Générateur ──
            z      = torch.randn(N, GAN_LATENT_DIM, device=device)
            loss_G = -discriminator(generator(z, yb), yb).mean()
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()
            g_losses.append(loss_G.item())

        vl_str = ""
        if val_loader is not None:
            generator.eval(); discriminator.eval()
            vg = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    z = torch.randn(xb.size(0), GAN_LATENT_DIM, device=device)
                    vg.append(-discriminator(generator(z, yb), yb).mean().item())
            vl_str = f" | ValG {np.mean(vg):.4f}"

        print(f"GAN Ep {ep+1:03d} | G {np.mean(g_losses):.4f} | D {np.mean(d_losses):.4f}{vl_str}")

print("Architecture GAN définie.")


# =========================================================
# 7. JUGES (entraînés UNE SEULE FOIS sur la magnitude)
# =========================================================

class DeepConvLSTM(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(),
        )
        self.lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True)
        self.fc   = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )
    def forward(self, x):
        h, _ = self.lstm(self.conv(x).transpose(1, 2))
        return self.fc(h[:, -1, :])


class CNNSimple(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, n_classes),
        )
    def forward(self, x):
        return self.net(x)


class MLPSimple(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(WINDOW_SIZE, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_classes),
        )
    def forward(self, x):
        return self.net(x)


def train_judge(model, train_loader, val_loader=None, epochs=JUDGE_EPOCHS):
    opt       = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit      = nn.CrossEntropyLoss()
    best_val  = float("inf")
    best_state = None
    pat_cnt   = 0
    PATIENCE  = 10

    for ep in range(epochs):
        model.train()
        tot_loss = tot_corr = tot_n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss   = crit(logits, yb)
            if not torch.isfinite(loss):
                return
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item() * xb.size(0)
            tot_corr += (logits.argmax(1) == yb).sum().item()
            tot_n    += xb.size(0)
        tl = tot_loss / tot_n
        ta = tot_corr / tot_n

        if val_loader is not None:
            vl, va = eval_judge(model, val_loader)
            scheduler.step(vl)
            print(f"[{model.__class__.__name__}] Ep {ep+1:03d} "
                  f"| TrLoss {tl:.4f} TrAcc {ta:.4f} "
                  f"| VaLoss {vl:.4f} VaAcc {va:.4f}")
            if vl < best_val:
                best_val   = vl
                best_state = copy.deepcopy(model.state_dict())
                pat_cnt    = 0
            else:
                pat_cnt += 1
            if pat_cnt >= PATIENCE:
                print(f"  Early stopping {model.__class__.__name__} à epoch {ep+1}")
                break
        else:
            print(f"[{model.__class__.__name__}] Ep {ep+1:03d} "
                  f"| TrLoss {tl:.4f} TrAcc {ta:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)


def eval_judge(model, loader):
    model.eval()
    crit = nn.CrossEntropyLoss()
    tot_loss = tot_corr = tot_n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss   = crit(logits, yb)
            tot_loss += loss.item() * xb.size(0)
            tot_corr += (logits.argmax(1) == yb).sum().item()
            tot_n    += xb.size(0)
    return tot_loss / tot_n, tot_corr / tot_n

print("Architectures juges définies.")


# =========================================================
# 8. ENTRAÎNEMENT PRINCIPAL
# =========================================================

# ── Juges (entraînés une seule fois sur données réelles magnitude) ──────────
judges = [
    DeepConvLSTM(n_classes).to(device),
    CNNSimple(n_classes).to(device),
    MLPSimple(n_classes).to(device),
]

for judge in judges:
    print(f"\n🟢 Entraînement Juge : {judge.__class__.__name__}")
    train_judge(judge, train_loader, val_loader, epochs=JUDGE_EPOCHS)

for judge in judges:
    judge.eval()

# ── GAN conditionnel (WGAN-GP) ────────────────────────────────────────────────
gan_gen  = ConditionalGenerator(n_classes, GAN_LATENT_DIM, GAN_LABEL_EMB).to(device)
gan_disc = ConditionalDiscriminator(n_classes, GAN_LABEL_EMB, WINDOW_SIZE).to(device)

print("\n🟣 Entraînement GAN conditionnel (WGAN-GP)")
train_gan(gan_gen, gan_disc, train_loader, val_loader, epochs=GAN_EPOCHS)

# Sauvegarde
torch.save(gan_gen.state_dict(), "gan_gen.pth")
for judge in judges:
    torch.save(judge.state_dict(), f"{judge.__class__.__name__}.pth")

print("\n[OK] Entraînement terminé (GAN + Juges).")


# =========================================================
# 9. ELDERNET GAIT SCORE
# =========================================================
# Inspiré de Brand et al. 2026 (npj Digital Medicine).
# Combine : autocorrélation (régularité), cadence, entropie spectrale.

def eldernet_gait_score(signal_norm_np, fs=FS, return_details=False):
    """
    signal_norm_np : (1, L) numpy array — signal normalisé magnitude
    Retourne un score scalaire [0, 1] :
        0 = signal apériodique (repos)
        1 = signal très périodique (marche régulière)
    Optionnel : return_details=True renvoie (score, cadence_spm, regularity)
    """
    mag = denorm_mag(signal_norm_np).flatten().astype(np.float64)
    mag = mag - np.mean(mag)

    # ── Autocorrélation normalisée ────────────────────────────────────────────
    corr = np.correlate(mag, mag, mode="full")[len(mag) - 1:]
    corr = corr / (corr[0] + 1e-9)

    min_lag = max(1, int(0.4 * fs))
    max_lag = min(len(corr) - 1, int(1.2 * fs))

    if max_lag > min_lag:
        best_lag   = np.argmax(corr[min_lag:max_lag]) + min_lag
        regularity = float(corr[best_lag])
        step_time  = best_lag / fs
        cadence    = float(np.clip(60.0 / (step_time + 1e-9), 40, 160))
    else:
        regularity = 0.0
        cadence    = 0.0

    # ── Entropie spectrale ────────────────────────────────────────────────────
    freqs, psd  = scipy_signal.periodogram(mag, fs=fs)
    psd_norm    = psd / (psd.sum() + 1e-9)
    sp_entropy  = -np.sum(psd_norm * np.log2(psd_norm + 1e-9))
    max_entropy = np.log2(len(psd_norm) + 1e-9)
    inv_entropy = 1.0 - (sp_entropy / (max_entropy + 1e-9))

    # ── Score composite ───────────────────────────────────────────────────────
    reg_clipped   = float(np.clip(regularity, 0, 1))
    cadence_score = float(np.clip((cadence - 40) / 120, 0, 1))
    score = 0.5 * reg_clipped + 0.3 * inv_entropy + 0.2 * cadence_score

    if return_details:
        return score, cadence, reg_clipped
    return score


def eldernet_gait_score_3ax(windows_3ax_norm, fs=FS):
    """
    windows_3ax_norm : (N, L, 3) numpy array normalisé
    Retourne le score moyen sur les 3 axes.
    """
    scores = []
    for win in windows_3ax_norm:
        for ax_idx in range(3):
            ax_sig    = win[:, ax_idx]
            ax_denorm = ax_sig * (ax_std[0, 0, ax_idx] + 1e-8) + ax_mean[0, 0, ax_idx]
            ax_denorm = ax_denorm - np.mean(ax_denorm)
            corr = np.correlate(ax_denorm, ax_denorm, mode="full")[len(ax_denorm) - 1:]
            corr = corr / (corr[0] + 1e-9)
            min_lag = max(1, int(0.4 * fs))
            max_lag = min(len(corr) - 1, int(1.2 * fs))
            if max_lag > min_lag:
                best_lag = np.argmax(corr[min_lag:max_lag]) + min_lag
                reg = float(np.clip(corr[best_lag], 0, 1))
            else:
                reg = 0.0
            scores.append(reg)
    return float(np.mean(scores)) if scores else 0.0

print("ElderNet Gait Score défini.")


# =========================================================
# 10. FONCTIONS D'ÉVALUATION PAR INFÉRENCE
# =========================================================

@torch.no_grad()
def judge_score_batch(gen_tensor, true_label_mapped):
    """
    gen_tensor : (N, 1, L) — déjà sur device, normalisé magnitude
    Retourne accuracy et confiance moyenne pour chaque juge.
    """
    results = {}
    for judge in judges:
        judge.eval()
        logits = judge(gen_tensor)
        probs  = torch.softmax(logits, 1)
        preds  = probs.argmax(1).cpu().numpy()
        confs  = probs.max(1).values.cpu().numpy()
        acc    = (preds == true_label_mapped).mean()
        conf   = confs.mean()
        results[judge.__class__.__name__] = dict(acc=float(acc), conf=float(conf))
    return results


def single_inference_run(class_id_mapped, n_samples=N_GEN_PER_RUN):
    """
    Effectue une inférence GAN pour une classe donnée.
    Retourne un dict de métriques (juges + gait score).
    """
    gen    = gan_gen.sample(n_samples, class_id_mapped, device)
    gen_np = gen.cpu().numpy()   # (N, 1, 50) normalisé

    judge_res       = judge_score_batch(gen, class_id_mapped)
    gait_scores_mag = [eldernet_gait_score(gen_np[i], FS) for i in range(n_samples)]

    return dict(
        judge_res      = judge_res,
        gait_score_mag = float(np.mean(gait_scores_mag)),
    )

print("Fonctions d'évaluation définies.")


# =========================================================
# 11. BOUCLE 100 RÉPÉTITIONS D'INFÉRENCE
# =========================================================

print(f"Lancement de {N_INFERENCE_RUNS} répétitions d'inférence...")

# Structure : results[class_mapped] = liste de métriques sur N_INFERENCE_RUNS
results = {c: [] for c in range(n_classes)}

for run in range(N_INFERENCE_RUNS):
    for cls in range(n_classes):
        res = single_inference_run(cls, N_GEN_PER_RUN)
        results[cls].append(res)

    if (run + 1) % 10 == 0:
        print(f"  Run {run+1}/{N_INFERENCE_RUNS} terminé")

print("✅ 100 runs terminés.")


# =========================================================
# 12. AGRÉGATION ET BOOTSTRAP CI PAR CLASSE
# =========================================================

activity_names = {0: "rest_inactive", 2: "walk", 3: "run_jog", 4: "stairs"}
label_names = [
    activity_names.get(inv_label_mapping[c], f"Label {inv_label_mapping[c]}")
    for c in range(n_classes)
]
judge_names = [j.__class__.__name__ for j in judges]


def bootstrap_ci(data, n_boot=2000, ci=0.95):
    data  = np.array(data)
    boots = np.array([np.mean(np.random.choice(data, len(data))) for _ in range(n_boot)])
    lo    = np.percentile(boots, 100 * (1 - ci) / 2)
    hi    = np.percentile(boots, 100 * (1 + ci) / 2)
    return lo, hi


# ── Tableau récapitulatif par classe ─────────────────────────────────────────
rows = []
for cls in range(n_classes):
    lname = label_names[cls]

    # Gait score
    gait_series = [r["gait_score_mag"] for r in results[cls]]
    gait_mean   = np.mean(gait_series)
    gait_std    = np.std(gait_series)
    gait_lo, gait_hi = bootstrap_ci(gait_series)

    rows.append({
        "Classe": lname, "Métrique": "GaitScore",
        "Moyenne": f"{gait_mean:.4f}", "Std": f"{gait_std:.4f}",
        "CI 95% lo": f"{gait_lo:.4f}", "CI 95% hi": f"{gait_hi:.4f}",
    })

    for jname in judge_names:
        acc_series  = [r["judge_res"][jname]["acc"]  for r in results[cls]]
        conf_series = [r["judge_res"][jname]["conf"] for r in results[cls]]

        for metric_name, series in [("JudgeAcc", acc_series), ("JudgeConf", conf_series)]:
            m   = np.mean(series)
            s   = np.std(series)
            lo, hi = bootstrap_ci(series)
            rows.append({
                "Classe": lname, "Métrique": f"{metric_name} ({jname})",
                "Moyenne": f"{m:.4f}", "Std": f"{s:.4f}",
                "CI 95% lo": f"{lo:.4f}", "CI 95% hi": f"{hi:.4f}",
            })

df_stats = pd.DataFrame(rows)

print("\n════ RÉSULTATS BOOTSTRAP CI (GAN) ════")
print(df_stats.to_string(index=False))


# ── Comparaison GAN généré vs données réelles (juges) ───────────────────────
print("\n════ COMPARAISON GAN vs DONNÉES RÉELLES (JudgeAcc sur test set) ════")

# Accuracy des juges sur données réelles (test set)
print(f"{'Classe':<16} {'Juge':<16} {'Réel':>8} {'GAN':>8} {'Δ':>8}")
print("-" * 56)

for cls in range(n_classes):
    lname = label_names[cls]
    mask  = y_test == cls
    X_real_cls = Xen_t[mask].to(device)
    y_real_cls = yte_t[mask].to(device)

    for jname, judge in zip(judge_names, judges):
        # Accuracy sur vraies données
        _, real_acc = eval_judge(judge, DataLoader(
            TensorDataset(X_real_cls, y_real_cls),
            batch_size=BATCH_SIZE))

        # Accuracy moyenne sur les 100 runs GAN
        gan_acc = np.mean([r["judge_res"][jname]["acc"] for r in results[cls]])
        delta   = gan_acc - real_acc
        print(f"{lname:<16} {jname:<16} {real_acc:>8.4f} {gan_acc:>8.4f} {delta:>+8.4f}")
    print()


# =========================================================
# 13. VISUALISATIONS GAN
# =========================================================

plt.style.use("default")
plt.rcParams["text.color"]      = "black"
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["xtick.color"]     = "black"
plt.rcParams["ytick.color"]     = "black"

GAN_COLOR = "#55A868"

# ── Figure 1 : GaitScore par classe (boxplot sur 100 runs) ───────────────────
fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5), sharey=False)
if n_classes == 1:
    axes = [axes]
fig.suptitle("ElderNet Gait Score — GAN Conditionnel\n(100 runs d'inférence)",
             fontsize=14, fontweight="bold")

for cls, ax in enumerate(axes):
    series = [r["gait_score_mag"] for r in results[cls]]
    bp = ax.boxplot([series], tick_labels=["GAN"], patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor(GAN_COLOR)
    bp["boxes"][0].set_alpha(0.75)

    lo, hi = bootstrap_ci(series)
    ax.axhline(np.mean(series), color="navy", linestyle="--", linewidth=1.2,
               label=f"Moy={np.mean(series):.3f}")
    ax.fill_between([0.5, 1.5], lo, hi, alpha=0.15, color="navy",
                    label=f"CI 95% [{lo:.3f}, {hi:.3f}]")

    ax.set_title(label_names[cls], fontsize=11)
    ax.set_ylabel("ElderNet Gait Score")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("gait_score_gan.png", dpi=150, bbox_inches="tight")
plt.show()


# ── Figure 2 : Judge Accuracy par classe (boxplot) ───────────────────────────
fig, axes = plt.subplots(n_classes, len(judge_names),
                         figsize=(5 * len(judge_names), 4 * n_classes))
if n_classes == 1:
    axes = [axes]
fig.suptitle("Judge Accuracy — GAN Conditionnel (100 runs)", fontsize=14, fontweight="bold")

for ci in range(n_classes):
    for ji, jname in enumerate(judge_names):
        ax     = axes[ci][ji]
        series = [r["judge_res"][jname]["acc"] for r in results[ci]]
        bp = ax.boxplot([series], tick_labels=["GAN"], patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor(GAN_COLOR)
        bp["boxes"][0].set_alpha(0.75)

        lo, hi = bootstrap_ci(series)
        ax.fill_between([0.5, 1.5], lo, hi, alpha=0.15, color="navy")
        ax.axhline(np.mean(series), color="navy", linestyle="--", linewidth=1.2)

        ax.set_title(f"{label_names[ci]} — {jname}", fontsize=9)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.1)
        ax.text(1, 1.05,
                f"Moy={np.mean(series):.3f}  CI[{lo:.3f},{hi:.3f}]",
                ha="center", fontsize=7.5, color="navy")
        ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("judge_accuracy_gan.png", dpi=150, bbox_inches="tight")
plt.show()


# ── Figure 3 : Barplot moyen (gait + accuracy tous juges) ────────────────────
fig, (ax_g, ax_a) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("GAN Conditionnel — Bilan par classe", fontsize=13, fontweight="bold")

x      = np.arange(n_classes)
gait_means = [np.mean([r["gait_score_mag"] for r in results[c]]) for c in range(n_classes)]
acc_means  = [
    np.mean([
        np.mean([r["judge_res"][jn]["acc"] for jn in judge_names])
        for r in results[c]
    ])
    for c in range(n_classes)
]

ax_g.bar(x, gait_means, color=GAN_COLOR, alpha=0.8, edgecolor="white")
ax_g.set_xticks(x); ax_g.set_xticklabels(label_names, fontsize=9)
ax_g.set_ylabel("ElderNet Gait Score moyen")
ax_g.set_ylim(0, 1.0); ax_g.grid(axis="y", alpha=0.3)
ax_g.set_title("Gait Score")

ax_a.bar(x, acc_means, color=GAN_COLOR, alpha=0.8, edgecolor="white")
ax_a.set_xticks(x); ax_a.set_xticklabels(label_names, fontsize=9)
ax_a.set_ylabel("Judge Accuracy moyenne (tous juges)")
ax_a.set_ylim(0, 1.1); ax_a.grid(axis="y", alpha=0.3)
ax_a.set_title("Judge Accuracy")

plt.tight_layout()
plt.savefig("gan_bilan_barplot.png", dpi=150, bbox_inches="tight")
plt.show()


# ── Figure 4 : Tableau récap (matplotlib table) ──────────────────────────────
col_keys  = ["Classe", "Métrique", "Moyenne", "Std", "CI 95% lo", "CI 95% hi"]
cell_data = df_stats[col_keys].values.tolist()

fig_h = max(4, len(cell_data) * 0.48 + 2)
fig, ax = plt.subplots(figsize=(16, fig_h))
ax.axis("off")
fig.suptitle("Tableau récapitulatif — GAN Conditionnel (100 runs, bootstrap CI 95%)",
             fontsize=12, fontweight="bold")

tbl = ax.table(cellText=cell_data, colLabels=col_keys, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.6)

col_widths = [0.16, 0.24, 0.12, 0.10, 0.14, 0.14]
for ci, w in enumerate(col_widths):
    for ri in range(len(cell_data) + 1):
        tbl[ri, ci].set_width(w)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("#f5f6fa" if r % 2 == 0 else "white")
        cell.set_text_props(color="black")

plt.tight_layout()
plt.savefig("summary_table_gan.png", dpi=150, bbox_inches="tight")
plt.show()


# =========================================================
# 14. VALIDATION EXPERTE — EXEMPLES VISUELS PAR CLASSE
# =========================================================

def expert_validation_gan(old_label_id, nom_activite):
    mapped   = label_mapping[old_label_id]
    gen      = gan_gen.sample(30, mapped, device)            # (30, 1, 50)
    gen_phys = denorm_mag(gen.cpu().numpy())                  # physique

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.38)
    fig.suptitle(f"GAN Conditionnel — {nom_activite} (label {old_label_id})",
                 fontsize=14, fontweight="bold")

    full = gen_phys.flatten()

    # Signal global
    ax_sig = fig.add_subplot(gs[0, :2])
    ax_sig.plot(full[:500], lw=0.8, color=GAN_COLOR)
    gait_s, cad, reg = eldernet_gait_score(gen.cpu().numpy()[0], FS, return_details=True)
    ax_sig.set_title(f"Signal généré | GaitScore={gait_s:.3f}  Cad={cad:.0f} spm  Reg={reg:.3f}")
    ax_sig.set_xlabel("Échantillons"); ax_sig.set_ylabel("Magnitude (unité physique)")
    ax_sig.grid(alpha=0.3)

    # Distribution des scores GaitScore sur 30 samples
    ax_dist = fig.add_subplot(gs[0, 2:])
    all_gait = [eldernet_gait_score(gen.cpu().numpy()[i], FS) for i in range(30)]
    ax_dist.hist(all_gait, bins=10, color=GAN_COLOR, alpha=0.8, edgecolor="white")
    ax_dist.axvline(np.mean(all_gait), color="navy", linestyle="--",
                    label=f"Moy={np.mean(all_gait):.3f}")
    ax_dist.set_title("Distribution GaitScore (30 fenêtres)"); ax_dist.legend(fontsize=9)
    ax_dist.grid(alpha=0.3)

    # Zoom sur une fenêtre
    ax_z = fig.add_subplot(gs[1, :2])
    ax_z.plot(full[250:375], marker=".", ms=3, color=GAN_COLOR)
    ax_z.set_title("Zoom — 1 fenêtre centrale"); ax_z.grid(alpha=0.3)

    # Spectrogramme
    ax_sp = fig.add_subplot(gs[1, 2:])
    f_, t_, Sxx = scipy_signal.spectrogram(full, fs=FS)
    ax_sp.pcolormesh(t_, f_, 10 * np.log10(Sxx + 1e-7), shading="gouraud")
    ax_sp.set_ylim(0, 10)
    ax_sp.set_title("Spectrogramme"); ax_sp.set_xlabel("Temps (s)"); ax_sp.set_ylabel("Fréquence (Hz)")

    # Verdicts des juges
    ax_j = fig.add_subplot(gs[2, :])
    ax_j.axis("off")
    txt = "VERDICTS JUGES :\n"
    for judge in judges:
        judge.eval()
        with torch.no_grad():
            probs = torch.softmax(judge(gen), 1).mean(0)
            pred  = inv_label_mapping[probs.argmax().item()]
            conf  = probs.max().item() * 100
            ok    = "[OK]" if pred == old_label_id else "[X]"
            txt  += (f"  {judge.__class__.__name__}: "
                     f"pred={pred}  conf={conf:.1f}%  {ok}\n")

        # Affichage des probabilités par classe
        prob_str = "  probs → " + "  ".join(
            f"cls{inv_label_mapping[i]}={probs[i].item()*100:.1f}%"
            for i in range(n_classes)
        )
        txt += prob_str + "\n"

    ax_j.text(0.01, 0.5, txt, fontsize=10, family="monospace",
              va="center", transform=ax_j.transAxes)

    plt.savefig(f"expert_gan_{nom_activite}.png", dpi=120, bbox_inches="tight")
    plt.show()


for old_lbl in label_mapping.keys():
    name = activity_names.get(old_lbl, f"Activity_{old_lbl}")
    expert_validation_gan(old_lbl, name)


# =========================================================
# 15. EXPORT DES RÉSULTATS
# =========================================================

df_stats.to_csv("resultats_gan.csv", index=False)
print("Résultats exportés → resultats_gan.csv")

print("\n════ BILAN GLOBAL GAN ════")
print(f"{'Classe':<16} {'GaitScore':>12} {'JudgeAcc moy':>14}")
print("-" * 44)
for cls in range(n_classes):
    gm   = np.mean([r["gait_score_mag"] for r in results[cls]])
    ja   = np.mean([
        np.mean([r["judge_res"][jn]["acc"] for jn in judge_names])
        for r in results[cls]
    ])
    print(f"{label_names[cls]:<16} {gm:>12.4f} {ja:>14.4f}")

print("\n════ NOTE INTERPRÉTATIVE ════")
print("  GaitScore  : régularité/périodicité du signal [0=aléatoire, 1=très régulier]")
print("  JudgeAcc   : % de fenêtres générées correctement classifiées")
print("  CI 95%     : intervalle de confiance bootstrap sur 100 runs")