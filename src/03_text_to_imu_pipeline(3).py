"""
Pipeline final — Text-to-IMU + Visualisation
=============================================
À partir d'un prompt utilisateur + une durée en secondes, ce script :
  1. classifie l'activité demandée (rest / walk / run_jog / other)
  2. demande à l'utilisateur combien de secondes générer
  3. génère le signal IMU 3 axes sur la durée demandée
  4. affiche les graphiques + sauvegarde optionnelle en CSV

Pré-requis : dossier `saved_models/` contenant :
  - diffusion_model.pth
  - metadata.json
  - prompt_classifier.pkl
  - prompt_label_mapping.json
"""

import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as scipy_signal
import tensorflow_hub as hub

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

SAVED_DIR = Path("saved_models")
USE_URL   = "https://tfhub.dev/google/universal-sentence-encoder/4"
MAX_SECS  = 3600

AXIS_NAMES  = ["acc_x", "acc_y", "acc_z"]
AXIS_COLORS = ["#4C72B0", "#DD8452", "#55A868"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")


# =============================================================================
# 1. ARCHITECTURE DU MODÈLE DE DIFFUSION
# =============================================================================

class ConditionalDiffusionNet(nn.Module):
    def __init__(self, n_classes, signal_len=50, time_emb_dim=32, label_emb_dim=16):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, label_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim), nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim), nn.ReLU(),
        )
        self.conv1 = nn.Conv1d(3, 64, 5, padding=2)
        self.conv2 = nn.Conv1d(64, 64, 5, padding=2)
        self.conv3 = nn.Conv1d(64, 64, 5, padding=2)
        self.conv4 = nn.Conv1d(64, 3, 5, padding=2)
        self.cond_project = nn.Linear(time_emb_dim + label_emb_dim, 64)
        self.relu = nn.ReLU()

    def forward(self, x, t_float, y):
        cond = torch.cat([self.time_mlp(t_float), self.label_emb(y)], 1)
        cp   = self.cond_project(cond).unsqueeze(-1)
        h    = self.relu(self.conv1(x)) + cp
        h    = self.relu(self.conv2(h)) + cp
        h    = self.relu(self.conv3(h))
        return self.conv4(h)


class DiffusionContext:
    def __init__(self, n_steps=200, device="cpu"):
        self.n_steps  = n_steps
        self.device   = device
        self.beta     = torch.linspace(1e-4, 0.02, n_steps, device=device)
        self.alpha    = 1.0 - self.beta
        self.alpha_cp = torch.cumprod(self.alpha, 0)

    @torch.no_grad()
    def sample(self, model, n_windows, class_id, signal_len):
        model.eval()
        x = torch.randn(n_windows, 3, signal_len, device=self.device)
        y = torch.full((n_windows,), class_id, dtype=torch.long, device=self.device)
        for i in reversed(range(self.n_steps)):
            tf  = torch.full((n_windows, 1), i, dtype=torch.float32, device=self.device)
            pn  = model(x, tf, y)
            bt  = self.beta[i]
            sat = torch.sqrt(self.alpha[i])
            smt = torch.sqrt(1 - self.alpha_cp[i])
            mu  = (1 / sat) * (x - (bt / smt) * pn)
            x   = mu + (torch.sqrt(bt) * torch.randn_like(x) if i > 0 else 0)
        return x


# =============================================================================
# 2. CHARGEMENT DES MODÈLES
# =============================================================================

print("\nChargement des modèles sauvegardés...")

with open(SAVED_DIR / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

WINDOW_SIZE   = metadata["window_size"]
FS            = metadata["fs"]
N_STEPS       = metadata["n_steps"]
TIME_EMB_DIM  = metadata["time_emb_dim"]
LABEL_EMB_DIM = metadata["label_emb_dim"]
n_classes_gen = metadata["n_classes"]

gen_label_mapping     = {int(k): v for k, v in metadata["label_mapping"].items()}
activity_names        = {int(k): v for k, v in metadata["activity_names"].items()}
ax_mean = np.array(metadata["ax_mean"]).reshape(1, 3, 1)
ax_std  = np.array(metadata["ax_std"]).reshape(1, 3, 1)
SECS_PER_WINDOW = WINDOW_SIZE / FS

def denorm_3ax(x):
    return x * (ax_std + 1e-8) + ax_mean

diff_model = ConditionalDiffusionNet(n_classes_gen, WINDOW_SIZE, TIME_EMB_DIM, LABEL_EMB_DIM).to(device)
diff_model.load_state_dict(torch.load(SAVED_DIR / "diffusion_model.pth", map_location=device))
diff_model.eval()
diff_context = DiffusionContext(N_STEPS, device)
print("  ✓ Modèle de diffusion chargé.")

prompt_clf = joblib.load(SAVED_DIR / "prompt_classifier.pkl")
with open(SAVED_DIR / "prompt_label_mapping.json", "r", encoding="utf-8") as f:
    d = json.load(f)
PROMPT_INV_MAPPING = {int(k): v for k, v in d["inv_mapping"].items()}
print("  ✓ Classifieur de prompt chargé.")

print("  Chargement de USE...")
embed = hub.load(USE_URL)
print("  ✓ USE chargé.")

PROMPT_TO_GEN_ACTIVITY    = {"rest": "rest_inactive", "walk": "walk", "run_jog": "run_jog"}
GEN_ACTIVITY_TO_OLD_LABEL = {v: k for k, v in activity_names.items()}


# =============================================================================
# 3. FONCTIONS
# =============================================================================

def classify_prompt(prompt):
    emb   = embed([prompt]).numpy()
    pred  = prompt_clf.predict(emb)[0]
    probs = prompt_clf.predict_proba(emb)[0]
    return PROMPT_INV_MAPPING[pred], float(probs[pred])


def ask_duration():
    while True:
        raw = input("\n⏱  Durée de génération souhaitée (en secondes, ex: 30 / 120) : ").strip()
        try:
            secs = float(raw)
            if secs <= 0:
                print(f"  ⚠ Entrez un nombre positif.")
                continue
            if secs > MAX_SECS:
                print(f"  ⚠ Maximum {MAX_SECS}s. Valeur ramenée à {MAX_SECS}s.")
                secs = MAX_SECS
            return secs
        except ValueError:
            print(f"  ⚠ '{raw}' n'est pas un nombre. Exemple : 60")


def generate_long_signal(class_id_mapped, n_windows):
    BATCH = 256
    all_windows = []
    remaining = n_windows
    while remaining > 0:
        batch_size = min(BATCH, remaining)
        gen_norm = diff_context.sample(diff_model, batch_size, class_id_mapped, WINDOW_SIZE)
        gen_phys = denorm_3ax(gen_norm.cpu().numpy())
        all_windows.append(gen_phys)
        remaining -= batch_size
        if n_windows > BATCH:
            print(f"  {n_windows - remaining}/{n_windows} fenêtres...", end="\r")
    signal = np.concatenate(all_windows, axis=0)       # (n_windows, 3, WINDOW_SIZE)
    signal = signal.transpose(1, 0, 2).reshape(3, -1)  # (3, total_samples)
    return signal


def plot_signal(signal, gen_activity_name, total_secs, prompt, confidence):
    time_axis = np.arange(signal.shape[1]) / FS
    n_total   = signal.shape[1]
    ZOOM_SECS = min(5, total_secs)
    zoom_pts  = int(ZOOM_SECS * FS)

    # ── Figure principale ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 7))
    fig.suptitle(
        f"Signal IMU généré — {gen_activity_name}\n"
        f"Prompt : {prompt!r}  |  Confiance : {confidence:.3f}  |  Durée : {total_secs:.1f}s",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.35)

    # Ligne 1 : signal complet
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(time_axis, signal[i], lw=0.6, color=AXIS_COLORS[i], alpha=0.85)
        ax.set_title(f"{AXIS_NAMES[i]} — signal complet", fontsize=10, fontweight="bold")
        ax.set_xlabel("Temps (s)"); ax.set_ylabel("Accélération (m/s²)")
        ax.grid(alpha=0.3)

    # Ligne 2 : zoom
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(time_axis[:zoom_pts], signal[i, :zoom_pts],
                lw=1.2, color=AXIS_COLORS[i], marker=".", ms=2)
        ax.set_title(f"{AXIS_NAMES[i]} — zoom {ZOOM_SECS:.0f}s", fontsize=10, fontweight="bold")
        ax.set_xlabel("Temps (s)"); ax.set_ylabel("Accélération (m/s²)")
        ax.grid(alpha=0.3)

    fname = f"imu_{gen_activity_name}_{int(total_secs)}s.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Figure sauvegardée → {fname}")

    # ── Spectrogramme ─────────────────────────────────────────────────────────
    fig2, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig2.suptitle(f"Spectrogramme — {gen_activity_name}  ({total_secs:.1f}s)",
                  fontsize=12, fontweight="bold")
    for i, ax in enumerate(axes):
        f_, t_, Sxx = scipy_signal.spectrogram(
            signal[i], fs=FS,
            nperseg=min(64, n_total),
            noverlap=min(32, n_total // 2)
        )
        pcm = ax.pcolormesh(t_, f_, 10 * np.log10(Sxx + 1e-10),
                            shading="gouraud", cmap="viridis")
        ax.set_ylim(0, FS / 2)
        ax.set_title(f"{AXIS_NAMES[i]}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Temps (s)"); ax.set_ylabel("Fréquence (Hz)")
        plt.colorbar(pcm, ax=ax, label="dB")

    fname2 = f"spectrogram_{gen_activity_name}_{int(total_secs)}s.png"
    plt.tight_layout()
    plt.savefig(fname2, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Spectrogramme sauvegardé → {fname2}")


# =============================================================================
# 4. PIPELINE PRINCIPAL
# =============================================================================

def text_to_imu(prompt, duration_secs=None):
    prompt_label, confidence = classify_prompt(prompt)
    print(f"\nPrompt        : {prompt!r}")
    print(f"Label détecté : {prompt_label}  (confiance : {confidence:.3f})")

    if prompt_label not in PROMPT_TO_GEN_ACTIVITY:
        print("⚠  Activité 'other' — aucune génération possible.")
        return

    if duration_secs is None:
        duration_secs = ask_duration()

    n_windows     = max(1, round(duration_secs / SECS_PER_WINDOW))
    total_samples = n_windows * WINDOW_SIZE
    total_secs    = total_samples / FS

    print(f"Durée cible   : {duration_secs:.1f}s → {n_windows} fenêtres → {total_samples} points ({total_secs:.1f}s réelles)")

    gen_activity_name = PROMPT_TO_GEN_ACTIVITY[prompt_label]
    old_label         = GEN_ACTIVITY_TO_OLD_LABEL[gen_activity_name]
    class_id_mapped   = gen_label_mapping[old_label]

    print(f"Génération '{gen_activity_name}'...")
    signal = generate_long_signal(class_id_mapped, n_windows)
    print(f"✓ Signal généré — shape {signal.shape}  ({total_secs:.1f}s à {FS} Hz)")

    plot_signal(signal, gen_activity_name, total_secs, prompt, confidence)

    save = input("\n💾 Sauvegarder en CSV ? (o/n) : ").strip().lower()
    if save == "o":
        import pandas as pd
        df = pd.DataFrame(signal.T, columns=["acc_x", "acc_y", "acc_z"])
        df["time_s"] = np.arange(len(df)) / FS
        fname = f"imu_{gen_activity_name}_{int(total_secs)}s.csv"
        df.to_csv(fname, index=False)
        print(f"  ✓ Sauvegardé → {fname}")


# =============================================================================
# 5. BOUCLE INTERACTIVE
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Text-to-IMU — Pipeline interactif")
    print("=" * 55)

    while True:
        prompt = input("\n📝 Entrez votre prompt (ou 'quit' pour quitter) : ").strip()
        if prompt.lower() in ("quit", "q", "exit"):
            print("Au revoir !")
            break
        if not prompt:
            continue
        text_to_imu(prompt)
