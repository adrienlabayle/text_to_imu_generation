import nbformat

nb = nbformat.v4.new_notebook()
cells = []

def code(src):
    return nbformat.v4.new_code_cell(src)

def md(src):
    return nbformat.v4.new_markdown_cell(src)

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""# Évaluation des Juges — DeepConvLSTM / CNNSimple / MLPSimple
## Accuracy, Confusion Matrix, Calibration, Analyse des erreurs
---
Ce notebook évalue la fiabilité des trois juges entraînés sur les données réelles de magnitude IMU.  
**Question centrale :** peut-on faire confiance aux scores de juges pour évaluer les signaux générés ?
"""))

# ─── Cell 1 : Imports + Config ───────────────────────────────────────────────
cells.append(md("## 1. Configuration & Imports"))
cells.append(code("""import gc
import sys
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
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, balanced_accuracy_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

print("===== DEBUG PYTHON / CUDA =====")
print("Python  :", sys.executable)
print("PyTorch :", torch.__version__)
print("CUDA    :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA requis.")

device = torch.device("cuda:0")
print("GPU     :", torch.cuda.get_device_name(0))

# ── Chemin données ────────────────────────────────────────────────────────────
PATH_PARQUET = r"C:\\Users\\adril\\Downloads\\unified_dataset_filtered_4_balanced.parquet"

# ── Hyper-paramètres fenêtrage (identiques au pipeline génératif) ─────────────
WINDOW_SIZE  = 50
STRIDE       = 25
BATCH_SIZE   = 4096
FS           = 50

# ── Juges ─────────────────────────────────────────────────────────────────────
JUDGE_EPOCHS = 50
EARLY_STOP   = 10

# ── Optionnel : charger des poids déjà entraînés (mettre False pour réentraîner)
LOAD_PRETRAINED = False

ACTIVITY_NAMES = {0: "rest_inactive", 2: "walk", 3: "run_jog", 4: "stairs"}

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
print("Configuration OK.")
"""))

# ─── Cell 2 : Chargement données ─────────────────────────────────────────────
cells.append(md("## 2. Chargement des données (magnitude + 3 axes)"))
cells.append(code("""X_mag  = []
X_3ax  = []
labels = []
groups = []

parquet_file = pq.ParquetFile(PATH_PARQUET)
print(f"Paquets à traiter : {parquet_file.num_row_groups}")

for i in range(parquet_file.num_row_groups):
    chunk = parquet_file.read_row_group(
        i,
        columns=["acc_x","acc_y","acc_z","global_activity_id",
                 "dataset","subject_id","session_id"],
    ).to_pandas()

    acc_mag = np.sqrt(
        chunk["acc_x"].values**2 +
        chunk["acc_y"].values**2 +
        chunk["acc_z"].values**2
    ).astype(np.float32)
    chunk["acc_mag"] = acc_mag

    for gk, gdf in chunk.groupby(["dataset","subject_id","session_id"]):
        mag_sig = gdf["acc_mag"].values.astype(np.float32)
        ax_sig  = gdf[["acc_x","acc_y","acc_z"]].values.astype(np.float32)
        acts    = gdf["global_activity_id"].values
        if len(mag_sig) < WINDOW_SIZE:
            continue
        for j in range(0, len(mag_sig) - WINDOW_SIZE + 1, STRIDE):
            X_mag.append(mag_sig[j:j+WINDOW_SIZE])
            X_3ax.append(ax_sig[j:j+WINDOW_SIZE])
            lbl = Counter(acts[j:j+WINDOW_SIZE]).most_common(1)[0][0]
            labels.append(lbl)
            groups.append(str(gk))
    del chunk; gc.collect()
    if (i+1) % 5 == 0:
        print(f"  Paquet {i+1}/{parquet_file.num_row_groups}...")

X_mag  = np.asarray(X_mag,  dtype=np.float32)
X_3ax  = np.asarray(X_3ax,  dtype=np.float32)
labels = np.asarray(labels)
groups = np.asarray(groups)

print(f"\\nFenêtres totales : {len(X_mag):,}")
for lbl, name in ACTIVITY_NAMES.items():
    n = np.sum(labels == lbl)
    if n > 0:
        print(f"  [{lbl}] {name} : {n:,}")
"""))

# ─── Cell 3 : Label mapping + Split + Normalisation ──────────────────────────
cells.append(md("## 3. Mapping labels · Split · Normalisation"))
cells.append(code("""# ── Mapping ───────────────────────────────────────────────────────────────────
unique_labels     = sorted(np.unique(labels))
label_mapping     = {old: new for new, old in enumerate(unique_labels)}
inv_label_mapping = {new: old for old, new in label_mapping.items()}
labels_mapped     = np.asarray([label_mapping[l] for l in labels], dtype=np.int64)
n_classes         = len(unique_labels)
class_names       = [ACTIVITY_NAMES.get(inv_label_mapping[c], f"cls{c}") for c in range(n_classes)]

print("Mapping :", label_mapping)
print("Classes :", class_names)

# ── Split (identique au pipeline génératif) ───────────────────────────────────
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
tv_idx, test_idx = next(gss_test.split(X_mag, labels_mapped, groups=groups))

X_tv = X_mag[tv_idx];  y_tv = labels_mapped[tv_idx];  g_tv = groups[tv_idx]
X_mag_test = X_mag[test_idx];  y_test = labels_mapped[test_idx]

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2222, random_state=SEED)
tr_idx, va_idx = next(gss_val.split(X_tv, y_tv, groups=g_tv))

X_mag_train = X_tv[tr_idx];  y_train = y_tv[tr_idx]
X_mag_val   = X_tv[va_idx];  y_val   = y_tv[va_idx]

print(f"Train : {len(X_mag_train):,}  Val : {len(X_mag_val):,}  Test : {len(X_mag_test):,}")

# ── Normalisation (z-score sur train) ─────────────────────────────────────────
mag_mean = np.mean(X_mag_train)
mag_std  = np.std(X_mag_train)

def norm_mag(x):
    return (x - mag_mean) / (mag_std + 1e-8)

Xtn = norm_mag(X_mag_train)
Xvn = norm_mag(X_mag_val)
Xen = norm_mag(X_mag_test)

def to_t(a, dtype=torch.float32):
    return torch.tensor(a, dtype=dtype)

# Tenseurs (N, 1, 50)
Xtn_t = to_t(Xtn).unsqueeze(1);  ytr_t = to_t(y_train, torch.long)
Xvn_t = to_t(Xvn).unsqueeze(1);  yva_t = to_t(y_val,   torch.long)
Xen_t = to_t(Xen).unsqueeze(1);  yte_t = to_t(y_test,  torch.long)

def make_loader(X, y, shuffle=False):
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, pin_memory=True)

train_loader = make_loader(Xtn_t, ytr_t, shuffle=True)
val_loader   = make_loader(Xvn_t, yva_t)
test_loader  = make_loader(Xen_t, yte_t)
print("Tenseurs prêts.")
"""))

# ─── Cell 4 : Architectures ───────────────────────────────────────────────────
cells.append(md("## 4. Architectures des juges"))
cells.append(code("""class DeepConvLSTM(nn.Module):
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
    def forward(self, x): return self.net(x)


class MLPSimple(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(WINDOW_SIZE, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_classes),
        )
    def forward(self, x): return self.net(x)

print("Architectures définies.")
"""))

# ─── Cell 5 : train_judge avec historique ─────────────────────────────────────
cells.append(md("## 5. Entraînement (avec suivi de l'historique train/val)"))
cells.append(code("""def train_judge(model, train_loader, val_loader=None, epochs=JUDGE_EPOCHS):
    \"\"\"Entraîne un juge et retourne l'historique complet (loss + accuracy).\"\"\"
    opt       = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit      = nn.CrossEntropyLoss()
    best_val  = float("inf")
    best_state = None
    pat_cnt   = 0
    history   = {"train_loss": [], "val_loss": [],
                 "train_acc":  [], "val_acc":  []}

    for ep in range(epochs):
        model.train()
        tot_loss = tot_corr = tot_n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss   = crit(logits, yb)
            if not torch.isfinite(loss): return history
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item() * xb.size(0)
            tot_corr += (logits.argmax(1) == yb).sum().item()
            tot_n    += xb.size(0)

        tl = tot_loss / tot_n
        ta = tot_corr / tot_n
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)

        if val_loader is not None:
            model.eval()
            vl, va = eval_judge(model, val_loader)
            scheduler.step(vl)
            history["val_loss"].append(vl)
            history["val_acc"].append(va)
            print(f"[{model.__class__.__name__:14s}] Ep {ep+1:03d} "
                  f"| TrainLoss {tl:.4f}  TrainAcc {ta:.4f} "
                  f"| ValLoss {vl:.4f}  ValAcc {va:.4f}")
            if vl < best_val:
                best_val   = vl
                best_state = copy.deepcopy(model.state_dict())
                pat_cnt    = 0
            else:
                pat_cnt += 1
            if pat_cnt >= EARLY_STOP:
                print(f"  ↳ Early stopping à epoch {ep+1}")
                break
        else:
            print(f"[{model.__class__.__name__:14s}] Ep {ep+1:03d} "
                  f"| TrainLoss {tl:.4f}  TrainAcc {ta:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


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

print("Fonctions d'entraînement définies.")
"""))

# ─── Cell 6 : Entraînement ────────────────────────────────────────────────────
cells.append(code("""judges = [
    DeepConvLSTM(n_classes).to(device),
    CNNSimple(n_classes).to(device),
    MLPSimple(n_classes).to(device),
]
judge_names = [j.__class__.__name__ for j in judges]
histories   = {}

if LOAD_PRETRAINED:
    for judge in judges:
        fname = f"{judge.__class__.__name__}.pth"
        judge.load_state_dict(torch.load(fname, map_location=device))
        judge.eval()
        print(f"[Chargé] {fname}")
else:
    for judge in judges:
        print(f"\\n{'='*60}")
        print(f"  Entraînement : {judge.__class__.__name__}")
        print(f"{'='*60}")
        hist = train_judge(judge, train_loader, val_loader, epochs=JUDGE_EPOCHS)
        histories[judge.__class__.__name__] = hist
        judge.eval()
        torch.save(judge.state_dict(), f"{judge.__class__.__name__}.pth")

print("\\n✅ Entraînement terminé.")
"""))

# ─── Cell 7 : Fonction d'évaluation complète ──────────────────────────────────
cells.append(md("## 6. Évaluation complète sur le test set"))
cells.append(code("""def evaluate_full(model, loader):
    \"\"\"
    Retourne toutes les prédictions + labels + confidences sur un DataLoader.
    \"\"\"
    model.eval()
    all_preds  = []
    all_labels = []
    all_confs  = []
    all_probs  = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs  = torch.softmax(logits, 1)
            preds  = probs.argmax(1).cpu().numpy()
            confs  = probs.max(1).values.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())
            all_confs.extend(confs)
            all_probs.append(probs.cpu().numpy())

    return (np.array(all_preds), np.array(all_labels),
            np.array(all_confs), np.vstack(all_probs))


# ── Calcul des métriques pour chaque juge ────────────────────────────────────
eval_results = {}

print(f"{'Juge':<18} {'Accuracy':>10} {'Balanced Acc':>14}")
print("-" * 46)

for judge in judges:
    name = judge.__class__.__name__
    preds, labels_true, confs, probs = evaluate_full(judge, test_loader)
    acc      = accuracy_score(labels_true, preds)
    bal_acc  = balanced_accuracy_score(labels_true, preds)
    eval_results[name] = dict(preds=preds, labels=labels_true,
                               confs=confs, probs=probs,
                               acc=acc, bal_acc=bal_acc)
    print(f"{name:<18} {acc:>10.4f} {bal_acc:>14.4f}")

print()
"""))

# ─── Cell 8 : Classification reports ─────────────────────────────────────────
cells.append(code("""print("=" * 60)
for name, res in eval_results.items():
    print(f"\\n{'='*60}")
    print(f"  Classification Report — {name}")
    print(f"{'='*60}")
    report = classification_report(
        res["labels"], res["preds"],
        target_names=class_names,
        digits=4
    )
    print(report)
"""))

# ─── Cell 9 : Confusion matrices ──────────────────────────────────────────────
cells.append(md("## 7. Matrices de confusion"))
cells.append(code("""fig, axes = plt.subplots(2, len(judges), figsize=(6 * len(judges), 11))
fig.suptitle("Matrices de confusion — Test Set", fontsize=15, fontweight="bold", y=1.01)

for col, (judge, name) in enumerate(zip(judges, judge_names)):
    res    = eval_results[name]
    labels = res["labels"]
    preds  = res["preds"]

    # ── Ligne 1 : counts bruts ────────────────────────────────────────────────
    cm_raw = confusion_matrix(labels, preds)
    ax_raw = axes[0][col]
    sns.heatmap(cm_raw, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax_raw, linewidths=0.5, cbar=False)
    ax_raw.set_title(f"{name}\\nCounts bruts  (Acc={res['acc']:.4f})",
                     fontsize=11, fontweight="bold")
    ax_raw.set_xlabel("Prédit"); ax_raw.set_ylabel("Réel")

    # ── Ligne 2 : normalisée (recall par ligne) ───────────────────────────────
    cm_norm = cm_raw.astype(float) / (cm_raw.sum(axis=1, keepdims=True) + 1e-9)
    ax_norm = axes[1][col]
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Oranges",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax_norm, linewidths=0.5, vmin=0, vmax=1, cbar=True)
    ax_norm.set_title(f"{name}\\nNormalisée (recall / ligne)",
                      fontsize=11, fontweight="bold")
    ax_norm.set_xlabel("Prédit"); ax_norm.set_ylabel("Réel")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# ─── Cell 10 : Courbes d'apprentissage ────────────────────────────────────────
cells.append(md("## 8. Courbes d'apprentissage (train vs val)"))
cells.append(code("""if not LOAD_PRETRAINED and histories:
    fig, axes = plt.subplots(2, len(judges), figsize=(6 * len(judges), 9))
    fig.suptitle("Courbes d'apprentissage", fontsize=14, fontweight="bold")

    for col, name in enumerate(judge_names):
        hist = histories[name]
        ep   = range(1, len(hist["train_loss"]) + 1)

        # Loss
        ax_l = axes[0][col]
        ax_l.plot(ep, hist["train_loss"], label="Train",  color="#4C72B0", lw=2)
        if hist["val_loss"]:
            ax_l.plot(ep, hist["val_loss"],  label="Val", color="#DD8452", lw=2, linestyle="--")
        ax_l.set_title(f"{name} — Loss", fontsize=11, fontweight="bold")
        ax_l.set_xlabel("Epoch"); ax_l.set_ylabel("Cross-Entropy Loss")
        ax_l.legend(); ax_l.grid(alpha=0.3)

        # Accuracy
        ax_a = axes[1][col]
        ax_a.plot(ep, hist["train_acc"], label="Train",  color="#4C72B0", lw=2)
        if hist["val_acc"]:
            ax_a.plot(ep, hist["val_acc"],  label="Val", color="#DD8452", lw=2, linestyle="--")
        ax_a.set_title(f"{name} — Accuracy", fontsize=11, fontweight="bold")
        ax_a.set_xlabel("Epoch"); ax_a.set_ylabel("Accuracy")
        ax_a.set_ylim(0, 1.05); ax_a.legend(); ax_a.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("Pas d'historique disponible (modèles chargés depuis fichier .pth).")
"""))

# ─── Cell 11 : Accuracy par classe ────────────────────────────────────────────
cells.append(md("## 9. Accuracy par classe — comparaison des 3 juges"))
cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Performances par classe — Juges", fontsize=13, fontweight="bold")

colors = ["#4C72B0", "#DD8452", "#55A868"]
x      = np.arange(n_classes)
width  = 0.25

# ── Accuracy par classe ───────────────────────────────────────────────────────
ax = axes[0]
for k, (name, color) in enumerate(zip(judge_names, colors)):
    res = eval_results[name]
    cm  = confusion_matrix(res["labels"], res["preds"])
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    bars = ax.bar(x + k * width, per_class_acc, width,
                  label=name, color=color, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x + width); ax.set_xticklabels(class_names, fontsize=10)
ax.set_ylabel("Recall (accuracy par classe)"); ax.set_ylim(0, 1.15)
ax.set_title("Recall par classe"); ax.legend(); ax.grid(axis="y", alpha=0.3)

# ── Accuracy globale ──────────────────────────────────────────────────────────
ax2 = axes[1]
accs     = [eval_results[n]["acc"]     for n in judge_names]
bal_accs = [eval_results[n]["bal_acc"] for n in judge_names]
x2 = np.arange(len(judge_names))
bars1 = ax2.bar(x2 - 0.2, accs,     0.35, label="Accuracy",          color=colors, alpha=0.85, edgecolor="white")
bars2 = ax2.bar(x2 + 0.2, bal_accs, 0.35, label="Balanced Accuracy", color=colors, alpha=0.5,  edgecolor="black", linestyle="--", linewidth=0.8)
for bar in list(bars1) + list(bars2):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
ax2.set_xticks(x2); ax2.set_xticklabels(judge_names, fontsize=10)
ax2.set_ylabel("Score"); ax2.set_ylim(0, 1.15)
ax2.set_title("Accuracy globale vs balanced"); ax2.legend(); ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("accuracy_per_class.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# ─── Cell 12 : Calibration ────────────────────────────────────────────────────
cells.append(md("""## 10. Calibration de la confiance
> **Reliability diagram** : un juge bien calibré a une confiance moyenne égale à son accuracy dans chaque bin.  
> Si la courbe est **au-dessus de la diagonale** → le juge est **trop confiant**.  
> Si elle est **en-dessous** → il est **trop prudent**.
"""))
cells.append(code("""def reliability_diagram(preds, labels, confs, n_bins=10, ax=None, title=""):
    \"\"\"Trace le reliability diagram + ECE (Expected Calibration Error).\"\"\"
    if ax is None:
        _, ax = plt.subplots()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_acc   = []
    bin_conf  = []
    bin_count = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            bin_acc.append(0); bin_conf.append((lo+hi)/2); bin_count.append(0)
            continue
        bin_acc.append((preds[mask] == labels[mask]).mean())
        bin_conf.append(confs[mask].mean())
        bin_count.append(mask.sum())

    bin_acc   = np.array(bin_acc)
    bin_conf  = np.array(bin_conf)
    bin_count = np.array(bin_count)

    # ECE
    n_total = len(preds)
    ece = np.sum((bin_count / n_total) * np.abs(bin_acc - bin_conf))

    # Gap (sur-confiance)
    bar_colors = ["#e74c3c" if a < c else "#2ecc71"
                  for a, c in zip(bin_acc, bin_conf)]
    ax.bar(bin_conf, bin_acc, width=1/n_bins, alpha=0.7,
           color=bar_colors, align="center", edgecolor="white", label="Accuracy / bin")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Parfaite calibration")
    ax.set_xlabel("Confiance moyenne"); ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_title(f"{title}\\nECE = {ece:.4f}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    return ece


fig, axes = plt.subplots(1, len(judges), figsize=(6 * len(judges), 5))
fig.suptitle("Calibration de la confiance (Reliability Diagram)", fontsize=13, fontweight="bold")

ece_scores = {}
for ax, (judge, name) in zip(axes, zip(judges, judge_names)):
    res = eval_results[name]
    ece = reliability_diagram(res["preds"], res["labels"], res["confs"],
                               n_bins=10, ax=ax, title=name)
    ece_scores[name] = ece

plt.tight_layout()
plt.savefig("calibration.png", dpi=150, bbox_inches="tight")
plt.show()

print("\\nExpected Calibration Error (ECE) — plus c'est bas, mieux c'est :")
for name, ece in ece_scores.items():
    flag = "✓ bien calibré" if ece < 0.05 else ("⚠ à surveiller" if ece < 0.10 else "✗ mal calibré")
    print(f"  {name:<18} ECE = {ece:.4f}  {flag}")
"""))

# ─── Cell 13 : Distribution de la confiance ───────────────────────────────────
cells.append(md("## 11. Distribution de la confiance par classe et par juge"))
cells.append(code("""fig, axes = plt.subplots(n_classes, len(judges),
                         figsize=(5 * len(judges), 3.5 * n_classes))
if n_classes == 1: axes = [axes]
fig.suptitle("Distribution de la confiance — Prédictions correctes vs incorrectes",
             fontsize=13, fontweight="bold")

for ci, cname in enumerate(class_names):
    for ji, (name, judge) in enumerate(zip(judge_names, judges)):
        ax  = axes[ci][ji]
        res = eval_results[name]
        mask_cls = res["labels"] == ci

        if mask_cls.sum() == 0:
            ax.axis("off"); continue

        conf_correct   = res["confs"][mask_cls & (res["preds"] == res["labels"])]
        conf_incorrect = res["confs"][mask_cls & (res["preds"] != res["labels"])]

        ax.hist(conf_correct,   bins=20, alpha=0.7, color="#2ecc71", label="Correct",   density=True)
        ax.hist(conf_incorrect, bins=20, alpha=0.7, color="#e74c3c", label="Incorrect", density=True)
        n_tot  = mask_cls.sum()
        n_ok   = len(conf_correct)
        ax.set_title(f"{cname} | {name}\\n"
                     f"Recall={n_ok/n_tot:.3f} ({n_ok}/{n_tot})",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Confiance"); ax.set_xlim(0, 1)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("confidence_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# ─── Cell 14 : Accord entre juges ─────────────────────────────────────────────
cells.append(md("## 12. Accord inter-juges (sur le test set)"))
cells.append(code("""# ── Taux d'accord entre chaque paire de juges ────────────────────────────────
print("Taux d'accord entre juges (% de prédictions identiques sur test set)")
print("-" * 50)

preds_matrix = np.stack([eval_results[n]["preds"] for n in judge_names], axis=1)
n_judges = len(judge_names)

agreement_table = np.zeros((n_judges, n_judges))
for i in range(n_judges):
    for j in range(n_judges):
        agreement_table[i, j] = (preds_matrix[:, i] == preds_matrix[:, j]).mean()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(agreement_table, annot=True, fmt=".3f", cmap="YlGnBu",
            xticklabels=judge_names, yticklabels=judge_names,
            ax=ax, vmin=0, vmax=1, linewidths=0.5)
ax.set_title("Taux d'accord inter-juges (test set)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("judge_agreement.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Vote majoritaire ─────────────────────────────────────────────────────────
from scipy import stats as sp_stats

majority_preds = sp_stats.mode(preds_matrix, axis=1, keepdims=False).mode
true_labels    = eval_results[judge_names[0]]["labels"]
majority_acc   = accuracy_score(true_labels, majority_preds)
majority_bal   = balanced_accuracy_score(true_labels, majority_preds)

print(f"\\nVote majoritaire (ensemble des 3 juges) :")
print(f"  Accuracy          : {majority_acc:.4f}")
print(f"  Balanced Accuracy : {majority_bal:.4f}")
print()
print(classification_report(true_labels, majority_preds, target_names=class_names, digits=4))
"""))

# ─── Cell 15 : Analyse des erreurs ────────────────────────────────────────────
cells.append(md("""## 13. Analyse des erreurs — signaux mal classifiés
> Visualisation de quelques signaux que les juges ont du mal à classer correctement.
"""))
cells.append(code("""def plot_misclassified(judge, judge_name, test_X_np, test_y_np, n_examples=4):
    \"\"\"
    Affiche des exemples de signaux mal classifiés avec la confiance du juge.
    test_X_np : (N, 50)  dénormalisé ou normalisé — on affiche la forme du signal.
    \"\"\"
    res = eval_results[judge_name]
    wrong_mask = res["preds"] != res["labels"]
    wrong_idx  = np.where(wrong_mask)[0]

    if len(wrong_idx) == 0:
        print(f"{judge_name} : aucune erreur sur le test set !")
        return

    # On sélectionne aléatoirement n_examples parmi les erreurs
    rng  = np.random.default_rng(SEED)
    pick = rng.choice(wrong_idx, size=min(n_examples, len(wrong_idx)), replace=False)

    fig, axes = plt.subplots(1, len(pick), figsize=(5 * len(pick), 4))
    if len(pick) == 1: axes = [axes]
    fig.suptitle(f"{judge_name} — Signaux mal classifiés (test set)",
                 fontsize=12, fontweight="bold")

    for ax, idx in zip(axes, pick):
        true_lbl = class_names[res["labels"][idx]]
        pred_lbl = class_names[res["preds"][idx]]
        conf     = res["confs"][idx]
        signal   = test_X_np[idx]

        ax.plot(signal, color="#e74c3c", lw=1.5)
        ax.set_title(f"Vrai : {true_lbl}\\nPrédit : {pred_lbl}  ({conf*100:.1f}%)",
                     fontsize=9, fontweight="bold", color="#c0392b")
        ax.set_xlabel("Échantillons"); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"misclassified_{judge_name}.png", dpi=120, bbox_inches="tight")
    plt.show()

# Xen est normalisé — on dénormalise pour l'affichage
X_test_display = X_mag_test  # valeurs brutes (physiques)

for judge, name in zip(judges, judge_names):
    plot_misclassified(judge, name, X_test_display, y_test, n_examples=6)
"""))

# ─── Cell 16 : Tableau récapitulatif final ────────────────────────────────────
cells.append(md("## 14. Tableau récapitulatif final"))
cells.append(code("""# ── Métriques par juge × classe ──────────────────────────────────────────────
rows = []
for name in judge_names:
    res  = eval_results[name]
    cm   = confusion_matrix(res["labels"], res["preds"])
    prec_rec_f1 = classification_report(res["labels"], res["preds"],
                                         target_names=class_names,
                                         output_dict=True)
    for ci, cname in enumerate(class_names):
        r = prec_rec_f1[cname]
        rows.append(dict(
            Juge       = name,
            Classe     = cname,
            Precision  = f"{r['precision']:.4f}",
            Recall     = f"{r['recall']:.4f}",
            F1         = f"{r['f1-score']:.4f}",
            Support    = int(r["support"]),
        ))
    rows.append(dict(
        Juge      = name,
        Classe    = "── GLOBAL ──",
        Precision = f"{res['acc']:.4f}",
        Recall    = f"{res['bal_acc']:.4f}",
        F1        = f"{ece_scores[name]:.4f}",
        Support   = int(len(res["labels"])),
    ))

df_recap = pd.DataFrame(rows)
df_recap.columns = ["Juge","Classe","Precision","Recall (Acc/Bal)","F1","Support"]
print(df_recap.to_string(index=False))

# ── Figure tableau matplotlib ─────────────────────────────────────────────────
col_keys  = list(df_recap.columns)
cell_data = df_recap.values.tolist()

fig_h = max(4, len(cell_data) * 0.45 + 2)
fig, ax = plt.subplots(figsize=(16, fig_h))
ax.axis("off")
fig.suptitle("Tableau récapitulatif — Évaluation des Juges (test set)",
             fontsize=12, fontweight="bold")

tbl = ax.table(cellText=cell_data, colLabels=col_keys, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.55)

# Couleurs
judge_colors = {j: c for j, c in zip(judge_names, ["#dbe9f7","#fde9d9","#d5f0dc"])}
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    else:
        juge_val = cell_data[r-1][0]
        base_color = judge_colors.get(juge_val, "#f5f6fa")
        if "GLOBAL" in str(cell_data[r-1][1]):
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(color="black", fontweight="bold")
        else:
            cell.set_facecolor(base_color if r % 2 == 0 else "white")
            cell.set_text_props(color="black")

plt.tight_layout()
plt.savefig("summary_table_judges.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# ─── Cell 17 : Export CSV ────────────────────────────────────────────────────
cells.append(md("## 15. Export"))
cells.append(code("""df_recap.to_csv("evaluation_juges.csv", index=False)
print("Exporté → evaluation_juges.csv")

print("\\n" + "="*60)
print("  BILAN — Peut-on faire confiance aux juges ?")
print("="*60)
for name in judge_names:
    res = eval_results[name]
    ece = ece_scores[name]
    print(f"\\n  {name}")
    print(f"    Accuracy globale   : {res['acc']:.4f}")
    print(f"    Balanced Accuracy  : {res['bal_acc']:.4f}")
    print(f"    ECE (calibration)  : {ece:.4f}", end="  ")
    print("✓ bien calibré" if ece < 0.05 else ("⚠ à surveiller" if ece < 0.10 else "✗ mal calibré"))

print(f"\\n  Vote majoritaire   : {majority_acc:.4f}  (bal: {majority_bal:.4f})")
print("\\n  Légende ECE :")
print("    < 0.05  : confiance fiable")
print("    0.05-0.10 : légèrement sur/sous-confiant")
print("    > 0.10  : confiance à prendre avec précaution")
"""))

# ─── Assemblage ───────────────────────────────────────────────────────────────
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    }
}

OUTPUT_PATH = r"C:\Users\adril\Downloads\evaluation_juges.ipynb"
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook généré : {OUTPUT_PATH}")