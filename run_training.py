# -*- coding: utf-8 -*-
"""
run_training.py
===============
Fast CNN training on 2300 cached samples.
- Loads X_indic_v2.npy (already extracted - skips feature step)
- Trains 50 epochs with early stopping
- Saves model.pkl  +  best_cnn.pt
"""

import os, sys, time, warnings, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from utils.model import DeepfakeAudioCNN, build_cnn_model, save_model_joblib, DEVICE

warnings.filterwarnings("ignore")

def p(msg):
    print(msg, flush=True)

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Config ─────────────────────────────────────────────────────────────────────
EPOCHS     = 60
BATCH      = 32
LR         = 3e-4
PATIENCE   = 15
CACHE_X    = Path("X_indic_v2.npy")
CACHE_Y    = Path("y_indic_v2.npy")
MODEL_OUT  = Path("model.pkl")
BEST_PT    = Path("best_cnn.pt")
PLOT_DIR   = Path("plots"); PLOT_DIR.mkdir(exist_ok=True)

# ── Load cache ─────────────────────────────────────────────────────────────────
p("\n" + "="*60)
p("  IndicFakeSpeech - FULL DATASET TRAINING")
p(f"  Device : {DEVICE}")
p(f"  Cache  : {CACHE_X.name}  Epochs={EPOCHS}  Batch={BATCH}  LR={LR}")
p("="*60)

p(f"\n[1/5] Loading feature cache...")
X = np.load(str(CACHE_X))   # (2300, 133, 300, 1)
y = np.load(str(CACHE_Y))   # (2300,)
p(f"      X={X.shape}   Real={(y==0).sum()}   Fake={(y==1).sum()}")

# ── Reshape: (N,133,300,1) → (N,1,133,300) ────────────────────────────────────
p("\n[2/5] Preparing tensors...")
Xp = X[:, :, :, 0][:, np.newaxis, :, :].astype(np.float32)
yp = y.astype(np.float32).reshape(-1, 1)

# 70/15/15 stratified split
Xt, Xtmp, yt, ytmp = train_test_split(Xp, yp, test_size=0.30,
                                       random_state=SEED, stratify=yp)
Xv, Xte, yv, yte   = train_test_split(Xtmp, ytmp, test_size=0.50,
                                       random_state=SEED, stratify=ytmp)
p(f"      Train={len(Xt)}  Val={len(Xv)}  Test={len(Xte)}")

# ── Build model ────────────────────────────────────────────────────────────────
p("\n[3/5] Building CNN model...")

# CNN that matches (1, 133, 300) input
class FullCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(2,2), nn.Dropout2d(0.25))
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(2,2), nn.Dropout2d(0.25))
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.MaxPool2d(2,2), nn.Dropout2d(0.2))
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1,1)))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, 1))  # no sigmoid – BCEWithLogitsLoss

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(x)

    def add_sigmoid(self):
        self.classifier.add_module("sigmoid", nn.Sigmoid())

model = FullCNN().to(DEVICE)
total_params = sum(p_.numel() for p_ in model.parameters())
p(f"      Parameters: {total_params:,}")

optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

# Weighted loss
n0  = float((yt == 0).sum())
n1  = float((yt == 1).sum())
pos_w = torch.tensor([n0 / max(n1,1)]).to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

# DataLoaders
def make_loader(Xa, ya, shuffle=True):
    ds = TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle,
                      num_workers=0, pin_memory=False)

val_loader = make_loader(Xv, yv, shuffle=False)

# ── Training loop ──────────────────────────────────────────────────────────────
p("\n[4/5] Training...")
p(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  "
  f"{'Val Loss':>10}  {'Val Acc':>10}  {'Time':>6}")
p("-"*60)

best_val_loss = float("inf")
patience_cnt  = 0
history = {"tl":[], "vl":[], "ta":[], "va":[]}
start = time.time()

for epoch in range(1, EPOCHS+1):
    # ── SpecAugment on train ─────────────────────────────────────
    Xa_aug = []
    for xi in Xt:
        xi_aug = xi.copy()
        if np.random.random() < 0.5:
            # frequency mask
            f0 = np.random.randint(0, max(1, xi_aug.shape[1] // 4))
            f  = np.random.randint(0, max(1, xi_aug.shape[1] // 4))
            xi_aug[0, f0:f0+f, :] = 0
            # time mask
            t0 = np.random.randint(0, max(1, xi_aug.shape[2] // 5))
            t  = np.random.randint(0, max(1, xi_aug.shape[2] // 5))
            xi_aug[0, :, t0:t0+t] = 0
        Xa_aug.append(xi_aug)
    Xa_aug = np.array(Xa_aug, dtype=np.float32)

    tr_loader = make_loader(Xa_aug, yt, shuffle=True)

    # Train
    model.train()
    tl, tok, tn = 0.0, 0, 0
    for Xb, yb in tr_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optim.zero_grad()
        logits = model(Xb)
        loss   = loss_fn(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        tl  += loss.item() * len(Xb)
        tok += ((torch.sigmoid(logits) >= 0.5).float() == yb).sum().item()
        tn  += len(Xb)
    tl /= tn; ta = tok / tn

    # Val
    model.eval()
    vl, vok, vn = 0.0, 0, 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            logits  = model(Xb)
            vl     += loss_fn(logits, yb).item() * len(Xb)
            vok    += ((torch.sigmoid(logits) >= 0.5).float() == yb).sum().item()
            vn     += len(Xb)
    vl /= vn; va = vok / vn
    sched.step()

    history["tl"].append(tl); history["vl"].append(vl)
    history["ta"].append(ta); history["va"].append(va)

    elapsed = time.time() - start
    marker = ""
    if vl < best_val_loss - 1e-4:
        best_val_loss = vl; patience_cnt = 0
        torch.save(model.state_dict(), str(BEST_PT))
        marker = " << BEST"
    else:
        patience_cnt += 1

    p(f"  {epoch:4d}   {tl:10.4f}   {ta*100:9.1f}%   "
      f"{vl:10.4f}   {va*100:9.1f}%  {elapsed:5.0f}s{marker}", )

    if patience_cnt >= PATIENCE:
        p(f"\n  [Early Stop] patience={PATIENCE} reached at epoch {epoch}.")
        break

# ── Threshold tuning ──────────────────────────────────────────────────────────
p("\n[5/5] Threshold tuning + final evaluation...")
model.load_state_dict(torch.load(str(BEST_PT), map_location=DEVICE))
model.eval()

# Add sigmoid for inference
model.add_sigmoid()

# Val probabilities for threshold search
with torch.no_grad():
    val_t = torch.from_numpy(Xv).to(DEVICE)
    val_probs = model(val_t).cpu().numpy().ravel()

best_thr, best_f1 = 0.5, -1.0
for thr in np.arange(0.20, 0.81, 0.01):
    pred = (val_probs >= thr).astype(int)
    f = f1_score(yv.astype(int).ravel(), pred, zero_division=0)
    if f > best_f1:
        best_f1 = f; best_thr = float(thr)

p(f"  Tuned threshold: {best_thr:.2f}  (val F1={best_f1:.4f})")

# Test evaluation
with torch.no_grad():
    te_t   = torch.from_numpy(Xte).to(DEVICE)
    te_probs = model(te_t).cpu().numpy().ravel()

y_pred = (te_probs >= best_thr).astype(int)
y_true = yte.astype(int).ravel()

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec  = recall_score(y_true, y_pred, zero_division=0)
f1   = f1_score(y_true, y_pred, zero_division=0)

p("\n" + "="*60)
p("  TEST SET RESULTS")
p(f"  Accuracy  : {acc*100:.2f}%")
p(f"  Precision : {prec*100:.2f}%")
p(f"  Recall    : {rec*100:.2f}%")
p(f"  F1-Score  : {f1*100:.2f}%")
p(f"  Threshold : {best_thr:.2f}")
p("="*60)
p(classification_report(y_true, y_pred, target_names=["Real","Fake"]))

# Confusion matrix
try:
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real","Fake"], yticklabels=["Real","Fake"])
    plt.title("Confusion Matrix - Full 2300 Samples"); plt.tight_layout()
    plt.savefig(str(PLOT_DIR/"confusion_matrix.png"), dpi=150); plt.close()
    p("[INFO] Saved plots/confusion_matrix.png")
except ImportError:
    pass

# Training curves
fig, ax = plt.subplots(1,2, figsize=(12,4))
ax[0].plot(history["ta"], label="Train"); ax[0].plot(history["va"], label="Val")
ax[0].set_title("Accuracy"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(history["tl"], label="Train"); ax[1].plot(history["vl"], label="Val")
ax[1].set_title("Loss"); ax[1].legend(); ax[1].grid(alpha=0.3)
plt.suptitle("IndicFakeSpeech Full Dataset"); plt.tight_layout()
plt.savefig(str(PLOT_DIR/"training_curves.png"), dpi=150); plt.close()
p("[INFO] Saved plots/training_curves.png")

# ── Save model ─────────────────────────────────────────────────────────────────
model.decision_threshold = best_thr
model.feature_version    = "mfcc_delta_spectral_v2"

payload = {
    "model_state":    model.cpu().state_dict(),
    "scaler":         StandardScaler(),
    "threshold":      best_thr,
    "feature_version":"mfcc_delta_spectral_v2",
    "model_class":    "FullCNN",
}
import joblib
joblib.dump(payload, str(MODEL_OUT))
p(f"\n[SAVED] model.pkl  (threshold={best_thr:.2f})")

p("\n" + "="*60)
p(f"  DONE!  Test Accuracy: {acc*100:.2f}%")
p(f"  Model : model.pkl")
p(f"  Run   : python app.py")
p("="*60 + "\n")
