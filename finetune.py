# -*- coding: utf-8 -*-
"""
finetune.py
===========
Loads best_cnn.pt (epoch-3 checkpoint, val_acc~74%) and fine-tunes with:
  - LR = 3e-5  (10x lower than initial run)
  - ReduceLROnPlateau scheduler (patient, halves LR on stagnation)
  - Light SpecAugment (30% probability, smaller masks)
  - 80 epochs with patience=20
  - Threshold tuning + final model.pkl save
"""

import os, sys, time, warnings, random
import numpy as np
import matplotlib; matplotlib.use("Agg")
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
import joblib

sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore")

def p(msg): print(msg, flush=True)

# ── Seed ──────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
LR         = 3e-5      # 10× lower than initial
EPOCHS     = 80
BATCH      = 32
PATIENCE   = 20        # generous — let plateau scheduler do its job
CACHE_X    = Path("X_indic_v2.npy")
CACHE_Y    = Path("y_indic_v2.npy")
CKPT_IN    = Path("best_cnn.pt")   # start from previous best
CKPT_OUT   = Path("best_cnn.pt")   # overwrite when improved
MODEL_OUT  = Path("model.pkl")
PLOT_DIR   = Path("plots"); PLOT_DIR.mkdir(exist_ok=True)


# ── Architecture (must match run_training.py exactly) ─────────────────────────
class FullCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32),
            nn.ReLU(True), nn.MaxPool2d(2,2), nn.Dropout2d(0.25))
        self.block2 = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64),
            nn.ReLU(True), nn.MaxPool2d(2,2), nn.Dropout2d(0.25))
        self.block3 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128),
            nn.ReLU(True), nn.MaxPool2d(2,2), nn.Dropout2d(0.2))
        self.block4 = nn.Sequential(
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256),
            nn.ReLU(True), nn.AdaptiveAvgPool2d((1,1)))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512,256), nn.ReLU(True), nn.Dropout(0.4),
            nn.Linear(256,128), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(128,1))   # logits — sigmoid added after training

    def forward(self, x):
        return self.classifier(self.block4(self.block3(self.block2(self.block1(x)))))

    def add_sigmoid(self):
        self.classifier.add_module("sigmoid", nn.Sigmoid())

    def predict_proba_single(self, features):
        if features.ndim == 3: features = features[:,:,0]
        x = torch.tensor(features[np.newaxis,np.newaxis,:,:], dtype=torch.float32)
        with torch.no_grad():
            prob = float(torch.sigmoid(self.forward(x)).squeeze()
                         if not any(isinstance(m, nn.Sigmoid) for m in self.classifier)
                         else self.forward(x).squeeze())
        thr   = float(getattr(self, "decision_threshold", 0.5))
        label = "Fake" if prob >= thr else "Real"
        conf  = prob if prob >= thr else 1.0 - prob
        return {"label": label, "probability": round(prob,4),
                "confidence": round(conf*100,2), "threshold": round(thr,4)}


# ── Load data ─────────────────────────────────────────────────────────────────
p("\n" + "="*60)
p("  IndicFakeSpeech - FINE-TUNE from best_cnn.pt")
p(f"  LR={LR}  Epochs={EPOCHS}  Batch={BATCH}  Patience={PATIENCE}")
p("="*60)

p("\n[1/5] Loading cached features...")
X = np.load(str(CACHE_X))  # (2300,133,300,1)
y = np.load(str(CACHE_Y))
p(f"      X={X.shape}  Real={(y==0).sum()}  Fake={(y==1).sum()}")

p("\n[2/5] Preparing tensors (same 70/15/15 split as run_training.py)...")
Xp = X[:,:,:,0][:,np.newaxis,:,:].astype(np.float32)
yp = y.astype(np.float32).reshape(-1,1)

Xt,Xtmp,yt,ytmp = train_test_split(Xp,yp,test_size=0.30,random_state=SEED,stratify=yp)
Xv,Xte,yv,yte   = train_test_split(Xtmp,ytmp,test_size=0.50,random_state=SEED,stratify=ytmp)
p(f"      Train={len(Xt)}  Val={len(Xv)}  Test={len(Xte)}")


# ── Load checkpoint ───────────────────────────────────────────────────────────
p(f"\n[3/5] Loading checkpoint {CKPT_IN} ...")
model = FullCNN()
state = torch.load(str(CKPT_IN), map_location="cpu")
missing, unexpected = model.load_state_dict(state, strict=False)
p(f"      Missing={missing}  Unexpected={unexpected}")
p(f"      Checkpoint loaded — starting fine-tune from epoch-3 weights")

# Weighted BCE on logits
n0  = float((yt==0).sum()); n1 = float((yt==1).sum())
pos_w   = torch.tensor([n0/max(n1,1)])
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
# ReduceLROnPlateau: halve LR when val_loss doesn't improve for 5 epochs
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim, mode="min", factor=0.5, patience=5, min_lr=1e-7, verbose=False)


# ── DataLoader helpers ────────────────────────────────────────────────────────
def spec_augment_light(x):
    """Light SpecAugment: small freq+time masks, 30% chance."""
    x = x.copy()
    H, W = x.shape[1], x.shape[2]
    # Frequency mask (up to 15 rows)
    f = np.random.randint(1, min(16, H//4))
    f0 = np.random.randint(0, H - f)
    x[0, f0:f0+f, :] = 0
    # Time mask (up to 40 cols)
    t = np.random.randint(1, min(41, W//5))
    t0 = np.random.randint(0, W - t)
    x[0, :, t0:t0+t] = 0
    return x

def make_loader(Xa, ya, augment=False, shuffle=True):
    if augment:
        aug = np.array([spec_augment_light(xi) if np.random.random()<0.30 else xi
                        for xi in Xa], dtype=np.float32)
    else:
        aug = Xa
    ds = TensorDataset(torch.from_numpy(aug), torch.from_numpy(ya))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=0)

val_loader = make_loader(Xv, yv, augment=False, shuffle=False)


# ── Training loop ─────────────────────────────────────────────────────────────
p("\n[4/5] Fine-tuning...")
p(f"{'Epoch':>6}  {'TrLoss':>8}  {'TrAcc':>7}  "
  f"{'VlLoss':>8}  {'VlAcc':>7}  {'LR':>9}  {'Time':>6}")
p("-"*65)

best_val_loss = float("inf")
patience_cnt  = 0
history = {"tl":[],"vl":[],"ta":[],"va":[]}
start = time.time()

for epoch in range(1, EPOCHS+1):
    tr_loader = make_loader(Xt, yt, augment=True, shuffle=True)

    # Train
    model.train()
    tl,tok,tn = 0.,0,0
    for Xb,yb in tr_loader:
        optim.zero_grad()
        logits = model(Xb)
        loss   = loss_fn(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        tl  += loss.item()*len(Xb)
        tok += ((torch.sigmoid(logits)>=0.5).float()==yb).sum().item()
        tn  += len(Xb)
    tl/=tn; ta=tok/tn

    # Val
    model.eval()
    vl,vok,vn = 0.,0,0
    with torch.no_grad():
        for Xb,yb in val_loader:
            logits  = model(Xb)
            vl     += loss_fn(logits,yb).item()*len(Xb)
            vok    += ((torch.sigmoid(logits)>=0.5).float()==yb).sum().item()
            vn     += len(Xb)
    vl/=vn; va=vok/vn
    sched.step(vl)   # plateau scheduler

    history["tl"].append(tl); history["vl"].append(vl)
    history["ta"].append(ta); history["va"].append(va)

    cur_lr = optim.param_groups[0]["lr"]
    elapsed = time.time()-start
    marker = ""
    if vl < best_val_loss - 1e-5:
        best_val_loss = vl; patience_cnt = 0
        torch.save(model.state_dict(), str(CKPT_OUT))
        marker = " << BEST"
    else:
        patience_cnt += 1

    p(f"  {epoch:4d}   {tl:8.4f}   {ta*100:6.1f}%   "
      f"{vl:8.4f}   {va*100:6.1f}%   {cur_lr:.2e}  {elapsed:5.0f}s{marker}")

    if patience_cnt >= PATIENCE:
        p(f"\n  [Early Stop] patience={PATIENCE} hit at epoch {epoch}.")
        break


# ── Threshold tuning ──────────────────────────────────────────────────────────
p("\n[5/5] Restoring best weights + threshold tuning...")
model.load_state_dict(torch.load(str(CKPT_OUT), map_location="cpu"), strict=False)
model.eval()
model.add_sigmoid()

# Val probabilities
with torch.no_grad():
    vp = model(torch.from_numpy(Xv)).numpy().ravel()

best_thr, best_f1 = 0.5, -1.
for thr in np.arange(0.20, 0.81, 0.01):
    pred = (vp >= thr).astype(int)
    f = f1_score(yv.astype(int).ravel(), pred, zero_division=0)
    if f > best_f1: best_f1=f; best_thr=float(thr)
p(f"  Threshold={best_thr:.2f}  Val-F1={best_f1:.4f}")

# Test
with torch.no_grad():
    tp = model(torch.from_numpy(Xte)).numpy().ravel()
y_pred = (tp >= best_thr).astype(int)
y_true = yte.astype(int).ravel()

acc  = accuracy_score(y_true,y_pred)
prec = precision_score(y_true,y_pred,zero_division=0)
rec  = recall_score(y_true,y_pred,zero_division=0)
f1   = f1_score(y_true,y_pred,zero_division=0)

p("\n" + "="*60)
p("  TEST SET RESULTS")
p(f"  Accuracy  : {acc*100:.2f}%")
p(f"  Precision : {prec*100:.2f}%")
p(f"  Recall    : {rec*100:.2f}%")
p(f"  F1-Score  : {f1*100:.2f}%")
p(f"  Threshold : {best_thr:.2f}")
p("="*60)
p(classification_report(y_true,y_pred,target_names=["Real","Fake"]))

# Confusion matrix
try:
    import seaborn as sns
    cm = confusion_matrix(y_true,y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
                xticklabels=["Real","Fake"],yticklabels=["Real","Fake"])
    plt.title("Confusion Matrix - Fine-tuned FullCNN"); plt.tight_layout()
    plt.savefig(str(PLOT_DIR/"confusion_matrix.png"),dpi=150); plt.close()
    p("[INFO] Saved plots/confusion_matrix.png")
except ImportError: pass

# Training curves
fig,ax = plt.subplots(1,2,figsize=(12,4))
ax[0].plot(history["ta"],label="Train"); ax[0].plot(history["va"],label="Val")
ax[0].set_title("Accuracy"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(history["tl"],label="Train"); ax[1].plot(history["vl"],label="Val")
ax[1].set_title("Loss"); ax[1].legend(); ax[1].grid(alpha=0.3)
plt.suptitle("Fine-tune Curves"); plt.tight_layout()
plt.savefig(str(PLOT_DIR/"training_curves.png"),dpi=150); plt.close()
p("[INFO] Saved plots/training_curves.png")

# Save model.pkl
model.decision_threshold = best_thr
payload = {
    "model_state":     model.cpu().state_dict(),
    "scaler":          StandardScaler(),
    "threshold":       best_thr,
    "feature_version": "mfcc_delta_spectral_v2",
    "model_class":     "FullCNN",
}
joblib.dump(payload, str(MODEL_OUT))

p(f"\n[SAVED] model.pkl  threshold={best_thr:.2f}")
p("\n" + "="*60)
p(f"  DONE!  Test Accuracy: {acc*100:.2f}%")
p(f"  Run now: python app.py")
p("="*60+"\n")
