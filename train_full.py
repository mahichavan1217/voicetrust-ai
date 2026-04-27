# -*- coding: utf-8 -*-
"""
train_full.py
=============
Full IndicFakeSpeech Dataset Training Script
- Trains on ALL 2300 clips (1150 real + 1150 fake)
- English + Hindi + Marathi multilingual support
- SpecAugment data augmentation
- Cosine LR schedule + early stopping
- Saves model.pkl for use with app.py

Usage:
    python train_full.py
    python train_full.py --epochs 50 --batch-size 32
"""

import os, sys, argparse, warnings, random, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from utils.preprocess         import preprocess_audio, TARGET_SR
from utils.feature_extraction import (
    FEATURE_ROWS,
    FEATURE_VERSION,
    MAX_FRAMES,
    extract_features_for_cnn,
)
from utils.augmentation       import spec_augment
from utils.model              import (DeepfakeAudioCNN, build_cnn_model,
                                      save_model_joblib, DEVICE)

warnings.filterwarnings("ignore")

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
DATASET_DIR = PROJECT_DIR / "dataset"
CACHE_X     = PROJECT_DIR / "X_indic_v2.npy"
CACHE_Y     = PROJECT_DIR / "y_indic_v2.npy"
MODEL_OUT   = PROJECT_DIR / "model.pkl"
BEST_PT     = PROJECT_DIR / "best_cnn.pt"
PLOT_DIR    = PROJECT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Train CNN on full IndicFakeSpeech dataset")
    p.add_argument("--dataset",    default="dataset",   help="Dataset root dir")
    p.add_argument("--model-out",  default="model.pkl", help="Output model path")
    p.add_argument("--epochs",     type=int, default=40, help="Max training epochs")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p.add_argument("--lr",         type=float, default=5e-4, help="Learning rate")
    p.add_argument("--no-cache",   action="store_true",  help="Force re-extract features")
    p.add_argument("--no-augment", action="store_true",  help="Disable SpecAugment")
    p.add_argument("--patience",   type=int, default=12, help="Early stopping patience")
    return p.parse_args()


# =============================================================================
# STEP 1 – Load all audio files from dataset/
# =============================================================================

def scan_dataset(dataset_root: Path):
    """
    Scan dataset/real/ and dataset/fake/ recursively for WAV files.
    Supports multilingual structure: real/english/, real/hindi/, real/marathi/
    """
    real_dir = dataset_root / "real"
    fake_dir = dataset_root / "fake"

    if not real_dir.exists() or not fake_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_root}.\n"
            "Expected: dataset/real/ and dataset/fake/ directories.\n"
            "Run dataset_generator.py first."
        )

    real_paths = sorted(real_dir.rglob("*.wav"))
    fake_paths = sorted(fake_dir.rglob("*.wav"))

    print(f"\n{'='*60}")
    print(f"  Dataset Scan: {dataset_root.resolve()}")
    print(f"{'='*60}")
    print(f"  Real clips found : {len(real_paths)}")
    print(f"  Fake clips found : {len(fake_paths)}")

    # Show language breakdown
    for lang in ["english", "hindi", "marathi"]:
        r = len(list((real_dir / lang).glob("*.wav"))) if (real_dir / lang).exists() else 0
        f = len(list((fake_dir / lang).glob("*.wav"))) if (fake_dir / lang).exists() else 0
        if r + f > 0:
            print(f"  {lang.capitalize():10} : Real={r}, Fake={f}")
    print(f"{'='*60}\n")

    return [str(p) for p in real_paths], [str(p) for p in fake_paths]


# =============================================================================
# STEP 2 – Feature Extraction (MFCC)
# =============================================================================

def extract_features(paths: list, label: int, desc: str):
    """Extract MFCC features for a list of audio files."""
    X, y, failed = [], [], 0
    for fp in tqdm(paths, desc=desc, ncols=80):
        try:
            wav = preprocess_audio(fp)
            if wav is None:
                failed += 1
                continue
            feat = extract_features_for_cnn(wav)
            X.append(feat)
            y.append(label)
        except Exception as e:
            failed += 1
    if failed > 0:
        print(f"  [WARN] {failed} files skipped during feature extraction")
    return X, y


def build_feature_dataset(real_paths, fake_paths, force_rebuild=False):
    """Extract features or load from cache."""
    if not force_rebuild and CACHE_X.exists() and CACHE_Y.exists():
        print(f"[INFO] Loading cached features from {CACHE_X.name} & {CACHE_Y.name}")
        X = np.load(str(CACHE_X))
        y = np.load(str(CACHE_Y))
        r = int((y == 0).sum())
        f = int((y == 1).sum())
        print(f"[INFO] Cache: X={X.shape}, Real={r}, Fake={f}")
        expected_shape = (FEATURE_ROWS, MAX_FRAMES, 1)
        shape_ok = X.ndim == 4 and X.shape[1:] == expected_shape
        count_ok = r >= len(real_paths) * 0.8 and f >= len(fake_paths) * 0.8
        if shape_ok and count_ok:
            return X, y
        print("[WARN] Cache is stale or incompatible. Re-extracting...")

    print(f"\n{'='*60}")
    print(f"  STEP 2 - Extracting Features ({FEATURE_VERSION})")
    print(f"{'='*60}")

    Xr, yr = extract_features(real_paths, label=0, desc="Real  features")
    Xf, yf = extract_features(fake_paths, label=1, desc="Fake  features")

    X = np.array(Xr + Xf, dtype=np.float32)
    y = np.array(yr + yf, dtype=np.float32)

    np.save(str(CACHE_X), X)
    np.save(str(CACHE_Y), y)
    print(f"[INFO] Features cached -> {CACHE_X.name} & {CACHE_Y.name}")
    print(f"[INFO] Dataset: X={X.shape}, Real={len(yr)}, Fake={len(yf)}")
    return X, y


# =============================================================================
# STEP 3 – Train CNN
# =============================================================================

def make_loader(Xa, ya, batch_size, shuffle=True, augment=False):
    """Create DataLoader, optionally applying SpecAugment."""
    if augment:
        aug_X = []
        for x in Xa:
            if np.random.random() < 0.6:
                aug_X.append(spec_augment(x[0])[np.newaxis])
            else:
                aug_X.append(x)
        Xa = np.array(aug_X, dtype=np.float32)
    ds = TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


def predict_probabilities(model, Xarray):
    """Return model probabilities for a numpy tensor shaped (N, 1, H, W)."""
    loader = DataLoader(torch.from_numpy(Xarray), batch_size=128, shuffle=False)
    probs = []
    model.eval()
    with torch.no_grad():
        for Xb in loader:
            Xb = Xb.to(DEVICE)
            probs.append(model(Xb).cpu().numpy().ravel())
    return np.concatenate(probs)


def find_best_threshold(y_true, probs):
    """Pick a validation threshold that maximizes fake-class F1."""
    y_true = y_true.astype(int).ravel()
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.arange(0.25, 0.76, 0.01):
        pred = (probs >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


def train_model(X, y, epochs, batch_size, lr, use_augment=True, patience=12):
    print(f"\n{'='*60}")
    print(f"  STEP 3 – Training CNN  (Device: {DEVICE})")
    print(f"  Epochs={epochs}, Batch={batch_size}, LR={lr}")
    print(f"  SpecAugment: {'ON' if use_augment else 'OFF'}")
    print(f"  Feature rows: {FEATURE_ROWS} | Version: {FEATURE_VERSION}")
    print(f"{'='*60}\n")

    # Reshape: (N, 40, 300, 1) -> (N, 1, 40, 300)
    Xp = X[:, :, :, 0][:, np.newaxis, :, :].astype(np.float32)
    yp = y.astype(np.float32).reshape(-1, 1)

    # 70 / 15 / 15 split — more training data
    Xt, Xtmp, yt, ytmp = train_test_split(Xp, yp, test_size=0.30,
                                           random_state=SEED, stratify=yp)
    Xv, Xte, yv, yte   = train_test_split(Xtmp, ytmp, test_size=0.50,
                                           random_state=SEED, stratify=ytmp)

    print(f"[INFO] Split -> Train={len(Xt)}, Val={len(Xv)}, Test={len(Xte)}")

    # Val loader (fixed, no augment)
    val_loader = make_loader(Xv, yv, batch_size, shuffle=False, augment=False)

    # Model
    model, optim, _ = build_cnn_model(learning_rate=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    # Weighted BCE to handle any class imbalance
    n0 = float((yt == 0).sum())
    n1 = float((yt == 1).sum())
    pos_w = torch.tensor([n0 / max(n1, 1)]).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    # Remove sigmoid from last layer (BCEWithLogitsLoss needs raw logits)
    model.classifier[-1] = nn.Identity()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_cnt  = 0
    PATIENCE      = patience

    start = time.time()
    for epoch in range(1, epochs + 1):
        # Rebuild train loader each epoch so SpecAugment masks vary
        tr_loader = make_loader(Xt, yt, batch_size, shuffle=True, augment=use_augment)

        # —— Train ————————————————————————————————————————————
        model.train()
        tr_loss, tr_ok, tr_n = 0.0, 0, 0
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            logits = model(Xb)
            loss   = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tr_loss += loss.item() * len(Xb)
            tr_ok   += ((torch.sigmoid(logits) >= 0.5).float() == yb).sum().item()
            tr_n    += len(Xb)
        tr_loss /= tr_n
        tr_acc   = tr_ok / tr_n

        # —— Validate ——————————————————————————————————————————
        model.eval()
        vl, vok, vn = 0.0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits  = model(Xb)
                vl     += loss_fn(logits, yb).item() * len(Xb)
                vok    += ((torch.sigmoid(logits) >= 0.5).float() == yb).sum().item()
                vn     += len(Xb)
        vl  /= vn
        va   = vok / vn
        sched.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va)

        elapsed = time.time() - start
        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"loss {tr_loss:.4f}  acc {tr_acc*100:.1f}% | "
              f"val_loss {vl:.4f}  val_acc {va*100:.1f}%  "
              f"[{elapsed:.0f}s]")

        # Early stopping & checkpoint
        if vl < best_val_loss - 1e-4:
            best_val_loss = vl
            patience_cnt  = 0
            torch.save(model.state_dict(), str(BEST_PT))
            print(f"           ^ [SAVED] best model (val_loss={vl:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n[INFO] Early stopping triggered at epoch {epoch}.")
                break

    # Restore best weights
    model.load_state_dict(torch.load(str(BEST_PT), map_location=DEVICE))
    model.eval()
    model.classifier[-1] = nn.Sigmoid()   # restore sigmoid for inference
    val_probs = predict_probabilities(model, Xv)
    threshold, val_f1 = find_best_threshold(yv, val_probs)
    model.decision_threshold = threshold
    print(f"\n[INFO] Training done. Best val_loss={best_val_loss:.4f}")
    print(f"[INFO] Tuned threshold={threshold:.2f} (val F1={val_f1:.4f})")
    return model, history, Xte, yte, threshold


# =============================================================================
# STEP 4 – Evaluate
# =============================================================================

def evaluate_model(model, Xte, yte, threshold=0.5):
    print(f"\n{'='*60}")
    print("  STEP 4 – Test Set Evaluation")
    print(f"{'='*60}")

    model.eval()
    Xp = torch.from_numpy(Xte).to(DEVICE)
    with torch.no_grad():
        probs = model(Xp).cpu().numpy().ravel()
    y_pred = (probs >= threshold).astype(int)
    y_true = yte.astype(int).ravel()

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n  {'-'*40}")
    print(f"  Accuracy  : {acc  * 100:.2f} %")
    print(f"  Precision : {prec * 100:.2f} %")
    print(f"  Recall    : {rec  * 100:.2f} %")
    print(f"  F1-Score  : {f1   * 100:.2f} %")
    print(f"  Threshold : {threshold:.2f}")
    print(f"  {'-'*40}")
    print()
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

    # Confusion matrix plot
    try:
        import seaborn as sns
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Real", "Fake"],
                    yticklabels=["Real", "Fake"])
        plt.title("Confusion Matrix - IndicFakeSpeech Full Dataset")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(str(PLOT_DIR / "confusion_matrix.png"), dpi=150)
        plt.close()
        print("[INFO] Saved plots/confusion_matrix.png")
    except ImportError:
        print("[WARN] seaborn not installed - skipping confusion matrix plot")

    return acc


def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history["train_acc"],  label="Train", linewidth=2)
    ax[0].plot(history["val_acc"],    label="Val",   linewidth=2)
    ax[0].set_title("Accuracy", fontsize=13)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(history["train_loss"], label="Train", linewidth=2)
    ax[1].plot(history["val_loss"],   label="Val",   linewidth=2)
    ax[1].set_title("Loss", fontsize=13)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    plt.suptitle("IndicFakeSpeech - Full Dataset Training", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(PLOT_DIR / "training_curves.png"), dpi=150)
    plt.close()
    print("[INFO] Saved plots/training_curves.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    args = parse_args()

    print("\n" + "=" * 60)
    print("  IndicFakeSpeech - FULL DATASET TRAINING")
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {args.epochs} | Batch : {args.batch_size} | LR : {args.lr}")
    print("=" * 60)

    dataset_root = Path(args.dataset)

    # Step 1: Scan dataset
    real_paths, fake_paths = scan_dataset(dataset_root)

    if len(real_paths) == 0 or len(fake_paths) == 0:
        print("[ERROR] No audio files found in dataset/real/ or dataset/fake/")
        print("        Please run dataset_generator.py first.")
        sys.exit(1)

    # Step 2: Feature extraction (with caching)
    X, y = build_feature_dataset(real_paths, fake_paths, force_rebuild=args.no_cache)

    real_count = int((y == 0).sum())
    fake_count = int((y == 1).sum())
    print(f"\n[INFO] Total samples: Real={real_count}, Fake={fake_count}, Total={len(y)}")

    # Step 3: Train
    model, history, X_test, y_test, threshold = train_model(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_augment=not args.no_augment,
        patience=args.patience
    )

    # Step 4: Evaluate
    accuracy = evaluate_model(model, X_test, y_test, threshold=threshold)
    plot_history(history)

    # Step 5: Save model
    scaler = StandardScaler()   # placeholder (features not scaled for CNN)
    save_model_joblib(
        model,
        scaler,
        path=args.model_out,
        threshold=threshold,
        feature_version=FEATURE_VERSION,
    )

    print("\n" + "=" * 60)
    print(f"  [DONE] Training Complete!")
    print(f"  [DONE] Test Accuracy : {accuracy*100:.2f}%")
    print(f"  [DONE] Model saved   : {args.model_out}")
    print(f"  [DONE] Plots saved   : plots/")
    print(f"\n  Now run:  python app.py")
    print("=" * 60 + "\n")
