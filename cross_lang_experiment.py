# -*- coding: utf-8 -*-
"""
cross_lang_experiment.py
========================
Research-grade Cross-Language Generalization Experiment for IndicFakeSpeech.

Setup (mirrors published deepfake-detection papers):
  TRAIN  : Hindi  (real + fake)  +  English (real + fake)
  TEST   : Marathi (real + fake) -- UNSEEN during training

Why this matters:
  If the model generalises to an unseen language, it proves the CNN learned
  *acoustic artefacts of synthesis* rather than *language-specific patterns*.
  This is a publishable research contribution.

Usage:
    python cross_lang_experiment.py
    python cross_lang_experiment.py --dataset dataset --epochs 30 --seed 42

Outputs:
    plots/cross_lang_confusion.png
    plots/cross_lang_curves.png
    cross_lang_results.txt          <- paste into your paper
"""

import os, sys, argparse, random, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from utils.preprocess         import preprocess_audio, TARGET_SR
from utils.feature_extraction import extract_features_for_cnn
from utils.augmentation       import spec_augment
from utils.model              import build_cnn_model, save_model_joblib, DEVICE

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# =============================================================================
# Argument parser
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-Language Deepfake Detection Experiment")
    p.add_argument("--dataset",    default="dataset",
                   help="Root of IndicFakeSpeech dataset")
    p.add_argument("--train-langs", nargs="+", default=["hindi", "english"],
                   help="Languages used for training (default: hindi english)")
    p.add_argument("--test-lang",   default="marathi",
                   help="Unseen language used ONLY for testing")
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--model-out",  default="model_crosslang.pkl")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


# =============================================================================
# Data loading
# =============================================================================

def collect_paths(dataset_root: Path, languages: list):
    """Return (real_paths, fake_paths) for given language list."""
    real_paths, fake_paths = [], []

    for lang in languages:
        r_dir = dataset_root / "real" / lang
        f_dir = dataset_root / "fake" / lang
        if r_dir.exists():
            real_paths += sorted(r_dir.glob("*.wav"))
        if f_dir.exists():
            fake_paths += sorted(f_dir.glob("*.wav"))

    return [str(p) for p in real_paths], [str(p) for p in fake_paths]


def extract_all(paths, label, desc):
    X, y = [], []
    for fp in tqdm(paths, desc=desc, ncols=70):
        try:
            wav  = preprocess_audio(fp)
            if wav is None:
                continue
            feat = extract_features_for_cnn(wav)
            X.append(feat); y.append(label)
        except Exception:
            pass
    return X, y


def build_xy(real_paths, fake_paths, split_name):
    print(f"\n[MFCC] {split_name}: {len(real_paths)} real + {len(fake_paths)} fake")
    n = min(len(real_paths), len(fake_paths))
    random.shuffle(real_paths); random.shuffle(fake_paths)
    real_paths, fake_paths = real_paths[:n], fake_paths[:n]

    Xr, yr = extract_all(real_paths, 0, f"{split_name}-real")
    Xf, yf = extract_all(fake_paths, 1, f"{split_name}-fake")
    X = np.array(Xr + Xf, dtype=np.float32)
    y = np.array(yr + yf, dtype=np.float32)
    print(f"  Shape: {X.shape} | Real={len(yr)}, Fake={len(yf)}")
    return X, y


# =============================================================================
# Training
# =============================================================================

def make_tensor(X, y, augment=False):
    """Convert numpy arrays to (channel-first) PyTorch tensors."""
    Xp = X[:, :, :, 0][:, np.newaxis, :, :].astype(np.float32)
    if augment:
        Xp = np.array([
            spec_augment(x[0])[np.newaxis] if random.random() < 0.6 else x
            for x in Xp
        ], dtype=np.float32)
    yp = y.astype(np.float32).reshape(-1, 1)
    return torch.from_numpy(Xp), torch.from_numpy(yp)


def train_model(X_tr, y_tr, X_val, y_val, epochs, batch_size):
    from sklearn.model_selection import train_test_split

    model, optim, _ = build_cnn_model(learning_rate=5e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    n0, n1  = float((y_tr == 0).sum()), float((y_tr == 1).sum())
    pw      = torch.tensor([n0 / max(n1, 1)]).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    model.classifier[-1] = nn.Identity()

    hist = {"tl": [], "vl": [], "ta": [], "va": []}
    best_val, pat = float("inf"), 0
    PATIENCE = 8

    Xv_t, yv_t = make_tensor(X_val, y_val)
    vl_ds = TensorDataset(Xv_t, yv_t)
    vl_loader = DataLoader(vl_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    for ep in range(1, epochs + 1):
        # Re-apply SpecAugment every epoch
        Xt_t, yt_t = make_tensor(X_tr, y_tr, augment=True)
        tr_ds = TensorDataset(Xt_t, yt_t)
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)

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

        model.eval()
        vl, vok, vn = 0.0, 0, 0
        with torch.no_grad():
            for Xb, yb in vl_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits  = model(Xb)
                vl     += loss_fn(logits, yb).item() * len(Xb)
                vok    += ((torch.sigmoid(logits) >= 0.5).float() == yb).sum().item()
                vn     += len(Xb)
        vl /= vn; va = vok / vn
        sched.step()

        hist["tl"].append(tl); hist["vl"].append(vl)
        hist["ta"].append(ta); hist["va"].append(va)
        print(f"  Epoch {ep:3d}/{epochs} | loss {tl:.4f} acc {ta*100:.1f}% "
              f"| val_loss {vl:.4f} val_acc {va*100:.1f}%")

        if vl < best_val - 1e-4:
            best_val = vl; pat = 0
            torch.save(model.state_dict(), "best_crosslang.pt")
        else:
            pat += 1
            if pat >= PATIENCE:
                print(f"  [INFO] Early stopping at epoch {ep}.")
                break

    model.load_state_dict(torch.load("best_crosslang.pt", map_location=DEVICE))
    model.eval()
    model.classifier[-1] = nn.Sigmoid()
    return model, hist


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model, X, y, split_name):
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score,
                                 confusion_matrix, classification_report)
    import seaborn as sns

    Xt, _ = make_tensor(X, y)
    with torch.no_grad():
        probs = model(Xt.to(DEVICE)).cpu().numpy().ravel()

    yp = (probs >= 0.5).astype(int)
    yt = y.astype(int).ravel()

    acc  = accuracy_score(yt, yp) * 100
    prec = precision_score(yt, yp, zero_division=0) * 100
    rec  = recall_score(yt, yp, zero_division=0) * 100
    f1   = f1_score(yt, yp, zero_division=0) * 100
    cm   = confusion_matrix(yt, yp)

    print(f"\n{'='*50}")
    print(f"  [{split_name.upper()}] Evaluation Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.2f}%")
    print(f"  Precision : {prec:.2f}%")
    print(f"  Recall    : {rec:.2f}%")
    print(f"  F1-Score  : {f1:.2f}%")
    print(f"{'='*50}")
    print(classification_report(yt, yp, target_names=["Real", "Fake"]))

    # Confusion matrix plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title(f"Confusion Matrix -- {split_name}")
    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    fname = f"plots/cross_lang_cm_{split_name.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  [Saved] {fname}")

    return {"split": split_name, "acc": acc, "prec": prec, "rec": rec, "f1": f1}


def plot_curves(hist, title="Cross-Language Training"):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hist["ta"], label="Train"); ax[0].plot(hist["va"], label="Val")
    ax[0].set_title("Accuracy"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(hist["tl"], label="Train"); ax[1].plot(hist["vl"], label="Val")
    ax[1].set_title("Loss");     ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/cross_lang_curves.png", dpi=150)
    plt.close()
    print("  [Saved] plots/cross_lang_curves.png")


def save_results_txt(results: list, train_langs, test_lang):
    """Write a results summary suitable for pasting into a paper."""
    lines = [
        "=" * 60,
        "  IndicFakeSpeech -- Cross-Language Experiment Results",
        "=" * 60,
        f"  Train languages : {', '.join(train_langs)}",
        f"  Test  language  : {test_lang}  (UNSEEN during training)",
        "",
        f"  {'Split':<25} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}",
        "  " + "-" * 55,
    ]
    for r in results:
        lines.append(
            f"  {r['split']:<25} {r['acc']:>7.2f}% {r['prec']:>7.2f}% "
            f"{r['rec']:>7.2f}% {r['f1']:>7.2f}%"
        )
    lines += [
        "  " + "-" * 55,
        "",
        "  Key Finding:",
    ]
    same_acc  = next((r["acc"] for r in results if "same" in r["split"].lower()), None)
    cross_acc = next((r["acc"] for r in results if "cross" in r["split"].lower()), None)
    if same_acc and cross_acc:
        drop = same_acc - cross_acc
        lines.append(f"  Accuracy drop (same -> cross-language): {drop:.2f}%")
        if cross_acc >= 70:
            lines.append("  => Model generalises to unseen language (publishable result).")
        else:
            lines.append("  => Consider more data or stronger augmentation.")
    lines += [
        "",
        "  BibTeX citation:",
        "  @inproceedings{indicfakespeech2024,",
        "    title={IndicFakeSpeech: Cross-Lingual Deepfake Audio Detection},",
        "    year={2024},",
        "    note={CNN trained on Hindi+English, tested on Marathi}",
        "  }",
        "=" * 60,
    ]

    out = "\n".join(lines)
    print("\n" + out)
    with open("cross_lang_results.txt", "w", encoding="utf-8") as f:
        f.write(out)
    print("\n  [Saved] cross_lang_results.txt  (paste into your paper!)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    SEED = args.seed
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    ROOT = Path(args.dataset)

    print("\n" + "=" * 60)
    print("  IndicFakeSpeech -- Cross-Language Experiment")
    print("=" * 60)
    print(f"  Train : {args.train_langs}")
    print(f"  Test  : {args.test_lang}  (UNSEEN)")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch_size}")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # Check test language exists
    test_real = list((ROOT / "real" / args.test_lang).glob("*.wav"))   if (ROOT / "real" / args.test_lang).exists() else []
    test_fake = list((ROOT / "fake" / args.test_lang).glob("*.wav"))   if (ROOT / "fake" / args.test_lang).exists() else []
    if not test_real or not test_fake:
        print(f"\n[ERROR] No data found for test language '{args.test_lang}'")
        print(f"  Run first:  python dataset_generator.py --langs mr --per-lang 150")
        sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────
    train_real, train_fake = collect_paths(ROOT, args.train_langs)
    test_real_p  = [str(p) for p in test_real]
    test_fake_p  = [str(p) for p in test_fake]

    print(f"\n  Train pool : {len(train_real)} real, {len(train_fake)} fake")
    print(f"  Test  pool : {len(test_real_p)} real, {len(test_fake_p)} fake")

    # ── Feature extraction ─────────────────────────────────────────────
    # Cache files
    cache_tr = Path("X_train_cl.npy")
    cache_te = Path("X_test_cl.npy")

    if cache_tr.exists() and cache_te.exists():
        print("\n[INFO] Loading cached MFCC features...")
        X_tr = np.load("X_train_cl.npy")
        y_tr = np.load("y_train_cl.npy")
        X_te = np.load("X_test_cl.npy")
        y_te = np.load("y_test_cl.npy")
        print(f"  Train: {X_tr.shape} | Test: {X_te.shape}")
    else:
        X_tr, y_tr = build_xy(train_real, train_fake, "TRAIN (hi+en)")
        X_te, y_te = build_xy(test_real_p, test_fake_p, "TEST  (mr)")
        np.save("X_train_cl.npy", X_tr); np.save("y_train_cl.npy", y_tr)
        np.save("X_test_cl.npy",  X_te); np.save("y_test_cl.npy",  y_te)
        print("[INFO] Features cached.")

    # ── Train/val split WITHIN training data ──────────────────────────
    from sklearn.model_selection import train_test_split
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_tr, y_tr, test_size=0.20, random_state=SEED, stratify=y_tr)
    print(f"\n  Train={len(X_tr2)}, Val={len(X_val)}, Test={len(X_te)}")

    # ── Train ──────────────────────────────────────────────────────────
    print(f"\n[TRAINING] on {args.train_langs} ...")
    model, hist = train_model(X_tr2, y_tr2, X_val, y_val,
                               args.epochs, args.batch_size)

    # ── Evaluate on same-distribution val set first ────────────────────
    results = []
    print("\n[EVAL] Same-language (hi+en validation set) ...")
    results.append(evaluate(model, X_val, y_val, "Same-Language (val)"))

    # ── Evaluate on UNSEEN language ────────────────────────────────────
    print(f"\n[EVAL] Cross-language ({args.test_lang} -- UNSEEN) ...")
    results.append(evaluate(model, X_te, y_te, f"Cross-Language ({args.test_lang})"))

    # ── Save model + plots + report ────────────────────────────────────
    plot_curves(hist)

    from sklearn.preprocessing import StandardScaler
    save_model_joblib(model, StandardScaler(), path=args.model_out)
    print(f"\n[Saved] model -> {args.model_out}")

    save_results_txt(results, args.train_langs, args.test_lang)
    print("\n[DONE] Experiment complete!")
    print("       Open cross_lang_results.txt for paper-ready table.")
