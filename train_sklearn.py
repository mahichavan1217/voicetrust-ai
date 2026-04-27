# -*- coding: utf-8 -*-
"""
Train a fast sklearn baseline on cached IndicFakeSpeech features.

This uses the existing X_indic_v2.npy / y_indic_v2.npy cache, flattens each
133x300 feature map, reduces it with TruncatedSVD, then trains LogisticRegression.
On the current 2300-clip cache this is much faster than CPU CNN training and
gives a stronger holdout score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils.feature_extraction import (
    FEATURE_VERSION,
    extract_features_from_file,
    flatten_feature_map,
)


SEED = 42
PROJECT_DIR = Path(__file__).parent
PLOT_DIR = PROJECT_DIR / "plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SVD + LogisticRegression on cached audio features"
    )
    parser.add_argument("--cache-x", default="X_indic_v2.npy")
    parser.add_argument("--cache-y", default="y_indic_v2.npy")
    parser.add_argument("--model-out", default="model.pkl")
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-refit", action="store_true",
                        help="Save the train-only model instead of refitting on train+val")
    parser.add_argument("--fit-full", action="store_true",
                        help="After holdout evaluation, save a final model fit on all samples")
    parser.add_argument("--extra-real", nargs="*", default=[],
                        help="Extra audio files to append as Real before training")
    parser.add_argument("--extra-fake", nargs="*", default=[],
                        help="Extra audio files to append as Fake before training")
    return parser.parse_args()


def build_model(components: int, seed: int):
    return make_pipeline(
        TruncatedSVD(n_components=components, random_state=seed),
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
            random_state=seed,
        ),
    )


def tune_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.20, 0.81, 0.01):
        pred = (probs >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(score)
    return best_threshold, best_f1


def evaluate(model, X, y, threshold: float) -> dict:
    probs = model.predict_proba(X)[:, 1]
    pred = (probs >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "report": classification_report(
            y, pred, target_names=["Real", "Fake"], zero_division=0
        ),
    }


def save_confusion_matrix(cm: list[list[int]]) -> None:
    PLOT_DIR.mkdir(exist_ok=True)
    matrix = np.array(cm)
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix - SVD Logistic Model")
    plt.xticks([0, 1], ["Real", "Fake"])
    plt.yticks([0, 1], ["Real", "Fake"])
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, str(matrix[row, col]),
                     ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def load_extra_audio(paths: list[str], label: int) -> tuple[np.ndarray | None, np.ndarray]:
    vectors = []
    labels = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_DIR / path
        feature = extract_features_from_file(str(path))
        if feature is None:
            print(f"[WARN] Skipped unreadable extra file: {path}")
            continue
        vectors.append(flatten_feature_map(feature)[0])
        labels.append(label)

    if not vectors:
        return None, np.array([], dtype=int)
    return np.array(vectors, dtype=np.float32), np.array(labels, dtype=int)


def main() -> None:
    args = parse_args()

    X_path = PROJECT_DIR / args.cache_x
    y_path = PROJECT_DIR / args.cache_y
    model_path = PROJECT_DIR / args.model_out

    print("\n" + "=" * 64)
    print("  IndicFakeSpeech - SVD + LogisticRegression Training")
    print(f"  Cache      : {X_path.name}")
    print(f"  Components : {args.components}")
    print("=" * 64)

    X_cache = np.load(X_path, mmap_mode="r")
    y = np.load(y_path).astype(int)
    X = flatten_feature_map(X_cache)

    extra_real_X, extra_real_y = load_extra_audio(args.extra_real, 0)
    extra_fake_X, extra_fake_y = load_extra_audio(args.extra_fake, 1)
    extra_blocks = []
    extra_labels = []
    if extra_real_X is not None:
        extra_blocks.append(extra_real_X)
        extra_labels.append(extra_real_y)
    if extra_fake_X is not None:
        extra_blocks.append(extra_fake_X)
        extra_labels.append(extra_fake_y)
    if extra_blocks:
        X = np.concatenate([X, *extra_blocks], axis=0)
        y = np.concatenate([y, *extra_labels], axis=0)
        print(
            f"[INFO] Added extras -> Real={len(extra_real_y)}  Fake={len(extra_fake_y)}"
        )

    if args.components >= min(X.shape):
        raise ValueError(
            f"--components must be less than min(samples, features)={min(X.shape)}"
        )

    print(f"[INFO] X={X.shape}  Real={(y == 0).sum()}  Fake={(y == 1).sum()}")

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=args.seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=args.seed, stratify=y_tmp
    )
    print(
        f"[INFO] Split -> Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}"
    )

    selection_model = build_model(args.components, args.seed)
    print("[INFO] Fitting selection model...")
    selection_model.fit(X_train, y_train)

    val_probs = selection_model.predict_proba(X_val)[:, 1]
    threshold, val_f1 = tune_threshold(y_val, val_probs)
    print(f"[INFO] Tuned threshold={threshold:.2f}  val_f1={val_f1:.4f}")

    if args.no_refit:
        evaluation_model = selection_model
        fit_scope = "train"
    else:
        evaluation_model = build_model(args.components, args.seed)
        print("[INFO] Refitting evaluation model on train+val...")
        evaluation_model.fit(
            np.concatenate([X_train, X_val], axis=0),
            np.concatenate([y_train, y_val], axis=0),
        )
        fit_scope = "train+val"

    metrics = evaluate(evaluation_model, X_test, y_test, threshold)
    save_confusion_matrix(metrics["confusion_matrix"])

    print("\n" + "=" * 64)
    print("  HOLDOUT TEST RESULTS")
    print(f"  Accuracy  : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision : {metrics['precision'] * 100:.2f}%")
    print(f"  Recall    : {metrics['recall'] * 100:.2f}%")
    print(f"  F1-Score  : {metrics['f1'] * 100:.2f}%")
    print(f"  Threshold : {threshold:.2f}")
    print("=" * 64)
    print(metrics["report"])

    final_model = evaluation_model
    if args.fit_full:
        final_model = build_model(args.components, args.seed)
        print("[INFO] Fitting deployment model on all available samples...")
        final_model.fit(X, y)
        fit_scope = "all"

    payload = {
        "model_type": "sklearn_svd_logreg",
        "model_class": "SklearnSVDLogReg",
        "sklearn_pipeline": final_model,
        "threshold": threshold,
        "feature_version": FEATURE_VERSION,
        "vectorizer": "flatten_feature_map_v1",
        "input_shape": tuple(X_cache.shape[1:]),
        "fit_scope": fit_scope,
        "extra_real": args.extra_real,
        "extra_fake": args.extra_fake,
        "metrics": {
            key: value
            for key, value in metrics.items()
            if key != "report"
        },
        "classification_report": metrics["report"],
    }
    joblib.dump(payload, model_path)

    metrics_path = PROJECT_DIR / "sklearn_results.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(payload["metrics"] | {"threshold": threshold}, handle, indent=2)

    print(f"[SAVED] Model   -> {model_path.name}")
    print(f"[SAVED] Metrics -> {metrics_path.name}")
    print("[SAVED] Plot    -> plots/confusion_matrix.png")


if __name__ == "__main__":
    main()
