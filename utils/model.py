"""
utils/model.py
--------------
CNN model for deepfake audio detection – built with PyTorch.

Architecture
------------
Input  : (batch, 1, N_MFCC, MAX_FRAMES)   – MFCC spectrogram as 2-D image
Block 1: Conv2d(1→32, 3×3)  + BN + ReLU + MaxPool(2×2) + Dropout(0.25)
Block 2: Conv2d(32→64, 3×3) + BN + ReLU + MaxPool(2×2) + Dropout(0.25)
Block 3: Conv2d(64→128,3×3) + BN + ReLU + AdaptiveAvgPool
Head   : Linear(128→256) + Dropout(0.4)
         Linear(256→128) + Dropout(0.3)
         Linear(128→1)   + Sigmoid        ← P(fake)
"""

import os
import joblib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Constants ───────────────────────────────────────────────────────────────
N_MFCC     = 40
MAX_FRAMES = 300
DEVICE     = torch.device("cpu")   # CPU-only (no CUDA required)


# ─── Model definition ────────────────────────────────────────────────────────

class DeepfakeAudioCNN(nn.Module):
    """
    Convolutional neural network for binary real/fake audio classification.
    Input shape: (batch, 1, N_MFCC, MAX_FRAMES)
    """

    def __init__(self, dropout1: float = 0.25,
                 dropout2: float = 0.4,
                 dropout3: float = 0.3):
        super().__init__()

        # ── Convolutional blocks ──────────────────────────────────
        self.block1 = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout1),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),   # collapses spatial dims → (B, 128, 1, 1)
        )

        # ── Classifier head ───────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)   # flatten (B, 128)
        return self.classifier(x)


def build_cnn_model(learning_rate: float = 1e-3):
    """
    Instantiate the CNN and return (model, optimiser, loss_fn).
    """
    model    = DeepfakeAudioCNN().to(DEVICE)
    optim    = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=1e-4)
    loss_fn  = nn.BCELoss()
    return model, optim, loss_fn


# ─── Save / Load ─────────────────────────────────────────────────────────────

def save_model_joblib(model: DeepfakeAudioCNN,
                      scaler,
                      path: str = "model.pkl",
                      threshold: float = 0.5,
                      feature_version: str | None = None) -> None:
    """
    Save CNN state-dict + sklearn scaler to a single joblib file.
    """
    payload = {
        "model_state": model.cpu().state_dict(),
        "scaler":      scaler,
        "threshold":   float(threshold),
        "feature_version": feature_version,
    }
    joblib.dump(payload, path)
    model.to(DEVICE)
    print(f"[INFO] Model saved -> {path}")


def load_model_joblib(path: str = "model.pkl"):
    """
    Load the CNN from a joblib file.

    Returns
    -------
    model  : DeepfakeAudioCNN (eval mode, weights restored)
    scaler : sklearn scaler or None
    """
    payload = joblib.load(path)
    model   = DeepfakeAudioCNN()
    model.load_state_dict(payload["model_state"])
    model.to(DEVICE)
    model.eval()
    scaler  = payload.get("scaler")
    model.decision_threshold = float(payload.get("threshold", 0.5))
    model.feature_version = payload.get("feature_version")
    print(f"[INFO] Model loaded <- {path}")
    return model, scaler


# --- Inference ---------------------------------------------------------------

def predict_single(model: DeepfakeAudioCNN,
                   features: np.ndarray,
                   threshold: float | None = None) -> dict:
    """
    Run inference on a single MFCC feature array.

    Parameters
    ----------
    features  : np.ndarray  shape (N_MFCC, MAX_FRAMES, 1)   or (N_MFCC, MAX_FRAMES)
    threshold : decision boundary (default 0.5)

    Returns
    -------
    dict: {label, probability, confidence}
    """
    model.eval()
    if threshold is None:
        threshold = float(getattr(model, "decision_threshold", 0.5))
    feat = features
    # Accept (H, W, 1) or (H, W) → convert to (1, 1, H, W)
    if feat.ndim == 3:
        feat = feat[:, :, 0]          # drop channel dim
    x    = torch.tensor(feat[np.newaxis, np.newaxis, :, :],
                        dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        prob = float(model(x).squeeze())
    label = "Fake" if prob >= threshold else "Real"
    conf  = prob if prob >= threshold else 1.0 - prob
    return {
        "label":       label,
        "probability": round(prob, 4),
        "confidence":  round(conf * 100, 2),
        "threshold":   round(float(threshold), 4),
    }
