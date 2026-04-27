"""
utils/feature_extraction.py
---------------------------
Extract fixed-size acoustic feature maps from preprocessed waveforms.

The model previously used only MFCCs. This version keeps MFCCs, then adds
delta, delta-delta, and spectral descriptors so the CNN sees more cues that
often separate genuine speech from synthesized speech.
"""

from __future__ import annotations

import numpy as np
import librosa

from utils.preprocess import TARGET_SR, CLIP_SAMPLES


N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 128
MAX_FRAMES = int(np.ceil(CLIP_SAMPLES / HOP_LENGTH))

# 40 MFCC + 40 delta + 40 delta-delta + 13 spectral rows.
FEATURE_ROWS = 133
FEATURE_VERSION = "mfcc_delta_spectral_v2"


def _fix_width(feature: np.ndarray, max_frames: int = MAX_FRAMES) -> np.ndarray:
    """Pad or truncate a feature matrix along the time axis."""
    if feature.shape[1] < max_frames:
        pad_width = max_frames - feature.shape[1]
        feature = np.pad(feature, ((0, 0), (0, pad_width)), mode="constant")
    else:
        feature = feature[:, :max_frames]
    return feature


def _safe_feature(fn, fallback_rows: int) -> np.ndarray:
    """Return an empty fixed feature block if librosa fails on an edge case."""
    try:
        return fn()
    except Exception:
        return np.zeros((fallback_rows, MAX_FRAMES), dtype=np.float32)


def _standardize_rows(feature: np.ndarray) -> np.ndarray:
    """Normalize every feature row independently across time."""
    mean = feature.mean(axis=1, keepdims=True)
    std = feature.std(axis=1, keepdims=True) + 1e-8
    return (feature - mean) / std


def extract_mfcc(
    waveform: np.ndarray,
    sr: int = TARGET_SR,
    n_mfcc: int = N_MFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Compute a fixed-width MFCC matrix."""
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=N_MELS,
    )
    return _fix_width(mfcc).astype(np.float32)


def extract_delta_mfcc(waveform: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Return stacked MFCC, delta MFCC, and delta-delta MFCC."""
    mfcc = extract_mfcc(waveform, sr)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2]).astype(np.float32)


def extract_spectral_features(
    waveform: np.ndarray,
    sr: int = TARGET_SR,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Extract fixed-width spectral descriptors used as extra CNN rows."""
    contrast = _safe_feature(
        lambda: librosa.feature.spectral_contrast(
            y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length
        ),
        fallback_rows=7,
    )
    centroid = _safe_feature(
        lambda: librosa.feature.spectral_centroid(
            y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length
        ),
        fallback_rows=1,
    )
    bandwidth = _safe_feature(
        lambda: librosa.feature.spectral_bandwidth(
            y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length
        ),
        fallback_rows=1,
    )
    rolloff = _safe_feature(
        lambda: librosa.feature.spectral_rolloff(
            y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length
        ),
        fallback_rows=1,
    )
    zcr = _safe_feature(
        lambda: librosa.feature.zero_crossing_rate(
            y=waveform, frame_length=n_fft, hop_length=hop_length
        ),
        fallback_rows=1,
    )
    rms = _safe_feature(
        lambda: librosa.feature.rms(
            y=waveform, frame_length=n_fft, hop_length=hop_length
        ),
        fallback_rows=1,
    )
    flatness = _safe_feature(
        lambda: librosa.feature.spectral_flatness(
            y=waveform, n_fft=n_fft, hop_length=hop_length
        ),
        fallback_rows=1,
    )

    blocks = [contrast, centroid, bandwidth, rolloff, zcr, rms, flatness]
    fixed = [_fix_width(np.asarray(block, dtype=np.float32)) for block in blocks]
    return np.vstack(fixed).astype(np.float32)


def extract_features_for_cnn(waveform: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Return a CNN-ready feature map with shape (FEATURE_ROWS, MAX_FRAMES, 1).
    """
    mfcc_stack = extract_delta_mfcc(waveform, sr)
    spectral = extract_spectral_features(waveform, sr)
    feature = np.vstack([mfcc_stack, spectral]).astype(np.float32)

    if feature.shape[0] != FEATURE_ROWS:
        raise ValueError(f"Expected {FEATURE_ROWS} feature rows, got {feature.shape[0]}")

    feature = _standardize_rows(feature)
    return feature[:, :, np.newaxis].astype(np.float32)


def flatten_feature_map(features: np.ndarray) -> np.ndarray:
    """Convert a CNN feature map or batch into sklearn-ready flat vectors."""
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim == 4:
        if arr.shape[-1] != 1:
            raise ValueError(f"Expected last channel size 1, got {arr.shape[-1]}")
        return arr[:, :, :, 0].reshape(arr.shape[0], -1).astype(np.float32, copy=False)
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        return arr.reshape(1, -1).astype(np.float32, copy=False)
    if arr.ndim == 2:
        return arr.reshape(1, -1).astype(np.float32, copy=False)
    raise ValueError(f"Expected feature map with 2-4 dims, got shape {arr.shape}")


def extract_features_from_file(filepath: str) -> np.ndarray | None:
    """Load, preprocess, and extract CNN features from a file path."""
    from utils.preprocess import preprocess_audio

    waveform = preprocess_audio(filepath)
    if waveform is None:
        return None
    return extract_features_for_cnn(waveform)
