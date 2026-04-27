"""
utils/feature_extraction.py
-----------------------------
Extract acoustic features from preprocessed waveforms.

Primary feature : MFCC spectrogram (40 coefficients × T frames)
Shape returned  : (40, T, 1)  – ready for a 2-D CNN

Additional features appended along the channel axis are optional but
improve discrimination between genuine and synthesised speech.
"""

import numpy as np
import librosa
from utils.preprocess import TARGET_SR, CLIP_SAMPLES


# ─── Feature hyper-parameters ────────────────────────────────────────────────
N_MFCC       = 40    # number of Mel-frequency cepstral coefficients
N_FFT        = 512   # FFT window size
HOP_LENGTH   = 160   # hop = 10 ms @ 16 kHz  (standard for speech)
N_MELS       = 128   # mel filters (used for mel-spectrogram variant)

# Fixed time-axis length  (frames = ceil(samples / hop))
MAX_FRAMES   = int(np.ceil(CLIP_SAMPLES / HOP_LENGTH))


# ─── Core extraction ─────────────────────────────────────────────────────────

def extract_mfcc(waveform: np.ndarray,
                 sr: int = TARGET_SR,
                 n_mfcc: int = N_MFCC,
                 n_fft: int = N_FFT,
                 hop_length: int = HOP_LENGTH) -> np.ndarray:
    """
    Compute MFCC matrix from a 1-D waveform.

    Returns
    -------
    mfcc : np.ndarray  shape (n_mfcc, MAX_FRAMES)
    """
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    # Ensure fixed width by padding / truncating along time axis
    if mfcc.shape[1] < MAX_FRAMES:
        pad_width = MAX_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_FRAMES]

    return mfcc.astype(np.float32)


def extract_delta_mfcc(waveform: np.ndarray,
                       sr: int = TARGET_SR) -> np.ndarray:
    """
    Return stacked [MFCC | ΔMFCC | ΔΔMFCC] array.
    Shape: (3*n_mfcc, MAX_FRAMES)
    """
    mfcc   = extract_mfcc(waveform, sr)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2])   # (120, T)


def extract_features_for_cnn(waveform: np.ndarray,
                              sr: int = TARGET_SR) -> np.ndarray:
    """
    Full feature pipeline for the CNN.

    Returns shape (40, MAX_FRAMES, 1) — MFCC spectrogram.
    NOTE: MFCC+Delta+Delta-Delta (120-dim) is also implemented (extract_delta_mfcc)
    but requires 2000+ clips per class to outperform plain MFCC.
    """
    mfcc = extract_mfcc(waveform, sr)   # (40, T)

    # Per-feature normalisation across time axis
    mean = mfcc.mean(axis=1, keepdims=True)
    std  = mfcc.std(axis=1, keepdims=True) + 1e-8
    mfcc = (mfcc - mean) / std

    # Add channel dimension → (40, T, 1)
    return mfcc[:, :, np.newaxis].astype(np.float32)


def extract_features_from_file(filepath: str) -> np.ndarray | None:
    """
    Convenience wrapper: load, preprocess, and extract CNN features
    from a file path.  Returns None on failure.
    """
    from utils.preprocess import preprocess_audio
    waveform = preprocess_audio(filepath)
    if waveform is None:
        return None
    return extract_features_for_cnn(waveform)
