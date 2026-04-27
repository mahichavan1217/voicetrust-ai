"""
utils/preprocess.py
--------------------
Audio preprocessing utilities.
- Load audio files
- Resample to 16 kHz
- Normalize amplitude
- Trim / pad to fixed duration
"""

import os
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Tuple


# ─── Constants ───────────────────────────────────────────────────────────────
TARGET_SR      = 16_000   # 16 kHz – common for speech models
CLIP_DURATION  = 3.0      # seconds – keep every clip exactly 3 s
CLIP_SAMPLES   = int(TARGET_SR * CLIP_DURATION)


# ─── Core helpers ────────────────────────────────────────────────────────────

def load_and_resample(filepath: str,
                      target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """
    Load any audio format supported by librosa/soundfile and
    resample to *target_sr*.

    Returns
    -------
    waveform : np.ndarray  shape (n_samples,)  float32, mono
    sr       : int         the target sample-rate
    """
    # mono=True collapses multi-channel audio to a single channel
    waveform, sr = librosa.load(filepath, sr=target_sr, mono=True)
    return waveform.astype(np.float32), sr


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    """
    Peak-normalize so the loudest sample has magnitude 1.0.
    Prevents division-by-zero on silent clips.
    """
    peak = np.max(np.abs(waveform))
    if peak > 1e-6:
        waveform = waveform / peak
    return waveform


def fix_length(waveform: np.ndarray,
               target_len: int = CLIP_SAMPLES) -> np.ndarray:
    """
    Pad (with zeros) or truncate *waveform* so its length == *target_len*.
    """
    if len(waveform) < target_len:
        # zero-pad at the end
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    else:
        waveform = waveform[:target_len]
    return waveform


def preprocess_audio(filepath: str,
                     target_sr: int = TARGET_SR,
                     target_len: int = CLIP_SAMPLES) -> Optional[np.ndarray]:
    """
    Full preprocessing pipeline for a single audio file:
        load → resample → normalize → fix-length

    Returns None if the file cannot be read.
    """
    try:
        waveform, _ = load_and_resample(filepath, target_sr)
        waveform     = normalize_waveform(waveform)
        waveform     = fix_length(waveform, target_len)
        return waveform
    except Exception as exc:
        print(f"[WARN] Could not process {filepath}: {exc}")
        return None


def save_wav(waveform: np.ndarray,
             filepath: str,
             sr: int = TARGET_SR) -> None:
    """Write a float32 waveform to a WAV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sf.write(filepath, waveform, sr)
