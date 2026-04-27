# -*- coding: utf-8 -*-
"""
utils/augmentation.py
======================
Audio & spectrogram data augmentation for improved generalisation.

Techniques used:
  1. SpecAugment  - mask frequency/time bands in MFCC (Google 2019)
  2. Time stretch  - randomly slow/speed up audio
  3. Pitch shift   - change pitch slightly
  4. Gaussian noise- add background noise

All augmentations are applied ON-THE-FLY during training.
"""

import numpy as np
import librosa

# ─── Waveform Augmentations ──────────────────────────────────────────────────

def add_gaussian_noise(audio: np.ndarray,
                        min_std: float = 0.002,
                        max_std: float = 0.012) -> np.ndarray:
    """Add random Gaussian noise (simulates mic/room noise)."""
    std = np.random.uniform(min_std, max_std)
    noise = np.random.normal(0, std, audio.shape).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0)


def time_stretch(audio: np.ndarray,
                  min_rate: float = 0.85,
                  max_rate: float = 1.15) -> np.ndarray:
    """Randomly stretch/compress audio in time (no pitch change)."""
    rate = np.random.uniform(min_rate, max_rate)
    return librosa.effects.time_stretch(audio, rate=rate)


def pitch_shift(audio: np.ndarray,
                sr: int = 16_000,
                min_steps: int = -3,
                max_steps: int = 3) -> np.ndarray:
    """Randomly shift pitch by ±3 semitones."""
    steps = np.random.uniform(min_steps, max_steps)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)


def random_gain(audio: np.ndarray,
                min_db: float = -6.0,
                max_db: float = 6.0) -> np.ndarray:
    """Random volume scaling (±6 dB)."""
    gain_db  = np.random.uniform(min_db, max_db)
    gain_lin = 10 ** (gain_db / 20.0)
    return np.clip(audio * gain_lin, -1.0, 1.0)


def random_crop_pad(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Randomly crop or pad to target length."""
    L = len(audio)
    if L > target_len:
        start = np.random.randint(0, L - target_len)
        return audio[start: start + target_len]
    elif L < target_len:
        pad_before = np.random.randint(0, target_len - L)
        return np.pad(audio, (pad_before, target_len - L - pad_before))
    return audio


def augment_waveform(audio: np.ndarray,
                      sr: int = 16_000,
                      p_noise: float   = 0.7,
                      p_stretch: float = 0.4,
                      p_pitch: float   = 0.3,
                      p_gain: float    = 0.5) -> np.ndarray:
    """
    Apply random combination of waveform augmentations.
    Each augmentation is applied independently with its own probability.
    """
    if np.random.random() < p_noise:
        audio = add_gaussian_noise(audio)
    if np.random.random() < p_stretch:
        audio = time_stretch(audio)
    if np.random.random() < p_pitch:
        audio = pitch_shift(audio, sr=sr)
    if np.random.random() < p_gain:
        audio = random_gain(audio)
    return audio


# ─── SpecAugment (on MFCC feature tensor) ───────────────────────────────────

def spec_augment(mfcc: np.ndarray,
                  freq_mask_param: int  = 10,
                  time_mask_param: int  = 30,
                  n_freq_masks: int     = 2,
                  n_time_masks: int     = 2) -> np.ndarray:
    """
    SpecAugment (Park et al., 2019) applied to MFCC tensor.

    Input shape : (n_mels, time)  or  (n_mels, time, 1)
    Output shape: same as input

    Parameters
    ----------
    freq_mask_param : max width of frequency mask
    time_mask_param : max width of time mask
    n_freq_masks    : number of frequency masks
    n_time_masks    : number of time masks
    """
    squeeze = False
    if mfcc.ndim == 3:
        mfcc = mfcc[:, :, 0]
        squeeze = True

    mfcc = mfcc.copy()
    n_mels, n_frames = mfcc.shape
    mean_val = mfcc.mean()

    # Frequency masking
    for _ in range(n_freq_masks):
        width = np.random.randint(0, min(freq_mask_param, n_mels))
        start = np.random.randint(0, n_mels - width + 1)
        mfcc[start: start + width, :] = mean_val

    # Time masking
    for _ in range(n_time_masks):
        width = np.random.randint(0, min(time_mask_param, n_frames))
        start = np.random.randint(0, n_frames - width + 1)
        mfcc[:, start: start + width] = mean_val

    if squeeze:
        mfcc = mfcc[:, :, np.newaxis]
    return mfcc


def augment_features(mfcc: np.ndarray,
                      p_spec: float = 0.6) -> np.ndarray:
    """Apply SpecAugment with probability p_spec."""
    if np.random.random() < p_spec:
        mfcc = spec_augment(mfcc)
    return mfcc
