"""
download_marathi_real.py
========================
Download real Marathi speech from OpenSLR 64
(Crowdsourced multilingual high-quality TTS data for 13 Indian languages)

Source: https://www.openslr.org/64/
Files : mr_in_female.zip (679 MB, ~7000 wav clips of real Marathi female speech)

Usage:
    python download_marathi_real.py
"""

import os, io, sys, zipfile, urllib.request
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm

TARGET_SR     = 16_000
CLIP_DURATION = 3.0
CLIP_SAMPLES  = int(TARGET_SR * CLIP_DURATION)
NOISE_STD     = 0.008
N_CLIPS       = 150  # How many Marathi clips to extract

ZIP_URL   = "https://www.openslr.org/resources/64/mr_in_female.zip"
CACHE_DIR = Path("dataset_cache/openslr64")
OUT_DIR   = Path("dataset/real/marathi")

OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def normalize(a):
    peak = np.max(np.abs(a))
    return a / peak if peak > 1e-6 else a


def fix_len(a):
    if len(a) < CLIP_SAMPLES:
        a = np.pad(a, (0, CLIP_SAMPLES - len(a)))
    return a[:CLIP_SAMPLES]


def add_noise(a, std=NOISE_STD):
    return normalize(a + np.random.normal(0, std, a.shape).astype(np.float32))


def process(arr, sr):
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != TARGET_SR:
        arr = librosa.resample(arr.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    arr = normalize(arr.astype(np.float32))
    arr = fix_len(arr)
    arr = add_noise(arr)
    return arr


def download_with_progress(url, dest_path):
    print(f"  Downloading: {url}")
    print(f"  To: {dest_path}")
    headers = {"User-Agent": "Mozilla/5.0"}

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as r:
        total = int(r.headers.get("content-length", 0))
        chunk = 1024 * 1024  # 1 MB chunks
        downloaded = 0

        with open(dest_path, "wb") as f:
            pbar = tqdm(total=total, unit="B", unit_scale=True, desc="  Downloading")
            while True:
                block = r.read(chunk)
                if not block:
                    break
                f.write(block)
                downloaded += len(block)
                pbar.update(len(block))
            pbar.close()
    print(f"  Download complete: {downloaded/(1024*1024):.1f} MB")


def main():
    zip_path = CACHE_DIR / "mr_in_female.zip"

    # Check existing clips
    existing = sorted(OUT_DIR.glob("real_mr_*.wav"))
    if len(existing) >= N_CLIPS:
        print(f"[OK] Already have {len(existing)} Marathi clips in {OUT_DIR}")
        return

    print("\n" + "="*60)
    print("  Downloading OpenSLR 64 — Real Marathi Speech")
    print("  Source : OpenSLR (Crowdsourced Indian TTS)")
    print(f"  Target : {N_CLIPS} clips @ 16kHz 3s WAV")
    print("="*60)

    # Download if not cached
    if not zip_path.exists():
        download_with_progress(ZIP_URL, str(zip_path))
    else:
        print(f"[OK] ZIP cached: {zip_path}")

    # Extract clips from ZIP
    print(f"\n  Extracting {N_CLIPS} Marathi clips...")
    saved = list(existing)
    start_idx = len(existing)
    extracted = 0

    with zipfile.ZipFile(str(zip_path), "r") as zf:
        wav_files = [f for f in zf.namelist()
                     if f.endswith(".wav") and "__MACOSX" not in f]
        print(f"  Found {len(wav_files)} WAV files in archive")

        pbar = tqdm(wav_files, desc="  Extracting", ncols=65)
        for zname in pbar:
            if len(saved) >= N_CLIPS:
                break
            try:
                idx = start_idx + extracted
                out_path = OUT_DIR / f"real_mr_{idx:05d}.wav"
                if out_path.exists():
                    saved.append(str(out_path))
                    extracted += 1
                    continue

                # Read WAV from zip in memory
                with zf.open(zname) as zf_member:
                    data = io.BytesIO(zf_member.read())
                arr, sr = sf.read(data)
                clip = process(arr, sr)
                sf.write(str(out_path), clip, TARGET_SR)
                saved.append(str(out_path))
                extracted += 1
            except Exception as e:
                pass
        pbar.close()

    print(f"\n{'='*60}")
    print(f"  DONE! Saved {len(saved)} real Marathi clips")
    print(f"  Location: {OUT_DIR.resolve()}")
    print(f"  Language: Marathi (mr) — Actual native speech")
    print(f"  Source  : OpenSLR 64 (Crowdsourced)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
