"""
download_hindi_real.py
========================
Download real Hindi speech from OpenSLR 103 (Hindi_test.tar.gz)
Replaces the English dummy files in dataset/real/hindi/

Source: https://www.openslr.org/103/
Files : Hindi_test.tar.gz (247 MB)

Usage:
    python download_hindi_real.py
"""

import os, io, sys, tarfile, urllib.request
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm

TARGET_SR     = 16_000
CLIP_DURATION = 3.0
CLIP_SAMPLES  = int(TARGET_SR * CLIP_DURATION)
NOISE_STD     = 0.008
N_CLIPS       = 500  # Extract 500 clips for the Hindi 'Real' dataset

URL       = "https://openslr.trmal.net/resources/103/Hindi_test.tar.gz"
CACHE_DIR = Path("dataset_cache/openslr103")
OUT_DIR   = Path("dataset/real/hindi")

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
    tar_path = CACHE_DIR / "Hindi_test.tar.gz"

    # Clean the directory first because it currently has English files named real_en_*.wav
    # We will delete them. But first, check if we already have 500 Hindi clips
    existing_hi = sorted(OUT_DIR.glob("real_hi_*.wav"))
    if len(existing_hi) >= N_CLIPS:
        print(f"[OK] Already have {len(existing_hi)} exact Hindi clips in {OUT_DIR}")
        return

    print("\n" + "="*60)
    print("  Downloading OpenSLR 103 — Real Hindi Speech")
    print(f"  Target : {N_CLIPS} clips @ 16kHz 3s WAV")
    print("="*60)

    # Download if not cached
    if not tar_path.exists():
        download_with_progress(URL, str(tar_path))
    else:
        print(f"[OK] Archive cached: {tar_path}")

    # Remove the dummy English files from the directory
    old_en_files = list(OUT_DIR.glob("real_en_*.wav"))
    for f in old_en_files:
        try:
            f.unlink()
        except:
            pass
    print(f"  Deleted {len(old_en_files)} placeholder English files from Hindi folder.")

    # Extract clips from Tar
    print(f"\n  Extracting {N_CLIPS} Hindi clips...")
    saved = []
    extracted = 0

    with tarfile.open(str(tar_path), "r:gz") as tar:
        # Find all valid audio files in the tarball
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".wav") and "__MACOSX" not in m.name]
        print(f"  Found {len(members)} WAV files in archive")

        pbar = tqdm(members, desc="  Extracting", ncols=65)
        for member in pbar:
            if len(saved) >= N_CLIPS:
                break
            try:
                out_path = OUT_DIR / f"real_hi_{extracted:05d}.wav"
                
                # Read WAV from tar in memory
                f = tar.extractfile(member)
                if f is not None:
                    arr, sr = sf.read(io.BytesIO(f.read()))
                    clip = process(arr, sr)
                    sf.write(str(out_path), clip, TARGET_SR)
                    saved.append(str(out_path))
                    extracted += 1
            except Exception as e:
                pass
        pbar.close()

    print(f"\n{'='*60}")
    print(f"  DONE! Saved {len(saved)} real Hindi clips")
    print(f"  Location: {OUT_DIR.resolve()}")
    print("  Language: Hindi (hi) — Actual native speech")
    print("  Source  : OpenSLR 103")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
