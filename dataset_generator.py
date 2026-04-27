# -*- coding: utf-8 -*-
"""
dataset_generator.py  -  IndicFakeSpeech Dataset Builder
==========================================================

Ek complete automatic dataset builder jo:
  ✅ Real audio: LibriSpeech (direct HTTP, no HuggingFace streaming)
  ✅ Fake audio: gTTS TTS (Hindi, English, Marathi)
  ✅ Noise augmentation (realistic)
  ✅ Proper multilingual folder structure
  ✅ metadata.csv (research-ready)
  ✅ dataset_report.txt

Output:
    dataset/
    ├── real/
    │   ├── hindi/
    │   └── english/
    ├── fake/
    │   ├── hindi/
    │   └── english/
    ├── metadata.csv
    └── dataset_report.txt

Usage:
    python dataset_generator.py                          # default 200 per lang
    python dataset_generator.py --per-lang 300           # more clips
    python dataset_generator.py --langs hi en --per-lang 150
    python dataset_generator.py --fake-only --per-lang 200
"""

import os, sys, csv, random, argparse, tempfile, tarfile, io, shutil, urllib.request
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from tqdm import tqdm

import librosa
import soundfile as sf

# ─── Constants ───────────────────────────────────────────────────────────────
TARGET_SR       = 16_000
CLIP_DURATION   = 3.0
CLIP_SAMPLES    = int(TARGET_SR * CLIP_DURATION)
NOISE_STD       = 0.008
SEED            = 42

random.seed(SEED)
np.random.seed(SEED)

# Language config: code -> (folder_name, gtts_lang_code)
LANG_CONFIG = {
    "hi": ("hindi",   "hi"),
    "mr": ("marathi", "mr"),
    "en": ("english", "en"),
    "ta": ("tamil",   "ta"),
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Audio helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def normalize(a):
    peak = np.max(np.abs(a))
    return a / peak if peak > 1e-6 else a

def fix_len(a):
    if len(a) < CLIP_SAMPLES:
        a = np.pad(a, (0, CLIP_SAMPLES - len(a)))
    return a[:CLIP_SAMPLES]

def add_noise(a, std=NOISE_STD):
    return normalize(a + np.random.normal(0, std, a.shape).astype(np.float32))

def process(arr, sr, aug=True):
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != TARGET_SR:
        arr = librosa.resample(arr.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    arr = normalize(arr.astype(np.float32))
    arr = fix_len(arr)
    if aug:
        arr = add_noise(arr)
    return arr

def save_wav(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, arr, TARGET_SR)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REAL AUDIO  (Direct HTTP LibriSpeech - no HuggingFace streaming needed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# LibriSpeech test-clean tarballs (~350 MB total, open access)
LIBRISPEECH_URLS = [
    "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "https://data.keithito.com/data/speech/test-clean.tar.gz",  # mirror
]

def _download_librispeech(cache_dir: Path) -> Path:
    """Download and extract LibriSpeech test-clean tarball."""
    extracted = cache_dir / "LibriSpeech" / "test-clean"
    if extracted.exists() and any(extracted.rglob("*.flac")):
        print(f"  [LS] Cache found at {extracted}")
        return extracted

    cache_dir.mkdir(parents=True, exist_ok=True)
    tar_path = cache_dir / "test-clean.tar.gz"

    print("  [LS] Downloading LibriSpeech test-clean (~350 MB)...")
    for url in LIBRISPEECH_URLS:
        try:
            print(f"  [LS] Trying: {url}")
            urllib.request.urlretrieve(
                url, str(tar_path),
                reporthook=lambda b, bs, ts: print(
                    f"\r       {min(b*bs, ts)/(1024*1024):.1f}/{ts/(1024*1024):.1f} MB",
                    end="", flush=True) if ts > 0 else None,
            )
            print()
            break
        except Exception as e:
            print(f"  [WARN] {url}: {e}")

    if not tar_path.exists():
        raise RuntimeError("Failed to download LibriSpeech from all mirrors.")

    print("  [LS] Extracting archive...")
    with tarfile.open(str(tar_path), "r:gz") as tar:
        tar.extractall(str(cache_dir))
    tar_path.unlink(missing_ok=True)   # free disk space
    print(f"  [LS] Extracted to {extracted}")
    return extracted


def collect_real_librispeech(out_dir: Path, n: int, aug: bool = True) -> list:
    """
    Collect n real speech clips from LibriSpeech (English audiobook speech).
    Downloads the tarball if not cached.
    """
    print(f"\n  [REAL] LibriSpeech English ({n} clips)...")
    existing = sorted(out_dir.glob("real_en_*.wav"))
    if len(existing) >= n:
        print(f"  [OK] {len(existing)} cached clips found.")
        return [(str(p), "english") for p in existing[:n]]

    try:
        ls_root = _download_librispeech(Path("dataset_cache"))
        flac_files = sorted(ls_root.rglob("*.flac"))
        random.shuffle(flac_files)
        print(f"  [LS] Found {len(flac_files)} FLAC files. Extracting clips...")

        saved = []
        idx   = len(existing)
        for fpath in tqdm(flac_files, desc="  LibriSpeech-real"):
            if idx >= n:
                break
            try:
                arr, sr = librosa.load(str(fpath), sr=None, mono=True)
                # Split into 3-second sub-clips if audio is long
                n_clips_in_file = max(1, int(len(arr) / (sr * CLIP_DURATION)))
                for ci in range(n_clips_in_file):
                    if idx >= n:
                        break
                    start = ci * int(sr * CLIP_DURATION)
                    chunk = arr[start: start + int(sr * CLIP_DURATION)]
                    if len(chunk) < sr * 0.5:   # skip very short chunks
                        continue
                    clip  = process(chunk, sr, aug=aug)
                    path  = str(out_dir / f"real_en_{idx:05d}.wav")
                    save_wav(clip, path)
                    saved.append((path, "english"))
                    idx += 1
            except Exception:
                pass

        print(f"  [OK] Saved {len(saved)} real English clips.")
        return saved
    except Exception as e:
        print(f"  [WARN] LibriSpeech failed: {e}")
        return []


def collect_real_hf(lang_code: str, out_dir: Path, n: int, aug: bool = True) -> list:
    """
    Try HuggingFace datasets (Common Voice) for a given language.
    Falls back to LibriSpeech for 'en'.
    """
    if lang_code == "en":
        return collect_real_librispeech(out_dir, n, aug)

    lang_name = LANG_CONFIG[lang_code][0]
    print(f"\n  [REAL] Trying HuggingFace Common Voice for {lang_name}...")

    existing = sorted(out_dir.glob(f"real_{lang_code}_*.wav"))
    if len(existing) >= n:
        print(f"  [OK] {len(existing)} cached clips found.")
        return [(str(p), lang_name) for p in existing[:n]]

    try:
        from datasets import load_dataset
        ds = load_dataset(
            "mozilla-foundation/common_voice_13_0", lang_code,
            split="train", streaming=True, trust_remote_code=False
        )
        saved = []
        idx   = len(existing)
        for sample in tqdm(ds, total=n, desc=f"  CV-{lang_name}"):
            if idx >= n:
                break
            try:
                arr  = np.array(sample["audio"]["array"], dtype=np.float32)
                sr   = sample["audio"]["sampling_rate"]
                clip = process(arr, sr, aug=aug)
                path = str(out_dir / f"real_{lang_code}_{idx:05d}.wav")
                save_wav(clip, path)
                saved.append((path, lang_name))
                idx += 1
            except Exception:
                pass
        print(f"  [OK] Saved {len(saved)} {lang_name} clips.")
        return saved
    except Exception as e:
        print(f"  [WARN] Common Voice {lang_code} failed: {e}")
        print(f"  [INFO] Please use dedicated scripts (e.g. download_hindi_real.py, download_marathi_real.py) to fetch real {lang_name} audio instead.")
        return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FAKE AUDIO  (gTTS TTS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Diverse sentence pools — varied text = diverse MFCC patterns
# RESEARCH TIP: variety in text reduces dataset bias
SENTENCES = {
    "hi": [
        "Namaste, mera naam Mahesh hai.",
        "Aaj mausam bahut accha hai.",
        "Yeh ek deepfake parikshan hai.",
        "Bharat ek mahan desh hai.",
        "Artificial intelligence bahut powerful hai.",
        "Machine learning se bahut kuch seekha.",
        "Voice cloning ek badi samasya hai.",
        "Hum is system ko improve karenge.",
        "Sangeet sunna mujhe bahut pasand hai.",
        "Vigyan aur technology ka bhavishya ujjwal hai.",
        "Deepfake detection ek zaruri kaam hai.",
        "Yeh audio computer se bana hai.",
        "Neural network speech patterns pehchanta hai.",
        "Data science ek exciting field hai.",
        "Is project ko paper mein include karenge.",
        "Kya tum meri madad karoge?",
        "Roz subah vyayam karna chahiye.",
        "Humara dataset bahut mazboot hai.",
        "Is vakya ko ek machine bol rahi hai.",
        "Deep learning aawaz pehchaan sakti hai.",
    ],
    "mr": [
        "Namaste, maze naav Mahesh aahe.",
        "Aaj havaman khup changla aahe.",
        "He ek deepfake chachani vaakya aahe.",
        "Bharat ek mahaan desh aahe.",
        "Krutrim buddhimatta khup shaktiishali aahe.",
        "Yantrasikshan pragatishil aahe.",
        "He audio sanganakavarun tayaar aahe.",
        "Aapan navi prannali vikasit karu.",
        "Sangeet aikane mala avdate.",
        "Aaj havamanaat paus ahe.",
    ],
    "en": [
        "Hello, this audio was generated by a computer.",
        "Deep learning can detect fake audio recordings.",
        "This is a synthesised text to speech sample.",
        "Artificial intelligence is transforming the world.",
        "Voice cloning technology poses security challenges.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models improve with more data.",
        "Please verify the authenticity of this recording.",
        "Natural language processing advances rapidly.",
        "Researchers are working on deepfake detection.",
        "This sentence was synthesized by a neural network.",
        "Audio deepfakes can spread dangerous misinformation.",
        "Convolutional networks excel at pattern recognition.",
        "We need diverse data for robust audio models.",
        "MFCC features capture important speech characteristics.",
        "This project builds a speech authenticity detector.",
        "Generated speech differs subtly from human voice.",
        "Feature extraction is the key to audio classification.",
        "Our dataset contains balanced real and fake samples.",
        "Deepfake audio detection is an active research area.",
    ],
    "ta": [
        "Vanakkam, en peyar Mahesh.",
        "Indru vaanam azhagaaga irukkiRadhu.",
        "Idu oru deepfake parisodhana vaakiyam.",
        "Bharatham oru periya naadu.",
        "Seyarkuriyel nuzulbugiyam vaLara thodangiRathu.",
    ],
}


def generate_fake_gtts(lang_code: str, out_dir: Path,
                        n: int, aug: bool = True) -> list:
    """Generate n fake TTS clips using gTTS."""
    from gtts import gTTS

    lang_name, gtts_lang = LANG_CONFIG[lang_code]
    pool = SENTENCES.get(lang_code, SENTENCES["en"])
    print(f"\n  [FAKE] gTTS {lang_name} ({n} clips)...")

    existing = sorted(out_dir.glob(f"fake_{lang_code}_*.wav"))
    if len(existing) >= n:
        print(f"  [OK] {len(existing)} cached clips found.")
        return [(str(p), lang_name) for p in existing[:n]]

    saved = [(str(p), lang_name) for p in existing]
    start_idx = len(existing)

    for i in tqdm(range(start_idx, n), desc=f"  gTTS-{lang_name}"):
        out_path = str(out_dir / f"fake_{lang_code}_{i:05d}.wav")
        if os.path.exists(out_path):
            saved.append((out_path, lang_name))
            continue

        # Rotate through sentences for diversity
        text = pool[i % len(pool)]
        try:
            tts = gTTS(text=text, lang=gtts_lang, slow=False)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            tts.save(tmp_path)
            arr, sr = librosa.load(tmp_path, sr=None, mono=True)
            clip = process(arr, sr, aug=aug)
            save_wav(clip, out_path)
            saved.append((out_path, lang_name))
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        except Exception as e:
            print(f"  [WARN] gTTS i={i}: {e}")

    print(f"  [OK] Generated {len(saved)} fake {lang_name} clips.")
    return saved


# Edge TTS voice map (Microsoft Neural voices - very realistic)
EDGE_VOICES = {
    "hi": ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"],
    "en": ["en-US-JennyNeural", "en-US-GuyNeural", "en-GB-SoniaNeural"],
    "mr": ["mr-IN-AarohiNeural"],
    "ta": ["ta-IN-PallaviNeural"],
}


def generate_fake_edge(lang_code: str, out_dir: Path,
                        n: int, aug: bool = True) -> list:
    """Generate n fake TTS clips using Microsoft Edge TTS (neural voices)."""
    import asyncio
    try:
        import edge_tts
    except ImportError:
        print("  [WARN] edge-tts not installed, skipping. Run: pip install edge-tts")
        return []

    lang_name = LANG_CONFIG[lang_code][0]
    voices    = EDGE_VOICES.get(lang_code, EDGE_VOICES["en"])
    pool      = SENTENCES.get(lang_code, SENTENCES["en"])
    print(f"\n  [FAKE] Edge-TTS {lang_name} (voices: {voices}, {n} clips)...")

    existing = sorted(out_dir.glob(f"edgefake_{lang_code}_*.wav"))
    if len(existing) >= n:
        print(f"  [OK] {len(existing)} cached Edge-TTS clips found.")
        return [(str(p), lang_name) for p in existing[:n]]

    saved     = [(str(p), lang_name) for p in existing]
    start_idx = len(existing)

    async def _gen(text, voice, mp3_path):
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(mp3_path)

    for i in tqdm(range(start_idx, n), desc=f"  EdgeTTS-{lang_name}"):
        out_path = str(out_dir / f"edgefake_{lang_code}_{i:05d}.wav")
        if os.path.exists(out_path):
            saved.append((out_path, lang_name)); continue
        text  = pool[i % len(pool)]
        voice = voices[i % len(voices)]
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            asyncio.run(_gen(text, voice, tmp_path))
            arr, sr = librosa.load(tmp_path, sr=None, mono=True)
            clip = process(arr, sr, aug=aug)
            save_wav(clip, out_path)
            saved.append((out_path, lang_name))
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        except Exception as e:
            print(f"  [WARN] Edge-TTS i={i}: {e}")

    print(f"  [OK] Generated {len(saved)} Edge-TTS fake {lang_name} clips.")
    return saved


def generate_fake_combined(lang_code: str, out_dir: Path,
                            n: int, aug: bool = True) -> list:
    """
    Generate n fake clips using BOTH gTTS and Edge TTS (50/50 split).
    Diversity of TTS engines = more robust model.
    """
    half = n // 2
    clips_gtts = generate_fake_gtts(lang_code, out_dir, half, aug)
    clips_edge = generate_fake_edge(lang_code, out_dir, n - half, aug)
    return clips_gtts + clips_edge


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# METADATA + REPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def write_csv(clips, csv_path):
    """clips: list of (path, lang_name, label:int, source:str)"""
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "label_name", "language",
                    "source", "duration_sec", "sample_rate"])
        for path, lang, label, src in clips:
            w.writerow([os.path.basename(path),
                        label, "real" if label==0 else "fake",
                        lang, src, CLIP_DURATION, TARGET_SR])
    print(f"\n[OK] metadata.csv -> {csv_path}")


def write_report(clips, report_path, langs, per_lang):
    lang_cnt  = Counter(c[1] for c in clips)
    label_cnt = Counter(c[2] for c in clips)
    src_cnt   = Counter(c[3] for c in clips)

    lines = [
        "=" * 60,
        "  IndicFakeSpeech Dataset Summary Report",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        f"Total clips    : {len(clips)}",
        f"Real (label=0) : {label_cnt[0]}",
        f"Fake (label=1) : {label_cnt[1]}",
        f"Format         : 16-bit PCM WAV, {TARGET_SR} Hz, {CLIP_DURATION}s",
        f"Noise Aug      : Gaussian noise std={NOISE_STD}",
        "",
        "Language Distribution:",
    ] + [f"  {k:<12}: {v}" for k,v in sorted(lang_cnt.items())] + [
        "",
        "Source Distribution:",
    ] + [f"  {k:<25}: {v}" for k,v in sorted(src_cnt.items())] + [
        "",
        "Citation (for your research paper):",
        '  @dataset{indicfakespeech2024,',
        f'    title={{IndicFakeSpeech: Indian Multilingual Deepfake Audio Dataset}},',
        f'    year={{2024}},',
        f'    note={{Languages: {", ".join(langs)}; Real: LibriSpeech/CommonVoice; Fake: gTTS}},',
        '  }',
        "=" * 60,
    ]
    text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"\n[OK] report -> {report_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args():
    p = argparse.ArgumentParser(description="IndicFakeSpeech Dataset Generator")
    p.add_argument("--langs",     nargs="+", default=["hi", "en"])
    p.add_argument("--per-lang",  type=int,  default=200)
    p.add_argument("--out-dir",   default="dataset")
    p.add_argument("--no-noise",  action="store_true")
    p.add_argument("--fake-only", action="store_true")
    p.add_argument("--real-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ROOT     = Path(args.out_dir)
    PER_LANG = args.per_lang
    LANGS    = args.langs
    AUG      = not args.no_noise

    print("\n" + "=" * 60)
    print("  IndicFakeSpeech Dataset Generator")
    print(f"  Languages  : {LANGS}")
    print(f"  Per lang   : {PER_LANG} real + {PER_LANG} fake")
    print(f"  Noise aug  : {AUG}")
    print(f"  Output     : {ROOT.resolve()}/")
    print("=" * 60)

    all_clips = []   # (path, lang_name, label, source)

    for lang in LANGS:
        if lang not in LANG_CONFIG:
            print(f"[WARN] Unknown lang '{lang}'. Valid: {list(LANG_CONFIG)}")
            continue

        lang_name = LANG_CONFIG[lang][0]
        real_dir  = ROOT / "real" / lang_name
        fake_dir  = ROOT / "fake" / lang_name
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  [{lang_name.upper()}]")
        print(f"{'='*60}")

        # --- REAL ---
        if not args.fake_only:
            real_clips = collect_real_hf(lang, real_dir, PER_LANG, AUG)
            src = "LibriSpeech" if lang == "en" else "CommonVoice"
            for path, lname in real_clips:
                all_clips.append((path, lname, 0, src))

        # --- FAKE (gTTS + Edge TTS combined) ---
        if not args.real_only:
            fake_clips = generate_fake_combined(lang, fake_dir, PER_LANG, AUG)
            for path, lname in fake_clips:
                src = "gTTS-TTS" if "edgefake" not in path else "EdgeTTS-Microsoft"
                all_clips.append((path, lname, 1, src))

    # --- Metadata ---
    write_csv(all_clips, str(ROOT / "metadata.csv"))
    write_report(all_clips, str(ROOT / "dataset_report.txt"), LANGS, PER_LANG)

    print(f"\n{'='*60}")
    print(f"  DONE! Dataset ready at: {ROOT.resolve()}")
    print(f"  Total   : {len(all_clips)} clips")
    print(f"  Real    : {sum(1 for c in all_clips if c[2]==0)}")
    print(f"  Fake    : {sum(1 for c in all_clips if c[2]==1)}")
    print(f"\n  Next step:")
    print(f"    python train_on_dataset.py --dataset {ROOT}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
