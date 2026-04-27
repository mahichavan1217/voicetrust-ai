# -*- coding: utf-8 -*-
"""
colab_train.py  -  Google Colab Training Script
================================================
Run this notebook in Google Colab for FREE GPU training!

Steps:
  1. Open Google Colab: https://colab.research.google.com
  2. Upload this file OR copy-paste cells below
  3. Runtime -> Change runtime type -> GPU (T4 recommended)
  4. Run All

What this script does:
  A. Install dependencies
  B. Generate IndicFakeSpeech dataset (LibriSpeech + gTTS)
  C. Train PyTorch CNN with SpecAugment on GPU
  D. Save model.pkl
  E. Download model.pkl to your local machine
"""

# =====================================================================
# CELL 1: Install dependencies
# =====================================================================
# Run in Colab:
#
# !pip install librosa soundfile gtts tqdm datasets torch torchvision \
#              scikit-learn joblib matplotlib seaborn flask flask-cors \
#              -q
#
# import torch
# print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")

# =====================================================================
# CELL 2: Clone / upload project files
# =====================================================================
# Option A: If your project is on GitHub:
#   !git clone https://github.com/YOUR_USERNAME/deepfake-audio-detector.git
#   %cd deepfake-audio-detector/project
#
# Option B: Upload files manually to Colab Files panel.
# Required files:
#   - utils/model.py
#   - utils/preprocess.py
#   - utils/feature_extraction.py
#   - utils/augmentation.py

# =====================================================================
# CELL 3: Quick dataset + training (paste this into a Colab cell)
# =====================================================================

COLAB_TRAINING_CODE = '''
import os, sys, random, io, base64, tarfile, urllib.request, tempfile
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import librosa, soundfile as sf
from gtts import gTTS

# ── Config ──────────────────────────────────────────────────────────
TARGET_SR    = 16_000
CLIP_SAMPLES = TARGET_SR * 3      # 3 seconds
PER_LANG     = 300                 # clips per class
EPOCHS       = 50
BATCH_SIZE   = 64
SEED         = 42
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

# ── Audio helpers ────────────────────────────────────────────────────
def normalize(a):
    p = np.max(np.abs(a)); return a / p if p > 1e-6 else a

def fix_len(a):
    if len(a) < CLIP_SAMPLES: a = np.pad(a, (0, CLIP_SAMPLES - len(a)))
    return a[:CLIP_SAMPLES]

def process(arr, sr):
    if arr.ndim > 1: arr = arr.mean(1)
    if sr != TARGET_SR: arr = librosa.resample(arr.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    return fix_len(normalize(arr.astype(np.float32)))

def save_wav(arr, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, arr, TARGET_SR)

# ── SpecAugment ──────────────────────────────────────────────────────
def spec_augment(mfcc, fq=10, tq=30, nf=2, nt=2):
    if mfcc.ndim == 3: mfcc = mfcc[:, :, 0]; sq = True
    else: sq = False
    mfcc = mfcc.copy(); m = mfcc.mean()
    H, W = mfcc.shape
    for _ in range(nf):
        w = np.random.randint(0, min(fq, H)); s = np.random.randint(0, H - w + 1)
        mfcc[s:s+w, :] = m
    for _ in range(nt):
        w = np.random.randint(0, min(tq, W)); s = np.random.randint(0, W - w + 1)
        mfcc[:, s:s+w] = m
    return mfcc[:, :, np.newaxis] if sq else mfcc

# ── Download real audio (LibriSpeech) ────────────────────────────────
def download_real(n, out_dir="data/real"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    existing = list(Path(out_dir).glob("*.wav"))
    if len(existing) >= n:
        return [str(p) for p in existing[:n]]

    print("Downloading LibriSpeech test-clean ...")
    url = "https://www.openslr.org/resources/12/test-clean.tar.gz"
    tar_path = "/tmp/ls.tar.gz"
    urllib.request.urlretrieve(url, tar_path)
    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall("/tmp/")

    flacs = list(Path("/tmp/LibriSpeech/test-clean").rglob("*.flac"))
    random.shuffle(flacs)
    saved, idx = [], 0
    for fpath in tqdm(flacs, desc="LibriSpeech"):
        if idx >= n: break
        arr, sr = librosa.load(str(fpath), sr=None, mono=True)
        n_sub = max(1, int(len(arr) / (sr * 3)))
        for ci in range(n_sub):
            if idx >= n: break
            chunk = arr[ci * sr * 3: (ci + 1) * sr * 3]
            if len(chunk) < sr: continue
            clip = process(chunk, sr)
            path = f"{out_dir}/real_{idx:05d}.wav"
            save_wav(clip, path); saved.append(path); idx += 1
    print(f"Saved {len(saved)} real clips.")
    return saved

# ── Generate fake audio (gTTS) ────────────────────────────────────────
SENTENCES = [
    "This sentence was synthesized by a computer.",
    "Deep learning can detect artificial speech patterns.",
    "Voice cloning poses significant detection challenges.",
    "Machine learning improves audio authentication.",
    "Artificial intelligence creates realistic sounding speech.",
    "Hello this is a text to speech demonstration.",
    "Researchers study spectral patterns in fake audio.",
    "Deepfake detection helps protect information integrity.",
    "Neural networks learn from millions of audio samples.",
    "Real human voice has unique irreproducible characteristics.",
    "Namaste yeh ek TTS audio hai.",
    "Aaj mausam bahut accha hai.",
    "Yeh audio artificial intelligence se bana hai.",
    "Machine learning deepfake audio pehchan sakti hai.",
    "Voice authentication ek zaruri technology hai.",
]

def generate_fake(n, out_dir="data/fake"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    existing = list(Path(out_dir).glob("*.wav"))
    if len(existing) >= n:
        return [str(p) for p in existing[:n]]
    saved = [str(p) for p in existing]
    start = len(existing)
    for i in tqdm(range(start, n), desc="gTTS fake"):
        path = f"{out_dir}/fake_{i:05d}.wav"
        if Path(path).exists(): saved.append(path); continue
        text = SENTENCES[i % len(SENTENCES)]
        lang = "hi" if "Namaste" in text or "yeh" in text else "en"
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tts.save(tmp.name); arr, sr = librosa.load(tmp.name, sr=None, mono=True)
                os.unlink(tmp.name)
            save_wav(process(arr, sr), path); saved.append(path)
        except Exception as e:
            print(f"  WARN gTTS {i}: {e}")
    print(f"Generated {len(saved)} fake clips.")
    return saved

# ── Feature extraction ────────────────────────────────────────────────
def extract(fp):
    try:
        arr, sr = librosa.load(fp, sr=TARGET_SR, mono=True)
        arr = fix_len(normalize(arr))
        mfcc = librosa.feature.mfcc(y=arr, sr=TARGET_SR, n_mfcc=40)
        mfcc = librosa.util.fix_length(mfcc, size=300, axis=1)
        return mfcc[:, :, np.newaxis].astype(np.float32)
    except: return None

# ── CNN Model ────────────────────────────────────────────────────────
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        def block(ci, co, k=3, p=1):
            return nn.Sequential(
                nn.Conv2d(ci, co, k, padding=p, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, k, padding=p, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout2d(0.25)
            )
        self.features = nn.Sequential(block(1,32), block(32,64), block(64,128))
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128*4, 256),
            nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
    def forward(self, x): return self.classifier(self.pool(self.features(x)))

# ── Main ─────────────────────────────────────────────────────────────
real_paths = download_real(PER_LANG)
fake_paths = generate_fake(PER_LANG)
n = min(len(real_paths), len(fake_paths))
print(f"Dataset: {n} real + {n} fake = {2*n} clips")

print("Extracting MFCCs ...")
Xr = [extract(fp) for fp in tqdm(real_paths[:n])]
Xf = [extract(fp) for fp in tqdm(fake_paths[:n])]
Xr = [x for x in Xr if x is not None]
Xf = [x for x in Xf if x is not None]

X = np.array(Xr + Xf, dtype=np.float32)
y = np.array([0]*len(Xr) + [1]*len(Xf), dtype=np.float32)

# SpecAugment on fake samples only (augment while building)
Xp = X[:, :, :, 0][:, np.newaxis, :, :]
yp = y.reshape(-1, 1)

from sklearn.model_selection import train_test_split
Xt, Xtmp, yt, ytmp = train_test_split(Xp, yp, test_size=0.30, stratify=yp, random_state=SEED)
Xv, Xte, yv, yte   = train_test_split(Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=SEED)
print(f"Train={len(Xt)}, Val={len(Xv)}, Test={len(Xte)}")

def loader(Xa, ya, shuffle=True, augment=False):
    if augment:
        Xa_aug = np.array([spec_augment(x[0])[np.newaxis] if np.random.random()<0.6 else x for x in Xa])
    else:
        Xa_aug = Xa
    ds = TensorDataset(torch.from_numpy(Xa_aug), torch.from_numpy(ya))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=2, pin_memory=True)

model   = CNN().to(DEVICE)
optim   = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
sched   = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
n0, n1  = float((yt==0).sum()), float((yt==1).sum())
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n0/max(n1,1)]).to(DEVICE))

best_val, pat = float("inf"), 0
PATIENCE = 10

for ep in range(1, EPOCHS+1):
    # Re-apply SpecAugment every epoch
    tr = loader(Xt, yt, shuffle=True, augment=True)
    vl_load = loader(Xv, yv, shuffle=False)

    model.train()
    tl, tok, tn = 0.0, 0, 0
    for Xb, yb in tr:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optim.zero_grad()
        logits = model(Xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        tl += loss.item()*len(Xb); tok += ((torch.sigmoid(logits)>=0.5).float()==yb).sum().item(); tn+=len(Xb)
    tl/=tn; ta=tok/tn

    model.eval()
    vl, vok, vn = 0.0, 0, 0
    with torch.no_grad():
        for Xb, yb in vl_load:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            logits = model(Xb)
            vl+=loss_fn(logits,yb).item()*len(Xb); vok+=((torch.sigmoid(logits)>=0.5).float()==yb).sum().item(); vn+=len(Xb)
    vl/=vn; va=vok/vn; sched.step()
    print(f"Epoch {ep:3d}/{EPOCHS} | loss {tl:.4f} acc {ta*100:.1f}% | val_loss {vl:.4f} val_acc {va*100:.1f}%")
    if vl < best_val-1e-4: best_val=vl; pat=0; torch.save(model.state_dict(), "best_colab.pt")
    else:
        pat+=1
        if pat>=PATIENCE: print(f"Early stop at {ep}"); break

model.load_state_dict(torch.load("best_colab.pt", map_location=DEVICE))
model.eval()

# Evaluate
from sklearn.metrics import accuracy_score, classification_report
model.eval()
Xtest_t = torch.from_numpy(Xte).to(DEVICE)
with torch.no_grad(): probs = torch.sigmoid(model(Xtest_t)).cpu().numpy().ravel()
ypred = (probs >= 0.5).astype(int)
print("\\n--- Test Results ---")
print(f"Accuracy: {accuracy_score(yte.ravel().astype(int), ypred)*100:.2f}%")
print(classification_report(yte.ravel().astype(int), ypred, target_names=["Real","Fake"]))

# Save model
import joblib
from sklearn.preprocessing import StandardScaler
torch.save(model.state_dict(), "model_state.pt")
joblib.dump({"model_state": model.state_dict(), "scaler": StandardScaler()}, "model.pkl")
print("\\nSaved model.pkl")

# Download from Colab
from google.colab import files
files.download("model.pkl")
print("model.pkl downloaded! Copy it to your project/ folder and restart Flask.")
'''

print("=" * 60)
print("  IndicFakeSpeech - Google Colab Training Notebook")
print("=" * 60)
print()
print("1. Go to: https://colab.research.google.com")
print("2. Create a new notebook")
print("3. Runtime -> Change runtime type -> GPU (T4)")
print()
print("CELL 1 - Install dependencies:")
print("""
!pip install librosa soundfile gtts tqdm datasets torch torchvision \\
             scikit-learn joblib matplotlib seaborn flask flask-cors -q
import torch
print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
""")
print("CELL 2 - Copy and paste the training code:")
print("(The full code is stored in the COLAB_TRAINING_CODE variable in this file)")
print()
print("CELL 3 - After training, model.pkl will be auto-downloaded.")
print("Copy model.pkl to your project/ folder and restart: python app.py")
print()
print("Estimated training time on Colab T4 GPU:")
print("  300 clips x 2 classes x 50 epochs = ~5 minutes")
print()
print("=" * 60)

# Save the training code to a separate .py file for easy copy-paste
with open("colab_cell_code.py", "w", encoding="utf-8") as f:
    f.write(COLAB_TRAINING_CODE)
print("Full Colab cell code saved to: colab_cell_code.py")
