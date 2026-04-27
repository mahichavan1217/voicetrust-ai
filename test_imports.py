# test_imports.py - run this to diagnose which import crashes
import sys
print("Python:", sys.version)
print("Checking imports one by one...")

steps = [
    ("numpy",        "import numpy as np; print('  numpy', np.__version__)"),
    ("matplotlib",   "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; print('  matplotlib OK')"),
    ("seaborn",      "import seaborn as sns; print('  seaborn OK')"),
    ("tqdm",         "from tqdm import tqdm; print('  tqdm OK')"),
    ("sklearn",      "from sklearn.model_selection import train_test_split; from sklearn.metrics import accuracy_score; print('  sklearn OK')"),
    ("torch",        "import torch; print('  torch', torch.__version__)"),
    ("torch.nn",     "import torch.nn as nn; from torch.utils.data import DataLoader, TensorDataset; print('  torch.nn OK')"),
    ("librosa",      "import librosa; print('  librosa', librosa.__version__)"),
    ("soundfile",    "import soundfile; print('  soundfile OK')"),
    ("datasets",     "from datasets import load_dataset; print('  datasets OK')"),
    ("gtts",         "from gtts import gTTS; print('  gtts OK')"),
    ("joblib",       "import joblib; print('  joblib OK')"),
    ("utils.preprocess",         "import sys; sys.path.insert(0,'.'); from utils.preprocess import preprocess_audio, save_wav, TARGET_SR; print('  utils.preprocess OK')"),
    ("utils.feature_extraction", "import sys; sys.path.insert(0,'.'); from utils.feature_extraction import extract_features_for_cnn; print('  utils.feature_extraction OK')"),
    ("utils.model",              "import sys; sys.path.insert(0,'.'); from utils.model import DeepfakeAudioCNN, build_cnn_model, save_model_joblib, DEVICE; print('  utils.model OK')"),
]

for name, code in steps:
    try:
        exec(code)
    except Exception as e:
        print(f"  FAILED [{name}]: {e}")

print("\nAll import checks complete!")
