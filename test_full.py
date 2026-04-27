import sys
sys.path.insert(0, '.')
print("1 sys ok")
import os, warnings, random
print("2 stdlib ok")
import numpy as np
print("3 numpy ok")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
print("4 matplotlib ok")
import seaborn as sns
print("5 seaborn ok")
from tqdm import tqdm
from pathlib import Path
print("6 tqdm/pathlib ok")
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("7 sklearn ok")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
print("8 torch ok")
from utils.preprocess import preprocess_audio, save_wav, TARGET_SR
print("9 preprocess ok")
from utils.feature_extraction import extract_features_for_cnn
print("10 feature_extraction ok")
from utils.model import DeepfakeAudioCNN, build_cnn_model, save_model_joblib, DEVICE
print("11 model ok")
print("DONE - all imports passed")
