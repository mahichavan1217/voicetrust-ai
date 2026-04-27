import sys, os, warnings, random
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils.preprocess import preprocess_audio, save_wav, TARGET_SR
from utils.feature_extraction import extract_features_for_cnn
from utils.model import DeepfakeAudioCNN, build_cnn_model, save_model_joblib, DEVICE
print('ALL OK without seaborn')
