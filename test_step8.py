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
import torch
print("7 torch ok")
from sklearn.model_selection import train_test_split
print("8 sklearn ok")
print("PASSED first 8")
