import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set paths to a sample of Real and Fake Indian language context audio
real_audio_path = 'dataset/real/hindi/real_hi_00001.wav'
fake_audio_path = 'dataset/fake/hindi/fake_hi_00001.wav'

# Load audio files
print("Loading real audio...")
y_real, sr_real = librosa.load(real_audio_path, sr=16000)
print("Loading fake audio...")
y_fake, sr_fake = librosa.load(fake_audio_path, sr=16000)

# Compute Mel Spectrograms
print("Computing Mel Spectrograms...")
S_real = librosa.feature.melspectrogram(y=y_real, sr=sr_real, n_mels=128, fmax=8000)
S_fake = librosa.feature.melspectrogram(y=y_fake, sr=sr_fake, n_mels=128, fmax=8000)

# Convert to log scale (dB)
S_dB_real = librosa.power_to_db(S_real, ref=np.max)
S_dB_fake = librosa.power_to_db(S_fake, ref=np.max)

# Create a figure for side-by-side comparison
plt.figure(figsize=(14, 6))

# Plot Real Speech Spectrogram
plt.subplot(1, 2, 1)
librosa.display.specshow(S_dB_real, sr=sr_real, x_axis='time', y_axis='mel', fmax=8000, cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title('Real Human Speech (Hindi)\nNotice the rich, irregular high-frequency harmonics', fontsize=12)
plt.tight_layout()

# Plot Synthetic Speech Spectrogram
plt.subplot(1, 2, 2)
librosa.display.specshow(S_dB_fake, sr=sr_fake, x_axis='time', y_axis='mel', fmax=8000, cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title('Synthetic Speech (gTTS Hindi)\nNotice the smoothed, overly uniform energy bands', fontsize=12)
plt.tight_layout()

# Save the plot
output_file = 'spectrogram_comparison_report.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\\n[DONE] Spectrogram comparison saved as: {output_file}")
