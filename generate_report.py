from pathlib import Path
from datetime import datetime

real_en = list(Path('dataset/real/english').glob('*.wav'))
real_hi = list(Path('dataset/real/hindi').glob('*.wav'))
real_mr = list(Path('dataset/real/marathi').glob('*.wav'))

fake_en = list(Path('dataset/fake/english').glob('*.wav'))
fake_hi = list(Path('dataset/fake/hindi').glob('*.wav'))
fake_mr = list(Path('dataset/fake/marathi').glob('*.wav'))

tot_real = len(real_en) + len(real_hi) + len(real_mr)
tot_fake = len(fake_en) + len(fake_hi) + len(fake_mr)
tot = tot_real + tot_fake

report = f"""============================================================
  IndicFakeSpeech Dataset Summary Report
  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
============================================================

Total clips    : {tot}
Real (label=0) : {tot_real}
Fake (label=1) : {tot_fake}
Format         : 16-bit PCM WAV, 16000 Hz, 3.0s

Language Distribution:
  English      : {len(real_en) + len(fake_en)} (Real: {len(real_en)}, Fake: {len(fake_en)})
  Hindi        : {len(real_hi) + len(fake_hi)} (Real: {len(real_hi)}, Fake: {len(fake_hi)})
  Marathi      : {len(real_mr) + len(fake_mr)} (Real: {len(real_mr)}, Fake: {len(fake_mr)})

Source Details:
  Real English : LibriSpeech (OpenSLR 12)
  Real Hindi   : ASR Challenge (OpenSLR 103)
  Real Marathi : Crowdsourced (OpenSLR 64)
  Fake All     : Generated via gTTS + EdgeTTS

Citation (for your research paper):
@dataset{{indicfakespeech2024,
  title     = {{IndicFakeSpeech: Multilingual Deepfake Audio Corpus}},
  author    = {{Major Project Team}},
  year      = {{2024}},
  size      = {{{tot} clips}},
}}
"""
with open('dataset/dataset_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print('Report regenerated successfully.')
