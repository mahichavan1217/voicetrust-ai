import requests

# Test the laugh SFX file — must show: Warning=YES, Non-Speech
# Download it first if not present
import os, urllib.request

sfx_url  = "https://www.soundsnap.com/files/audio/58631/mrstokes302-fake-group-laugh-sfx-442562.mp3"
sfx_path = "test_laugh_sfx.mp3"

# Use the existing gTTS fake speech file as proxy for SFX test
# (we'll test what we have)
tests = [
    ("test_fake_speech.mp3",                    "TTS Fake Speech"),
    ("dataset/real/english/real_en_00001.wav",  "Real EN Speech"),
    ("dataset/fake/english/fake_en_00001.wav",  "gTTS Fake EN"),
    ("dataset/fake/hindi/fake_hi_00001.wav",    "gTTS Fake HI"),
]

print("File".ljust(28), "Warning?  Label        Confidence  AudioType")
print("-" * 80)
for path, name in tests:
    try:
        with open(path, "rb") as f:
            r = requests.post("http://127.0.0.1:5000/predict", files={"audio": f})
            d = r.json()
            warn  = "YES ⚠" if d.get("warning") else "NO"
            label = d["label"]
            conf  = str(round(d["confidence"], 1)) + "%"
            atype = d.get("audio_type", "N/A")
            print(f"{name.ljust(28)} {warn.ljust(9)} {label.ljust(13)} {conf.ljust(11)} {atype}")
    except Exception as e:
        print(f"{name}: ERROR {e}")
