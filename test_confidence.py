import requests

tests = [
    ("dataset/fake/english/fake_en_00001.wav", "Fake EN gTTS"),
    ("dataset/fake/hindi/fake_hi_00001.wav",   "Fake HI gTTS"),
    ("dataset/real/english/real_en_00001.wav", "Real EN"),
    ("dataset/real/hindi/real_en_00001.wav",   "Real HI"),
    ("test_fake_speech.mp3",                   "TTS test file"),
]

print("File".ljust(25), "Expected  Got       Confidence")
print("-" * 60)
for path, name in tests:
    try:
        with open(path, "rb") as f:
            r = requests.post("http://127.0.0.1:5000/predict", files={"audio": f})
            d = r.json()
            expected = "FAKE" if ("fake" in name.lower() or "tts" in name.lower()) else "REAL"
            got = d["label"].upper()
            ok = "OK" if got == expected else "WRONG"
            print(f"{name.ljust(25)} {expected.ljust(9)} {got.ljust(9)} {d['confidence']}%  [{ok}]")
    except Exception as e:
        print(f"{name}: ERROR {e}")
