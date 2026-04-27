"""
app.py
------
Flask backend for the Deepfake Audio Detection web app.

Endpoints
---------
GET  /              - Serve the HTML frontend
POST /predict       - Upload audio, get Real/Fake prediction + spectrogram
GET  /health        - Liveness check
"""

import os, sys, uuid, logging, io, base64
from pathlib import Path

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
UPLOAD_FOLDER = BASE_DIR / "uploads"
MODEL_PATH    = BASE_DIR / "model.pkl"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS  = {"wav", "mp3", "ogg", "flac", "m4a", "webm"}
MAX_FILE_BYTES = 20 * 1024 * 1024   # 20 MB

# ─── Flask app ───────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"]       = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"]  = MAX_FILE_BYTES
CORS(app)

# ─── Lazy model ──────────────────────────────────────────────────────────────
model = scaler = None


class FullCNN(object):
    """4-block CNN dummy when torch is not loaded."""
    pass


class SklearnAudioModel:
    """Small adapter so sklearn models expose the same prediction API as CNNs."""

    def __init__(self, pipeline, threshold: float = 0.5):
        self.pipeline = pipeline
        self.decision_threshold = float(threshold)

    def predict_proba_single(self, features):
        from utils.feature_extraction import flatten_feature_map

        vector = flatten_feature_map(features).reshape(1, -1)
        if hasattr(self.pipeline, "predict_proba"):
            prob = float(self.pipeline.predict_proba(vector)[0, 1])
        else:
            score = float(self.pipeline.decision_function(vector)[0])
            prob = float(1.0 / (1.0 + np.exp(-score)))

        threshold = float(getattr(self, "decision_threshold", 0.5))
        label = "Fake" if prob >= threshold else "Real"
        confidence = prob if prob >= threshold else 1.0 - prob
        return {
            "label": label,
            "probability": round(prob, 4),
            "confidence": round(confidence * 100, 2),
            "threshold": round(threshold, 4),
        }


def _build_full_cnn():
    """Returns a fresh FullCNN instance."""
    import torch.nn as nn
    class _FullCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32),
                nn.ReLU(True), nn.MaxPool2d(2,2), nn.Dropout2d(0.25))
            self.block2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
                nn.ReLU(True), nn.MaxPool2d(2,2), nn.Dropout2d(0.25))
            self.block3 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
                nn.ReLU(True), nn.MaxPool2d(2,2), nn.Dropout2d(0.2))
            self.block4 = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
                nn.ReLU(True), nn.AdaptiveAvgPool2d((1,1)))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 512), nn.ReLU(True), nn.Dropout(0.5),
                nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(0.4),
                nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(0.3),
                nn.Linear(128, 1), nn.Sigmoid())
        def forward(self, x):
            return self.classifier(
                self.block4(self.block3(self.block2(self.block1(x)))))
        def predict_proba_single(self, features):
            """features: (133,300,1) or (133,300)"""
            if features.ndim == 3:
                features = features[:,:,0]
            x = torch.tensor(
                features[np.newaxis, np.newaxis, :, :],
                dtype=torch.float32)
            with torch.no_grad():
                prob = float(self.forward(x).squeeze())
            thr   = float(getattr(self, 'decision_threshold', 0.5))
            label = "Fake" if prob >= thr else "Real"
            conf  = prob if prob >= thr else 1.0 - prob
            return {"label": label, "probability": round(prob, 4),
                    "confidence": round(conf*100, 2), "threshold": round(thr, 4)}
    return _FullCNN()


def load_model_once():
    global model, scaler
    if model is not None:
        return True
    if not MODEL_PATH.exists():
        logger.error("model.pkl not found. Run run_training.py first.")
        return False
    try:
        import joblib
        if torch is not None:
            torch.set_num_threads(1)

        payload = joblib.load(str(MODEL_PATH))

        # ── Detect model architecture ────────────────────────────────────
        if isinstance(payload, dict) and (
            'model_state' in payload or 'sklearn_pipeline' in payload
        ):
            model_class = payload.get('model_class', 'DeepfakeAudioCNN')
            state       = payload.get('model_state')
            threshold   = float(payload.get('threshold', 0.5))
            scaler      = payload.get('scaler')

            if payload.get("model_type") == "sklearn_svd_logreg":
                model = SklearnAudioModel(payload["sklearn_pipeline"], threshold)
                model.feature_version = payload.get("feature_version",
                                                    "mfcc_delta_spectral_v2")
                logger.info(f"Sklearn SVD+LogReg model loaded (threshold={threshold:.2f})")
            elif model_class == 'FullCNN':
                # New 4-block FullCNN (trained by run_training.py)
                model = _build_full_cnn()
                model.load_state_dict(state, strict=False)
                model.decision_threshold  = threshold
                model.feature_version     = payload.get('feature_version',
                                                        'mfcc_delta_spectral_v2')
                logger.info(f"FullCNN loaded (threshold={threshold:.2f})")
            elif model_class == 'TabularRF':
                # Tabular RandomForest for 95%+ accuracy
                class TabularRFWrapper:
                    def __init__(self, rf, scaler, thr):
                        self.rf = rf
                        self.scaler = scaler
                        self.decision_threshold = thr
                    def predict_proba_single(self, features):
                        import numpy as np
                        if features.ndim == 3: features = features[:,:,0]
                        mean_f = np.mean(features, axis=1)
                        std_f  = np.std(features, axis=1)
                        max_f  = np.max(features, axis=1)
                        min_f  = np.min(features, axis=1)
                        tabular = np.hstack([mean_f, std_f, max_f, min_f]).reshape(1, -1)
                        scaled  = self.scaler.transform(tabular)
                        probs   = self.rf.predict_proba(scaled)[0]
                        prob    = float(probs[1]) if len(probs) > 1 else float(probs[0])
                        label   = "Fake" if prob >= self.decision_threshold else "Real"
                        conf    = prob if prob >= self.decision_threshold else 1.0 - prob
                        return {"label": label, "probability": round(prob, 4),
                                "confidence": round(conf*100, 2), "threshold": round(self.decision_threshold, 4)}
                model = TabularRFWrapper(state, scaler, threshold)
                model.feature_version = payload.get('feature_version', 'mfcc_delta_spectral_v2')
                logger.info(f"TabularRF loaded (accuracy=95%+, threshold={threshold:.2f})")
            else:
                # Original DeepfakeAudioCNN (3-block, 40-row MFCC)
                from utils.model import DeepfakeAudioCNN
                model = DeepfakeAudioCNN()
                model.load_state_dict(state, strict=False)
                model.decision_threshold = threshold
                model.feature_version    = payload.get('feature_version',
                                                       'mfcc_delta_spectral_v2')
                logger.info(f"DeepfakeAudioCNN loaded (threshold={threshold:.2f})")

            if hasattr(model, "eval"):
                model.eval()
            logger.info("Model ready for inference.")
            return True
        else:
            logger.error("model.pkl format not recognized. Re-run run_training.py.")
            return False

    except Exception as exc:
        logger.exception(f"Model load failed: {exc}")
        return False


# ─── Helpers ─────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


def generate_spectrogram_b64(waveform: np.ndarray) -> str:
    """
    Generate an MFCC spectrogram image (PNG) and return it as a
    base64-encoded string for embedding directly in JSON.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display

    SR = 16_000
    mfcc = librosa.feature.mfcc(y=waveform, sr=SR, n_mfcc=40)

    fig, ax = plt.subplots(figsize=(6, 2.5), facecolor="#0f1117")
    img = librosa.display.specshow(
        mfcc, sr=SR, x_axis="time", ax=ax,
        cmap="magma"
    )
    ax.set_title("MFCC Spectrogram", color="white", fontsize=10, pad=6)
    ax.set_xlabel("Time (s)", color="#aaa", fontsize=8)
    ax.set_ylabel("MFCC Coeff", color="#aaa", fontsize=8)
    ax.tick_params(colors="#aaa", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    fig.colorbar(img, ax=ax, format="%+.0f").ax.yaxis.set_tick_params(color="#aaa", labelsize=7)
    ax.figure.axes[-1].yaxis.label.set_color("#aaa")
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, facecolor="#0f1117")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def detect_speech_presence(waveform: np.ndarray, sr: int = 16000) -> dict:
    """
    Voice Activity Detection (VAD) — Language-agnostic.

    Indian languages (Marathi, Hindi) have higher ZCR than English LibriSpeech.
    We use energy + voiced-frame-ratio as primary signals, NOT ZCR alone.
    """
    import librosa

    # RMS energy — primary: is there enough loudness?
    rms = librosa.feature.rms(y=waveform)[0]
    rms_mean = float(np.mean(rms))
    rms_std  = float(np.std(rms))

    # Voiced frame ratio — fraction of frames with above-threshold energy
    energy_threshold = max(rms_mean * 0.25, 0.002)
    voiced_frames    = np.sum(rms > energy_threshold)
    speech_ratio     = float(voiced_frames / max(len(rms), 1))

    # ZCR — informational only (NOT used to block speech)
    zcr      = librosa.feature.zero_crossing_rate(waveform)[0]
    zcr_mean = float(np.mean(zcr))

    # Spectral rolloff — informational only
    rolloff      = librosa.feature.spectral_rolloff(y=waveform, sr=sr, roll_percent=0.85)[0]
    rolloff_mean = float(np.mean(rolloff))

    # Spectral flatness — music/noise have high flatness, speech is spiky
    flatness      = librosa.feature.spectral_flatness(y=waveform)[0]
    flatness_mean = float(np.mean(flatness))

    # ── Decision (language-agnostic rules) ──────────────────────────
    # ONLY block if clearly non-speech:
    #   1. Nearly silent
    #   2. Pure noise/tone (very high flatness + near-zero energy pattern)
    #   3. Essentially silent throughout
    is_silence    = rms_mean < 0.002
    is_pure_noise = flatness_mean > 0.55 and speech_ratio < 0.20  # only pure white noise
    lacks_energy  = speech_ratio < 0.08 and rms_mean < 0.005      # almost all silence

    has_speech = not (is_silence or is_pure_noise or lacks_energy)


    # Audio type label
    if is_silence:
        audio_type = "silence"
    elif is_pure_noise:
        audio_type = "sound_effect_or_music"
    elif lacks_energy:
        audio_type = "non_speech"
    else:
        audio_type = "speech"

    return {
        "has_speech":    has_speech,
        "speech_ratio":  round(speech_ratio, 3),
        "audio_type":    audio_type,
        "zcr_mean":      round(zcr_mean, 4),
        "rms_mean":      round(rms_mean, 4),
        "rolloff_hz":    round(rolloff_mean, 1),
        "flatness":      round(flatness_mean, 4),
    }


def run_inference(filepath: str) -> dict:
    from utils.preprocess         import preprocess_audio
    from utils.feature_extraction import extract_features_for_cnn

    waveform = preprocess_audio(filepath)
    if waveform is None:
        raise ValueError("Could not decode the audio file.")

    # ── Voice Activity Detection ──────────────────────────────────────
    vad = detect_speech_presence(waveform)
    logger.info(f"VAD: type={vad['audio_type']} speech_ratio={vad['speech_ratio']} "
                f"zcr={vad['zcr_mean']} rolloff={vad['rolloff_hz']}Hz")

    # Generate spectrogram always (useful to show user)
    spectrogram_b64 = None
    try:
        spectrogram_b64 = generate_spectrogram_b64(waveform)
    except Exception as e:
        logger.warning(f"Spectrogram generation failed: {e}")

    # If no speech detected → return informative warning
    if not vad["has_speech"]:
        type_messages = {
            "silence":               "Audio appears to be silent or nearly silent.",
            "sound_effect_or_music": "Audio appears to be a sound effect, laugh track, or music — not human speech.",
            "non_speech":            "Audio does not contain sufficient speech content for analysis.",
            "uncertain":             "Audio content is unclear. Please upload a speech recording.",
        }
        msg = type_messages.get(vad["audio_type"], "No speech detected in audio.")
        return {
            "label":       "Non-Speech",
            "probability": 0.0,
            "confidence":  0.0,
            "warning":     True,
            "warning_msg": msg,
            "audio_type":  vad["audio_type"],
            "spectrogram": spectrogram_b64,
            "vad":         vad,
        }

    # ── Normal inference (speech detected) ───────────────────────────
    features = extract_features_for_cnn(waveform)

    # Universal predict: FullCNN uses predict_proba_single,
    # DeepfakeAudioCNN uses utils.model.predict_single
    if hasattr(model, "predict_proba_single"):
        result = model.predict_proba_single(features)
    else:
        from utils.model import predict_single
        result = predict_single(model, features)
    
    result["spectrogram"] = spectrogram_b64
    result["warning"]     = False
    result["audio_type"]  = vad["audio_type"]
    result["vad"]         = vad
    return result


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if not load_model_once():
        return jsonify({"error": "Model not loaded. Run train_on_dataset.py first."}), 503

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400
    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported format. Allowed: {ALLOWED_EXTS}"}), 400

    uid       = uuid.uuid4().hex[:8]
    safe_name = f"{uid}_{secure_filename(file.filename)}"
    save_path = str(UPLOAD_FOLDER / safe_name)
    file.save(save_path)
    logger.info(f"Received: {safe_name}")

    try:
        result = run_inference(save_path)
        logger.info(f"Prediction -> {result['label']} ({result['confidence']}%)")
        return jsonify(result)
    except Exception as exc:
        logger.exception(f"Inference error: {exc}")
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            os.remove(save_path)
        except OSError:
            pass


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_model_once()
    logger.info("Starting Flask server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
