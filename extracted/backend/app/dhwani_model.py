"""
dhwani_model.py
----------------
Wraps a REAL pretrained deepfake-audio detection model ("Dhwani"), trained
on actual genuine vs. actual AI-generated/cloned speech -- unlike the
placeholder RandomForest in ml_model.py, which was trained on synthetic
proxy feature vectors and cannot reliably catch real clones.

Model: https://huggingface.co/ayush2635/Dhwani-Multilingual-Deepfake-Audio-Detection-Model
  - Architecture: Wav2Vec2 XLS-R (300M) front-end + AASIST back-end
    (AASIST is a published, competition-grade anti-spoofing architecture)
  - Trained on: Mozilla Common Voice (real) + real TTS-generated fakes,
    covering English, Hindi, Tamil, Telugu, Malayalam
  - Format: ONNX, 16kHz mono float32, fixed 3-second (48,000 sample) window

SETUP (required -- this file is NOT included in the repo, ~1.2GB):
  1. Visit the model page above, go to "Files and versions"
  2. Download `best_model.onnx`
  3. Place it at: backend/app/models/dhwani_best_model.onnx
That's it -- this module auto-detects the file and activates. If the file
isn't present, DHWANI_LOADED stays False and risk_engine.py falls back to
the heuristic + placeholder-ML blend (with an honest lower confidence).

Known limitation (stated by the model's own authors): may show lower
confidence against very new/unseen zero-shot cloning architectures not in
its training distribution. This is expected and worth stating plainly in
a demo -- no detector is complete against every current and future clone.
"""

from __future__ import annotations
import os
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "dhwani_best_model.onnx")
TARGET_SR = 16000
WINDOW_SAMPLES = 48000  # exactly 3 seconds at 16kHz, fixed by the model

_session = None
DHWANI_LOADED = False

if os.path.exists(MODEL_PATH):
    try:
        import onnxruntime as ort
        _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        DHWANI_LOADED = True
    except Exception as e:
        print(f"[dhwani_model] Found model file but failed to load it: {e}")
        _session = None
        DHWANI_LOADED = False
else:
    print(
        f"[dhwani_model] No model file at {MODEL_PATH} -- running on heuristic + "
        f"placeholder ML only. See dhwani_model.py docstring to enable real detection."
    )


def _prepare_window(y: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16kHz mono if needed, then pad/truncate to exactly 3s."""
    import librosa
    if sr != TARGET_SR:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    if len(y) > WINDOW_SAMPLES:
        y = y[:WINDOW_SAMPLES]
    elif len(y) < WINDOW_SAMPLES:
        y = np.pad(y, (0, WINDOW_SAMPLES - len(y)), mode="constant")
    # per-sample normalization, as specified by the model card
    y = (y - np.mean(y)) / np.sqrt(np.var(y) + 1e-5)
    return y.astype(np.float32).reshape(1, WINDOW_SAMPLES)


def dhwani_predict_proba(y: np.ndarray, sr: int) -> float:
    """Returns probability (0-1) that the audio is AI-generated/cloned,
    using the real pretrained model. Raises if the model isn't loaded --
    callers should check DHWANI_LOADED first."""
    if _session is None:
        raise RuntimeError("Dhwani model not loaded")

    x = _prepare_window(y, sr)
    input_name = _session.get_inputs()[0].name
    logits = _session.run(None, {input_name: x})[0]
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    return float(probs[0][1])  # index 1 = "fake" class per the model card
