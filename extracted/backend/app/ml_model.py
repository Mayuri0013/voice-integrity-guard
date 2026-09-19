"""
ml_model.py
-----------
Pluggable ML classifier slot for the risk engine.

IMPORTANT / READ THIS:
The shipped model (models/baseline_model.joblib) is trained on a
PROGRAMMATICALLY SYNTHESIZED proxy dataset (see train_baseline.py) that
mimics the *statistical direction* of known differences between natural
and TTS/vocoder speech (e.g. lower pitch/jitter/shimmer variance,
different spectral flatness). It exists so the full pipeline (train ->
save -> load -> predict -> blend) is real and working end-to-end today.

It is NOT trained on real human vs. real cloned-voice recordings, so its
raw accuracy claims should not be overstated to judges/evaluators as
"production accuracy." Presented honestly, it demonstrates a working,
retrainable architecture -- the exact place you plug in ASVspoof 2019/2021
or your own recorded genuine-vs-cloned dataset later. Retraining is a
single command: `python train_baseline.py --data-dir /path/to/real_data`
once you extend train_baseline.py's loader to read real labeled audio
instead of the synthetic generator.
"""

from __future__ import annotations
import os
import joblib
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "baseline_model.joblib")

FEATURE_ORDER = [
    "spectral_flatness_mean", "spectral_flatness_std", "spectral_centroid_mean",
    "spectral_bandwidth_mean", "spectral_rolloff_mean", "zcr_mean", "zcr_std",
] + [f"mfcc{i}_mean" for i in range(1, 14)] + [f"mfcc{i}_std" for i in range(1, 14)] + [
    "pitch_mean_hz", "pitch_std_hz", "jitter_proxy", "voiced_ratio",
    "shimmer_proxy", "rms_std", "pause_ratio", "duration_sec",
]

_model = None
MODEL_LOADED = False

if os.path.exists(MODEL_PATH):
    try:
        _model = joblib.load(MODEL_PATH)
        MODEL_LOADED = True
    except Exception:
        _model = None
        MODEL_LOADED = False


def features_to_vector(features: dict) -> np.ndarray:
    return np.array([[features.get(f, 0.0) for f in FEATURE_ORDER]], dtype=np.float32)


def ml_predict_proba(features: dict) -> float:
    """Returns probability (0-1) that the sample is synthetic/cloned."""
    if _model is None:
        raise RuntimeError("Model not loaded")
    x = features_to_vector(features)
    proba = _model.predict_proba(x)[0]
    # class order established in train_baseline.py: 0=genuine, 1=synthetic
    return float(proba[1])
