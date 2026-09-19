"""
risk_engine.py
--------------
Combines two scoring strategies into one final impersonation risk score
(0-100), matching the "Real-Time Risk Scoring Engine" component of the
problem statement:

  1. HEURISTIC SCORER  -- interpretable, rule-based, works out of the box
     with zero training data. Encodes patterns reported in voice-spoofing
     literature: natural human speech has more micro-variation (jitter,
     shimmer, pitch variance) than most neural TTS/vocoder output, and
     vocoder artifacts often show up as unusually flat/regular spectral
     energy. This is the layer you can defend & explain to judges today.

  2. ML SCORER (pluggable) -- a trained classifier (RandomForest baseline
     shipped here, trained on a *synthetic proxy dataset* -- see
     train_baseline.py) that outputs a probability. This slot is built so
     that when your team obtains a real labeled corpus (ASVspoof 2019/2021,
     your own recorded genuine vs. cloned samples, etc.) you retrain
     ml_model.py's classifier and drop it in -- no other code changes.

Final score = weighted blend of both, plus contextual risk boosters
(unknown caller, high-value transaction context, etc.) as described in
the "Contextual enrichment" bullet of the problem statement.
"""

from __future__ import annotations
import numpy as np
from .ml_model import ml_predict_proba, MODEL_LOADED
from .dhwani_model import dhwani_predict_proba, DHWANI_LOADED

# ---- Heuristic reference ranges for natural human speech ----
# These are reasonable, literature-informed starting points (NOT
# calibrated on a labeled dataset). Document clearly: recalibrate these
# once real labeled data is available.
NATURAL_RANGES = {
    "jitter_proxy": (0.015, 0.09),       # natural speech shows measurable jitter
    "shimmer_proxy": (0.05, 0.35),       # natural amplitude micro-variation
    "pitch_std_hz": (8.0, 55.0),         # natural pitch contour variability
    "spectral_flatness_mean": (0.0, 0.35),  # very high flatness -> noise-like/vocoder artifact
    "zcr_std": (0.01, 0.09),
}

WEIGHTS = {
    "jitter_proxy": 0.22,
    "shimmer_proxy": 0.18,
    "pitch_std_hz": 0.22,
    "spectral_flatness_mean": 0.20,
    "zcr_std": 0.18,
}

# Below these thresholds, there isn't enough actual voiced speech in the
# clip to say anything meaningful. Without this gate, "no pitch detected"
# (jitter/shimmer/pitch_std all default to 0) gets misread by the
# heuristic engine as "suspiciously smooth -> AI-like", which is why
# background noise / silence was scoring as MEDIUM risk -- it isn't smooth
# speech, it's an absence of speech, and those are not the same thing.
MIN_VOICED_RATIO = 0.15
MIN_RMS_MEAN = 0.004
MIN_DURATION_SEC = 0.6


def has_sufficient_speech(features: dict) -> bool:
    if not features:
        return False
    if features.get("duration_sec", 0.0) < MIN_DURATION_SEC:
        return False
    if features.get("voiced_ratio", 0.0) < MIN_VOICED_RATIO:
        return False
    if features.get("rms_mean", 0.0) < MIN_RMS_MEAN:
        return False
    return True


def _deviation_score(value: float, low: float, high: float) -> float:
    """Returns 0.0 (well within natural range) to 1.0 (far outside it,
    i.e. suspiciously smooth/regular -> synthetic-like)."""
    if value >= low and value <= high:
        return 0.0
    if value < low:
        span = max(low, 1e-6)
        return float(np.clip((low - value) / span, 0, 1))
    span = max(high, 1e-6)
    return float(np.clip((value - high) / (2 * span), 0, 1))


def heuristic_score(features: dict) -> tuple[float, dict]:
    """Returns (score 0-100, per-feature breakdown) using only the
    'too smooth / too regular' direction for jitter/shimmer/pitch_std
    (since suspiciously LOW variation is the synthetic signal), and
    both directions for spectral flatness / zcr."""
    if not features:
        return 0.0, {}

    breakdown = {}
    total = 0.0
    total_weight = 0.0

    for feat, weight in WEIGHTS.items():
        val = features.get(feat, None)
        if val is None:
            continue
        low, high = NATURAL_RANGES[feat]
        if feat in ("jitter_proxy", "shimmer_proxy", "pitch_std_hz"):
            # only penalize being BELOW natural range (too smooth = synthetic-like)
            dev = _deviation_score(val, low, high) if val < low else 0.0
        else:
            dev = _deviation_score(val, low, high)
        breakdown[feat] = round(dev * 100, 1)
        total += dev * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0, {}

    score = (total / total_weight) * 100
    return float(np.clip(score, 0, 100)), breakdown


def contextual_boost(context: dict | None) -> float:
    """Additive risk points from call metadata (0-20 extra points),
    matching the 'Contextual enrichment' bullet."""
    if not context:
        return 0.0
    boost = 0.0
    if context.get("unknown_caller"):
        boost += 8
    if context.get("high_value_transaction"):
        boost += 8
    if context.get("privileged_request"):  # e.g. "share OTP", "approve transfer"
        boost += 4
    return min(boost, 20.0)


def compute_risk(features: dict, context: dict | None = None, raw_audio=None, sr: int | None = None) -> dict:
    """Main entry point: blends heuristic + ML score + context into a
    final risk score and a human-readable recommendation, matching the
    'Alerting and User Interaction Layer' bullet.

    ML scoring priority:
      1. Dhwani (real pretrained deepfake-audio model) if loaded and
         raw_audio/sr are provided -- this is the signal that actually
         generalizes to real AI-cloned voices.
      2. Placeholder RandomForest (ml_model.py) as a fallback -- trained
         on synthetic proxy data only, kept for when Dhwani's model file
         isn't present, but should not be trusted for real accuracy claims.
    """
    h_score, breakdown = heuristic_score(features)

    # If there isn't enough real voiced speech in this clip, don't run any
    # scoring at all -- report it plainly instead of guessing.
    if not has_sufficient_speech(features):
        return {
            "risk_score": 0.0,
            "risk_level": "LOW",
            "heuristic_score": 0.0,
            "ml_score": None,
            "ml_source": None,
            "context_boost": 0.0,
            "feature_breakdown": {},
            "recommendation": (
                "No clear speech detected in this audio window (mostly silence "
                "or background noise). Speak closer to the microphone and try again."
            ),
            "insufficient_audio": True,
        }

    ml_score = None
    ml_source = None

    if DHWANI_LOADED and raw_audio is not None and sr is not None:
        try:
            ml_score = dhwani_predict_proba(raw_audio, sr) * 100
            ml_source = "dhwani"
        except Exception:
            ml_score = None

    if ml_score is None and MODEL_LOADED and features:
        try:
            ml_score = ml_predict_proba(features) * 100
            ml_source = "placeholder"
        except Exception:
            ml_score = None

    if ml_score is not None:
        if ml_source == "dhwani":
            # trust the real model more heavily than the hand-tuned heuristics
            blended = 0.25 * h_score + 0.75 * ml_score
        else:
            blended = 0.5 * h_score + 0.5 * ml_score
    else:
        blended = h_score

    ctx_boost = contextual_boost(context)
    final_score = float(np.clip(blended + ctx_boost, 0, 100))

    if final_score >= 70:
        level = "HIGH"
        recommendation = (
            "High likelihood of synthetic/cloned voice. Do NOT proceed with "
            "sensitive action. Trigger callback verification on a known-good "
            "number and escalate to supervisor."
        )
    elif final_score >= 40:
        level = "MEDIUM"
        recommendation = (
            "Some indicators of possible voice manipulation. Recommend secondary "
            "verification (MFA / callback) before proceeding."
        )
    else:
        level = "LOW"
        recommendation = "No strong indicators of synthetic voice detected. Proceed with normal caution."

    return {
        "risk_score": round(final_score, 1),
        "risk_level": level,
        "heuristic_score": round(h_score, 1),
        "ml_score": round(ml_score, 1) if ml_score is not None else None,
        "ml_source": ml_source,  # "dhwani" (real model) | "placeholder" | None
        "context_boost": ctx_boost,
        "feature_breakdown": breakdown,
        "recommendation": recommendation,
        "insufficient_audio": False,
    }
