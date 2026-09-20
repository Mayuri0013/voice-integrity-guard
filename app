"""
speaker_verification.py
------------------------
"Cross-session consistency checks" from the problem statement: compares
a live/test voice sample against a known genuine reference sample of the
same claimed speaker, and reports how likely they are to be the same
person's voice.

This answers a DIFFERENT question than the deepfake detector (dhwani_model.py):
  - Deepfake detector: "does this audio sound AI-generated at all?"
  - Speaker verification (this module): "does this voice match THIS
    specific person's known voice?"

Both matter for the full problem statement -- a clone could theoretically
be detected as "not obviously synthetic" by the deepfake model but still
fail a speaker-match check against the real account holder's enrolled
voice, and vice versa.

MODEL: Resemblyzer (a lightweight, pretrained speaker-embedding encoder,
~17MB, bundled with the package -- no separate download needed, unlike
Dhwani). It converts a voice clip into a fixed-length numeric "voiceprint"
vector; comparing two voiceprints with cosine similarity tells you how
likely they are to be the same speaker.

CALIBRATION NOTE: the MATCH_THRESHOLD below (0.75) is a commonly-used
community default for Resemblyzer embeddings, not a value calibrated on
a labeled dataset by us. As with the deepfake heuristics, recalibrate
this once you have real labeled same-speaker / different-speaker pairs
to validate against (see tools/batch_test.py for the same style of
validation harness, adaptable to speaker-pair testing).
"""

from __future__ import annotations
import numpy as np

MATCH_THRESHOLD = 0.75  # cosine similarity above this = same-speaker match

_encoder = None
ENCODER_LOADED = False

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    _encoder = VoiceEncoder()
    ENCODER_LOADED = True
except Exception as e:
    print(f"[speaker_verification] Could not load Resemblyzer voice encoder: {e}")
    ENCODER_LOADED = False


def _embed(y: np.ndarray, sr: int) -> np.ndarray:
    """Resemblyzer expects its own preprocessing (VAD trimming, resampling
    to 16kHz internally) -- preprocess_wav handles a raw waveform + rate."""
    from resemblyzer import preprocess_wav
    wav = preprocess_wav(y, source_sr=sr)
    return _encoder.embed_utterance(wav)


def compare_voices(y1: np.ndarray, sr1: int, y2: np.ndarray, sr2: int) -> dict:
    """Compares two voice samples and returns a similarity score (0-100),
    a match verdict, and a recommendation -- mirroring the same
    score/level/recommendation shape as risk_engine.compute_risk so the
    frontend can reuse similar rendering logic."""
    if not ENCODER_LOADED:
        return {
            "error": "Speaker verification model not loaded on the server.",
        }

    emb1 = _embed(y1, sr1)
    emb2 = _embed(y2, sr2)

    similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))
    similarity_pct = round(similarity * 100, 1)
    is_match = similarity >= MATCH_THRESHOLD

    if similarity >= 0.85:
        level = "STRONG_MATCH"
        recommendation = "Very likely the same speaker. Voice identity confirmed with high confidence."
    elif similarity >= MATCH_THRESHOLD:
        level = "LIKELY_MATCH"
        recommendation = "Likely the same speaker, but confidence is moderate. Consider a secondary check for high-stakes actions."
    elif similarity >= 0.55:
        level = "UNCERTAIN"
        recommendation = "Inconclusive. Could be the same speaker in a noisy/short clip, or a different speaker. Do not rely on this alone -- request additional verification."
    else:
        level = "NO_MATCH"
        recommendation = "Very likely a different speaker (or a voice clone that doesn't fully match the reference). Do not treat this caller as verified."

    return {
        "similarity_score": similarity_pct,
        "is_match": is_match,
        "match_level": level,
        "threshold_used": MATCH_THRESHOLD * 100,
        "recommendation": recommendation,
    }
