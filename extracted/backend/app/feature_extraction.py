"""
feature_extraction.py
----------------------
Extracts real acoustic, spectral, and prosodic features from raw audio.
These are the discriminative signals the risk engine uses to judge
whether a voice sample looks "naturally human" or shows patterns
associated with AI/TTS/voice-cloning generation.

Feature groups (mirrors the problem statement's "Multi-Layer Voice
Authenticity Analysis"):
  1. Spectral / acoustic artifacts   -> spectral_flatness, spectral_centroid,
                                         spectral_bandwidth, spectral_rolloff,
                                         zero_crossing_rate, mfcc stats
  2. Prosody / behavioral micro-variation -> pitch (F0) mean/std, jitter,
                                         shimmer, pause ratio, speaking rate proxy
"""

from __future__ import annotations
import numpy as np
import librosa

SAMPLE_RATE = 16000  # standard rate for speech analysis; we resample everything to this


def load_audio_from_bytes(audio_bytes: bytes) -> np.ndarray:
    """Decode raw audio bytes (wav/flac/etc via soundfile backend) into a
    mono float32 waveform resampled to SAMPLE_RATE."""
    import io
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)  # downmix to mono
    if sr != SAMPLE_RATE:
        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
    return data


def _safe(val, default=0.0):
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return default
    return float(val)


def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Compute the full feature vector for a waveform chunk.

    Returns a flat dict of named scalar features -- deliberately flat
    (not raw audio) so it can be logged/stored without exposing the
    underlying voice recording (see privacy.py).
    """
    if y is None or len(y) < int(0.25 * sr):
        # too short a chunk to say anything meaningful
        return {}

    y = y.astype(np.float32)
    # trim near-silence so silence doesn't dilute stats
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    if len(y_trimmed) < int(0.1 * sr):
        y_trimmed = y

    features = {}

    # ---- 1. Spectral / synthesis-artifact features ----
    stft = np.abs(librosa.stft(y_trimmed, n_fft=512, hop_length=160))
    flatness = librosa.feature.spectral_flatness(S=stft)
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y_trimmed)

    features["spectral_flatness_mean"] = _safe(np.mean(flatness))
    features["spectral_flatness_std"] = _safe(np.std(flatness))
    features["spectral_centroid_mean"] = _safe(np.mean(centroid))
    features["spectral_bandwidth_mean"] = _safe(np.mean(bandwidth))
    features["spectral_rolloff_mean"] = _safe(np.mean(rolloff))
    features["zcr_mean"] = _safe(np.mean(zcr))
    features["zcr_std"] = _safe(np.std(zcr))

    # MFCCs -- 13 coefficients, mean+std each (summarized, not raw)
    mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc{i+1}_mean"] = _safe(np.mean(mfcc[i]))
        features[f"mfcc{i+1}_std"] = _safe(np.std(mfcc[i]))

    # ---- 2. Prosody / micro-variation features ----
    # Pitch (F0) tracking via pyin (probabilistic YIN) -- robust for speech
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y_trimmed, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
        f0_voiced = f0_voiced[~np.isnan(f0_voiced)] if f0_voiced.size else f0_voiced
    except Exception:
        f0_voiced = np.array([])

    if f0_voiced.size > 3:
        features["pitch_mean_hz"] = _safe(np.mean(f0_voiced))
        features["pitch_std_hz"] = _safe(np.std(f0_voiced))
        # jitter proxy: average absolute frame-to-frame pitch deviation, normalized
        diffs = np.abs(np.diff(f0_voiced))
        features["jitter_proxy"] = _safe(np.mean(diffs) / (np.mean(f0_voiced) + 1e-6))
        voiced_ratio = float(np.count_nonzero(voiced_flag)) / max(len(voiced_flag), 1)
        features["voiced_ratio"] = _safe(voiced_ratio)
    else:
        features["pitch_mean_hz"] = 0.0
        features["pitch_std_hz"] = 0.0
        features["jitter_proxy"] = 0.0
        features["voiced_ratio"] = 0.0

    # shimmer proxy: frame-to-frame amplitude envelope variation
    rms = librosa.feature.rms(y=y_trimmed, frame_length=512, hop_length=160)[0]
    if rms.size > 3 and np.mean(rms) > 1e-6:
        amp_diffs = np.abs(np.diff(rms))
        features["shimmer_proxy"] = _safe(np.mean(amp_diffs) / (np.mean(rms) + 1e-6))
        features["rms_std"] = _safe(np.std(rms))
    else:
        features["shimmer_proxy"] = 0.0
        features["rms_std"] = 0.0
    features["rms_mean"] = _safe(np.mean(rms)) if rms.size else 0.0

    # pause ratio: fraction of frames below an energy threshold (speaking rhythm proxy)
    if rms.size:
        thresh = 0.1 * np.max(rms) if np.max(rms) > 0 else 0
        pause_ratio = float(np.count_nonzero(rms < thresh)) / rms.size
        features["pause_ratio"] = _safe(pause_ratio)
    else:
        features["pause_ratio"] = 0.0

    features["duration_sec"] = _safe(len(y_trimmed) / sr)

    return features


FEATURE_ORDER = None  # populated lazily by risk_engine/ml_model to keep vector order consistent
