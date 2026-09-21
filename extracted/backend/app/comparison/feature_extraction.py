"""
Multi-Layer Voice Authenticity Analysis.

Extracts real acoustic + prosodic features from a waveform: MFCCs, spectral
flatness/centroid/bandwidth/rolloff, pitch (via librosa.pyin), jitter/shimmer
proxies, and pause ratio. These are genuine computations on the raw signal,
not mocked values.
"""
import numpy as np
import librosa

SAMPLE_RATE = 16000


def load_audio(path_or_buffer, sr=SAMPLE_RATE):
    y, _ = librosa.load(path_or_buffer, sr=sr, mono=True)
    return y


def _safe(val, default=0.0):
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return default
    return float(val)


def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    if y is None or len(y) < sr * 0.3:
        # too short / empty -> return neutral feature vector
        return _empty_features()

    y = librosa.util.normalize(y)

    # --- Spectral features ---
    spec_flatness = librosa.feature.spectral_flatness(y=y)[0]
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # --- Pitch tracking (pyin) ---
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        f0_voiced = f0[voiced_flag] if f0 is not None else np.array([])
        f0_voiced = f0_voiced[~np.isnan(f0_voiced)]
    except Exception:
        f0_voiced = np.array([])

    if len(f0_voiced) > 3:
        pitch_mean = np.mean(f0_voiced)
        pitch_std = np.std(f0_voiced)
        # jitter proxy: mean abs relative frame-to-frame pitch change
        diffs = np.abs(np.diff(f0_voiced))
        jitter_proxy = np.mean(diffs / (f0_voiced[:-1] + 1e-6))
    else:
        pitch_mean, pitch_std, jitter_proxy = 0.0, 0.0, 0.0

    # --- Amplitude / shimmer proxy ---
    rms = librosa.feature.rms(y=y)[0]
    shimmer_proxy = np.std(rms) / (np.mean(rms) + 1e-6)

    # --- Pause ratio (silence detection) ---
    intervals = librosa.effects.split(y, top_db=30)
    voiced_samples = sum(e - s for s, e in intervals)
    pause_ratio = 1.0 - (voiced_samples / max(len(y), 1))

    features = {
        "mfcc_mean": [_safe(x) for x in np.mean(mfcc, axis=1)],
        "mfcc_std": [_safe(x) for x in np.std(mfcc, axis=1)],
        "spectral_flatness_mean": _safe(np.mean(spec_flatness)),
        "spectral_flatness_std": _safe(np.std(spec_flatness)),
        "spectral_centroid_mean": _safe(np.mean(spec_centroid)),
        "spectral_bandwidth_mean": _safe(np.mean(spec_bandwidth)),
        "spectral_rolloff_mean": _safe(np.mean(spec_rolloff)),
        "pitch_mean_hz": _safe(pitch_mean),
        "pitch_std_hz": _safe(pitch_std),
        "jitter_proxy": _safe(jitter_proxy),
        "shimmer_proxy": _safe(shimmer_proxy),
        "pause_ratio": _safe(pause_ratio),
        "duration_sec": _safe(len(y) / sr),
    }
    return features


def _empty_features() -> dict:
    return {
        "mfcc_mean": [0.0] * 13,
        "mfcc_std": [0.0] * 13,
        "spectral_flatness_mean": 0.0,
        "spectral_flatness_std": 0.0,
        "spectral_centroid_mean": 0.0,
        "spectral_bandwidth_mean": 0.0,
        "spectral_rolloff_mean": 0.0,
        "pitch_mean_hz": 0.0,
        "pitch_std_hz": 0.0,
        "jitter_proxy": 0.0,
        "shimmer_proxy": 0.0,
        "pause_ratio": 0.0,
        "duration_sec": 0.0,
    }


def features_to_vector(features: dict) -> np.ndarray:
    """Flatten the feature dict into a fixed-order numeric vector for ML input."""
    vec = list(features["mfcc_mean"]) + list(features["mfcc_std"]) + [
        features["spectral_flatness_mean"],
        features["spectral_flatness_std"],
        features["spectral_centroid_mean"],
        features["spectral_bandwidth_mean"],
        features["spectral_rolloff_mean"],
        features["pitch_std_hz"],
        features["jitter_proxy"],
        features["shimmer_proxy"],
        features["pause_ratio"],
    ]
    return np.array(vec, dtype=np.float32)
