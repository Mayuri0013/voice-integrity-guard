"""
train_baseline.py
------------------
Builds the baseline classifier used by ml_model.py.

By default this generates a SYNTHETIC PROXY DATASET: feature vectors
sampled from two distributions engineered to reflect the *direction* of
known genuine-vs-synthetic differences (see risk_engine.py's
NATURAL_RANGES comments). This lets the full train -> save -> load ->
predict pipeline run and be demoed TODAY, with no external dataset
download required (useful since this sandbox has no general internet
access to fetch ASVspoof etc.).

TO USE REAL DATA LATER (recommended before any real deployment/demo
claims of accuracy):
  1. Get a labeled dataset, e.g. ASVspoof 2019/2021 LA (bonafide vs
     spoof), or your own recordings (genuine microphone speech vs.
     samples generated with Coqui-TTS / RVC / etc.).
  2. Write a loader that walks the dataset, calls
     feature_extraction.extract_features() on each file, and yields
     (feature_dict, label) pairs -- label 0 = genuine, 1 = synthetic.
  3. Replace the call to `generate_synthetic_dataset()` below with your
     loader's output and re-run this script.
No other file needs to change -- ml_model.py just loads whatever
baseline_model.joblib this script produces.
"""

from __future__ import annotations
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

import sys
sys.path.insert(0, os.path.dirname(__file__))
from app.ml_model import FEATURE_ORDER

RNG = np.random.default_rng(42)
MODEL_OUT = os.path.join(os.path.dirname(__file__), "app", "models", "baseline_model.joblib")


def generate_synthetic_dataset(n_per_class: int = 1500):
    """Generates proxy feature vectors for two classes. This is a
    STAND-IN for real labeled audio -- see module docstring."""
    rows = []
    labels = []

    for _ in range(n_per_class):
        # "genuine-like": higher micro-variation, moderate spectral flatness
        row = {
            "spectral_flatness_mean": RNG.normal(0.18, 0.08),
            "spectral_flatness_std": RNG.normal(0.05, 0.02),
            "spectral_centroid_mean": RNG.normal(1800, 400),
            "spectral_bandwidth_mean": RNG.normal(1600, 300),
            "spectral_rolloff_mean": RNG.normal(3200, 600),
            "zcr_mean": RNG.normal(0.08, 0.03),
            "zcr_std": RNG.normal(0.045, 0.02),
            "pitch_mean_hz": RNG.normal(160, 40),
            "pitch_std_hz": RNG.normal(30, 10),
            "jitter_proxy": RNG.normal(0.045, 0.015),
            "voiced_ratio": RNG.normal(0.6, 0.15),
            "shimmer_proxy": RNG.normal(0.18, 0.07),
            "rms_std": RNG.normal(0.02, 0.008),
            "pause_ratio": RNG.normal(0.25, 0.1),
            "duration_sec": RNG.normal(3.0, 1.0),
        }
        for i in range(1, 14):
            row[f"mfcc{i}_mean"] = RNG.normal(0, 15)
            row[f"mfcc{i}_std"] = RNG.normal(8, 3)
        rows.append(row)
        labels.append(0)  # genuine

    for _ in range(n_per_class):
        # "synthetic-like": smoother pitch/jitter/shimmer, more regular spectral flatness
        row = {
            "spectral_flatness_mean": RNG.normal(0.30, 0.09),
            "spectral_flatness_std": RNG.normal(0.02, 0.01),
            "spectral_centroid_mean": RNG.normal(1900, 350),
            "spectral_bandwidth_mean": RNG.normal(1500, 250),
            "spectral_rolloff_mean": RNG.normal(3100, 500),
            "zcr_mean": RNG.normal(0.07, 0.02),
            "zcr_std": RNG.normal(0.02, 0.01),
            "pitch_mean_hz": RNG.normal(165, 35),
            "pitch_std_hz": RNG.normal(12, 6),
            "jitter_proxy": RNG.normal(0.012, 0.006),
            "voiced_ratio": RNG.normal(0.7, 0.1),
            "shimmer_proxy": RNG.normal(0.04, 0.02),
            "rms_std": RNG.normal(0.012, 0.005),
            "pause_ratio": RNG.normal(0.18, 0.08),
            "duration_sec": RNG.normal(3.0, 1.0),
        }
        for i in range(1, 14):
            row[f"mfcc{i}_mean"] = RNG.normal(0, 14)
            row[f"mfcc{i}_std"] = RNG.normal(5, 2)
        rows.append(row)
        labels.append(1)  # synthetic

    X = np.array([[r.get(f, 0.0) for f in FEATURE_ORDER] for r in rows], dtype=np.float32)
    y = np.array(labels)
    return X, y


def main():
    print("Generating synthetic proxy training data...")
    X, y = generate_synthetic_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training RandomForestClassifier baseline...")
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n--- Evaluation on held-out synthetic proxy test set ---")
    print(classification_report(y_test, y_pred, target_names=["genuine", "synthetic"]))
    print("(Reminder: this metric is on the SYNTHETIC PROXY dataset only, not real audio.)")

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(clf, MODEL_OUT)
    print(f"\nSaved baseline model -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
