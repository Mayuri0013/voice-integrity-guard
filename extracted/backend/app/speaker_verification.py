"""
Speaker verification using SpeechBrain ECAPA-TDNN.

This module is used ONLY by the two-audio comparison feature.
It does not affect Dhwani's single-audio risk analysis.
"""

import numpy as np
import torch
import librosa

from speechbrain.inference.speaker import EncoderClassifier


SR = 16000
MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"

# Same threshold used in Vaishnavi's original comparison feature
SAME_SPEAKER_THRESHOLD = 0.25


_classifier = None


def _get_classifier():
    global _classifier

    if _classifier is None:
        _classifier = EncoderClassifier.from_hparams(
            source=MODEL_ID,
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
        )

    return _classifier


def _load_mono_16k(path: str):
    """
    Load audio exactly as the original comparison implementation:
    mono + 16 kHz.
    """
    y, _ = librosa.load(
        path,
        sr=SR,
        mono=True,
    )

    return y


def get_embedding(path: str) -> np.ndarray:
    """
    Generate ECAPA-TDNN speaker embedding for one audio file.
    """

    classifier = _get_classifier()

    y = _load_mono_16k(path)

    signal = torch.from_numpy(
        y.astype(np.float32)
    ).unsqueeze(0)

    with torch.no_grad():
        embedding = classifier.encode_batch(signal)

    return embedding.squeeze().cpu().numpy()


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    return float(
        np.dot(a, b)
        /
        (
            np.linalg.norm(a)
            *
            np.linalg.norm(b)
            + 1e-9
        )
    )


def compare(
    genuine_path: str,
    test_path: str,
) -> dict:

    embedding1 = get_embedding(genuine_path)
    embedding2 = get_embedding(test_path)

    similarity = cosine_similarity(
        embedding1,
        embedding2,
    )

    similarity_percent = round(
        (similarity + 1) / 2 * 100,
        1,
    )

    return {
        "cosine_similarity": round(similarity, 4),
        "similarity_percent": similarity_percent,
        "same_speaker_likely": (
            similarity > SAME_SPEAKER_THRESHOLD
        ),
    }