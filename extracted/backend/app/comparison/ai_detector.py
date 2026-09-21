"""
Real pretrained AI-generated / deepfake voice detector.

Uses MelodyMachine/Deepfake-audio-detection-V2, a wav2vec2 model fine-tuned
specifically to classify real vs AI-generated speech (99.7% accuracy on its
own evaluation set). Replaces the old RandomForest baseline, which was only
ever trained on synthetic placeholder data.

First call downloads the model (~360MB) and caches it -- do this once
tonight with working internet, not for the first time tomorrow.
"""
from transformers import pipeline

MODEL_ID = "MelodyMachine/Deepfake-audio-detection-V2"

_pipe = None


def _get_pipe():
    global _pipe
    if _pipe is None:
        _pipe = pipeline("audio-classification", model=MODEL_ID)
    return _pipe


def predict_fake_probability(y, sr: int = 16000) -> float:
    """
    Returns probability [0,1] that the given waveform is AI-generated/fake.
    `y` is a mono float32 numpy array.
    """
    pipe = _get_pipe()
    results = pipe({"array": y, "sampling_rate": sr})
    print(f"[DEBUG] model output: {results}")

    fake_keywords = ("fake", "spoof", "synthetic", "ai", "generated", "label_1", "1")
    real_keywords = ("real", "bonafide", "genuine", "human", "label_0", "0")

    for r in results:
        label = r["label"].strip().lower()
        if any(k in label for k in fake_keywords):
            return float(r["score"])
    for r in results:
        label = r["label"].strip().lower()
        if any(k in label for k in real_keywords):
            return float(1.0 - r["score"])

    print(f"[ai_detector] Unrecognized label scheme: {results}")
    return float(results[0]["score"])