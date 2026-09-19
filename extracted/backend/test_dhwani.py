"""
test_dhwani.py -- standalone check for the Dhwani ONNX detector.

Run this BEFORE touching risk_engine.py / main.py.
Goal: confirm the model loads, confirm its real input/output contract,
and confirm which output index means "fake".

Usage:
    pip install onnxruntime librosa numpy soundfile
    python test_dhwani.py path\to\dhwani_best_model.onnx path\to\audio.wav [more.wav ...]
"""

import sys
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("Missing onnxruntime. Run: pip install onnxruntime")

try:
    import librosa
except ImportError:
    sys.exit("Missing librosa. Run: pip install librosa soundfile")


SR = 16000
WINDOW = 48000  # 3 seconds, per the model card


def softmax(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    e = np.exp(x - x.max())
    return e / e.sum()


def inspect(session):
    print("=" * 60)
    print("MODEL CONTRACT")
    print("=" * 60)
    for i in session.get_inputs():
        print(f"  INPUT   name={i.name!r}  shape={i.shape}  type={i.type}")
    for o in session.get_outputs():
        print(f"  OUTPUT  name={o.name!r}  shape={o.shape}  type={o.type}")
    print()


def load_audio(path):
    """Load, force mono, resample to 16k."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)


def windows(y):
    """
    Split into non-overlapping 3s windows.
    The model only accepts 48000 samples, so longer audio must be chunked.
    A short clip is zero-padded to one full window.
    """
    if len(y) <= WINDOW:
        out = np.zeros(WINDOW, dtype=np.float32)
        out[: len(y)] = y
        return [out]
    n = len(y) // WINDOW
    return [y[i * WINDOW : (i + 1) * WINDOW] for i in range(n)]


def zscore(chunk):
    """Zero-mean, unit-variance normalization (standard wav2vec2 preprocessing)."""
    return (chunk - np.mean(chunk)) / np.sqrt(np.var(chunk) + 1e-5)


def run_file(session, in_name, path, label):
    y = load_audio(path)
    chunks = windows(y)

    print(f"file      : {path}")
    print(f"duration  : {len(y)/SR:.2f}s  ->  {len(chunks)} window(s)")

    for mode, prep in (("RAW", lambda c: c), ("NORMALIZED", zscore)):
        probs = []
        for chunk in chunks:
            shaped = prep(chunk).astype(np.float32).reshape(1, -1)
            out = session.run(None, {in_name: shaped})[0]
            raw = np.asarray(out).ravel()
            probs.append(softmax(raw) if raw.size > 1 else np.array([1 - raw[0], raw[0]]))
        mean_prob = np.mean(probs, axis=0)
        print(f"  [{mode:10s}] raw_logits(first win)={np.round(raw,4).tolist()}  "
              f"mean_probs: idx0={mean_prob[0]:.4f} idx1={mean_prob[-1]:.4f}")

    print("-" * 60)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)

    model_path, audio_paths = sys.argv[1], sys.argv[2:]

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inspect(session)
    in_name = session.get_inputs()[0].name

    for i, p in enumerate(audio_paths):
        try:
            run_file(session, in_name, p, label=f"clip{i}")
        except Exception as e:
            print(f"file      : {p}\nFAILED    : {e}\n" + "-" * 60)

    print()
    print("WHAT TO DO WITH THIS OUTPUT")
    print("Compare RAW vs NORMALIZED rows per file. If NORMALIZED gives")
    print("different (non-saturated) numbers between your two clips, that's")
    print("the correct preprocessing -- use it. If BOTH modes still give near-")
    print("identical output on both files, the demo files themselves are too")
    print("similar for the model to tell apart -- you need a real recorded")
    print("voice clip and a real TTS-generated clip to identify the fake index.")


if __name__ == "__main__":
    main()
