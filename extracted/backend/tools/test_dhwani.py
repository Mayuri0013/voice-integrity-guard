"""
test_dhwani.py -- standalone check for the Dhwani ONNX detector.

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
    """Load, force mono, resample to 16k. This is the critical check --
    if librosa can't decode the file properly (e.g. a corrupt or unusual
    .ogg container), this either raises loudly or returns near-silence,
    which is exactly what we're checking for here."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)


def windows(y):
    if len(y) <= WINDOW:
        out = np.zeros(WINDOW, dtype=np.float32)
        out[: len(y)] = y
        return [out]
    n = len(y) // WINDOW
    return [y[i * WINDOW : (i + 1) * WINDOW] for i in range(n)]


def run_file(session, in_name, path):
    y = load_audio(path)
    print(f"file        : {path}")
    print(f"duration    : {len(y)/SR:.2f}s  ({len(y)} samples)")
    print(f"peak amp    : {np.max(np.abs(y)):.4f}   (near 0.0 = likely silent/broken decode)")
    print(f"rms energy  : {np.sqrt(np.mean(y**2)):.4f}   (near 0.0 = likely silent/broken decode)")

    chunks = windows(y)
    raws, probs = [], []
    for chunk in chunks:
        out = session.run(None, {in_name: chunk.reshape(1, -1)})[0]
        raw = np.asarray(out).ravel()
        raws.append(raw)
        probs.append(softmax(raw) if raw.size > 1 else np.array([1 - raw[0], raw[0]]))

    mean_prob = np.mean(probs, axis=0)
    print(f"windows     : {len(chunks)}")
    print(f"raw logits  : {np.round(raws[0], 4).tolist()}  (first window)")
    print(f"mean probs  : idx0(genuine)={mean_prob[0]:.4f}  idx1(fake)={mean_prob[-1]:.4f}")
    print("-" * 60)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)

    model_path, audio_paths = sys.argv[1], sys.argv[2:]
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inspect(session)

    in_name = session.get_inputs()[0].name
    for p in audio_paths:
        try:
            run_file(session, in_name, p)
        except Exception as e:
            print(f"file        : {p}\nFAILED      : {e}\n" + "-" * 60)


if __name__ == "__main__":
    main()
