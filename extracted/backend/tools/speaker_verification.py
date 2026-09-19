"""
speaker_verification.py
------------------------
Answers a DIFFERENT question than dhwani_model.py.

dhwani_model.py asks: "Is THIS clip synthetic/AI-generated?"
This module asks:      "Do THESE TWO clips belong to the same speaker?"

Together they cover the real attack scenario: someone submits a clip
claiming to be a known person (e.g. a customer's enrolled voice sample).
We check (a) is the incoming clip a deepfake at all, and (b) does it
even sound like the claimed person, using a real pretrained speaker
embedding model (ECAPA-TDNN, trained on VoxCeleb1+2, EER 0.69%).

Model: speechbrain/spkrec-ecapa-voxceleb
  - Downloaded automatically on first use (~80MB), cached locally after
  - Requires: pip install speechbrain torchaudio
    (this pulls in torch -- unlike torchcodec earlier, speechbrain's own
    install path is well-tested and matches its torch version automatically,
    so this should install cleanly without the earlier version conflict)

USAGE:
    from speaker_verification import compare_speakers
    result = compare_speakers("reference.wav", "suspect.wav")
    # result = {"same_speaker": True/False, "similarity_score": 0.0-1.0}
"""

from __future__ import annotations
import os
from functools import lru_cache

# Similarity threshold: SpeechBrain's own verify_files() uses ~0.25 by
# default (on raw cosine similarity, not normalized 0-1), but that default
# is tuned for VoxCeleb's specific test conditions. Treat this as a
# starting point to validate against your own real/impostor pairs before
# trusting it in a demo -- same lesson as the risk_engine.py thresholds.
DEFAULT_THRESHOLD = 0.25


@lru_cache(maxsize=1)
def _get_verifier():
    """
    Loads the model once and caches it -- this is a ~80MB download on
    first call (cached to disk after), and loading it fresh per-request
    would be slow. lru_cache ensures it only happens once per server run.

    local_strategy=COPY avoids Windows' symlink permission requirement
    (creating symlinks needs admin rights or Developer Mode on Windows by
    default). Copying uses a bit more disk space than symlinking, but for
    an ~80MB model that's negligible, and it works for every user out of
    the box with no elevation needed.
    """
    from speechbrain.inference.speaker import SpeakerRecognition
    from speechbrain.utils.fetching import LocalStrategy
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        local_strategy=LocalStrategy.COPY,
    )


def _load_as_tensor(path: str):
    """
    Loads audio with soundfile (reliable on Windows) instead of letting
    torchaudio load it internally -- torchaudio's own backend has known
    Windows issues ("System error" on perfectly valid WAV files) that
    have nothing to do with the file itself. soundfile has worked
    reliably throughout this project, so we reuse it here and hand the
    model an already-loaded tensor instead.
    """
    import numpy as np
    import soundfile as sf
    import torch

    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:  # collapse stereo to mono if needed
        data = np.mean(data, axis=1)
    tensor = torch.from_numpy(data).unsqueeze(0)  # shape: (1, num_samples)
    return tensor, sr


def compare_speakers(reference_path: str, suspect_path: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Compares two audio files and estimates whether they're the same speaker.

    Returns:
        {
            "same_speaker": bool,
            "similarity_score": float,   # cosine similarity, roughly -1 to 1
            "threshold_used": float,
            "source": "speechbrain-ecapa" | "error"
            "error": str (only present on failure)
        }
    """
    if not os.path.exists(reference_path):
        return {"source": "error", "error": f"reference file not found: {reference_path}"}
    if not os.path.exists(suspect_path):
        return {"source": "error", "error": f"suspect file not found: {suspect_path}"}

    try:
        import torch.nn.functional as F

        verifier = _get_verifier()
        ref_tensor, ref_sr = _load_as_tensor(reference_path)
        sus_tensor, sus_sr = _load_as_tensor(suspect_path)

        # The model expects 16kHz; resample if the file differs.
        ref_tensor = _resample_if_needed(ref_tensor, ref_sr)
        sus_tensor = _resample_if_needed(sus_tensor, sus_sr)

        ref_emb = verifier.encode_batch(ref_tensor)
        sus_emb = verifier.encode_batch(sus_tensor)

        score = F.cosine_similarity(ref_emb.squeeze(1), sus_emb.squeeze(1)).item()
        same_speaker = score >= threshold
        return {
            "same_speaker": same_speaker,
            "similarity_score": round(score, 4),
            "threshold_used": threshold,
            "source": "speechbrain-ecapa",
        }
    except Exception as e:
        # Mirror dhwani_model.py's philosophy: never silently return a
        # default verdict on failure. Surface the error so it's visible
        # in the API response and logs, not hidden behind a fake "0" score.
        return {"source": "error", "error": str(e)}


def _resample_if_needed(tensor, sr, target_sr: int = 16000):
    if sr == target_sr:
        return tensor
    import torchaudio.functional as AF
    return AF.resample(tensor, sr, target_sr)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python speaker_verification.py reference.wav suspect.wav")
        sys.exit(1)
    result = compare_speakers(sys.argv[1], sys.argv[2])
    print(result)
