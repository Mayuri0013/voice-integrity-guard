"""
download_dataset.py
--------------------
Downloads a small, pre-labeled real-vs-AI-generated speech dataset from
Hugging Face and saves it directly into genuine/ and fake/ folders, ready
for batch_test.py -- no manual zip/extract/sort steps needed.

Dataset used: garystafford/deepfake-audio-detection
  - 1,866 samples total (~933 real, ~933 fake), FLAC 16kHz mono
  - Real: human speech pulled from YouTube recordings
  - Fake: generated using ElevenLabs, Amazon Polly, Hexgrad Kokoro, Hume AI,
    Speechify, and Luvvoice -- genuinely diverse TTS sources, including
    ElevenLabs, the same tool you've already been testing with
  - Much smaller download than ASVspoof's multi-GB package -- typically
    finishes in a few minutes on a normal connection

USAGE:
  pip install datasets soundfile
  python download_dataset.py --out-dir my_dataset --limit 150
This creates:
    my_dataset/genuine/*.wav
    my_dataset/fake/*.wav

NOTE: this version deliberately avoids the datasets library's automatic
audio-decoding path, which pulls in torchcodec -> torch (a multi-GB
dependency you don't need). Instead we request raw audio bytes and decode
them ourselves with soundfile, which you already have installed.
"""

from __future__ import annotations
import argparse
import io
import os

import soundfile as sf
from datasets import load_dataset, Audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="my_dataset", help="Where to create genuine/ and fake/ folders")
    parser.add_argument("--limit", type=int, default=150, help="Max files per class (default 150 = 300 total)")
    args = parser.parse_args()

    genuine_dir = os.path.join(args.out_dir, "genuine")
    fake_dir = os.path.join(args.out_dir, "fake")
    os.makedirs(genuine_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    print("Downloading garystafford/deepfake-audio-detection from Hugging Face...")
    print("(This pulls a few hundred MB, much smaller than ASVspoof's full package.)")
    dataset = load_dataset("garystafford/deepfake-audio-detection", split="train")

    # Ask for raw bytes instead of an auto-decoded array. The auto-decode path
    # requires torchcodec -> torch, a multi-GB dependency we don't need here;
    # we decode ourselves with soundfile below, which is already installed.
    dataset = dataset.cast_column("audio", Audio(decode=False))

    genuine_count = 0
    fake_count = 0

    for item in dataset:
        if genuine_count >= args.limit and fake_count >= args.limit:
            break

        audio = item["audio"]
        label = item["label"]  # 0 = real/genuine, 1 = fake, per this dataset's convention

        # With decode=False, audio is {"path": ..., "bytes": ...}. "bytes" is
        # present when the file is embedded in the parquet shard (usual case);
        # "path" is a local cache path when it isn't. Handle both.
        if audio.get("bytes") is not None:
            data, samplerate = sf.read(io.BytesIO(audio["bytes"]))
        elif audio.get("path"):
            data, samplerate = sf.read(audio["path"])
        else:
            continue  # no audio available for this row, skip it

        if label == 0 and genuine_count < args.limit:
            path = os.path.join(genuine_dir, f"genuine_{genuine_count:04d}.wav")
            sf.write(path, data, samplerate)
            genuine_count += 1
        elif label == 1 and fake_count < args.limit:
            path = os.path.join(fake_dir, f"fake_{fake_count:04d}.wav")
            sf.write(path, data, samplerate)
            fake_count += 1

    print(f"\nDone. Saved {genuine_count} genuine file(s) -> {genuine_dir}")
    print(f"Saved {fake_count} fake file(s) -> {fake_dir}")
    print("\nNext step:")
    print(f'  python batch_test.py --genuine-dir "{genuine_dir}" --fake-dir "{fake_dir}" --csv report.csv')


if __name__ == "__main__":
    main()
