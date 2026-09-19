"""
sort_asvspoof.py
-----------------
ASVspoof datasets don't come pre-sorted into "genuine" and "fake" folders
-- they ship as one flat folder of audio files, plus a separate protocol
(key) text file that maps each filename to its label. This script reads
that protocol file and copies each audio file into the right folder for
batch_test.py to use.

ASVspoof protocol files are whitespace-separated text, one line per file,
typically formatted like:
    LA_0079 LA_D_1047731 - - bonafide
    LA_0079 LA_D_1047731 - - spoof
The label is the LAST column: "bonafide" (genuine) or "spoof" (fake).
(Some ASVspoof editions use "bona-fide"/"human" for genuine and an attack
ID like "A01" for fake -- adjust GENUINE_LABELS below if your protocol
file's genuine label text looks different; open the file in a text editor
and check the last column's values if this script reports 0 matches.)

USAGE:
  python sort_asvspoof.py \
    --protocol path/to/ASVspoof2019.LA.cm.dev.trl.txt \
    --audio-dir path/to/ASVspoof2019_LA_dev/flac \
    --out-dir my_dataset \
    --limit 200
This creates:
    my_dataset/genuine/*.flac
    my_dataset/fake/*.flac
"""

from __future__ import annotations
import argparse
import os
import shutil

GENUINE_LABELS = {"bonafide", "bona-fide", "human", "genuine", "real"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, help="Path to the ASVspoof protocol/key .txt file")
    parser.add_argument("--audio-dir", required=True, help="Folder containing the .flac/.wav audio files")
    parser.add_argument("--out-dir", required=True, help="Where to create genuine/ and fake/ subfolders")
    parser.add_argument("--limit", type=int, default=None, help="Max files per class (for a quick subset)")
    parser.add_argument("--audio-ext", default=".flac", help="Audio file extension (default .flac)")
    args = parser.parse_args()

    genuine_out = os.path.join(args.out_dir, "genuine")
    fake_out = os.path.join(args.out_dir, "fake")
    os.makedirs(genuine_out, exist_ok=True)
    os.makedirs(fake_out, exist_ok=True)

    genuine_count = 0
    fake_count = 0
    missing = 0
    skipped_lines = 0

    with open(args.protocol, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                skipped_lines += 1
                continue

            filename_stem = parts[1]  # typically the 2nd column is the audio filename (no extension)
            label = parts[-1].lower()

            src = os.path.join(args.audio_dir, filename_stem + args.audio_ext)
            if not os.path.exists(src):
                missing += 1
                continue

            is_genuine = label in GENUINE_LABELS

            if is_genuine:
                if args.limit and genuine_count >= args.limit:
                    continue
                shutil.copy2(src, os.path.join(genuine_out, filename_stem + args.audio_ext))
                genuine_count += 1
            else:
                if args.limit and fake_count >= args.limit:
                    continue
                shutil.copy2(src, os.path.join(fake_out, filename_stem + args.audio_ext))
                fake_count += 1

    print(f"Sorted {genuine_count} genuine file(s) -> {genuine_out}")
    print(f"Sorted {fake_count} fake file(s) -> {fake_out}")
    if missing:
        print(f"WARNING: {missing} file(s) listed in the protocol were not found in --audio-dir")
    if skipped_lines:
        print(f"WARNING: {skipped_lines} line(s) in the protocol file couldn't be parsed")
    if genuine_count == 0 and fake_count == 0:
        print(
            "\nNo files matched. Open your protocol file in a text editor and check the "
            "label column's exact text -- you may need to add it to GENUINE_LABELS in this script."
        )


if __name__ == "__main__":
    main()
