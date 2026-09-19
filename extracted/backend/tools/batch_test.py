"""
batch_test.py
-------------
Runs every audio file in two folders (genuine speech and fake/AI-generated
speech) through the running Voice Integrity Guard API, compares the
system's prediction against the known ground-truth label, and prints a
real accuracy report -- this is what turns "we tried it on a few files"
into "we measured X% accuracy across N labeled samples."

USAGE:
  1. Start your server first:  uvicorn app.main:app --reload --port 8000
  2. Organize your test files into two folders:
       my_dataset/genuine/   <- real human speech (.wav/.flac/.mp3)
       my_dataset/fake/      <- AI-generated/cloned speech
  3. Run:
       python batch_test.py --genuine-dir my_dataset/genuine --fake-dir my_dataset/fake

  Optional flags:
    --api-base   default http://localhost:8000
    --threshold  risk_score threshold above which a clip is predicted
                 "fake" (default 50 -- matches the MEDIUM/HIGH boundary
                 region; adjust if you want a stricter/looser cutoff)
    --limit      only test the first N files per folder (useful for a
                 quick smoke test before running the full set)
    --csv        path to write a full per-file CSV report (optional)

OUTPUT:
  - A confusion matrix (true positive / false positive / true negative /
    false negative counts)
  - Overall accuracy, precision, recall, F1
  - A breakdown of which files were misclassified, so you can look at
    *why* (e.g. "struggles with compressed audio") for your presentation
"""

from __future__ import annotations
import argparse
import csv as csv_module
import os
import sys
import time

import requests

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def list_audio_files(folder: str, limit: int | None = None) -> list[str]:
    if not os.path.isdir(folder):
        print(f"ERROR: folder not found: {folder}")
        return []
    files = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
    ]
    if limit:
        files = files[:limit]
    return files


def analyze_file(api_base: str, filepath: str) -> dict | None:
    try:
        with open(filepath, "rb") as f:
            files = {"file": (os.path.basename(filepath), f)}
            resp = requests.post(f"{api_base}/api/analyze", files=files, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            print(f"  [server error] {filepath}: {data['error']}")
            return None
        return data["risk_result"]
    except Exception as e:
        print(f"  [request failed] {filepath}: {e}")
        return None


def run_batch(api_base: str, genuine_dir: str, fake_dir: str, threshold: float, limit: int | None):
    genuine_files = list_audio_files(genuine_dir, limit)
    fake_files = list_audio_files(fake_dir, limit)

    print(f"Found {len(genuine_files)} genuine file(s), {len(fake_files)} fake file(s).")
    if not genuine_files and not fake_files:
        print("Nothing to test. Check your --genuine-dir / --fake-dir paths.")
        return []

    results = []
    total = len(genuine_files) + len(fake_files)
    done = 0

    for path in genuine_files:
        done += 1
        print(f"[{done}/{total}] analyzing (genuine): {os.path.basename(path)}")
        r = analyze_file(api_base, path)
        if r is not None:
            results.append({"file": path, "true_label": "genuine", **r})

    for path in fake_files:
        done += 1
        print(f"[{done}/{total}] analyzing (fake): {os.path.basename(path)}")
        r = analyze_file(api_base, path)
        if r is not None:
            results.append({"file": path, "true_label": "fake", **r})

    return results


def summarize(results: list[dict], threshold: float, csv_path: str | None):
    if not results:
        print("No results to summarize.")
        return

    tp = fp = tn = fn = 0
    misclassified = []

    for r in results:
        predicted_fake = r["risk_score"] >= threshold
        actually_fake = r["true_label"] == "fake"

        if actually_fake and predicted_fake:
            tp += 1
        elif actually_fake and not predicted_fake:
            fn += 1
            misclassified.append(r)
        elif not actually_fake and predicted_fake:
            fp += 1
            misclassified.append(r)
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total samples tested: {total}")
    print(f"Risk-score threshold used for 'fake' prediction: {threshold}")
    print()
    print("Confusion matrix:")
    print(f"  True Positive  (fake correctly flagged):     {tp}")
    print(f"  False Negative (fake MISSED, said genuine):  {fn}")
    print(f"  True Negative  (genuine correctly cleared):  {tn}")
    print(f"  False Positive (genuine wrongly flagged):    {fp}")
    print()
    print(f"Accuracy:  {accuracy*100:.1f}%")
    print(f"Precision: {precision*100:.1f}%  (of clips flagged fake, % actually fake)")
    print(f"Recall:    {recall*100:.1f}%  (of actual fakes, % correctly caught)")
    print(f"F1 score:  {f1*100:.1f}%")

    ml_sources = {}
    for r in results:
        src = r.get("ml_source") or "none"
        ml_sources[src] = ml_sources.get(src, 0) + 1
    print(f"\nML source breakdown: {ml_sources}")
    if "dhwani" not in ml_sources:
        print("WARNING: no results used 'dhwani' as ml_source -- the real model may not be loaded.")

    if misclassified:
        print(f"\n{len(misclassified)} misclassified file(s):")
        for r in misclassified:
            print(f"  [{r['true_label']:>7}] score={r['risk_score']:>5.1f} "
                  f"level={r['risk_level']:<6} source={r.get('ml_source')} -> {os.path.basename(r['file'])}")

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv_module.DictWriter(
                f, fieldnames=["file", "true_label", "risk_score", "risk_level", "ml_source", "predicted_fake"]
            )
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "file": r["file"],
                    "true_label": r["true_label"],
                    "risk_score": r["risk_score"],
                    "risk_level": r["risk_level"],
                    "ml_source": r.get("ml_source"),
                    "predicted_fake": r["risk_score"] >= threshold,
                })
        print(f"\nFull per-file report written to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch-test Voice Integrity Guard accuracy against labeled audio.")
    parser.add_argument("--genuine-dir", required=True, help="Folder of real human speech files")
    parser.add_argument("--fake-dir", required=True, help="Folder of AI-generated/cloned speech files")
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--limit", type=int, default=None, help="Only test first N files per folder")
    parser.add_argument("--csv", default=None, help="Optional path to write a full CSV report")
    args = parser.parse_args()

    try:
        requests.get(f"{args.api_base}/api/health", timeout=5)
    except Exception:
        print(f"ERROR: could not reach {args.api_base} -- is the server running?")
        sys.exit(1)

    start = time.time()
    results = run_batch(args.api_base, args.genuine_dir, args.fake_dir, args.threshold, args.limit)
    summarize(results, args.threshold, args.csv)
    print(f"\nDone in {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
