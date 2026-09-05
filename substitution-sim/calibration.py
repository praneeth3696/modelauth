"""Does the anytime-valid guarantee actually hold on real streams?

This is the plot that matters more than any delay number. A detector claiming
P(false alarm) <= alpha is making a checkable promise; sweeping alpha across
orders of magnitude on clean null streams and plotting nominal against achieved
either shows the promise kept (points on or below the diagonal) or shows it
broken. Asserting the theorem is not evidence. This is.

Reported per tier, over all null streams:
  achieved = fraction of null streams that ever raise a flag.

For the Ville-valid e-process this must sit at or below alpha. For the CUSUM
form, whose reset breaks the martingale, it is expected to run higher -- that
gap is exactly what the reset buys and costs, and it is worth showing.

Usage:  python calibration.py
"""

import csv
import os
import sys

import numpy as np

from config import MODEL_PAIRS
from data_loader import DATA_DIR, load_numeric_stream
from detector_compression import eprocess_detector, mdl_cusum_detector

ALPHAS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
N_REPS = 15
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "final-analysis", "figures", "calibration.csv")


def null_streams(difficulty, n_reps=N_REPS):
    streams = []
    for rep in range(n_reps):
        path = os.path.join(DATA_DIR, f"{difficulty}_null_rep{rep}.jsonl")
        if not os.path.exists(path):
            break
        recs = load_numeric_stream(path)
        streams.append(([r["numeric_answer"] for r in recs],
                        [r["prompt"][:16] for r in recs]))
    return streams


DETECTORS = [
    ("e-process (Ville)", eprocess_detector),
    ("MDL-CUSUM (reset)", mdl_cusum_detector),
]


def main():
    rows = []
    for difficulty in MODEL_PAIRS:
        streams = null_streams(difficulty)
        if not streams:
            continue
        print(f"\n=== {difficulty} ({len(streams)} null streams) ===")
        print(f"{'detector':<22} {'nominal':>10} {'achieved':>10} {'streams':>9}  verdict")
        print("-" * 68)
        for name, fn in DETECTORS:
            for alpha in ALPHAS:
                fired = 0
                for values, ctxs in streams:
                    res = fn(values, contexts=ctxs, alpha=alpha)
                    if any(d["flagged"] for d in res):
                        fired += 1
                achieved = fired / len(streams)
                ok = achieved <= alpha or fired == 0
                # With only 15 streams, an achieved rate below 1/15 is
                # indistinguishable from zero; say so rather than imply
                # resolution the sample size does not support.
                verdict = ("held" if ok else
                           "resolution-limited" if alpha < 1.0 / len(streams)
                           else "VIOLATED")
                print(f"{name:<22} {alpha:>10.0e} {achieved:>10.3f} "
                      f"{fired:>4}/{len(streams):<4} {verdict}")
                rows.append({
                    "difficulty": difficulty, "detector": name,
                    "nominal_alpha": alpha, "achieved": achieved,
                    "streams_fired": fired, "n_streams": len(streams),
                    "verdict": verdict,
                })

    if not rows:
        print("No null streams found.")
        return 1
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT_CSV}")
    print("\nNote: with 15 null streams per tier the smallest resolvable rate is")
    print("1/15 = 0.067, so every alpha below that can only be confirmed as "
          "'no stream fired'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
