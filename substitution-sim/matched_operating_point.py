"""The fair comparison: every detector at the same false-alarm budget.

The headline table compares detectors at their default settings, which is not a
like-for-like read. Adaptive CUSUM at h=5.0 reaches 8.50 probes on the easy tier
but raises false alarms on 7 of 14 clean streams; the e-process raises none. One
of those is not operating at the other's error rate.

This script tunes every threshold-bearing detector to the tightest setting that
raises NO false alarm, then reports delay and power at that common operating
point. Tuning uses held-out rep14 only; the reported numbers come from reps 0-13,
so no threshold is chosen on the streams it is scored against.

The result to read carefully: at a matched zero-false-alarm budget, the tuned
CUSUM is FASTER than the e-process. What the e-process buys is not speed -- it
is that its threshold is derived from alpha rather than swept, and in deployment
there is no labelled substitution stream to sweep against. The tuned number here
is available only because we have ground truth that a real monitor does not.

Usage:  python matched_operating_point.py
"""

import csv
import os
import sys

import numpy as np

from config import MODEL_PAIRS, SWITCH_POINT
from data_loader import DATA_DIR, load_numeric_stream
from detector_cusum import adaptive_cusum_detector
from detector_variance_cusum import variance_cusum_detector
from detector_v1 import sliding_window_detector
from detector_compression import eprocess_detector, mdl_cusum_detector
from detector_baselines_2026 import js_fingerprint_detector

HELD_OUT_REP = 14
N_TEST = 14
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "final-analysis", "figures", "matched_operating_point.csv")

# (label, fn, kwarg name, candidate grid ordered most-sensitive first)
TUNABLE = [
    ("adaptive CUSUM", adaptive_cusum_detector, "h",
     [5, 6, 7, 8, 9, 10, 12, 14, 17, 20, 25]),
    ("variance CUSUM", variance_cusum_detector, "h",
     [5, 6, 7, 8, 9, 10, 12, 14, 17, 20, 25]),
    ("KS sliding window", sliding_window_detector, "alpha",
     [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]),
    ("JS fingerprint [Bruckner]", js_fingerprint_detector, "threshold",
     [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]),
    ("MDL-CUSUM", mdl_cusum_detector, "alpha",
     [1e-2, 1e-3, 1e-4, 1e-6, 1e-8]),
    ("e-process (Ville)", eprocess_detector, "alpha",
     [1e-2, 1e-3, 1e-4, 1e-6, 1e-8]),
]


def load(difficulty, condition, rep):
    path = os.path.join(DATA_DIR, f"{difficulty}_{condition}_rep{rep}.jsonl")
    if not os.path.exists(path):
        return None
    recs = load_numeric_stream(path)
    return ([r["numeric_answer"] for r in recs], [r["prompt"][:16] for r in recs])


def run(fn, stream, kwargs):
    values, ctxs = stream
    names = fn.__code__.co_varnames[:fn.__code__.co_argcount]
    if "contexts" in names:
        return fn(values, contexts=ctxs, **kwargs)
    return fn(values, **kwargs)


def main():
    rows = []
    for difficulty in MODEL_PAIRS:
        tune_null = load(difficulty, "null", HELD_OUT_REP)
        if tune_null is None:
            continue
        test_null = [s for s in (load(difficulty, "null", i) for i in range(N_TEST)) if s]
        test_sub = [s for s in (load(difficulty, "substitution", i) for i in range(N_TEST)) if s]
        if not test_null or not test_sub:
            continue

        print(f"\n=== {difficulty} ===")
        print(f"{'detector':<26} {'tuned':>10} {'delay':>8} {'power':>8} {'FA streams':>11}")
        print("-" * 68)

        for label, fn, key, grid in TUNABLE:
            # tightest setting on the grid that stays silent on the held-out null
            chosen = None
            for cand in grid:
                res = run(fn, tune_null, {key: cand})
                if not any(d["flagged"] for d in res):
                    chosen = cand
                    break
            if chosen is None:
                print(f"{label:<26} {'none':>10}   no setting silences the held-out null")
                rows.append({"difficulty": difficulty, "detector": label,
                             "tuned_value": None, "mean_delay": None,
                             "power": None, "fa_streams": None,
                             "n_null": len(test_null)})
                continue

            delays = []
            for s in test_sub:
                res = run(fn, s, {key: chosen})
                hit = next((d["index"] for d in res
                            if d["flagged"] and d["index"] >= SWITCH_POINT), None)
                if hit is not None:
                    delays.append(hit - SWITCH_POINT)
            fa = sum(1 for s in test_null
                     if any(d["flagged"] for d in run(fn, s, {key: chosen})))

            delay = float(np.mean(delays)) if delays else None
            power = len(delays) / len(test_sub)
            print(f"{label:<26} {key}={chosen:<7.4g} "
                  f"{(f'{delay:.2f}' if delay else '--'):>8} "
                  f"{power*100:>7.1f}% {fa:>6}/{len(test_null):<4}")
            rows.append({"difficulty": difficulty, "detector": label,
                         "tuned_value": chosen, "mean_delay": delay,
                         "power": power, "fa_streams": fa,
                         "n_null": len(test_null)})

    if not rows:
        print("No data.")
        return 1
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT_CSV}")
    print("\nThresholds were chosen on held-out rep14 and scored on reps 0-13.")
    print("A tuned threshold needs labelled data that a deployed monitor does not")
    print("have; the e-process row needs only alpha, which is why it is the one")
    print("number here that transfers to deployment unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
