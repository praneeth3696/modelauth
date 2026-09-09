import os
import csv
import numpy as np
from data_loader import load_numeric_stream
from detector_v1 import sliding_window_detector
from detector_cusum import adaptive_cusum_detector
from detector_das_cusum import das_cusum_detector
from detector_fixed_reference import fixed_reference_detector, build_reference_distribution

def compute_metrics(detector_fn, sub_streams, null_streams, true_switch, **kwargs):
    """
    sub_streams / null_streams: list of numeric_answer lists (multiple repetitions)
    Returns average detection delay (post-switch) and false-alarm rate across repetitions.
    """
    delays = []
    for stream in sub_streams:
        results = detector_fn(stream, **kwargs)
        # Fix negative delay bug: count first flag occurring AT or AFTER true_switch
        first_valid_flag = next((d for d in results if d["flagged"] and d["index"] >= true_switch), None)
        if first_valid_flag:
            delays.append(first_valid_flag["index"] - true_switch)
        else:
            delays.append(None)  # missed detection

    false_alarm_counts = []
    for stream in null_streams:
        results = detector_fn(stream, **kwargs)
        # False alarm count: flags before true_switch (or in null streams)
        flags = sum(1 for d in results if d["flagged"] and d["index"] < true_switch)
        false_alarm_counts.append(flags / len(results) if results else 0.0)

    detected = [d for d in delays if d is not None]
    return {
        "mean_delay": float(np.mean(detected)) if detected else None,
        "detection_rate": float(len(detected) / len(delays)) if delays else 0.0,
        "mean_false_alarm_rate": float(np.mean(false_alarm_counts)) if false_alarm_counts else 0.0,
    }

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def load_all_reps(difficulty, condition, n_reps=14, data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    streams = []
    for rep in range(n_reps):
        fname = os.path.join(data_dir, f"{difficulty}_{condition}_rep{rep}.jsonl")
        if not os.path.exists(fname):
            break
        records = load_numeric_stream(fname)
        streams.append([r["numeric_answer"] for r in records])
    return streams

if __name__ == "__main__":
    DIFFICULTIES = ["medium", "hard"]
    results_table = []

    for difficulty in DIFFICULTIES:
        sub_streams = load_all_reps(difficulty, "substitution", n_reps=14)
        null_streams = load_all_reps(difficulty, "null", n_reps=14)

        if not sub_streams or not null_streams:
            continue

        # Held-out reference stream for fixed-reference detector (rep14)
        ref_file = os.path.join(DATA_DIR, f"{difficulty}_null_rep14.jsonl")
        reference_dist = None
        if os.path.exists(ref_file):
            ref_records = load_numeric_stream(ref_file)
            reference_dist = build_reference_distribution([r["numeric_answer"] for r in ref_records])

        methods = [
            ("v1 naive", sliding_window_detector, {"window_size": 20}),
            ("adaptive CUSUM", adaptive_cusum_detector, {"warmup": 40, "k": 0.5, "h": 5.0}),
            ("DAS-CUSUM", das_cusum_detector, {"warmup": 40, "k": 0.5, "h": 5.0}),
        ]

        if reference_dist is not None:
            methods.append(
                ("fixed-reference", lambda stream, **kw: fixed_reference_detector(stream, reference_dist, **kw), {"batch_size": 20})
            )

        for name, fn, kwargs in methods:
            metrics = compute_metrics(fn, sub_streams, null_streams, true_switch=200, **kwargs)
            results_table.append({"difficulty": difficulty, "method": name, **metrics})

    out_csv = "../final-analysis/figures/summary_table.csv"

    if results_table:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results_table[0].keys())
            writer.writeheader()
            writer.writerows(results_table)

    for row in results_table:
        print(row)

