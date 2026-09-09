import os
import sys
import re
import numpy as np
from scipy.stats import ks_2samp

# Add parent directory and substitution-sim to path for imports
SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "substitution-sim"))
if SIM_DIR not in sys.path:
    sys.path.insert(0, SIM_DIR)

from data_loader import load_stream, load_numeric_stream
from detector_cusum import adaptive_cusum_detector
from evaluate import compute_metrics, load_all_reps

def audit_data_completeness(data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(SIM_DIR, "data")
    
    report = []
    if not os.path.exists(data_dir):
        print(f"[warn] Data directory not found: {data_dir}")
        return report

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".jsonl"):
            continue
        filepath = os.path.join(data_dir, fname)
        records = load_stream(filepath)
        n_total = len(records)
        n_failed = sum(1 for r in records if r.get("failed"))
        n_unparseable = 0
        for r in records:
            if r.get("answer") is None or not re.search(r'\d+', r.get("answer") or ""):
                n_unparseable += 1
        
        usable_pct = round(100 * (n_total - n_unparseable) / n_total, 1) if n_total > 0 else 0.0
        report.append({
            "file": fname,
            "n_total": n_total,
            "n_failed": n_failed,
            "n_unparseable": n_unparseable,
            "pct_usable": usable_pct,
        })
    return report

def check_model_separability(difficulty="easy", n_reps=5, data_dir=None):
    """Compare model A vs model B answer distributions directly, ignoring switch timing."""
    if data_dir is None:
        data_dir = os.path.join(SIM_DIR, "data")

    a_answers, b_answers = [], []
    for rep in range(n_reps):
        null_file = os.path.join(data_dir, f"{difficulty}_null_rep{rep}.jsonl")
        sub_file = os.path.join(data_dir, f"{difficulty}_substitution_rep{rep}.jsonl")
        
        if os.path.exists(null_file):
            records = load_numeric_stream(null_file)
            a_answers.extend([r["numeric_answer"] for r in records])
        
        if os.path.exists(sub_file):
            records = load_numeric_stream(sub_file)
            # Second half of substitution stream is model_b (post index 200)
            b_answers.extend([r["numeric_answer"] for r in records if r["index"] >= 200])

    if not a_answers or not b_answers:
        return {"difficulty": difficulty, "ks_stat": None, "p_value": None, "status": "Insufficient Data"}

    stat, p_value = ks_2samp(a_answers, b_answers)
    return {
        "difficulty": difficulty,
        "n_samples_a": len(a_answers),
        "n_samples_b": len(b_answers),
        "ks_stat": round(stat, 4),
        "p_value": p_value,
        "separable": p_value < 0.05
    }

def check_tier_ordering(difficulties=["easy"]):
    results = {}
    for difficulty in difficulties:
        sub_streams = load_all_reps(difficulty, "substitution")
        null_streams = load_all_reps(difficulty, "null")
        if not sub_streams:
            results[difficulty] = "No streams loaded"
            continue

        
        metrics = compute_metrics(
            adaptive_cusum_detector,
            sub_streams,
            null_streams,
            true_switch=200,
            warmup=40,
            k=0.5,
            h=5.0
        )
        results[difficulty] = metrics
    return results

def audit_all_repetitions(difficulty="easy", condition="substitution", n_reps=20, data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(SIM_DIR, "data")
    
    anomalies = []
    for rep in range(n_reps):
        fname = os.path.join(data_dir, f"{difficulty}_{condition}_rep{rep}.jsonl")
        if os.path.exists(fname):
            records = load_numeric_stream(fname)
            if len(records) < 300:
                anomalies.append((fname, len(records)))
    return anomalies
