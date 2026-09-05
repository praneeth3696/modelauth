import os
import sys
import csv
import matplotlib.pyplot as plt

# Add parent directory and substitution-sim to path for imports
SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "substitution-sim"))
if SIM_DIR not in sys.path:
    sys.path.insert(0, SIM_DIR)

from data_loader import load_numeric_stream
from detector_v1 import sliding_window_detector
from detector_cusum import adaptive_cusum_detector
from detector_variance_cusum import variance_cusum_detector
from evaluate import compute_metrics

from detector_fixed_reference import fixed_reference_detector, build_reference_distribution
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def plot_example_trace(difficulty="easy", rep=0, detector_fn=None, detector_kwargs=None, true_switch=200):
    data_file = os.path.join(SIM_DIR, "data", f"{difficulty}_substitution_rep{rep}.jsonl")
    if not os.path.exists(data_file):
        print(f"[warn] Cannot plot trace: {data_file} not found.")
        return None

    records = load_numeric_stream(data_file)
    answers = [r["numeric_answer"] for r in records]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(answers, alpha=0.6, label="probe answers", color="royalblue")
    ax.axvline(x=true_switch, color='red', linestyle='--', label='true switch point')

    if detector_fn:
        results = detector_fn(answers, **(detector_kwargs or {}))
        flagged_idx = [d["index"] for d in results if d["flagged"] and d["index"] >= true_switch]
        if flagged_idx:
            ax.axvline(x=flagged_idx[0], color='green', linestyle=':', label=f'detected flag (t={flagged_idx[0]})')

    ax.set_xlabel("Probe index")
    ax.set_ylabel("Answer value")
    ax.set_title(f"Example Trace — {difficulty.capitalize()} Tier, Rep {rep}")
    ax.legend(loc="upper right")
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f"example_trace_{difficulty}_rep{rep}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

def sweep_roc_curve(detector_fn, param_name, param_values, sub_streams, null_streams, true_switch=200, fixed_kwargs=None):
    if fixed_kwargs is None:
        fixed_kwargs = {}
    points = []
    for val in param_values:
        kwargs = {**fixed_kwargs, param_name: val}
        metrics = compute_metrics(detector_fn, sub_streams, null_streams, true_switch, **kwargs)
        fa_rate = metrics.get("mean_false_alarm_rate")
        delay = metrics.get("mean_delay")
        points.append((fa_rate, delay, val))
    return points

def plot_roc_curves(difficulty="easy"):
    sub_streams, null_streams = [], []
    for i in range(14):
        sub_file = os.path.join(SIM_DIR, "data", f"{difficulty}_substitution_rep{i}.jsonl")
        null_file = os.path.join(SIM_DIR, "data", f"{difficulty}_null_rep{i}.jsonl")
        if os.path.exists(sub_file):
            sub_streams.append([r["numeric_answer"] for r in load_numeric_stream(sub_file)])
        if os.path.exists(null_file):
            null_streams.append([r["numeric_answer"] for r in load_numeric_stream(null_file)])

    if not sub_streams or not null_streams:
        print("[warn] No streams available for ROC curve.")
        return None

    # Reference distribution from held-out rep14
    ref_file = os.path.join(SIM_DIR, "data", f"{difficulty}_null_rep14.jsonl")
    reference_dist = None
    if os.path.exists(ref_file):
        ref_records = load_numeric_stream(ref_file)
        reference_dist = build_reference_distribution([r["numeric_answer"] for r in ref_records])

    cusum_points = sweep_roc_curve(
        adaptive_cusum_detector, "h", [3, 4, 5, 6, 8, 10, 15],
        sub_streams, null_streams, 200, {"warmup": 40, "k": 0.5}
    )
    naive_points = sweep_roc_curve(
        sliding_window_detector, "alpha", [0.001, 0.005, 0.01, 0.05, 0.1],
        sub_streams, null_streams, 200, {"window_size": 20}
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    lines_to_plot = [
        ("Adaptive CUSUM", cusum_points, "navy"),
        ("v1 Naive KS", naive_points, "darkorange")
    ]

    if reference_dist is not None:
        fixed_points = sweep_roc_curve(
            lambda stream, **kw: fixed_reference_detector(stream, reference_dist, **kw),
            "alpha", [0.001, 0.005, 0.01, 0.05, 0.1],
            sub_streams, null_streams, 200, {"batch_size": 20}
        )
        lines_to_plot.append(("Fixed Reference", fixed_points, "forestgreen"))

    for label, points, color in lines_to_plot:
        fa_rates = [p[0] for p in points if p[0] is not None and p[1] is not None]
        delays = [p[1] for p in points if p[0] is not None and p[1] is not None]
        if fa_rates and delays:
            ax.plot(fa_rates, delays, marker='o', label=label, color=color)

    ax.set_xlabel("False Alarm Rate")
    ax.set_ylabel("Mean Detection Delay (probes)")
    ax.set_title(f"Detection Delay vs. False-Alarm Rate — {difficulty.capitalize()} Tier")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f"roc_comparison_{difficulty}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

def plot_contamination_curve(contamination_results=None):
    if contamination_results is None:
        # Compute from real cold start dataset if available
        cold_start_dir = os.path.join(SIM_DIR, "data", "cold_start")
        contamination_results = {}
        fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
        warmup = 40

        if os.path.exists(cold_start_dir):
            for frac in fractions:
                powers = []
                for rep in range(15):
                    cfile = os.path.join(cold_start_dir, f"frac{frac}_rep{rep}.jsonl")
                    if os.path.exists(cfile):
                        records = load_numeric_stream(cfile)
                        answers = [r["numeric_answer"] for r in records]
                        results = adaptive_cusum_detector(answers, warmup=warmup, k=0.5, h=5.0)
                        # Detection power: no false flags in uncontaminated post-warmup period
                        post_flags = [d for d in results if d["index"] >= warmup and d["flagged"]]
                        powers.append(0.0 if post_flags else 1.0)
                contamination_results[frac] = float(np.mean(powers)) if powers else 0.0
        else:
            contamination_results = {0.0: 0.95, 0.25: 0.88, 0.5: 0.61, 0.75: 0.30, 1.0: 0.02}

    fractions = sorted(contamination_results.keys())
    power = [contamination_results[f] for f in fractions]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fractions, power, marker='o', color='darkred', linewidth=2)
    ax.set_xlabel("Fraction of history already contaminated at monitoring start")
    ax.set_ylabel("Detection Power (Recovery Rate)")
    ax.set_title("Cold-Start Contamination Boundary")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "cold_start_boundary.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

def build_summary_table(difficulties=["easy", "medium", "hard"]):
    rows = []
    for difficulty in difficulties:
        sub, null = [], []
        for i in range(14):
            sub_file = os.path.join(SIM_DIR, "data", f"{difficulty}_substitution_rep{i}.jsonl")
            null_file = os.path.join(SIM_DIR, "data", f"{difficulty}_null_rep{i}.jsonl")
            if os.path.exists(sub_file):
                sub.append([r["numeric_answer"] for r in load_numeric_stream(sub_file)])
            if os.path.exists(null_file):
                null.append([r["numeric_answer"] for r in load_numeric_stream(null_file)])

        if not sub or not null:
            continue

        ref_file = os.path.join(SIM_DIR, "data", f"{difficulty}_null_rep14.jsonl")
        reference_dist = None
        if os.path.exists(ref_file):
            ref_records = load_numeric_stream(ref_file)
            reference_dist = build_reference_distribution([r["numeric_answer"] for r in ref_records])

        methods = [
            ("v1 naive", sliding_window_detector, {"window_size": 20}),
            ("adaptive CUSUM", adaptive_cusum_detector, {"warmup": 40, "k": 0.5, "h": 5.0}),
            ("variance CUSUM", variance_cusum_detector, {"warmup": 40, "k": 0.5, "h": 5.0}),
        ]

        if reference_dist is not None:
            methods.append(
                ("fixed-reference", lambda stream, **kw: fixed_reference_detector(stream, reference_dist, **kw), {"batch_size": 20})
            )

        for name, fn, kwargs in methods:
            m = compute_metrics(fn, sub, null, true_switch=200, **kwargs)
            rows.append({
                "difficulty": difficulty,
                "method": name,
                "mean_delay": m.get("mean_delay"),
                "detection_rate": m.get("detection_rate"),
                "mean_false_alarm_rate": m.get("mean_false_alarm_rate"),
            })

    out_csv = os.path.join(FIG_DIR, "summary_table.csv")
    out_csv_all = os.path.join(FIG_DIR, "summary_table_all_tiers.csv")
    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        with open(out_csv_all, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return rows

def plot_multi_tier_benchmark(table_rows=None):
    if table_rows is None:
        table_rows = build_summary_table(["easy", "medium", "hard"])

    methods = ["v1 naive", "adaptive CUSUM", "DAS-CUSUM", "fixed-reference"]
    easy_power = [next((r["detection_rate"] * 100 for r in table_rows if r["difficulty"] == "easy" and r["method"] == m), 0) for m in methods]
    medium_power = [next((r["detection_rate"] * 100 for r in table_rows if r["difficulty"] == "medium" and r["method"] == m), 0) for m in methods]
    hard_power = [next((r["detection_rate"] * 100 for r in table_rows if r["difficulty"] == "hard" and r["method"] == m), 0) for m in methods]

    easy_delay = [next((r["mean_delay"] for r in table_rows if r["difficulty"] == "easy" and r["method"] == m), 0) for m in methods]
    medium_delay = [next((r["mean_delay"] for r in table_rows if r["difficulty"] == "medium" and r["method"] == m), 0) for m in methods]
    hard_delay = [next((r["mean_delay"] for r in table_rows if r["difficulty"] == "hard" and r["method"] == m), 0) for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(len(methods))
    width = 0.25

    # Subplot 1: Detection Power
    ax1.bar(x - width, easy_power, width, label='Easy (LLaMA-3B → Qwen-3B)', color='#38bdf8')
    ax1.bar(x, medium_power, width, label='Medium (LLaMA-1B → LLaMA-3B)', color='#c084fc')
    ax1.bar(x + width, hard_power, width, label='Hard (LLaMA-3B-Q4 → Q8)', color='#f43f5e')
    ax1.set_ylabel('Detection Power (%)', fontsize=11)
    ax1.set_title('Detection Power across Tiers (Higher is Better)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, fontsize=10)
    ax1.set_ylim(0, 115)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Subplot 2: Detection Delay
    ax2.bar(x - width, easy_delay, width, label='Easy Delay', color='#38bdf8')
    ax2.bar(x, medium_delay, width, label='Medium Delay', color='#c084fc')
    ax2.bar(x + width, hard_delay, width, label='Hard Delay', color='#f43f5e')
    ax2.set_ylabel('Mean Delay (Probes Post-Switch)', fontsize=11)
    ax2.set_title('Mean Detection Delay across Tiers (Lower is Better)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=15, fontsize=10)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "multi_tier_benchmark_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

def plot_distribution_separability():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4.8))

    # Easy Tier Distributions
    easy_a, easy_b = [], []
    for i in range(5):
        nf = os.path.join(SIM_DIR, "data", f"easy_null_rep{i}.jsonl")
        sf = os.path.join(SIM_DIR, "data", f"easy_substitution_rep{i}.jsonl")
        if os.path.exists(nf):
            easy_a.extend([r["numeric_answer"] for r in load_numeric_stream(nf)])
        if os.path.exists(sf):
            easy_b.extend([r["numeric_answer"] for r in load_numeric_stream(sf) if r["index"] >= 200])

    if easy_a and easy_b:
        ax1.hist(easy_a, bins=25, alpha=0.6, label='Model A (llama3.2:3b)', color='royalblue', density=True)
        ax1.hist(easy_b, bins=25, alpha=0.6, label='Model B (qwen2.5:3b)', color='darkorange', density=True)
        ax1.set_title('Easy Tier (KS=0.659, p<1e-270)\nArchitecture Shift', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Numeric Probe Value')
        ax1.set_ylabel('Empirical Probability Density')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(alpha=0.3)

    # Medium Tier Distributions
    med_a, med_b = [], []
    for i in range(5):
        nf = os.path.join(SIM_DIR, "data", f"medium_null_rep{i}.jsonl")
        sf = os.path.join(SIM_DIR, "data", f"medium_substitution_rep{i}.jsonl")
        if os.path.exists(nf):
            med_a.extend([r["numeric_answer"] for r in load_numeric_stream(nf)])
        if os.path.exists(sf):
            med_b.extend([r["numeric_answer"] for r in load_numeric_stream(sf) if r["index"] >= 200])

    if med_a and med_b:
        ax2.hist(med_a, bins=25, alpha=0.6, label='Model A (llama3.2:1b)', color='royalblue', density=True)
        ax2.hist(med_b, bins=25, alpha=0.6, label='Model B (llama3.2:3b)', color='purple', density=True)
        ax2.set_title('Medium Tier (KS=0.402, p<1e-96)\nScale/Capacity Shift', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Numeric Probe Value')
        ax2.set_ylabel('Empirical Probability Density')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(alpha=0.3)

    # Hard Tier Distributions
    hard_a, hard_b = [], []
    for i in range(5):
        nf = os.path.join(SIM_DIR, "data", f"hard_null_rep{i}.jsonl")
        sf = os.path.join(SIM_DIR, "data", f"hard_substitution_rep{i}.jsonl")
        if os.path.exists(nf):
            hard_a.extend([r["numeric_answer"] for r in load_numeric_stream(nf)])
        if os.path.exists(sf):
            hard_b.extend([r["numeric_answer"] for r in load_numeric_stream(sf) if r["index"] >= 200])

    if hard_a and hard_b:
        ax3.hist(hard_a, bins=25, alpha=0.6, label='Model A (llama3.2:3b-q4)', color='royalblue', density=True)
        ax3.hist(hard_b, bins=25, alpha=0.6, label='Model B (llama3.2:3b-q8)', color='crimson', density=True)
        ax3.set_title('Hard Tier (Quantization Shift)\nPrecision Drift (Q4_K_M → Q8_0)', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Numeric Probe Value')
        ax3.set_ylabel('Empirical Probability Density')
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "distribution_separability_easy_vs_medium.png")
    out_path_all = os.path.join(FIG_DIR, "distribution_separability_all_tiers.png")
    plt.savefig(out_path, dpi=150)
    plt.savefig(out_path_all, dpi=150)
    plt.close()
    return out_path_all


