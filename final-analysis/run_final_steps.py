import os
import sys

# Ensure module pathing
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from sanity_checks import (
    audit_data_completeness,
    check_model_separability,
    check_tier_ordering,
    audit_all_repetitions,
)
from visualizations import (
    plot_example_trace,
    plot_roc_curves,
    plot_contamination_curve,
    build_summary_table,
    plot_detector_benchmark,
    plot_distribution_separability,
    FIG_DIR,
)
from detector_cusum import adaptive_cusum_detector

def main():
    print("=" * 70)
    print("      EASY TIER: SANITY-CHECKING & VISUALIZATION SUITE")
    print("=" * 70)
    print()

    # -------------------------------------------------------------
    # STEP 1: SANITY CHECKS
    # -------------------------------------------------------------
    print("--- [STEP 1.1] Data Completeness Audit ---")
    completeness_report = audit_data_completeness()
    if completeness_report:
        for row in completeness_report[:10]:
            print(f"File: {row['file']:<35} | Total: {row['n_total']:<3} | Failed: {row['n_failed']:<2} | Unparseable: {row['n_unparseable']:<2} | Usable: {row['pct_usable']}%")
        if len(completeness_report) > 10:
            print(f"... and {len(completeness_report) - 10} more files audited.")
    else:
        print("[warn] No .jsonl data files found in data directory.")
    print()

    print("--- [STEP 1.2] Model Separability Check (Model A vs Model B) ---")
    separability = check_model_separability("easy")
    print(f"Easy Tier Separability: KS-Stat = {separability.get('ks_stat')}, p-value = {separability.get('p_value')}, Separable = {separability.get('separable')}")
    print()

    print("--- [STEP 1.3] Check Tier Metrics ---")
    ordering = check_tier_ordering(["easy"])
    print(f"Easy Tier Metrics: {ordering}")
    print()

    print("--- [STEP 1.4] Repetition Anomaly Audit ---")
    anomalies = audit_all_repetitions("easy", "substitution")
    if anomalies:
        print(f"[warn] Easy Tier: Found {len(anomalies)} short streams (<300 records): {anomalies}")
    else:
        print(f"[OK] Easy Tier: All checked streams contain expected record counts (>=300 records).")
    print()

    # -------------------------------------------------------------
    # STEP 2: VISUALIZATION IN DEPTH
    # -------------------------------------------------------------
    print("=" * 70)
    print("--- [STEP 2] Building Figures & Summary Tables ---")
    print("=" * 70)

    # Figure 1: Example Trace for Easy
    trace_path = plot_example_trace("easy", 0, adaptive_cusum_detector, {"warmup": 40, "k": 0.5, "h": 5.0})
    if trace_path:
        print(f"[OK] Figure (Easy Trace) generated: {os.path.relpath(trace_path, SCRIPT_DIR)}")

    # Figure 2: ROC Curve for Easy
    roc_path = plot_roc_curves("easy")
    if roc_path:
        print(f"[OK] Figure (Easy ROC) generated: {os.path.relpath(roc_path, SCRIPT_DIR)}")

    # Figure 3: Contamination Curve
    contam_path = plot_contamination_curve()
    if contam_path:
        print(f"[OK] Figure 3 (Contamination Boundary) generated: {os.path.relpath(contam_path, SCRIPT_DIR)}")

    # Figure 4: Detector Comparison Bar Chart
    comp_path = plot_detector_benchmark()
    if comp_path:
        print(f"[OK] Figure 4 (Detector Comparison) generated: {os.path.relpath(comp_path, SCRIPT_DIR)}")

    # Figure 5: Model Output Distribution Separability
    dist_path = plot_distribution_separability()
    if dist_path:
        print(f"[OK] Figure 5 (Distribution Separability) generated: {os.path.relpath(dist_path, SCRIPT_DIR)}")

    # Table 1: Summary CSV
    table_rows = build_summary_table(["easy"])
    summary_path = os.path.join(FIG_DIR, "summary_table.csv")
    print(f"[OK] Table 1 generated: {os.path.relpath(summary_path, SCRIPT_DIR)}")
    print("\nSummary Table Output:")
    for r in table_rows:
        print(r)

    print()
    print("=" * 70)
    print("      EASY TIER EXECUTION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()

