"""Multi-tier detector benchmark, scored against the information-theoretic floor.

Changes from the previous version:
  - Adds the oracle LR-CUSUM upper bound (knows both PMFs; nothing deployable
    can match it) so the gap between "achievable" and "achieved" is visible,
    and scores every detector against its measured delay on the same streams.
    The Lorden bound is reported alongside as a reference but is NOT the
    denominator -- see the note at the ratio, it is a different criterion.
  - Suppresses that ratio below 50% power, where a mean over a few detected
    streams would otherwise read as though a weak detector beat the oracle.
  - Counts null-stream false alarms over the WHOLE stream, not only the region
    before t=200. A null stream has no switch, so a flag at t=300 is a false
    alarm too; the old version discarded those.
  - Reports streams_with_false_alarm alongside the rate, because operationally
    one alarm on a clean endpoint costs the same as ten.
  - Skips tiers with no data instead of silently emitting an easy-only table.
"""

import csv
import os
import sys

import numpy as np

from config import MODEL_PAIRS, SWITCH_POINT
from data_loader import DATA_DIR, load_numeric_stream, split_by_model
from detector_v1 import sliding_window_detector
from detector_cusum import adaptive_cusum_detector
from detector_variance_cusum import variance_cusum_detector
from detector_fixed_reference import fixed_reference_detector, build_reference_distribution
from detector_oracle import oracle_lr_cusum, fit_oracle_pmfs
from detector_compression import eprocess_detector, mdl_cusum_detector
from detector_baselines_2026 import energy_distance_detector, js_fingerprint_detector
from stats_categorical import channel_report, estimate_pmf, lorden_bound

ALPHA = 1e-4          # anytime-valid target for the oracle
N_TEST_REPS = 14      # rep14 is held out for reference/oracle fitting
HELD_OUT_REP = 14
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "final-analysis", "figures", "summary_table_all_tiers.csv")


def stream_path(difficulty, condition, rep):
    return os.path.join(DATA_DIR, f"{difficulty}_{condition}_rep{rep}.jsonl")


def load_all_reps(difficulty, condition, n_reps=N_TEST_REPS):
    streams = []
    for rep in range(n_reps):
        path = stream_path(difficulty, condition, rep)
        if not os.path.exists(path):
            break
        recs = load_numeric_stream(path)
        streams.append(([r["numeric_answer"] for r in recs],
                        [r["prompt"][:16] for r in recs]))
    return streams


def _call(detector_fn, values, ctxs, kwargs):
    """Pass contexts only to detectors that accept them."""
    if "contexts" in detector_fn.__code__.co_varnames[:detector_fn.__code__.co_argcount]:
        return detector_fn(values, contexts=ctxs, **kwargs)
    return detector_fn(values, **kwargs)


def compute_metrics(detector_fn, sub_streams, null_streams, true_switch, **kwargs):
    delays = []
    for values, ctxs in sub_streams:
        results = _call(detector_fn, values, ctxs, kwargs)
        first = next((d for d in results if d["flagged"] and d["index"] >= true_switch), None)
        delays.append(first["index"] - true_switch if first else None)

    fa_rates, streams_with_fa = [], 0
    for values, ctxs in null_streams:
        results = _call(detector_fn, values, ctxs, kwargs)
        # A null stream never switches, so ANY flag is a false alarm.
        n_flags = sum(1 for d in results if d["flagged"])
        fa_rates.append(n_flags / len(results) if results else 0.0)
        if n_flags:
            streams_with_fa += 1

    detected = [d for d in delays if d is not None]
    return {
        "mean_delay": float(np.mean(detected)) if detected else None,
        "detection_rate": len(detected) / len(delays) if delays else 0.0,
        "mean_false_alarm_rate": float(np.mean(fa_rates)) if fa_rates else 0.0,
        "streams_with_false_alarm": streams_with_fa,
        "n_null_streams": len(null_streams),
    }


def tier_channel(difficulty):
    """Estimate the tier's KL/TV/Lorden bound from the held-out streams only."""
    null_path = stream_path(difficulty, "null", HELD_OUT_REP)
    sub_path = stream_path(difficulty, "substitution", HELD_OUT_REP)
    if not (os.path.exists(null_path) and os.path.exists(sub_path)):
        return None, None, None

    a_vals = [r["numeric_answer"] for r in load_numeric_stream(null_path)]
    by_model = split_by_model(sub_path)
    model_a, model_b = MODEL_PAIRS[difficulty]
    b_vals = by_model.get(model_b, [])
    if not a_vals or not b_vals:
        return None, None, None

    report = channel_report(a_vals, b_vals, alpha=ALPHA)
    return report, estimate_pmf(a_vals), estimate_pmf(b_vals)


def main():
    rows = []
    channel_rows = []

    for difficulty in MODEL_PAIRS:
        sub_streams = load_all_reps(difficulty, "substitution")
        null_streams = load_all_reps(difficulty, "null")
        if not sub_streams or not null_streams:
            print(f"[skip] {difficulty}: no data on disk "
                  f"({len(sub_streams)} sub / {len(null_streams)} null streams)")
            continue

        report, p_a, p_b = tier_channel(difficulty)
        if report:
            model_a, model_b = MODEL_PAIRS[difficulty]
            print(f"\n=== {difficulty}: {model_a} -> {model_b} ===")
            print(f"  KL(B||A)  = {report['kl_b_given_a']:.3f} nats/probe")
            print(f"  TV        = {report['total_variation']:.3f}   "
                  f"Bayes error = {report['bayes_error']:.3f}")
            print(f"  entropy   = {report['entropy_a_bits']:.2f} / "
                  f"{report['entropy_b_bits']:.2f} bits (A / B)")
            print(f"  Lorden bound at alpha={ALPHA:g}: "
                  f"{report['lorden_bound_probes']:.1f} probes")
            channel_rows.append({"difficulty": difficulty, **{
                k: v for k, v in report.items() if k != "alpha"}})
        optimum = report["lorden_bound_probes"] if report else None

        methods = [
            ("KS sliding window", sliding_window_detector, {"window_size": 20}),
            ("energy distance [Leshin]", energy_distance_detector, {"alpha": ALPHA}),
            ("JS fingerprint [Bruckner]", js_fingerprint_detector, {}),
            ("MDL-CUSUM", mdl_cusum_detector, {"alpha": ALPHA}),
            ("e-process (Ville)", eprocess_detector, {"alpha": ALPHA}),
            ("e-process conditional", eprocess_detector,
             {"alpha": ALPHA, "conditional": True}),
            ("adaptive CUSUM", adaptive_cusum_detector, {"warmup": 40, "k": 0.5, "h": 5.0}),
            ("variance CUSUM", variance_cusum_detector, {"warmup": 40, "k": 0.5, "h": 5.0}),
        ]

        ref_path = stream_path(difficulty, "null", HELD_OUT_REP)
        if os.path.exists(ref_path):
            ref = build_reference_distribution(
                [r["numeric_answer"] for r in load_numeric_stream(ref_path)])
            methods.append((
                "fixed reference",
                lambda s, _r=ref, **kw: fixed_reference_detector(s, _r, **kw),
                {"batch_size": 20},
            ))

        if p_a is not None and p_b is not None:
            methods.append((
                "oracle LR-CUSUM",
                lambda s, _a=p_a, _b=p_b, **kw: oracle_lr_cusum(s, _a, _b, **kw),
                {"alpha": ALPHA},
            ))

        measured = []
        for name, fn, kwargs in methods:
            m = compute_metrics(fn, sub_streams, null_streams, SWITCH_POINT, **kwargs)
            measured.append((name, m))

        # The empirical optimum is the oracle's own measured delay on these same
        # streams -- same metric, same data, so the ratio is apples to apples.
        #
        # The Lorden bound is NOT that, and must not be used as the denominator.
        # Lorden's criterion is the worst case over change points and pre-change
        # histories, sup_v ess sup E[(tau-v)+ | F_v], and h/KL is asymptotic in h.
        # A mean delay measured from a known switch point with the statistic reset
        # to zero is an average-case quantity, and it can legitimately come in
        # below the worst-case reference -- the oracle does exactly that here,
        # helped by overshoot: when qwen emits 42, a token llama almost never
        # produces, one probe carries several nats. Keep it as a reference column.
        oracle_delay = next((m["mean_delay"] for n, m in measured
                             if n == "oracle LR-CUSUM" and m["mean_delay"]), None)

        # Mean delay is averaged over detected streams only, so at low power it
        # describes a handful of lucky runs rather than the detector. On the hard
        # tier this makes adaptive CUSUM look like it beats the oracle (0.6x) while
        # detecting 28.6% of the time with false alarms on 8 of 14 clean streams --
        # it is firing near-randomly, not detecting early. Suppress the ratio below
        # this power floor rather than print a number that invites the wrong read.
        MIN_POWER_FOR_RATIO = 0.5
        oracle_reliable = next(
            (m["detection_rate"] for n, m in measured if n == "oracle LR-CUSUM"), 0.0
        ) >= MIN_POWER_FOR_RATIO

        for name, m in measured:
            comparable = (m["detection_rate"] >= MIN_POWER_FOR_RATIO and oracle_reliable)
            ratio = (100.0 * m["mean_delay"] / oracle_delay
                     if oracle_delay and m["mean_delay"] is not None and comparable
                     else None)
            rows.append({
                "difficulty": difficulty,
                "method": name,
                "mean_delay": m["mean_delay"],
                "pct_of_oracle": round(ratio, 1) if ratio is not None else None,
                "detection_rate": m["detection_rate"],
                "mean_false_alarm_rate": m["mean_false_alarm_rate"],
                "streams_with_false_alarm": m["streams_with_false_alarm"],
                "n_null_streams": m["n_null_streams"],
                "oracle_delay_probes": round(oracle_delay, 2) if oracle_delay else None,
                "lorden_worstcase_ref": round(optimum, 2) if optimum else None,
            })

    if not rows:
        print("No tier had usable data. Run run_experiments.py first.")
        return 1

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    if channel_rows:
        ch_csv = os.path.join(os.path.dirname(OUT_CSV), "channel_report.csv")
        with open(ch_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(channel_rows[0].keys()))
            w.writeheader()
            w.writerows(channel_rows)

    hdr = (f"\n{'tier':<8} {'method':<20} {'delay':>7} {'xoracle':>8} "
           f"{'power':>7} {'FA rate':>9} {'FA streams':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        delay = f"{r['mean_delay']:.2f}" if r["mean_delay"] is not None else "--"
        pct = f"{r['pct_of_oracle']/100:.1f}x" if r["pct_of_oracle"] is not None else "--"
        print(f"{r['difficulty']:<8} {r['method']:<20} {delay:>7} {pct:>8} "
              f"{r['detection_rate']*100:>6.1f}% {r['mean_false_alarm_rate']*100:>8.3f}% "
              f"{r['streams_with_false_alarm']:>6}/{r['n_null_streams']:<4}")
    print("\nxoracle = mean delay relative to the oracle LR-CUSUM on the same streams,")
    print("shown only where that detector and the oracle both clear 50% power: a mean")
    print("over a few detected streams is not comparable to one over all of them.")
    print("The Lorden column in the CSV is a worst-case asymptotic reference, not a")
    print("denominator: it is a different criterion from the average delay measured here.")
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
