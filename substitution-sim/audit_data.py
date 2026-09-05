"""Phase 0 data audit: completeness, parse failures, and per-tier separability.

Run this before trusting any detector output. It answers two questions:
  1. How much of each stream actually survives parsing, and what is being lost?
  2. Is there any signal in the channel at all for this tier -- i.e. do the two
     models differ enough that a detector could succeed even in principle?

A tier with near-zero KL is a data problem, not a detector problem, and no
amount of detector work will fix it.
"""

import os
import sys
from collections import Counter

from config import MODEL_PAIRS
from data_loader import DATA_DIR, audit_file, load_numeric_stream, split_by_model
from stats_categorical import channel_report

ALPHA = 1e-4


def completeness():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".jsonl"))
    if not files:
        print(f"No .jsonl files in {DATA_DIR}")
        return []

    print(f"{'file':<34} {'total':>6} {'ok':>6} {'range':>6} {'unparse':>8} "
          f"{'failed':>7} {'usable':>8}")
    print("-" * 80)
    reports, all_offenders = [], Counter()
    for name in files:
        rep = audit_file(os.path.join(DATA_DIR, name))
        reports.append(rep)
        all_offenders.update(rep["offenders"])
        flag = "  <-- CHECK" if rep["pct_usable"] < 95 else ""
        print(f"{rep['file']:<34} {rep['n_total']:>6} {rep['n_ok']:>6} "
              f"{rep['n_out_of_range']:>6} {rep['n_unparseable']:>8} "
              f"{rep['n_failed']:>7} {rep['pct_usable']:>7.1f}%{flag}")

    tot = sum(r["n_total"] for r in reports)
    bad = sum(r["n_out_of_range"] + r["n_unparseable"] + r["n_failed"] for r in reports)
    print(f"\n{len(reports)} files, {tot} probes, {bad} rejected "
          f"({100*bad/tot:.2f}%)" if tot else "")
    if all_offenders:
        print("\nRejected answers, most common first:")
        for text, n in all_offenders.most_common(12):
            print(f"  {n:>5} x {text}")
    return reports


def separability():
    print("\n" + "=" * 80)
    print("PER-TIER SEPARABILITY (held-out rep14)")
    print("=" * 80)
    any_tier = False
    for difficulty, (model_a, model_b) in MODEL_PAIRS.items():
        null_p = os.path.join(DATA_DIR, f"{difficulty}_null_rep14.jsonl")
        sub_p = os.path.join(DATA_DIR, f"{difficulty}_substitution_rep14.jsonl")
        if not (os.path.exists(null_p) and os.path.exists(sub_p)):
            print(f"\n{difficulty:<8} NO DATA -- run run_experiments.py for this tier")
            continue
        any_tier = True
        a_vals = [r["numeric_answer"] for r in load_numeric_stream(null_p)]
        b_vals = split_by_model(sub_p).get(model_b, [])
        if not b_vals:
            print(f"\n{difficulty:<8} no {model_b} samples in the substitution stream")
            continue

        rep = channel_report(a_vals, b_vals, alpha=ALPHA)
        print(f"\n{difficulty}: {model_a} -> {model_b}   (n={rep['n_a']} / {rep['n_b']})")
        print(f"  entropy         {rep['entropy_a_bits']:.2f} / {rep['entropy_b_bits']:.2f} bits")
        print(f"  KL(B||A)        {rep['kl_b_given_a']:.3f} nats/probe")
        print(f"  total variation {rep['total_variation']:.3f}")
        print(f"  Bayes error     {rep['bayes_error']:.3f}  (single probe)")
        print(f"  Lorden bound    {rep['lorden_bound_probes']:.1f} probes "
              f"at alpha={ALPHA:g}")

        top_a = Counter(a_vals).most_common(5)
        top_b = Counter(b_vals).most_common(5)
        print(f"  top A: {top_a}")
        print(f"  top B: {top_b}")

        if rep["kl_b_given_a"] < 0.05:
            print("  VERDICT: channel is nearly blind. The numeric probe cannot "
                  "carry this tier;\n           a richer probe family is required "
                  "before detector work is meaningful.")
        elif rep["kl_b_given_a"] < 0.5:
            print("  VERDICT: weak but usable. Expect delays in the hundreds of probes.")
        else:
            print("  VERDICT: strong channel. Detection should take single-digit probes.")
    return any_tier


if __name__ == "__main__":
    completeness()
    ok = separability()
    sys.exit(0 if ok else 1)
