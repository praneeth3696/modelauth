"""Probe selection as experimental design, not intuition.

Ranks every surveyed probe by information per token billed, then picks a
portfolio under a budget with a diversity constraint. Runs entirely offline from
run_probe_survey.py's output.

Three quantities per probe:

  KL / TV      pairwise divergence between models on that probe's symbol
               distribution. This is the evidence the probe yields per use.
  cost         max_tokens, which is what the provider bills. A probe worth twice
               the evidence at three times the price is a worse buy, and the
               existing three templates were never compared on this axis.
  parse_rate   fraction of generations that produced a usable symbol. A probe
               with high divergence and a 40% parse rate is worse than its KL
               suggests, because most requests yield nothing.

Diversity matters for a reason specific to this problem. A provider that
recognises one probe can cache it away (see adversary.ProbeAwareCache), so a
portfolio concentrated in one family is one string-match from useless. Selection
therefore caps how much of the budget any single family may take.

Usage:
    python probe_selection.py
    python probe_selection.py --budget 40 --max-per-family 3
    python probe_selection.py --covert-only
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

from data_loader import DATA_DIR
from probe_bank import PROBE_BANK

SURVEY_DIR = os.path.join(DATA_DIR, "probe_survey")
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "final-analysis", "figures", "probe_leaderboard.csv")
BY_ID = {p["id"]: p for p in PROBE_BANK}


def load_survey():
    """{probe_id: {model: Counter(symbol)}} plus per-probe parse stats."""
    files = sorted(glob.glob(os.path.join(SURVEY_DIR, "*.jsonl")))
    if not files:
        return None, None
    dists = defaultdict(lambda: defaultdict(Counter))
    stats = defaultdict(lambda: {"n": 0, "parsed": 0})
    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                pid, model, sym = r["probe_id"], r["model"], r["symbol"]
                stats[pid]["n"] += 1
                if sym is not None:
                    stats[pid]["parsed"] += 1
                    dists[pid][model][sym] += 1
    return dists, stats


def pmf(counter, support, alpha=0.5):
    arr = np.array([counter.get(s, 0) for s in support], dtype=float)
    return (arr + alpha) / (arr.sum() + alpha * len(support))


def score_probe(pid, per_model):
    """Worst-case pairwise separation across all model pairs.

    Worst case, not average: a probe that separates four models brilliantly and
    two not at all is only as good as the pair it cannot tell apart, because the
    adversary picks which substitution to make.
    """
    probe = BY_ID.get(pid)
    if probe is None:
        return None
    support = getattr(probe["parser"], "support", None)
    models = sorted(per_model)
    if support is None or len(models) < 2:
        return None

    pmfs = {m: pmf(per_model[m], support) for m in models}
    worst_kl, worst_tv, worst_pair = np.inf, np.inf, None
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            p, q = pmfs[models[i]], pmfs[models[j]]
            kl = float(np.sum(q * np.log(q / p)))
            tv = float(0.5 * np.sum(np.abs(p - q)))
            if kl < worst_kl:
                worst_kl, worst_tv, worst_pair = kl, tv, (models[i], models[j])
    return {"worst_kl": worst_kl, "worst_tv": worst_tv,
            "worst_pair": worst_pair, "n_models": len(models)}


def select(ranked, budget, max_per_family):
    """Greedy pick by information per token, capped per family."""
    chosen, spent, used = [], 0, Counter()
    for row in ranked:
        probe = BY_ID[row["probe_id"]]
        if used[probe["family"]] >= max_per_family:
            continue
        if spent + probe["max_tokens"] > budget:
            continue
        chosen.append(row)
        used[probe["family"]] += 1
        spent += probe["max_tokens"]
    return chosen, spent


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=60, help="total max_tokens")
    ap.add_argument("--max-per-family", type=int, default=3)
    ap.add_argument("--covert-only", action="store_true")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args(argv)

    dists, stats = load_survey()
    if dists is None:
        print(f"No survey data in {SURVEY_DIR}.")
        print("This step needs Ollama. Run:  python run_probe_survey.py")
        print("Everything here is offline once that has been run once.")
        return 1

    ranked = []
    for pid, per_model in dists.items():
        sc = score_probe(pid, per_model)
        if sc is None:
            continue
        probe = BY_ID[pid]
        if args.covert_only and not probe["covert"]:
            continue
        st = stats[pid]
        parse_rate = st["parsed"] / st["n"] if st["n"] else 0.0
        # Effective information per request: unparsed responses yield nothing.
        eff = sc["worst_kl"] * parse_rate
        ranked.append({
            "probe_id": pid, "family": probe["family"], "covert": probe["covert"],
            "cost_tokens": probe["max_tokens"],
            "worst_kl": round(sc["worst_kl"], 4),
            "worst_tv": round(sc["worst_tv"], 4),
            "parse_rate": round(parse_rate, 3),
            "effective_kl": round(eff, 4),
            "kl_per_token": round(eff / probe["max_tokens"], 5),
            "hardest_pair": " vs ".join(sc["worst_pair"]),
            "n_samples": st["n"],
        })

    if not ranked:
        print("Survey data present but no probe scored. Check parsers.")
        return 1

    ranked.sort(key=lambda r: -r["kl_per_token"])

    print(f"PROBE LEADERBOARD -- {len(ranked)} probes, ranked by information per token")
    print(f"{'probe':<22} {'family':<18} {'cov':>4} {'tok':>4} "
          f"{'worstKL':>8} {'parse':>6} {'KL/tok':>8}")
    print("-" * 78)
    for r in ranked[:args.top]:
        print(f"{r['probe_id']:<22} {r['family']:<18} "
              f"{'yes' if r['covert'] else '':>4} {r['cost_tokens']:>4} "
              f"{r['worst_kl']:>8.3f} {r['parse_rate']:>6.2f} {r['kl_per_token']:>8.4f}")

    chosen, spent = select(ranked, args.budget, args.max_per_family)
    print(f"\nPORTFOLIO -- budget {args.budget} tokens, "
          f"max {args.max_per_family} per family")
    print(f"{'probe':<22} {'family':<18} {'tok':>4} {'KL/tok':>8}")
    print("-" * 56)
    for r in chosen:
        print(f"{r['probe_id']:<22} {r['family']:<18} "
              f"{r['cost_tokens']:>4} {r['kl_per_token']:>8.4f}")
    total_kl = sum(r["effective_kl"] for r in chosen)
    print(f"\n{len(chosen)} probes, {spent}/{args.budget} tokens, "
          f"summed effective KL {total_kl:.3f} nats per full sweep")
    if total_kl > 0:
        print(f"At alpha=1e-4 that is roughly {np.log(1e4)/total_kl:.1f} sweeps "
              f"to detection against the hardest model pair.")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    for r in ranked:
        r["selected"] = r["probe_id"] in {c["probe_id"] for c in chosen}
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ranked[0].keys()))
        w.writeheader()
        w.writerows(ranked)
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
