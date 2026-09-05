"""The detectability frontier: how much cheating buys how much impunity.

Runs each detector against each provider strategy and reports probes-to-detection
alongside what the provider saved. The output is an exchange rate: at routing
fraction r the provider banks (1-r) of the inference bill and buys N probes of
impunity before the monitor fires.

Offline by default -- resamples from logged response pools, so it needs no
serving stack. That destroys within-session run structure, so absolute delays
run pessimistic; read the shape of the curve, not the intercept. Pass --online
to route live traffic through Ollama instead, which is the number for a paper.

Usage:
    python run_adversarial.py                  # offline, easy tier
    python run_adversarial.py --tier medium
    python run_adversarial.py --trials 60
    python run_adversarial.py --online         # requires Ollama
"""

import argparse
import csv
import os
import sys

import numpy as np

from config import MODEL_PAIRS, PROBE_TEMPLATES
from adversary import (Honest, FullSwap, PartialRoute, ProbeAwareCache,
                       DistributionMatched, simulate, build_pools)
from detector_compression import eprocess_detector, mixture_alternative_detector
from detector_cusum import adaptive_cusum_detector

SWITCH = 200
LENGTH = 1200
ALPHA = 1e-4
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "final-analysis", "figures", "adversarial_frontier.csv")

DETECTORS = [
    ("e-process", eprocess_detector, {"alpha": ALPHA}),
    ("mixture-alternative", mixture_alternative_detector, {"alpha": ALPHA}),
    ("adaptive CUSUM", adaptive_cusum_detector, {"h": 7.0}),
]


def detect(fn, values, ctxs, kwargs):
    names = fn.__code__.co_varnames[:fn.__code__.co_argcount]
    res = fn(values, contexts=ctxs, **kwargs) if "contexts" in names else fn(values, **kwargs)
    return next((d["index"] for d in res if d["flagged"] and d["index"] >= SWITCH), None)


def strategies(pool_a, pool_b):
    out = [("honest", Honest(pool_a)), ("full swap", FullSwap(pool_a, pool_b, SWITCH))]
    for r in (0.75, 0.5, 0.25, 0.1, 0.05):
        out.append((f"route {int(r*100)}%", PartialRoute(pool_a, pool_b, SWITCH, r)))
    out.append(("probe-aware cache", ProbeAwareCache(pool_a, pool_b, SWITCH)))
    out.append(("probe-cache 50% miss",
                ProbeAwareCache(pool_a, pool_b, SWITCH, detect_rate=0.5)))
    out.append(("distribution-matched", DistributionMatched(pool_a, pool_b, SWITCH)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="easy", choices=list(MODEL_PAIRS))
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--online", action="store_true",
                    help="route live traffic through Ollama instead of resampling")
    args = ap.parse_args(argv)

    if args.online:
        print("Online mode needs Ollama and regenerates traffic per strategy.")
        print("Not run here; see adversary.py -- the strategy objects are shared,")
        print("so wiring them to probe_client.probe is the only change needed.")
        return 2

    pool_a, pool_b = build_pools(args.tier)
    if not pool_a or not pool_b:
        print(f"No logged data for tier '{args.tier}'.")
        return 1
    print(f"tier={args.tier}  pools: A={len(pool_a)} B={len(pool_b)}  "
          f"trials={args.trials}  offline resampling\n")

    rows = []
    hdr = (f"{'strategy':<22} {'saving':>7} " +
           " ".join(f"{n:>22}" for n, _, _ in DETECTORS))
    print(hdr)
    print("-" * len(hdr))

    # The honest row is the control: whatever a detector "finds" there is its
    # own false-alarm rate. Any strategy it flags at or below that rate is not
    # being detected, however confident the raw power number looks.
    honest_power = {}

    for label, prov in strategies(pool_a, pool_b):
        cells, saving = [], None
        for dname, fn, kwargs in DETECTORS:
            delays, fired = [], 0
            for trial in range(args.trials):
                values, ctxs, truth, cost = simulate(
                    prov, LENGTH, SWITCH, seed=trial, prompts=PROBE_TEMPLATES)
                saving = 1.0 - cost
                hit = detect(fn, values, ctxs, kwargs)
                if hit is not None:
                    fired += 1
                    delays.append(hit - SWITCH)
            power = fired / args.trials
            delay = float(np.mean(delays)) if delays else None
            if label == "honest":
                honest_power[dname] = power
            base = honest_power.get(dname, 0.0)
            excess = max(0.0, power - base)
            mark = "" if label == "honest" else ("  " if excess > 0.1 else " !")
            cells.append(f"{(f'{delay:.0f}' if delay else '--'):>6} "
                         f"@{power*100:>4.0f}% +{excess*100:>3.0f}{mark:<2}")
            rows.append({"tier": args.tier, "strategy": label,
                         "provider_saving": round(saving, 3) if saving else 0.0,
                         "detector": dname, "mean_delay": delay,
                         "power": power, "honest_false_alarm": base,
                         "excess_power": round(excess, 3), "trials": args.trials})
        print(f"{label:<22} {saving*100:>6.0f}% " + " ".join(cells))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    write_header = True
    mode = "w"
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV) as f:
            existing = [r for r in csv.DictReader(f) if r["tier"] != args.tier]
        if existing:
            rows = existing + rows
    with open(OUT_CSV, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT_CSV}")
    print("\nsaving = fraction of the full-price inference bill the provider keeps.")
    print("'--' means the strategy was never detected in any trial.")
    print("+N = power ABOVE the detector's own rate on honest traffic. '!' marks")
    print("cells where that excess is under 10 points: the detector is firing at")
    print("roughly its false-alarm rate, which is not detection however high the")
    print("raw power looks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
