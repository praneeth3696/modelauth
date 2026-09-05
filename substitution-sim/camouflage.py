"""Can a provider tell your audit probes from its ordinary traffic?

No prior work in this area states this constraint, and it is now the binding
one. fpverify ships the "pick a random number" probe in a public repository, so
a provider that reads the monitoring literature can filter audit-shaped requests
with a single string match, serve those honestly from cache, and route the rest
to the cheap model. adversary.ProbeAwareCache measures what that costs the
monitor: on the easy tier it takes the e-process from 100% power to zero while
the provider keeps essentially the whole saving.

So probe detectability belongs next to power as a reported metric. A probe that
yields 3 nats and is trivially filterable is worth less in deployment than one
yielding 0.5 nats that reads as ordinary traffic.

This module scores detectability two ways:

  keyword     the one-line filter a provider would actually write first. Cheap,
              and it catches every published probe.
  classifier  logistic regression on character n-grams, trained to separate the
              probe bank from an organic-traffic corpus. This is the stronger
              adversary and the number to report.

The organic corpus matters. A default one ships here so the module runs, but it
is small and hand-written, so treat its absolute numbers as indicative. Point
--organic at a real prompt log for a result worth publishing.

Usage:
    python camouflage.py
    python camouflage.py --organic path/to/prompts.txt
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter

import numpy as np

from probe_bank import PROBE_BANK

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "final-analysis", "figures", "camouflage.csv")

# Small stand-in corpus of ordinary developer/assistant traffic. Replace with a
# real log via --organic; these 40 lines are enough to run, not to conclude.
DEFAULT_ORGANIC = [
    "Write a Python function that reverses a string.",
    "What is the capital of Portugal?",
    "Summarise this paragraph in one sentence.",
    "How do I center a div with flexbox?",
    "Explain the difference between TCP and UDP.",
    "Fix this SQL query, it returns duplicates.",
    "What should I name a variable holding a user's age?",
    "Translate 'good morning' into Japanese.",
    "Write a regex that matches an email address.",
    "How many worker threads for a small web service?",
    "Give me a one-line git command to undo the last commit.",
    "What timeout in seconds for a database query?",
    "Draft a polite decline to a meeting invitation.",
    "Convert this JSON to YAML.",
    "Why is my Docker build so slow?",
    "Recommend a minimum password length.",
    "Explain closures to someone new to JavaScript.",
    "What HTTP status code should a missing record return?",
    "Write a unit test for this add function.",
    "Suggest a git branch name for a bugfix.",
    "Is it better to use a list or a set here?",
    "How do I read a CSV file in pandas?",
    "Rewrite this sentence to be more concise.",
    "What does this error message mean?",
    "Give me an example integer for a unit test.",
    "Tabs or spaces?",
    "Suggest a port number for a local dev server.",
    "How many times should an HTTP client retry?",
    "What is the time complexity of quicksort?",
    "Write a docstring for this function.",
    "Should I use REST or GraphQL for this API?",
    "Name three benefits of code review.",
    "Default page size for an API listing?",
    "How do I mock a network call in pytest?",
    "What delimiter for a config file: comma, colon, or equals?",
    "Explain what a race condition is.",
    "Make this function handle empty input.",
    "How many bullet points should a summary slide have?",
    "What is the difference between git merge and rebase?",
    "Give me a random seed for reproducibility.",
]

PROBE_KEYWORDS = [
    r"\brandom\b", r"\bpick one\b", r"\bchoose\b.*\bat random\b",
    r"\bnumber only\b", r"\bjust the digits\b", r"\bone word\b",
    r"\bno explanation\b", r"\banswer with the (number|letter) alone\b",
]


def keyword_flags(text):
    t = text.lower()
    return any(re.search(k, t) for k in PROBE_KEYWORDS)


def char_ngrams(text, n=(3, 4)):
    t = " " + text.lower().strip() + " "
    out = Counter()
    for k in n:
        for i in range(len(t) - k + 1):
            out[t[i:i + k]] += 1
    return out


def train_classifier(pos_texts, neg_texts, seed=0, epochs=300, lr=0.5):
    """Logistic regression on hashed char n-grams. No sklearn dependency."""
    rng = np.random.default_rng(seed)
    DIM = 2048

    def vec(text):
        v = np.zeros(DIM)
        for g, c in char_ngrams(text).items():
            v[hash(g) % DIM] += c
        n = np.linalg.norm(v)
        return v / n if n else v

    X = np.array([vec(t) for t in pos_texts + neg_texts])
    y = np.array([1.0] * len(pos_texts) + [0.0] * len(neg_texts))

    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(0.7 * len(X))
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]

    w, b = np.zeros(DIM), 0.0
    for _ in range(epochs):
        z = Xtr @ w + b
        p = 1 / (1 + np.exp(-z))
        gw = Xtr.T @ (p - ytr) / len(Xtr)
        gb = float((p - ytr).mean())
        w -= lr * gw
        b -= lr * gb

    pte = 1 / (1 + np.exp(-(Xte @ w + b)))
    pred = (pte > 0.5).astype(float)
    acc = float((pred == yte).mean()) if len(yte) else float("nan")
    return acc, (lambda t: float(1 / (1 + np.exp(-(vec(t) @ w + b)))))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--organic", default=None,
                    help="file of organic prompts, one per line")
    args = ap.parse_args(argv)

    if args.organic:
        with open(args.organic, encoding="utf-8") as f:
            organic = [l.strip() for l in f if l.strip()]
        source = args.organic
    else:
        organic = DEFAULT_ORGANIC
        source = "built-in stand-in corpus (40 lines)"

    probes = [p["prompt"] for p in PROBE_BANK]
    print(f"{len(probes)} probes vs {len(organic)} organic prompts")
    print(f"organic source: {source}\n")

    kw_probe = sum(keyword_flags(p) for p in probes) / len(probes)
    kw_organic = sum(keyword_flags(o) for o in organic) / len(organic)
    print("KEYWORD FILTER (the one-line version a provider writes first)")
    print(f"  probes caught      {kw_probe*100:5.1f}%")
    print(f"  organic misrouted  {kw_organic*100:5.1f}%")

    covert = [p for p in PROBE_BANK if p["covert"]]
    overt = [p for p in PROBE_BANK if not p["covert"]]
    kw_covert = sum(keyword_flags(p["prompt"]) for p in covert) / max(1, len(covert))
    kw_overt = sum(keyword_flags(p["prompt"]) for p in overt) / max(1, len(overt))
    print(f"  of overt probes    {kw_overt*100:5.1f}% caught  ({len(overt)} probes)")
    print(f"  of covert probes   {kw_covert*100:5.1f}% caught  ({len(covert)} probes)")

    acc, score = train_classifier(probes, organic)
    print(f"\nLEARNED CLASSIFIER (char n-grams, held-out accuracy {acc*100:.1f}%)")
    print("  A provider at this accuracy can filter probes without a keyword list.")

    rows = []
    for p in PROBE_BANK:
        rows.append({
            "probe_id": p["id"], "family": p["family"],
            "declared_covert": p["covert"],
            "keyword_flagged": keyword_flags(p["prompt"]),
            "classifier_prob_is_probe": round(score(p["prompt"]), 3),
            "prompt": p["prompt"],
        })
    rows.sort(key=lambda r: r["classifier_prob_is_probe"])

    print("\nLEAST DETECTABLE PROBES (lowest classifier score)")
    print(f"{'probe':<22} {'kw':>4} {'p(probe)':>9}  prompt")
    print("-" * 92)
    for r in rows[:12]:
        print(f"{r['probe_id']:<22} {'Y' if r['keyword_flagged'] else '.':>4} "
              f"{r['classifier_prob_is_probe']:>9.3f}  {r['prompt'][:52]}")

    print("\nMOST DETECTABLE PROBES")
    print(f"{'probe':<22} {'kw':>4} {'p(probe)':>9}  prompt")
    print("-" * 92)
    for r in rows[-8:]:
        print(f"{r['probe_id']:<22} {'Y' if r['keyword_flagged'] else '.':>4} "
              f"{r['classifier_prob_is_probe']:>9.3f}  {r['prompt'][:52]}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT_CSV}")
    print("\nPair this with probe_selection.py: the portfolio worth deploying")
    print("maximises information per token subject to staying below a")
    print("detectability ceiling, not information alone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
