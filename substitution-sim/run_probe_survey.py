"""Sample every candidate probe against every model. REQUIRES OLLAMA.

This is the one step in Phase 3 that cannot be done from logged data: nobody has
ever sent these probes to these models. Everything downstream -- the information
leaderboard, portfolio selection, the camouflage evaluation -- runs off the
JSONL this writes, and all of it works offline once this has been run once.

Cost: len(PROBE_BANK) x n_models x samples requests. At the defaults that is
about 200 x 5 x 60 = 60,000 generations of <=10 tokens. On a local 3B model
expect a few hours; it is fully resumable, so interrupt it freely.

Usage:
    python run_probe_survey.py --check              # what would run, no calls
    python run_probe_survey.py                      # all models in config
    python run_probe_survey.py --samples 100
    python run_probe_survey.py --models llama3.2:3b qwen2.5:3b
    python run_probe_survey.py --families covert_task random_number

Output: data/probe_survey/<model>.jsonl, one record per generation:
    {"probe_id", "family", "prompt", "model", "raw", "symbol", "covert"}
`symbol` is the parsed categorical value, or null when the response did not
parse -- keep those, the parse-failure rate is itself a probe quality signal.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

from config import MODEL_PAIRS
from data_loader import DATA_DIR
from probe_bank import PROBE_BANK

SURVEY_DIR = os.path.join(DATA_DIR, "probe_survey")
OLLAMA_TAGS = "http://localhost:11434/api/tags"


def all_models():
    seen = []
    for a, b in MODEL_PAIRS.values():
        for m in (a, b):
            if m not in seen:
                seen.append(m)
    return seen


def available():
    try:
        import requests
        r = requests.get(OLLAMA_TAGS, timeout=4)
        r.raise_for_status()
        return {m["name"] for m in r.json().get("models", [])}
    except Exception as e:
        print(f"[error] Ollama not reachable at {OLLAMA_TAGS}: {e}")
        return None


def load_done(path):
    """Return {(probe_id): count} already recorded, so a rerun resumes."""
    done = defaultdict(int)
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done[json.loads(line)["probe_id"]] += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
    return done


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=60,
                    help="generations per probe per model")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    probes = PROBE_BANK
    if args.families:
        probes = [p for p in probes if p["family"] in args.families]
    if not probes:
        print("No probes match those families.")
        return 2

    models = args.models or all_models()
    have = available()
    if have is None:
        print("        Start it with `ollama serve`, then re-run.")
        return 1
    missing = [m for m in models if m not in have]
    for m in missing:
        print(f"  [skip] {m} not pulled   ->  ollama pull {m}")
    models = [m for m in models if m in have]
    if not models:
        print("\nNo requested model is available.")
        return 1

    total = len(probes) * len(models) * args.samples
    print(f"{len(probes)} probes x {len(models)} models x {args.samples} samples "
          f"= {total} generations")
    for m in models:
        print(f"  {m}")
    if args.check:
        print("\n--check: nothing sent.")
        return 0

    from probe_client import probe as call_probe

    os.makedirs(SURVEY_DIR, exist_ok=True)
    for model in models:
        path = os.path.join(SURVEY_DIR, f"{model.replace(':', '_').replace('/', '_')}.jsonl")
        done = load_done(path)
        pending = sum(max(0, args.samples - done[p["id"]]) for p in probes)
        print(f"\n=== {model} ===  {pending} generations to go "
              f"({len(done)} probes already have samples)")
        if not pending:
            continue

        with open(path, "a") as f:
            for pi, p in enumerate(probes, 1):
                need = args.samples - done[p["id"]]
                if need <= 0:
                    continue
                for _ in range(need):
                    raw = call_probe(model, p["prompt"],
                                     temperature=args.temperature,
                                     max_tokens=p["max_tokens"])
                    sym = p["parser"](raw) if raw is not None else None
                    f.write(json.dumps({
                        "probe_id": p["id"], "family": p["family"],
                        "prompt": p["prompt"], "model": model,
                        "raw": raw, "symbol": sym, "covert": p["covert"],
                    }) + "\n")
                f.flush()
                if pi % 10 == 0:
                    print(f"  [{pi}/{len(probes)}] {p['id']}", flush=True)
        print(f"  wrote {path}")

    print("\nDone. Next: python probe_selection.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
