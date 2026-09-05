"""Resumable experiment runner.

Usage:
    python run_experiments.py                # all tiers in config
    python run_experiments.py medium hard    # only these tiers
    python run_experiments.py --check        # preflight only, generate nothing

Fixes over the previous version:
  - MAX_TOKENS is passed through from config. It was not, so the generator kept
    its own default of 5 and every truncation fix in config.py was a no-op.
  - Writes to substitution-sim/data regardless of the working directory.
  - Preflights model availability against the Ollama tag list, so a missing
    model fails in one second instead of retrying for hours across 400 probes.
"""

import json
import os
import sys

from config import (MODEL_PAIRS, TOTAL_REQUESTS, SWITCH_POINT, N_REPETITIONS,
                    TEMPERATURE, MAX_TOKENS)
from data_loader import DATA_DIR

OLLAMA_TAGS = "http://localhost:11434/api/tags"


def available_models():
    """Names Ollama currently serves, or None if it is not reachable."""
    try:
        import requests
        r = requests.get(OLLAMA_TAGS, timeout=4)
        r.raise_for_status()
        return {m["name"] for m in r.json().get("models", [])}
    except Exception as e:
        print(f"[error] Ollama not reachable at {OLLAMA_TAGS}: {e}")
        return None


def preflight(tiers):
    have = available_models()
    if have is None:
        print("        Start it with `ollama serve`, then re-run.")
        return None

    print(f"Ollama is serving {len(have)} model(s).")
    runnable = []
    for tier in tiers:
        model_a, model_b = MODEL_PAIRS[tier]
        missing = [m for m in (model_a, model_b) if m not in have]
        if missing:
            print(f"  [skip] {tier:<7} missing: {', '.join(missing)}")
            for m in missing:
                print(f"           ollama pull {m}")
        else:
            print(f"  [ok]   {tier:<7} {model_a} -> {model_b}")
            runnable.append(tier)
    return runnable


def main(argv):
    check_only = "--check" in argv
    requested = [a for a in argv if not a.startswith("-")]
    tiers = requested or list(MODEL_PAIRS)

    unknown = [t for t in tiers if t not in MODEL_PAIRS]
    if unknown:
        print(f"Unknown tier(s): {', '.join(unknown)}. "
              f"Known: {', '.join(MODEL_PAIRS)}")
        return 2

    runnable = preflight(tiers)
    if runnable is None:
        return 1
    if check_only:
        return 0 if runnable else 1
    if not runnable:
        print("\nNothing to run. Pull the missing models first.")
        return 1

    # Imported here, not at module scope, so --check works without the
    # generation-time dependencies installed.
    from simulator import generate_probe_stream

    os.makedirs(DATA_DIR, exist_ok=True)
    total = len(runnable) * 2 * N_REPETITIONS
    done = 0

    for tier in runnable:
        model_a, model_b = MODEL_PAIRS[tier]
        for condition in ("substitution", "null"):
            for rep in range(N_REPETITIONS):
                done += 1
                path = os.path.join(DATA_DIR, f"{tier}_{condition}_rep{rep}.jsonl")
                if os.path.exists(path):
                    continue

                print(f"[{done}/{total}] generating {os.path.basename(path)} "
                      f"({TOTAL_REQUESTS} probes)...", flush=True)
                stream = generate_probe_stream(
                    model_a=model_a,
                    model_b=model_b,
                    total_requests=TOTAL_REQUESTS,
                    switch_point=SWITCH_POINT if condition == "substitution" else None,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                failures = sum(1 for r in stream if r["failed"])
                if failures > TOTAL_REQUESTS * 0.1:
                    print(f"  [warn] {failures}/{TOTAL_REQUESTS} probes failed; "
                          f"not writing {os.path.basename(path)}")
                    continue

                tmp = path + ".partial"
                with open(tmp, "w") as f:
                    for r in stream:
                        f.write(json.dumps(r) + "\n")
                os.replace(tmp, path)   # never leave a half-written stream behind
                print(f"  wrote {os.path.basename(path)} ({failures} failed probes)")

    print("\nDone. Next: python audit_data.py && python evaluate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
