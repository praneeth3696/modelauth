import json
import os
import re
from collections import Counter

from config import VALID_RANGE

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Parse outcomes. Only OK values enter a detector stream; the rest are counted
# so that silent data loss shows up in the audit instead of in the results.
OK = "ok"
OUT_OF_RANGE = "out_of_range"
UNPARSEABLE = "unparseable"
FAILED = "failed"


def resolve(filepath):
    """Accept either an absolute path or one relative to substitution-sim/data."""
    if os.path.isabs(filepath) or os.path.exists(filepath):
        return filepath
    return os.path.join(DATA_DIR, os.path.basename(filepath))


def load_stream(filepath):
    records = []
    with open(resolve(filepath)) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_numeric_answer(answer_text):
    """Return (value, status). Value is None unless status is OK.

    The old version returned int(re.search(r'\\d+', text)) with no range check,
    so a max_tokens truncation like "854" (a fragment of a longer answer) was
    accepted as a legitimate draw and then standardized against.
    """
    if answer_text is None:
        return None, FAILED
    match = re.search(r"\d+", answer_text)
    if not match:
        return None, UNPARSEABLE
    value = int(match.group())
    lo, hi = VALID_RANGE
    if not (lo <= value <= hi):
        return None, OUT_OF_RANGE
    return value, OK


def load_numeric_stream(filepath, verbose=False):
    """Load a stream, annotating each record with numeric_answer and parse_status."""
    records = load_stream(filepath)
    for r in records:
        value, status = parse_numeric_answer(r.get("answer"))
        r["numeric_answer"] = value
        r["parse_status"] = status
    valid = [r for r in records if r["parse_status"] == OK]
    if verbose:
        counts = Counter(r["parse_status"] for r in records)
        dropped = len(records) - len(valid)
        if dropped:
            print(f"[audit] {os.path.basename(filepath)}: kept {len(valid)}/{len(records)} "
                  f"({dict(counts)})")
    return valid


def load_values(filepath):
    """Just the usable integers, in order."""
    return [r["numeric_answer"] for r in load_numeric_stream(filepath)]


def audit_file(filepath):
    """Per-file parse breakdown, for the completeness report."""
    records = load_stream(filepath)
    counts = Counter()
    offenders = Counter()
    for r in records:
        _, status = parse_numeric_answer(r.get("answer"))
        counts[status] += 1
        if status in (OUT_OF_RANGE, UNPARSEABLE):
            offenders[repr(r.get("answer"))] += 1
    n = len(records)
    return {
        "file": os.path.basename(filepath),
        "n_total": n,
        "n_ok": counts[OK],
        "n_out_of_range": counts[OUT_OF_RANGE],
        "n_unparseable": counts[UNPARSEABLE],
        "n_failed": counts[FAILED],
        "pct_usable": round(100 * counts[OK] / n, 2) if n else 0.0,
        "offenders": offenders,
    }


def split_by_model(filepath):
    """Return {model_name: [values]} using the recorded ground truth.

    Used to estimate per-model PMFs from held-out streams for the oracle detector.
    """
    out = {}
    for r in load_numeric_stream(filepath):
        out.setdefault(r["true_model"], []).append(r["numeric_answer"])
    return out
