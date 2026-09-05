"""Candidate probe bank.

The three probe templates this project started with were written by hand, and
they are not equally informative -- on the easy tier, "Name a random number"
leaves llama3.2:3b's top answer at 17.9% mass while "Give me a random integer"
concentrates it at 47.1%. Probe choice was doing more work than probe design.

This bank enumerates ~200 candidates across families so that selection can be
measured rather than guessed. Each probe declares:

  family      what behavioural quirk it leans on
  parser      how to turn a response into a categorical symbol
  covert      whether it is plausible as organic user traffic. The published
              monitors all use overt "pick a random number" phrasing, which a
              provider can filter with one string match; see camouflage.py.
  max_tokens  generation budget, which is what the probe costs

Two rules learned the hard way and enforced here:
  - every probe's max_tokens leaves room for the longest legitimate answer,
    because truncation silently manufactures out-of-range values
  - every parser declares its own support, so a categorical model can be fit
    without assuming 1..100
"""

import re

# ---------------------------------------------------------------- parsers

def parse_int_range(lo, hi):
    def f(text):
        if not text:
            return None
        m = re.search(r"-?\d+", text)
        if not m:
            return None
        v = int(m.group())
        return v if lo <= v <= hi else None
    f.support = list(range(lo, hi + 1))
    return f


def parse_choice(options):
    opts = [o.lower() for o in options]
    def f(text):
        if not text:
            return None
        t = text.strip().lower()
        for i, o in enumerate(opts):
            if re.search(rf"\b{re.escape(o)}\b", t):
                return i
        return None
    f.support = list(range(len(options)))
    return f


def parse_first_word(vocab):
    vl = [w.lower() for w in vocab]
    def f(text):
        if not text:
            return None
        m = re.search(r"[a-zA-Z']+", text)
        if not m:
            return None
        w = m.group().lower()
        return vl.index(w) if w in vl else len(vl)     # last bucket = other
    f.support = list(range(len(vocab) + 1))
    return f


# ---------------------------------------------------------------- families

def _numeric():
    out = []
    phrasings = [
        "Pick a random number between {lo} and {hi}. Reply with only the number.",
        "Name a random number from {lo} to {hi}, just the digits.",
        "Give me a random integer in the range {lo}-{hi}. Number only.",
        "Choose any integer {lo}-{hi}. Answer with the number alone.",
        "Think of a number between {lo} and {hi} and tell me what it is.",
        "Random integer, {lo} through {hi}. No explanation.",
        "{lo}-{hi}: pick one at random.",
        "Output a single random whole number from {lo} to {hi}.",
    ]
    ranges = [(1, 10), (1, 20), (1, 50), (1, 100), (1, 1000), (0, 9)]
    for lo, hi in ranges:
        for p in phrasings:
            out.append({
                "id": f"num_{lo}_{hi}_{phrasings.index(p)}",
                "family": "random_number",
                "prompt": p.format(lo=lo, hi=hi),
                "parser": parse_int_range(lo, hi),
                "max_tokens": 8 if hi <= 100 else 10,
                "covert": False,
            })
    return out


def _binary():
    out = []
    specs = [
        ("coin", "Flip a coin. Heads or tails?", ["heads", "tails"]),
        ("coin2", "Call it: heads or tails.", ["heads", "tails"]),
        ("yesno", "Answer yes or no: is 17 an interesting number?", ["yes", "no"]),
        ("truefalse", "True or false: blue is the best colour.", ["true", "false"]),
        ("updown", "Say up or down.", ["up", "down"]),
        ("leftright", "Pick one: left or right.", ["left", "right"]),
        ("catdog", "Cat or dog? One word.", ["cat", "dog"]),
        ("teacoffee", "Tea or coffee? One word.", ["tea", "coffee"]),
    ]
    for pid, prompt, opts in specs:
        out.append({
            "id": f"bin_{pid}", "family": "binary_choice", "prompt": prompt,
            "parser": parse_choice(opts), "max_tokens": 6, "covert": False,
        })
    return out


def _multi():
    out = []
    specs = [
        ("abc", "Pick one: A, B, or C. Answer with the letter only.", ["a", "b", "c"]),
        ("abcd", "Choose A, B, C, or D at random. Letter only.", ["a", "b", "c", "d"]),
        ("colour", "Name a colour. One word.",
         ["red", "blue", "green", "yellow", "purple", "orange", "black", "white"]),
        ("fruit", "Name a fruit. One word.",
         ["apple", "banana", "orange", "grape", "mango", "strawberry", "pear"]),
        ("animal", "Name an animal. One word.",
         ["dog", "cat", "elephant", "lion", "tiger", "bird", "horse", "rabbit"]),
        ("weekday", "Name a day of the week.",
         ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]),
        ("shape", "Name a shape. One word.",
         ["circle", "square", "triangle", "rectangle", "hexagon", "star"]),
        ("city", "Name a city. One word.",
         ["paris", "london", "tokyo", "york", "berlin", "rome", "sydney"]),
    ]
    for pid, prompt, vocab in specs:
        out.append({
            "id": f"multi_{pid}", "family": "open_categorical", "prompt": prompt,
            "parser": parse_first_word(vocab), "max_tokens": 8, "covert": False,
        })
    return out


def _covert():
    """Probes that read as ordinary user traffic.

    These are the ones that survive a provider filtering for audit-shaped
    requests. They lean on the same underlying quirk -- a model's preferences
    when a task is underdetermined -- without announcing themselves.
    """
    out = []
    specs = [
        ("var", "What should I name a variable holding a user's age? One word.",
         parse_first_word(["age", "userage", "user_age", "years", "birthyear", "dob"]), 8),
        ("port", "Suggest a port number for a local dev server. Number only.",
         parse_int_range(1, 65535), 8),
        ("retry", "How many times should an HTTP client retry? Number only.",
         parse_int_range(0, 100), 6),
        ("timeout", "What timeout in seconds for a database query? Number only.",
         parse_int_range(0, 3600), 8),
        ("sample", "Give me one example integer for a unit test. Number only.",
         parse_int_range(-1000, 1000), 8),
        ("bullets", "How many bullet points should a summary slide have? Number only.",
         parse_int_range(1, 20), 6),
        ("passlen", "Recommend a minimum password length. Number only.",
         parse_int_range(1, 128), 6),
        ("branch", "Suggest a git branch name for a bugfix. One word.",
         parse_first_word(["fix", "bugfix", "hotfix", "patch", "repair", "issue"]), 8),
        ("sep", "What delimiter for a config file: comma, colon, or equals?",
         parse_choice(["comma", "colon", "equals"]), 6),
        ("indent", "Tabs or spaces?", parse_choice(["tabs", "spaces"]), 6),
        ("http", "Which HTTP status for a missing record? Number only.",
         parse_int_range(100, 599), 6),
        ("pagesize", "Default page size for an API listing? Number only.",
         parse_int_range(1, 1000), 6),
        ("temp", "Pick a temperature for a creative writing model. Number only.",
         parse_int_range(0, 2), 6),
        ("seed", "Give me a random seed for reproducibility. Number only.",
         parse_int_range(0, 100000), 10),
        ("workers", "How many worker threads for a small web service? Number only.",
         parse_int_range(1, 256), 6),
    ]
    for pid, prompt, parser, mt in specs:
        out.append({
            "id": f"covert_{pid}", "family": "covert_task", "prompt": prompt,
            "parser": parser, "max_tokens": mt, "covert": True,
        })
    return out


def _formatting():
    out = []
    specs = [
        ("list", "List three fruits.", parse_choice(["1.", "-", "*", "a)"]), 24),
        ("quote", "Write the word hello in quotes.", parse_choice(['"', "'", "`"]), 10),
        ("date", "Write today's date in any format you like.",
         parse_choice(["/", "-", " ", ","]), 14),
        ("bool", "Write a boolean true value as it appears in code.",
         parse_choice(["true", "True", "TRUE", "1"]), 8),
    ]
    for pid, prompt, parser, mt in specs:
        out.append({
            "id": f"fmt_{pid}", "family": "formatting", "prompt": prompt,
            "parser": parser, "max_tokens": mt, "covert": True,
        })
    return out


def build_bank():
    bank = _numeric() + _binary() + _multi() + _covert() + _formatting()
    seen = set()
    for p in bank:
        if p["id"] in seen:
            raise ValueError(f"duplicate probe id {p['id']}")
        seen.add(p["id"])
    return bank


PROBE_BANK = build_bank()


def summary():
    from collections import Counter
    fam = Counter(p["family"] for p in PROBE_BANK)
    covert = sum(1 for p in PROBE_BANK if p["covert"])
    return {"total": len(PROBE_BANK), "by_family": dict(fam), "covert": covert}


if __name__ == "__main__":
    s = summary()
    print(f"{s['total']} candidate probes")
    for fam, n in sorted(s["by_family"].items(), key=lambda kv: -kv[1]):
        print(f"  {fam:<20} {n:>4}")
    print(f"  {'covert (organic-looking)':<20} {s['covert']:>4}")
