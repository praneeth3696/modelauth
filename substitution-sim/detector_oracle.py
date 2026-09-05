"""Oracle likelihood-ratio CUSUM -- the upper bound on achievable performance.

This detector cheats: it is handed both the pre-change PMF and the post-change
PMF, estimated from held-out streams. No deployable monitor knows the second
one. It exists so that every real detector can be reported as a fraction of what
was actually attainable on this channel, rather than only against its neighbours
in the table.

Moustakides (1986) showed the LR-CUSUM below is exactly optimal under Lorden's
(1971) criterion, so its measured delay should land close to lorden_bound().
"""

import numpy as np

from stats_categorical import _INDEX, estimate_pmf


def _logratio_table(p_b, p_a):
    return np.log(p_b / p_a)


def oracle_lr_cusum(values, p_a, p_b, alpha=1e-4):
    """Page's CUSUM on the exact log-likelihood ratio.

    S_t = max(0, S_{t-1} + log(p_b(x_t) / p_a(x_t))), flag when S_t > log(1/alpha).

    The threshold is anytime-valid: exp(S_t) is a non-negative supermartingale
    under the null, so Ville's inequality bounds the false-alarm probability over
    the entire run by alpha. There is nothing to tune.
    """
    h = np.log(1.0 / alpha)
    llr = _logratio_table(np.asarray(p_b, float), np.asarray(p_a, float))
    S = 0.0
    results = []
    for t, x in enumerate(values):
        idx = _INDEX.get(x)
        if idx is None:          # outside 1..100 should never reach here
            results.append({"index": t, "stat": S, "flagged": False})
            continue
        S = max(0.0, S + float(llr[idx]))
        flagged = S > h
        results.append({"index": t, "stat": S, "flagged": flagged})
        if flagged:
            S = 0.0
    return results


def fit_oracle_pmfs(held_out_null_file, held_out_sub_file, switch_point):
    """Estimate p_a and p_b from streams excluded from the test set.

    p_a comes from a clean null stream; p_b from the post-switch tail of a
    substitution stream. Both files must be held out of evaluation or the oracle
    is training on its own test data.
    """
    from data_loader import load_numeric_stream

    a_vals = [r["numeric_answer"] for r in load_numeric_stream(held_out_null_file)]

    sub = load_numeric_stream(held_out_sub_file)
    b_vals = [r["numeric_answer"] for r in sub if r["index"] >= switch_point]

    return estimate_pmf(a_vals), estimate_pmf(b_vals), len(a_vals), len(b_vals)
