"""Compression detectors: the model as a codebook, done prequentially.

An honest endpoint keeps compressing under the codebook you learned from it. A
substituted endpoint stops. The accumulated excess code length is the evidence.

ON THE CONSTRUCTION, because two plausible-looking versions of this are wrong.

Wrong version 1 -- excess surprisal. The statistic -log p(x) - H(p) is mean-zero
under the null but exp of it has expectation K*exp(-H) >> 1, so Ville does not
apply and log(1/alpha) as a threshold buys nothing.

Wrong version 2 -- frozen null vs adaptive alternative. Freeze p_0 on a warmup
window, run an adaptive KT codebook against it, accumulate log q/p_0. This looks
like a likelihood ratio but is not a martingale, because E[q/p_0] = 1 requires
p_0 to be the TRUE null law rather than a 40-sample estimate of it. The adaptive
codebook compresses better purely by having seen more data, so the statistic
drifts up on perfectly clean streams: measured +92.9 nats over one easy-tier
null stream, and a 100% false-alarm rate at every alpha from 1e-1 to 1e-6.

What is actually correct is to make BOTH codes prequential:

    null        q_all -- one KT codebook fed every observation from t=0
    alternative q_v   -- a KT codebook restarted at candidate change point v

    logLR(v, t) = sum_{i=v..t} [ log q_v(x_i | x_v..x_{i-1})
                                 - log q_all(x_i | x_0..x_{i-1}) ]

Each prequential code defines a genuine joint probability distribution over
sequences, so for any two of them E_P[Q/P] = 1 exactly, and Ville's inequality
applies to the ratio. The null being tested is "one stationary categorical law
generated this whole stream", which is precisely the hypothesis we care about,
and it needs no oracle knowledge of what that law is. Under the null the
restarted code has strictly less data, so the statistic drifts negative -- the
model-complexity penalty MDL is supposed to charge, and it makes the test
conservative rather than anti-conservative.

Detectors here:

  eprocess_detector      mixture over the unknown change point (Shiryaev-Roberts
                         form): R_t = sum_v exp(logLR(v,t)). Under H0 each term
                         has expectation 1 so E[R_t] = t, and thresholding at
                         t/alpha is anytime-valid. Nothing to tune but alpha.
  mdl_cusum_detector     CUSUM form, restarting the alternative when the
                         statistic returns to zero. Cheaper and rearms cleanly;
                         false-alarm control is empirical, not Ville.
  mixture_alternative_detector
                         alternative is (1-r)*null + r*something-else, for
                         partial routing, where a clean-swap alternative is
                         testing the wrong hypothesis.

conditional=True keeps one codebook per probe template, so template rotation
does not read as a distribution shift.
"""

import numpy as np
from collections import Counter

from stats_categorical import K

KT_ALPHA = 0.5          # Krichevsky-Trofimov add-1/2
DEFAULT_WINDOW = 200    # how far back candidate change points are kept


class _Codebook:
    """KT predictive PMF over the fixed 1..100 support, optionally per context."""

    __slots__ = ("kt_alpha", "counts", "totals")

    def __init__(self, kt_alpha=KT_ALPHA):
        self.kt_alpha = kt_alpha
        self.counts = {}
        self.totals = {}

    def prob(self, value, ctx="_"):
        c = self.counts.get(ctx)
        n = self.totals.get(ctx, 0)
        hit = c[value] if c else 0
        return (hit + self.kt_alpha) / (n + self.kt_alpha * K)

    def observe(self, value, ctx="_"):
        d = self.counts.get(ctx)
        if d is None:
            d = self.counts[ctx] = Counter()
        d[value] += 1
        self.totals[ctx] = self.totals.get(ctx, 0) + 1


def _contexts(values, contexts, conditional):
    if conditional and contexts is not None:
        return list(contexts)
    return ["_"] * len(values)


def eprocess_detector(values, contexts=None, warmup=40, alpha=1e-4,
                      conditional=False, window=DEFAULT_WINDOW):
    """Shiryaev-Roberts mixture over change points. Anytime-valid via Ville."""
    if len(values) <= warmup:
        return []
    ctxs = _contexts(values, contexts, conditional)

    null = _Codebook()
    for i in range(warmup):
        null.observe(values[i], ctxs[i])

    cands = []      # (codebook, cumulative logLR)
    out = []
    for t in range(warmup, len(values)):
        x, ctx = values[t], ctxs[t]
        log_null = np.log(null.prob(x, ctx))

        cands.append((_Codebook(), 0.0))
        updated = []
        for book, acc in cands:
            acc += np.log(book.prob(x, ctx)) - log_null
            book.observe(x, ctx)
            updated.append((book, acc))
        cands = updated[-window:]

        null.observe(x, ctx)

        accs = np.array([a for _, a in cands])
        m = accs.max()
        R = float(np.exp(m) * np.exp(accs - m).sum()) if np.isfinite(m) else 0.0

        n_seen = t - warmup + 1
        flagged = R > n_seen / alpha
        out.append({"index": t, "stat": R, "log_stat": float(m), "flagged": flagged})
        if flagged:
            cands = []
    return out


def mdl_cusum_detector(values, contexts=None, warmup=40, alpha=1e-4,
                       conditional=False):
    """CUSUM on the prequential log-ratio. Rearms; guarantee is empirical."""
    if len(values) <= warmup:
        return []
    ctxs = _contexts(values, contexts, conditional)

    null = _Codebook()
    for i in range(warmup):
        null.observe(values[i], ctxs[i])

    h = np.log(1.0 / alpha)
    alt = _Codebook()
    S = 0.0
    out = []
    for t in range(warmup, len(values)):
        x, ctx = values[t], ctxs[t]
        inc = np.log(alt.prob(x, ctx)) - np.log(null.prob(x, ctx))
        alt.observe(x, ctx)
        null.observe(x, ctx)
        S = max(0.0, S + inc)
        flagged = S > h
        out.append({"index": t, "stat": S, "flagged": flagged})
        if flagged:
            S = 0.0
            alt = _Codebook()
    return out


def mixture_alternative_detector(values, contexts=None, warmup=40, alpha=1e-4,
                                 route_grid=(0.05, 0.1, 0.25, 0.5, 1.0),
                                 conditional=False):
    """Partial routing: the alternative is a mixture, not a clean swap.

    Under routing at rate r the post-change law is (1-r)*p_null + r*p_other, so
    a detector whose alternative says "everything changed" is testing the wrong
    thing and bleeds power as r falls. One CUSUM per candidate r, max taken,
    with a Bonferroni factor over the grid.
    """
    if len(values) <= warmup:
        return []
    ctxs = _contexts(values, contexts, conditional)

    null = _Codebook()
    for i in range(warmup):
        null.observe(values[i], ctxs[i])

    h = np.log(len(route_grid) / alpha)
    books = {r: _Codebook() for r in route_grid}
    accs = {r: 0.0 for r in route_grid}
    out = []
    for t in range(warmup, len(values)):
        x, ctx = values[t], ctxs[t]
        p_null = null.prob(x, ctx)
        best = 0.0
        for r in route_grid:
            mixed = (1.0 - r) * p_null + r * books[r].prob(x, ctx)
            accs[r] = max(0.0, accs[r] + np.log(mixed) - np.log(p_null))
            books[r].observe(x, ctx)
            best = max(best, accs[r])
        null.observe(x, ctx)
        flagged = best > h
        out.append({"index": t, "stat": best, "flagged": flagged})
        if flagged:
            for r in route_grid:
                accs[r] = 0.0
                books[r] = _Codebook()
    return out
