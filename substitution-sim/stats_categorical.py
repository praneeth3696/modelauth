"""Categorical treatment of the probe channel.

The probe responses are not draws from a distribution on the real line. They are
draws from a sharply peaked categorical distribution over a few dozen tokens:
llama3.2:3b answers 53 about a third of the time, qwen2.5:3b answers 42 about a
third of the time, and the numeric ordering between those two carries nothing.
Everything here works on the PMF directly so the divergences and the optimality
bound are the real ones rather than mean/variance proxies.
"""

import numpy as np
from collections import Counter

from config import VALID_RANGE

LO, HI = VALID_RANGE
SUPPORT = list(range(LO, HI + 1))
K = len(SUPPORT)
_INDEX = {v: i for i, v in enumerate(SUPPORT)}


def estimate_pmf(values, alpha=0.5):
    """Add-alpha (Krichevsky-Trofimov at alpha=0.5) PMF over the full 1..100 support.

    Smoothing over the whole support, not just observed values, keeps the
    log-ratio finite when the substituted model emits a token the baseline never
    produced -- which is exactly the high-evidence event we want to reward.
    """
    counts = Counter(values)
    c = np.array([counts.get(v, 0) for v in SUPPORT], dtype=float)
    return (c + alpha) / (c.sum() + alpha * K)


def kl_divergence(p, q):
    """KL(p || q) in nats."""
    return float(np.sum(p * np.log(p / q)))


def total_variation(p, q):
    return float(0.5 * np.sum(np.abs(p - q)))


def entropy(p):
    return float(-np.sum(p * np.log(p)))


def bayes_error(p, q):
    """Optimal single-probe misclassification rate between the two models."""
    return float((1.0 - total_variation(p, q)) / 2.0)


def lorden_bound(p_b, p_a, alpha):
    """Lower bound on expected detection delay for ANY sequential detector.

    Lorden (1971): the minimax expected delay of a change from p_a to p_b is
    ~ h / KL(p_b || p_a), where an anytime-valid threshold h = log(1/alpha)
    buys a false-alarm probability of at most alpha over the whole run.

    Returns delay in probes. This is the number every detector gets scored
    against; "% of optimal" below 100 would mean a bug, not a breakthrough.
    """
    kl = kl_divergence(p_b, p_a)
    if kl <= 0:
        return float("inf")
    return float(np.log(1.0 / alpha) / kl)


def channel_report(values_a, values_b, alpha=1e-4):
    """Everything you need to say how hard a given tier actually is."""
    p_a = estimate_pmf(values_a)
    p_b = estimate_pmf(values_b)
    return {
        "n_a": len(values_a),
        "n_b": len(values_b),
        "entropy_a_bits": entropy(p_a) / np.log(2),
        "entropy_b_bits": entropy(p_b) / np.log(2),
        "kl_b_given_a": kl_divergence(p_b, p_a),
        "kl_a_given_b": kl_divergence(p_a, p_b),
        "total_variation": total_variation(p_a, p_b),
        "bayes_error": bayes_error(p_a, p_b),
        "lorden_bound_probes": lorden_bound(p_b, p_a, alpha),
        "alpha": alpha,
    }
