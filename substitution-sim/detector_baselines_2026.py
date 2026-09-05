"""Reimplementations of the two closest 2026 approaches, as named comparators.

These are our reconstructions from the papers' descriptions, not the authors'
code, and they are adapted to this probe channel (single-token integers rather
than embedded free text). Treat the numbers as "this family of method on our
data", not as a reproduction of the published results.

  energy_distance_detector
      after Leshin, Shah, Timmis & Kang, "Behavioral Fingerprints for LLM
      Endpoint Stability and Identity", arXiv:2603.19022 (Mar 2026).
      Periodic fingerprints compared to a self-baselined reference by energy
      distance with a permutation test, aggregated sequentially as e-values.
      Their fingerprints embed free-text responses; ours are integers, so the
      energy distance runs on the raw values.

  js_fingerprint_detector
      after Bruckner, "One Token Is Enough", arXiv:2607.10252 (Jul 2026).
      Jensen-Shannon divergence between a single-token answer PMF and a
      reference fingerprint. Published as one-shot identification; run here in
      a sliding window so it is comparable on a detection-delay axis, which is
      a use the paper does not claim.
"""

import numpy as np

from stats_categorical import estimate_pmf


def _energy_distance(x, y):
    """E-distance between two 1-D samples: 2*E|X-Y| - E|X-X'| - E|Y-Y'|."""
    x = np.asarray(x, float)[:, None]
    y = np.asarray(y, float)[:, None]
    d_xy = np.abs(x - y.T).mean()
    d_xx = np.abs(x - x.T).mean()
    d_yy = np.abs(y - y.T).mean()
    return 2 * d_xy - d_xx - d_yy


def energy_distance_detector(values, warmup=80, block=40, alpha=1e-4,
                             n_perm=199, rng_seed=0):
    """Self-baselined fingerprint comparison with a permutation test.

    Every `block` observations, compare the new block against the frozen
    baseline by energy distance, convert the permutation p-value to an e-value
    (1/p is a valid e-value under the null), and accumulate the product.
    Flags when the running product exceeds 1/alpha.
    """
    if len(values) <= warmup + block:
        return []
    rng = np.random.default_rng(rng_seed)
    baseline = np.asarray(values[:warmup], float)
    h = np.log(1.0 / alpha)

    log_e = 0.0
    out = []
    for end in range(warmup + block, len(values) + 1, block):
        blk = np.asarray(values[end - block:end], float)
        observed = _energy_distance(baseline, blk)

        pooled = np.concatenate([baseline, blk])
        n_b = len(baseline)
        count = 0
        for _ in range(n_perm):
            rng.shuffle(pooled)
            if _energy_distance(pooled[:n_b], pooled[n_b:]) >= observed:
                count += 1
        p = (count + 1) / (n_perm + 1)
        log_e += np.log(1.0 / p)          # 1/p is an e-value under the null
        flagged = log_e > h
        out.append({"index": end, "log_e": log_e, "p_value": p, "flagged": flagged})
        if flagged:
            log_e = 0.0
    return out


def _js_divergence(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        return float(np.sum(a * np.log(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def js_fingerprint_detector(values, warmup=80, window=40, threshold=0.30):
    """Sliding-window Jensen-Shannon divergence against a frozen fingerprint.

    threshold is in nats. Bruckner reports same-model JS around 0.227 and
    different-model around 0.463 on his corpus with base-e logs; 0.30 sits
    between those and is the natural cut. It is a fixed cut, not a calibrated
    error rate -- which is the point of including it as a comparator.
    """
    if len(values) <= warmup + window:
        return []
    reference = estimate_pmf(values[:warmup])
    out = []
    for t in range(warmup + window, len(values) + 1):
        current = estimate_pmf(values[t - window:t])
        js = _js_divergence(current, reference)
        out.append({"index": t, "js": js, "flagged": js > threshold})
    return out
