"""Variance-sensitive CUSUM on a chi-square style statistic.

Renamed from detector_das_cusum.py / das_cusum_detector. This is NOT DAS-CUSUM:
DAS-CUSUM (Dynamically Adjusted Sensitivity) modulates the allowance k from a
running drift estimate, which nothing here does. What this implements is a CUSUM
on 0.5*(z^2 - 1), which is mean-zero under the null and grows under either a
mean or a variance shift. Calling it DAS-CUSUM in the write-up was a citation
error; the old name is kept as a deprecated alias at the bottom of this file.
"""

import numpy as np


def variance_cusum_detector(numeric_answers, warmup=40, k=0.5, h=5.0):
    answers = np.array(numeric_answers, dtype=float)
    results = []
    baseline_window = list(answers[:warmup])
    pos_cusum = 0.0

    for t in range(warmup, len(answers)):
        mu = np.mean(baseline_window)
        sigma = np.std(baseline_window) + 1e-6

        z = (answers[t] - mu) / sigma
        symmetric_stat = 0.5 * (z**2 - 1)

        pos_cusum = max(0, pos_cusum + symmetric_stat - k)
        flagged = pos_cusum > h

        results.append({"index": t, "variance_cusum": pos_cusum, "flagged": flagged})

        if flagged:
            pos_cusum = 0.0
            baseline_window = list(answers[max(0, t - warmup):t])
        else:
            baseline_window.append(answers[t])
            if len(baseline_window) > warmup * 2:
                baseline_window.pop(0)

    return results


def das_cusum_detector(*args, **kwargs):
    """Deprecated alias. Use variance_cusum_detector; this is not DAS-CUSUM."""
    import warnings
    warnings.warn(
        "das_cusum_detector is a misnomer and is deprecated; "
        "use variance_cusum_detector",
        DeprecationWarning,
        stacklevel=2,
    )
    return variance_cusum_detector(*args, **kwargs)
