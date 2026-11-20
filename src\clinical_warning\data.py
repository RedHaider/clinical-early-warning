from __future__ import annotations

import numpy as np


def make_sequences(n: int = 1600, steps: int = 8, seed: int = 27):
    rng = np.random.default_rng(seed)
    age_group = rng.integers(0, 2, n)
    baseline = rng.normal(size=(n, 4))
    slope = rng.normal(scale=0.18, size=(n, 4))
    time = np.arange(steps)[None, :, None]
    seq = baseline[:, None, :] + time * slope[:, None, :] + rng.normal(scale=0.3, size=(n, steps, 4))
    logit = 1.6 * slope[:, 0] - 1.3 * slope[:, 1] + 1.0 * slope[:, 2] + 0.45 * baseline[:, 3] + 0.3 * age_group - 1.0
    probability = 1.0 / (1.0 + np.exp(-logit))
    label = rng.binomial(1, probability)
    return seq, age_group, label


def features(seq, trend_aware: bool = True):
    last = seq[:, -1, :]
    if not trend_aware:
        return last
    slope = (seq[:, -1, :] - seq[:, 0, :]) / (seq.shape[1] - 1)
    variability = seq.std(axis=1)
    return np.column_stack([last, slope, variability])
