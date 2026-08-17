"""Cluster bootstrap over scenario templates (the unit of randomization).

Examples from the same template are NOT independent; all resampling is done at
the template level. Returns percentile 95% CIs by default.
"""
from __future__ import annotations

import numpy as np


def cluster_bootstrap_auc(score: np.ndarray, y: np.ndarray, template_ids: np.ndarray,
                          n_reps: int = 1000, seed: int = 0):
    """Bootstrap CI for AUROC with clusters = templates.

    score: predicted score per example; y: 0/1 labels; template_ids: cluster id.
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    clusters = np.unique(template_ids)
    n = len(clusters)
    est = roc_auc_score(y, score)
    reps = []
    for _ in range(n_reps):
        pick = rng.integers(0, n, size=n)
        idx = np.concatenate([np.where(template_ids == clusters[c])[0] for c in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        reps.append(roc_auc_score(y[idx], score[idx]))
    reps = np.array(reps)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(lo), float(hi), float(est)


def cluster_bootstrap_mean_delta(values: np.ndarray, template_ids: np.ndarray,
                                 n_reps: int = 1000, seed: int = 0):
    """Bootstrap CI for the mean of a per-example delta, clustered by template."""
    rng = np.random.default_rng(seed)
    clusters = np.unique(template_ids)
    n = len(clusters)
    est = float(values.mean())
    reps = []
    for _ in range(n_reps):
        pick = rng.integers(0, n, size=n)
        idx = np.concatenate([np.where(template_ids == clusters[c])[0] for c in pick])
        reps.append(values[idx].mean())
    reps = np.array(reps)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(lo), float(hi), est
