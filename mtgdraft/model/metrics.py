from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def log_loss_per_game(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss_per_game(y, p).mean())


def auc(y: np.ndarray, p: np.ndarray) -> float:
    r = rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def paired_diff(y, p_a, p_b, groups: np.ndarray) -> tuple[float, float]:
    """Mean log-loss of a minus b, with a standard error clustered by group (draft)."""
    d = log_loss_per_game(y, p_a) - log_loss_per_game(y, p_b)
    _, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=d)
    g = len(sums)
    mean = d.mean()
    resid = sums - mean * np.bincount(inv)
    se = np.sqrt(g / (g - 1) * (resid ** 2).sum()) / len(d)
    return float(mean), float(se)
