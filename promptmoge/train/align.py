"""Robust scale / shift alignment utilities (numpy, CPU) used for LS-anchored baselines and scoring."""
from __future__ import annotations

import numpy as np


def fit_scale_shift_l2(pred: np.ndarray, target: np.ndarray):
    """Closed-form LS (a, b) minimising ||a*pred + b - target||^2."""
    pred = pred.astype(np.float64); target = target.astype(np.float64)
    n = pred.size
    sx, sy = pred.sum(), target.sum()
    sxx, sxy = pred @ pred, pred @ target
    den = n * sxx - sx * sx
    if den <= 0 or not np.isfinite(den):
        return 1.0, 0.0
    a = (n * sxy - sx * sy) / den
    return float(a), float((sy - a * sx) / n)


def fit_scale_shift_robust(pred: np.ndarray, target: np.ndarray, iters: int = 5, trim: float = 0.2):
    """Trimmed / IRLS-L1 scale+shift: refit on the (1-trim) fraction of lowest relative residuals.

    Robust to LiDAR outliers (edges, flying pixels) that an L2 fit would chase."""
    a, b = fit_scale_shift_l2(pred, target)
    for _ in range(iters):
        r = np.abs(a * pred + b - target) / np.maximum(target, 1e-3)
        keep = r <= np.quantile(r, 1.0 - trim)
        if keep.sum() < 16:
            break
        a, b = fit_scale_shift_l2(pred[keep], target[keep])
    return a, b


def fit_scale_only_robust(pred: np.ndarray, target: np.ndarray, iters: int = 5, trim: float = 0.2):
    """Median-ratio style robust scale (weighted L1 in log space ~ median of log ratio)."""
    keep = (pred > 0) & (target > 0)
    lr = np.log(target[keep]) - np.log(pred[keep])
    return float(np.exp(np.median(lr)))
