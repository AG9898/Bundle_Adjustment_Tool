"""Robust residual costs and their iteratively reweighted least-squares weights."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .models import LossName


def robust_cost_and_weights(
    residuals: npt.NDArray[np.float64], loss: LossName, huber_delta: float
) -> tuple[float, npt.NDArray[np.float64]]:
    """Return sum cost and one IRLS weight per 2D observation residual."""
    if residuals.ndim != 2 or residuals.shape[1] != 2:
        raise ValueError("residuals must have shape (N, 2)")
    squared_norms = np.sum(residuals * residuals, axis=1)
    if loss == "squared":
        return float(0.5 * np.sum(squared_norms)), np.ones(len(residuals), dtype=np.float64)
    norms = np.sqrt(squared_norms)
    inlier = norms <= huber_delta
    cost = np.where(inlier, 0.5 * squared_norms, huber_delta * (norms - 0.5 * huber_delta))
    weights = np.ones_like(norms)
    nonzero_outlier = ~inlier & (norms > 0.0)
    weights[nonzero_outlier] = huber_delta / norms[nonzero_outlier]
    return float(np.sum(cost)), weights
