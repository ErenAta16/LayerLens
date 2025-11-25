"""
aggregators.py
--------------
Helper module separated to enable moving the score aggregation function
to the Cython side.
"""

from typing import Dict

try:
    import numpy as np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - fallback to pure Python if NumPy is not available
    np = None  # type: ignore[assignment]

try:
    from ._aggregators import aggregate_scores_c  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - fallback if not compiled
    aggregate_scores_c = None


def aggregate_scores(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Combines scores from different analyses using weighted average.
    Calls Cython implementation if available, otherwise falls back to Python code.
    """

    keys = list(metrics.keys())

    if np is not None:
        metric_arr = np.array([metrics[k] for k in keys], dtype=np.float64)
        weight_arr = np.array([weights.get(k, 0.0) for k in keys], dtype=np.float64)

        if aggregate_scores_c is not None:
            return float(aggregate_scores_c(metric_arr, weight_arr))

        total_weight = weight_arr.sum() or 1.0
        weighted_sum = float(np.dot(metric_arr, weight_arr))
        return weighted_sum / total_weight

    # Pure Python computation if NumPy is not available
    total_weight = 0.0
    weighted_sum = 0.0
    for key in keys:
        weight = weights.get(key, 0.0)
        total_weight += weight
        weighted_sum += metrics[key] * weight

    return weighted_sum / (total_weight or 1.0)

