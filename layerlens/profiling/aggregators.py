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
    
    If weights are empty or all zero, uses equal weights for all metrics.
    """

    keys = list(metrics.keys())
    if not keys:
        return 0.0

    if np is not None:
        # PERFORMANCE FIX: Use np.fromiter for better memory efficiency
        # Pre-allocate arrays to avoid intermediate list creation
        metric_arr = np.fromiter((metrics[k] for k in keys), dtype=np.float64, count=len(keys))
        weight_arr = np.fromiter((weights.get(k, 0.0) for k in keys), dtype=np.float64, count=len(keys))

        if aggregate_scores_c is not None:
            return float(aggregate_scores_c(metric_arr, weight_arr))

        total_weight = weight_arr.sum()
        # If no weights provided or all zero, use equal weights
        if total_weight == 0.0:
            return float(metric_arr.mean())
        
        weighted_sum = float(np.dot(metric_arr, weight_arr))
        return weighted_sum / total_weight

    # Pure Python computation if NumPy is not available
    total_weight = 0.0
    weighted_sum = 0.0
    for key in keys:
        weight = weights.get(key, 0.0)
        total_weight += weight
        weighted_sum += metrics[key] * weight

    # If no weights provided or all zero, use equal weights
    if total_weight == 0.0:
        return sum(metrics.values()) / len(metrics)

    return weighted_sum / total_weight

