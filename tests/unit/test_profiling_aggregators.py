"""
Unit tests for profiling aggregators.
"""

import pytest
import numpy as np
from layerlens.profiling.aggregators import aggregate_scores


def test_aggregate_scores_basic():
    """Test basic score aggregation."""
    metrics = {
        "gradient_energy": 0.5,
        "fisher": 0.3,
        "proxy_eval": 0.2,
    }
    weights = {
        "gradient_energy": 0.4,
        "fisher": 0.4,
        "proxy_eval": 0.2,
    }
    
    result = aggregate_scores(metrics, weights)
    expected = 0.5 * 0.4 + 0.3 * 0.4 + 0.2 * 0.2
    assert abs(result - expected) < 1e-6


def test_aggregate_scores_missing_weights():
    """Test aggregation with missing weights (should use equal weights)."""
    metrics = {
        "gradient_energy": 0.5,
        "fisher": 0.3,
    }
    weights = {}  # Empty weights
    
    result = aggregate_scores(metrics, weights)
    expected = (0.5 + 0.3) / 2.0
    assert abs(result - expected) < 1e-6


def test_aggregate_scores_partial_weights():
    """Test aggregation with partial weights."""
    metrics = {
        "gradient_energy": 0.5,
        "fisher": 0.3,
        "proxy_eval": 0.2,
    }
    weights = {
        "gradient_energy": 1.0,  # Only one weight provided
    }
    
    result = aggregate_scores(metrics, weights)
    # Should normalize to use only available weights
    assert result > 0
    assert result <= 1.0

