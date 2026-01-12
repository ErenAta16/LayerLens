"""
feature_engineering.py
----------------------
Helpers that compute derived features per layer.
This layer provides additional signals to optimization and profiling modules.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

from ..models import LayerSpec


@dataclass
class LayerFeatures:
    """
    Attributes that can be used during modeling.
    - param_share: Ratio of layer parameters to total model
    - attention_flag: Whether it's an attention layer
    - scaled_sensitivity: Combination of profile scores (e.g., utility)
    """

    param_share: float
    attention_flag: float
    scaled_sensitivity: float

    def to_dict(self) -> Dict[str, float]:
        """Helper for JSON writing."""

        return asdict(self)


def compute_layer_features(
    layer: LayerSpec,
    total_params: int,
    sensitivity: float,
) -> LayerFeatures:
    """
    Generates layer feature vector.

    Args:
        layer: Layer definition.
        total_params: Total number of model parameters.
        sensitivity: Computed utility or score.
    """

    param_share = layer.hidden_size / max(total_params, 1)
    attention_flag = 1.0 if layer.supports_attention else 0.0
    scaled_sensitivity = min(max(sensitivity, 0.0), 1.0)

    return LayerFeatures(
        param_share=param_share,
        attention_flag=attention_flag,
        scaled_sensitivity=scaled_sensitivity,
    )

