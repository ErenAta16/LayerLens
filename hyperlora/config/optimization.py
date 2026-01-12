"""
optimization.py
---------------
Contains OptimizationConfig data class for optimization constraints and objectives.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .latency import LatencyProfile


@dataclass
class OptimizationConfig:
    """
    Defines constraints and objectives of the rank/method selection problem.
    """

    max_trainable_params: int
    max_flops: float
    max_vram_gb: float
    latency_target_ms: float
    candidate_methods: List[str] = field(
        default_factory=lambda: ["lora", "adapter", "prefix", "none"]
    )
    method_penalties: Dict[str, float] = field(
        default_factory=lambda: {"lora": 1.0, "adapter": 1.2, "prefix": 1.1, "none": 0.5}
    )
    objective_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "utility": 0.5,
            "cost": 0.2,
            "flop": 0.1,
            "vram": 0.1,
            "latency": 0.1,
        }
    )
    # Optional detailed latency profile. When provided, the solver will use it
    # to approximate per-layer latency instead of a naive rank * constant model.
    latency_profile: Optional[LatencyProfile] = None
    randomness_seed: Optional[int] = None

