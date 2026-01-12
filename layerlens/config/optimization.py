"""
optimization.py
---------------
Contains OptimizationConfig data class for optimization constraints and objectives.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .latency import LatencyProfile
from ..exceptions import ConfigurationError


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
    
    def __post_init__(self) -> None:
        """
        Validates configuration values.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Validate constraints
        if self.max_trainable_params <= 0:
            raise ConfigurationError(
                f"max_trainable_params must be positive, got {self.max_trainable_params}"
            )
        
        if self.max_flops <= 0:
            raise ConfigurationError(
                f"max_flops must be positive, got {self.max_flops}"
            )
        
        if self.max_vram_gb <= 0:
            raise ConfigurationError(
                f"max_vram_gb must be positive, got {self.max_vram_gb}"
            )
        
        if self.latency_target_ms <= 0:
            raise ConfigurationError(
                f"latency_target_ms must be positive, got {self.latency_target_ms}"
            )
        
        # Validate candidate methods
        if not self.candidate_methods:
            raise ConfigurationError("candidate_methods cannot be empty")
        
        valid_methods = {"lora", "adapter", "prefix", "none"}
        invalid_methods = set(self.candidate_methods) - valid_methods
        if invalid_methods:
            raise ConfigurationError(
                f"invalid candidate_methods: {sorted(invalid_methods)}. "
                f"Valid methods: {sorted(valid_methods)}"
            )
        
        # Validate objective weights sum to reasonable range (0.5-2.0)
        total_weight = sum(self.objective_weights.values())
        if total_weight <= 0:
            raise ConfigurationError(
                f"objective_weights must have positive total, got {total_weight}"
            )

