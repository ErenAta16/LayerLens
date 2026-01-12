"""
config.py
----------
Contains configuration data classes used throughout the project.
Purpose is to represent shared parameters in profiling and optimization stages
in a modular and type-safe manner.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ProfilingConfig:
    """
    Parameters for layer sensitivity analyses.
    This class carries configuration that will be transferred to Cython modules in the future.
    """

    calibration_batch_size: int = 64
    gradient_window: int = 128
    use_proxy_finetune: bool = True
    proxy_steps: int = 5
    fisher_trace_samples: int = 8
    metrics: List[str] = field(
        default_factory=lambda: ["gradient_energy", "fisher", "proxy_eval"]
    )
    metric_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "gradient_energy": 0.4,
            "fisher": 0.4,
            "proxy_eval": 0.2,
        }
    )

    def __post_init__(self) -> None:
        """
        Validates config values.
        """
        # Fisher trace samples: minimum 1, maximum 1000 (for performance)
        if self.fisher_trace_samples < 1:
            raise ValueError(
                f"fisher_trace_samples must be at least 1, got {self.fisher_trace_samples}"
            )
        if self.fisher_trace_samples > 1000:
            raise ValueError(
                f"fisher_trace_samples must be at most 1000, got {self.fisher_trace_samples}"
            )

        # Calibration batch size: must be positive
        if self.calibration_batch_size <= 0:
            raise ValueError(
                f"calibration_batch_size must be positive, got {self.calibration_batch_size}"
            )

        # Gradient window: must be positive
        if self.gradient_window <= 0:
            raise ValueError(
                f"gradient_window must be positive, got {self.gradient_window}"
            )

        # Proxy steps: must be positive
        if self.proxy_steps <= 0:
            raise ValueError(
                f"proxy_steps must be positive, got {self.proxy_steps}"
            )


@dataclass
class LatencyProfile:
    """
    Describes a more realistic latency model for a given deployment setting.

    The idea is to capture the dominant factors that affect per-layer latency so
    that the optimizer can reason about the latency budget in milliseconds,
    instead of using a fixed `rank * const` heuristic.
    """

    # High-level characteristics
    device_type: str = "gpu"  # "gpu" or "cpu"
    model_family: str = "llm"  # e.g. "llm", "yolo", "vit"

    # Workload characteristics
    batch_size: int = 1
    sequence_length: int = 2048  # for LLMs
    input_resolution: int = 640  # for vision models (e.g. YOLO, ViT), shortest side

    # Base latency behaviour (all in milliseconds)
    base_ms_per_layer: float = 0.5
    ms_per_rank_unit: float = 0.02
    io_overhead_ms: float = 0.0


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

