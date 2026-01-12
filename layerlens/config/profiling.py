"""
profiling.py
------------
Contains ProfilingConfig data class for layer sensitivity analysis configuration.
"""

from dataclasses import dataclass, field
from typing import List, Dict

from ..exceptions import ConfigurationError


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
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Fisher trace samples: minimum 1, maximum 1000 (for performance)
        if self.fisher_trace_samples < 1:
            raise ConfigurationError(
                f"fisher_trace_samples must be at least 1, got {self.fisher_trace_samples}"
            )
        if self.fisher_trace_samples > 1000:
            raise ConfigurationError(
                f"fisher_trace_samples must be at most 1000, got {self.fisher_trace_samples}"
            )

        # Calibration batch size: must be positive
        if self.calibration_batch_size <= 0:
            raise ConfigurationError(
                f"calibration_batch_size must be positive, got {self.calibration_batch_size}"
            )

        # Gradient window: must be positive
        if self.gradient_window <= 0:
            raise ConfigurationError(
                f"gradient_window must be positive, got {self.gradient_window}"
            )

        # Proxy steps: must be positive
        if self.proxy_steps <= 0:
            raise ConfigurationError(
                f"proxy_steps must be positive, got {self.proxy_steps}"
            )
        
        # Validate metric weights if provided
        if self.metric_weights:
            total_weight = sum(self.metric_weights.values())
            if total_weight <= 0:
                raise ConfigurationError(
                    f"metric_weights must have positive total, got {total_weight}"
                )

