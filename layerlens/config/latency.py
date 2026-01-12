"""
latency.py
----------
Contains LatencyProfile data class for realistic latency modeling.
"""

from dataclasses import dataclass

from ..exceptions import ConfigurationError


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
    
    def __post_init__(self) -> None:
        """
        Validates latency profile values.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Validate device type
        valid_devices = {"gpu", "cpu"}
        if self.device_type.lower() not in valid_devices:
            raise ConfigurationError(
                f"device_type must be one of {valid_devices}, got {self.device_type}"
            )
        
        # Validate model family
        valid_families = {"llm", "yolo", "vit", "vision"}
        if self.model_family.lower() not in valid_families:
            # Allow but warn about unknown families
            pass
        
        # Validate numeric values
        if self.batch_size <= 0:
            raise ConfigurationError(
                f"batch_size must be positive, got {self.batch_size}"
            )
        
        if self.sequence_length <= 0:
            raise ConfigurationError(
                f"sequence_length must be positive, got {self.sequence_length}"
            )
        
        if self.input_resolution <= 0:
            raise ConfigurationError(
                f"input_resolution must be positive, got {self.input_resolution}"
            )
        
        if self.base_ms_per_layer < 0:
            raise ConfigurationError(
                f"base_ms_per_layer must be non-negative, got {self.base_ms_per_layer}"
            )
        
        if self.ms_per_rank_unit < 0:
            raise ConfigurationError(
                f"ms_per_rank_unit must be non-negative, got {self.ms_per_rank_unit}"
            )
        
        if self.io_overhead_ms < 0:
            raise ConfigurationError(
                f"io_overhead_ms must be non-negative, got {self.io_overhead_ms}"
            )

