"""
latency.py
----------
Contains LatencyProfile data class for realistic latency modeling.
"""

from dataclasses import dataclass


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

