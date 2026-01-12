"""
Pytest configuration and fixtures for LayerLens tests.
"""

import pytest
import numpy as np
from pathlib import Path

from layerlens.config import ProfilingConfig, OptimizationConfig, LatencyProfile
from layerlens.models import ModelSpec, LayerSpec


@pytest.fixture
def sample_profiling_config():
    """Sample ProfilingConfig for testing."""
    return ProfilingConfig(
        calibration_batch_size=32,
        gradient_window=64,
        fisher_trace_samples=8,
    )


@pytest.fixture
def sample_optimization_config():
    """Sample OptimizationConfig for testing."""
    return OptimizationConfig(
        max_trainable_params=50000,
        max_flops=1e9,
        max_vram_gb=8.0,
        latency_target_ms=100.0,
    )


@pytest.fixture
def sample_llm_latency_profile():
    """Sample LLM LatencyProfile for testing."""
    return LatencyProfile(
        device_type="gpu",
        model_family="llm",
        batch_size=4,
        sequence_length=2048,
        base_ms_per_layer=0.4,
        ms_per_rank_unit=0.015,
        io_overhead_ms=5.0,
    )


@pytest.fixture
def sample_yolo_latency_profile():
    """Sample YOLO LatencyProfile for testing."""
    return LatencyProfile(
        device_type="gpu",
        model_family="yolo",
        batch_size=2,
        input_resolution=640,
        base_ms_per_layer=0.8,
        ms_per_rank_unit=0.025,
        io_overhead_ms=8.0,
    )


@pytest.fixture
def sample_model_spec():
    """Sample ModelSpec for testing."""
    layers = [
        LayerSpec(
            name=f"layer.{i}",
            hidden_size=768,
            layer_type="transformer",
            supports_attention=True,
            metadata={"layer_index": i}
        )
        for i in range(12)
    ]
    return ModelSpec(
        model_name="test-model",
        total_params=110_000_000,
        layers=layers
    )


@pytest.fixture
def sample_activation_cache():
    """Sample activation cache for testing."""
    return {
        f"layer.{i}": {
            "grad_norm": 0.5 + np.random.random() * 0.3,
            "fisher_trace": 0.3 + np.random.random() * 0.2,
            "proxy_gain": 0.1 + np.random.random() * 0.2,
        }
        for i in range(12)
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory for tests."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir

