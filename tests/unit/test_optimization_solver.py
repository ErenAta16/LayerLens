"""
Unit tests for optimization solver.
"""

import pytest
from hyperlora.optimization.solver import AllocationSolver, AllocationResult
from hyperlora.config import OptimizationConfig, ProfilingConfig
from hyperlora.meta import LayerSpec


@pytest.fixture
def solver_config():
    return OptimizationConfig(
        max_trainable_params=50000,
        max_flops=1e9,
        max_vram_gb=8.0,
        latency_target_ms=100.0,
    )


@pytest.fixture
def solver(solver_config):
    return AllocationSolver(solver_config)


def test_solver_initialization(solver_config):
    """Test solver initialization."""
    solver = AllocationSolver(solver_config)
    assert solver.config == solver_config


def test_solver_method_selection(solver):
    """Test method selection based on utility."""
    # Low utility should select "none" or lower priority method
    method = solver._select_method(0.01)
    assert method in ["lora", "adapter", "prefix", "none"]
    
    # Higher utility should select higher priority method
    method_high = solver._select_method(0.1)
    assert method_high in ["lora", "adapter", "prefix", "none"]


def test_solver_rank_estimation(solver):
    """Test rank estimation."""
    layer = LayerSpec(
        name="test_layer",
        hidden_size=768,
        layer_type="transformer",
    )
    
    rank = solver._estimate_rank(layer, 0.05)
    assert rank >= 1.0
    assert rank <= layer.hidden_size / 10


def test_solver_latency_estimation(solver):
    """Test latency estimation."""
    # Without latency profile, should use simple heuristic
    latency = solver._estimate_latency(10.0)
    assert latency == 10.0 * 0.01  # rank * 0.01


def test_solver_latency_estimation_with_profile(solver_config):
    """Test latency estimation with LatencyProfile."""
    from hyperlora.config import LatencyProfile
    
    latency_profile = LatencyProfile(
        device_type="gpu",
        model_family="llm",
        base_ms_per_layer=0.4,
        ms_per_rank_unit=0.015,
    )
    solver_config.latency_profile = latency_profile
    solver = AllocationSolver(solver_config)
    
    latency = solver._estimate_latency(10.0)
    # Should use profile-based calculation
    assert latency > 0
    assert latency < 100.0  # Reasonable upper bound


def test_solver_solve_basic(solver, sample_model_spec, sample_activation_cache):
    """Test basic solve functionality."""
    # Convert activation cache to sensitivity scores format
    sensitivity_scores = {}
    for layer in sample_model_spec.layers:
        activations = sample_activation_cache.get(layer.name, {})
        sensitivity_scores[layer.name] = {
            "gradient_energy": activations.get("grad_norm", 0.0) / layer.hidden_size,
            "fisher": activations.get("fisher_trace", 0.0) / (layer.hidden_size ** 0.5),
            "proxy_eval": activations.get("proxy_gain", 0.0),
        }
    
    results = solver.solve(sample_model_spec.layers, sensitivity_scores)
    
    assert len(results) == len(sample_model_spec.layers)
    assert all(isinstance(r, AllocationResult) for r in results)
    assert all(r.rank >= 0 for r in results)
    assert all(r.method in ["lora", "adapter", "prefix", "none"] for r in results)

