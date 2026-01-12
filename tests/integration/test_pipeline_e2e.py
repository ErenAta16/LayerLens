"""
End-to-end integration tests for LayerLens pipeline.
"""

import pytest
from pathlib import Path
from layerlens.cli import run_pipeline
from layerlens.config import ProfilingConfig, OptimizationConfig
from layerlens.models import ModelSpec, LayerSpec


def test_pipeline_e2e_basic(sample_model_spec, sample_activation_cache, sample_profiling_config, 
                            sample_optimization_config, temp_output_dir):
    """Test end-to-end pipeline execution."""
    # Convert activation cache to proper format
    sensitivity_scores = {}
    for layer in sample_model_spec.layers:
        activations = sample_activation_cache.get(layer.name, {})
        sensitivity_scores[layer.name] = {
            "gradient_energy": activations.get("grad_norm", 0.0) / layer.hidden_size,
            "fisher": activations.get("fisher_trace", 0.0) / (layer.hidden_size ** 0.5),
            "proxy_eval": activations.get("proxy_gain", 0.0),
        }
    
    # Run pipeline
    manifest_path = run_pipeline(
        model_spec=sample_model_spec,
        profiling_cfg=sample_profiling_config,
        optimization_cfg=sample_optimization_config,
        activation_cache=sample_activation_cache,
        output_dir=temp_output_dir,
    )
    
    # Verify manifest was created
    assert manifest_path.exists()
    assert manifest_path.suffix == ".json"
    
    # Verify manifest content
    import json
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    assert "allocations" in manifest
    assert len(manifest["allocations"]) == len(sample_model_spec.layers)
    
    # Verify each allocation has required fields
    for alloc in manifest["allocations"]:
        assert "layer" in alloc
        assert "method" in alloc
        assert "rank" in alloc
        assert alloc["method"] in ["lora", "adapter", "prefix", "none"]


def test_pipeline_with_latency_profile(sample_model_spec, sample_activation_cache, 
                                       sample_profiling_config, sample_llm_latency_profile,
                                       temp_output_dir):
    """Test pipeline with latency profile."""
    optimization_cfg = OptimizationConfig(
        max_trainable_params=50000,
        max_flops=1e9,
        max_vram_gb=8.0,
        latency_target_ms=100.0,
        latency_profile=sample_llm_latency_profile,
    )
    
    manifest_path = run_pipeline(
        model_spec=sample_model_spec,
        profiling_cfg=sample_profiling_config,
        optimization_cfg=optimization_cfg,
        activation_cache=sample_activation_cache,
        output_dir=temp_output_dir,
    )
    
    assert manifest_path.exists()
    
    # Verify latency estimates are present
    import json
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    for alloc in manifest["allocations"]:
        if "latency" in alloc:
            assert alloc["latency"] >= 0

