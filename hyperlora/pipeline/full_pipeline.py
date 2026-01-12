"""
full_pipeline.py
----------------
Contains the main end-to-end pipeline orchestration function that runs:
Profiling → Optimization → Manifest generation.

This module provides the core orchestration flow for LayerLens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..config import ProfilingConfig, OptimizationConfig
from ..meta import ModelSpec, LayerSpec
from ..profiling import (
    GradientEnergyAnalyzer,
    FisherInformationAnalyzer,
    ProxyFineTuneAnalyzer,
    aggregate_scores,
)
from ..optimization import AllocationSolver
from ..output import ManifestWriter
from ..features import compute_layer_features


def run_pipeline(
    model_spec: ModelSpec,
    profiling_cfg: ProfilingConfig,
    optimization_cfg: OptimizationConfig,
    activation_cache: Dict[str, Dict[str, float]],
    output_dir: Path,
) -> Path:
    """
    Run the complete LayerLens pipeline: profiling → optimization → manifest generation.
    
    Args:
        model_spec: Model specification with layer information
        profiling_cfg: Profiling configuration
        optimization_cfg: Optimization configuration with constraints
        activation_cache: Dictionary mapping layer names to activation metrics
        output_dir: Directory to save the generated manifest
        
    Returns:
        Path to the generated manifest JSON file
    """
    # 1. Profiling: Compute layer scores.
    gradient_analyzer = GradientEnergyAnalyzer(profiling_cfg)
    fisher_analyzer = FisherInformationAnalyzer(profiling_cfg)
    proxy_analyzer = ProxyFineTuneAnalyzer(profiling_cfg)

    sensitivity_scores: Dict[str, Dict[str, float]] = {}
    for layer in model_spec.layers:
        activations = activation_cache.get(layer.name, {})
        metrics = {
            "gradient_energy": gradient_analyzer.score(layer, activations),
            "fisher": fisher_analyzer.score(layer, activations),
            "proxy_eval": proxy_analyzer.score(layer, activations),
        }
        sensitivity_scores[layer.name] = metrics
    
    # Normalize utility scores across all layers to 0-1 range for method selection
    # This ensures method selection thresholds work correctly regardless of raw metric scale
    all_utilities = []
    for layer_name, metrics in sensitivity_scores.items():
        if profiling_cfg.metric_weights:
            weights = profiling_cfg.metric_weights
        else:
            weights = {key: 1.0 / len(metrics) for key in metrics.keys()}
        utility = aggregate_scores(metrics, weights)
        all_utilities.append((layer_name, utility))
    
    if all_utilities:
        utilities_only = [u for _, u in all_utilities]
        min_util = min(utilities_only)
        max_util = max(utilities_only)
        util_range = max_util - min_util if max_util > min_util else 1.0
        
        # Normalize utilities to 0-1 range and store back
        normalized_scores = {}
        for layer_name, metrics in sensitivity_scores.items():
            if profiling_cfg.metric_weights:
                weights = profiling_cfg.metric_weights
            else:
                weights = {key: 1.0 / len(metrics) for key in metrics.keys()}
            raw_utility = aggregate_scores(metrics, weights)
            # Normalize: (value - min) / range
            normalized_utility = (raw_utility - min_util) / util_range if util_range > 0 else 0.0
            # Scale to 0-0.1 range to match threshold expectations
            normalized_utility = normalized_utility * 0.1
            # Store normalized utility in a way that solver can use
            metrics_normalized = metrics.copy()
            metrics_normalized["_normalized_utility"] = normalized_utility
            normalized_scores[layer_name] = metrics_normalized
        
        sensitivity_scores = normalized_scores

    # 2. Optimization: Select method and rank.
    solver = AllocationSolver(optimization_cfg, profiling_cfg)
    allocations = solver.solve(model_spec.layers, sensitivity_scores)

    # 3. Feature engineering: Compute layer feature sets.
    layer_features = {
        layer.name: compute_layer_features(
            layer=layer,
            total_params=model_spec.total_params,
            sensitivity=next(
                (a.utility_score for a in allocations if a.layer_name == layer.name), 0.0
            ),
        ).to_dict()
        for layer in model_spec.layers
    }

    # 4. Extract gradient/Fisher data for validation
    gradient_norms = {}
    fisher_traces = {}
    for layer in model_spec.layers:
        activations = activation_cache.get(layer.name, {})
        gradient_norms[layer.name] = activations.get("grad_norm", 0.0)
        fisher_traces[layer.name] = activations.get("fisher_trace", 0.0)
    
    # 5. Manifest generation: Save JSON output.
    writer = ManifestWriter(output_dir)
    manifest_path = writer.write(
        allocations,
        file_name=f"{model_spec.model_name}_plan",
        extra_metadata={
            "layer_features": layer_features,
            "gradient_norms": gradient_norms,
            "fisher_traces": fisher_traces,
        },
    )
    return manifest_path

