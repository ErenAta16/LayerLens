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
from ..models import ModelSpec, LayerSpec
from ..profiling import (
    GradientEnergyAnalyzer,
    FisherInformationAnalyzer,
    ProxyFineTuneAnalyzer,
    aggregate_scores,
)
from ..optimization import AllocationSolver
from ..output import ManifestWriter
from ..features import compute_layer_features
from ..utils.validation import validate_model_spec, validate_activation_cache, validate_config
from ..utils.logging import get_logger
from ..exceptions import (
    ModelSpecError,
    ActivationCacheError,
    ConfigurationError,
    ProfilingError,
    OptimizationError,
    ManifestError,
)

logger = get_logger()


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
        
    Raises:
        ModelSpecError: If model specification is invalid
        ActivationCacheError: If activation cache is invalid
        ConfigurationError: If configuration is invalid
        ProfilingError: If profiling fails
        OptimizationError: If optimization fails
        ManifestError: If manifest generation fails
    """
    logger.info(f"Starting LayerLens pipeline for model: {model_spec.model_name}")
    
    # Input validation
    try:
        logger.debug("Validating inputs...")
        validate_model_spec(model_spec)
        validate_activation_cache(activation_cache, model_spec)
        validate_config(profiling_cfg, optimization_cfg)
        logger.debug("Input validation passed")
    except (ModelSpecError, ActivationCacheError, ConfigurationError) as e:
        logger.error(f"Input validation failed: {e}")
        raise
    
    # Validate output directory
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Output directory ready: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        raise ManifestError(f"Cannot create output directory: {e}") from e
    
    # 1. Profiling: Compute layer scores.
    try:
        logger.info("Step 1/5: Profiling layer sensitivities...")
        gradient_analyzer = GradientEnergyAnalyzer(profiling_cfg)
        fisher_analyzer = FisherInformationAnalyzer(profiling_cfg)
        proxy_analyzer = ProxyFineTuneAnalyzer(profiling_cfg)

        sensitivity_scores: Dict[str, Dict[str, float]] = {}
        for layer in model_spec.layers:
            try:
                activations = activation_cache.get(layer.name, {})
                metrics = {
                    "gradient_energy": gradient_analyzer.score(layer, activations),
                    "fisher": fisher_analyzer.score(layer, activations),
                    "proxy_eval": proxy_analyzer.score(layer, activations),
                }
                sensitivity_scores[layer.name] = metrics
            except Exception as e:
                logger.error(f"Failed to profile layer {layer.name}: {e}")
                raise ProfilingError(f"Profiling failed for layer {layer.name}: {e}") from e
        
        logger.info(f"Profiling completed for {len(sensitivity_scores)} layers")
    except Exception as e:
        if isinstance(e, ProfilingError):
            raise
        logger.error(f"Profiling step failed: {e}")
        raise ProfilingError(f"Profiling failed: {e}") from e
    
    # Normalize utility scores across all layers to 0-1 range for method selection
    # This ensures method selection thresholds work correctly regardless of raw metric scale
    # PERFORMANCE FIX: Single pass calculation to avoid duplicate aggregate_scores() calls
    all_utilities = []
    
    # Pre-compute weights once if not provided (avoid repeated dict creation)
    default_weights = None
    if profiling_cfg.metric_weights:
        weights_template = profiling_cfg.metric_weights
    else:
        # Lazy initialization: compute once when needed
        weights_template = None
    
    # Single loop: calculate utilities once
    for layer_name, metrics in sensitivity_scores.items():
        if profiling_cfg.metric_weights:
            weights = weights_template
        else:
            # Compute default weights once per unique metric set size
            if weights_template is None or len(weights_template) != len(metrics):
                weights_template = {key: 1.0 / len(metrics) for key in metrics.keys()}
            weights = weights_template
        utility = aggregate_scores(metrics, weights)
        all_utilities.append((layer_name, utility))
    
    if all_utilities:
        utilities_only = [u for _, u in all_utilities]
        min_util = min(utilities_only)
        max_util = max(utilities_only)
        util_range = max_util - min_util if max_util > min_util else 1.0
        
        # Normalize utilities using pre-computed values (no recalculation)
        normalized_scores = {}
        for layer_name, utility in all_utilities:
            # Normalize: (value - min) / range
            normalized_utility = (utility - min_util) / util_range if util_range > 0 else 0.0
            # Scale to 0-0.1 range to match threshold expectations
            normalized_utility = normalized_utility * 0.1
            # Store normalized utility (avoid unnecessary dict copy)
            metrics = sensitivity_scores[layer_name]
            normalized_scores[layer_name] = {
                **metrics,
                "_normalized_utility": normalized_utility
            }
        
        sensitivity_scores = normalized_scores
        logger.debug(f"Normalized {len(normalized_scores)} utility scores")

    # 2. Optimization: Select method and rank.
    try:
        logger.info("Step 2/5: Running optimization...")
        solver = AllocationSolver(optimization_cfg, profiling_cfg)
        allocations = solver.solve(model_spec.layers, sensitivity_scores)
        
        if not allocations:
            raise OptimizationError("Optimization returned no allocations")
        
        logger.info(f"Optimization completed: {len(allocations)} allocations")
    except Exception as e:
        if isinstance(e, OptimizationError):
            raise
        logger.error(f"Optimization step failed: {e}")
        raise OptimizationError(f"Optimization failed: {e}") from e

    # 3. Feature engineering: Compute layer feature sets.
    try:
        logger.info("Step 3/5: Computing layer features...")
        # PERFORMANCE FIX: O(N²) → O(N) by converting allocations to dict first
        alloc_dict = {a.layer_name: a for a in allocations}
        layer_features = {
            layer.name: compute_layer_features(
                layer=layer,
                total_params=model_spec.total_params,
                sensitivity=alloc_dict[layer.name].utility_score if layer.name in alloc_dict else 0.0,
            ).to_dict()
            for layer in model_spec.layers
        }
        logger.debug(f"Computed features for {len(layer_features)} layers")
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise ProfilingError(f"Feature engineering failed: {e}") from e

    # 4. Extract gradient/Fisher data for validation
    try:
        logger.info("Step 4/5: Extracting validation data...")
        gradient_norms = {}
        fisher_traces = {}
        for layer in model_spec.layers:
            activations = activation_cache.get(layer.name, {})
            gradient_norms[layer.name] = activations.get("grad_norm", 0.0)
            fisher_traces[layer.name] = activations.get("fisher_trace", 0.0)
    except Exception as e:
        logger.error(f"Failed to extract validation data: {e}")
        raise ProfilingError(f"Validation data extraction failed: {e}") from e
    
    # 5. Manifest generation: Save JSON output.
    try:
        logger.info("Step 5/5: Generating manifest...")
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
        logger.info(f"Pipeline completed successfully. Manifest: {manifest_path}")
        return manifest_path
    except Exception as e:
        logger.error(f"Manifest generation failed: {e}")
        raise ManifestError(f"Failed to generate manifest: {e}") from e

