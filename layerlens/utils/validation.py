"""
validation.py
-------------
Input validation utilities for LayerLens components.
"""

from typing import Dict, List, Any
from pathlib import Path

from ..exceptions import (
    ModelSpecError,
    ActivationCacheError,
    ConfigurationError,
)
from ..models import ModelSpec, LayerSpec
from ..config import ProfilingConfig, OptimizationConfig


def validate_model_spec(model_spec: ModelSpec) -> None:
    """
    Validate model specification.
    
    Args:
        model_spec: Model specification to validate
        
    Raises:
        ModelSpecError: If model specification is invalid
    """
    if not isinstance(model_spec, ModelSpec):
        raise ModelSpecError(f"model_spec must be ModelSpec instance, got {type(model_spec)}")
    
    if not model_spec.model_name:
        raise ModelSpecError("model_spec.model_name cannot be empty")
    
    if not model_spec.layers:
        raise ModelSpecError("model_spec.layers cannot be empty")
    
    if model_spec.total_params <= 0:
        raise ModelSpecError(f"model_spec.total_params must be positive, got {model_spec.total_params}")
    
    # Validate each layer
    layer_names = set()
    for i, layer in enumerate(model_spec.layers):
        if not isinstance(layer, LayerSpec):
            raise ModelSpecError(f"layer at index {i} must be LayerSpec instance, got {type(layer)}")
        
        if not layer.name:
            raise ModelSpecError(f"layer at index {i} has empty name")
        
        if layer.name in layer_names:
            raise ModelSpecError(f"duplicate layer name: {layer.name}")
        layer_names.add(layer.name)
        
        if layer.hidden_size <= 0:
            raise ModelSpecError(
                f"layer '{layer.name}' has invalid hidden_size: {layer.hidden_size}"
            )


def validate_activation_cache(
    activation_cache: Dict[str, Dict[str, float]],
    model_spec: ModelSpec,
    required_keys: List[str] | None = None,
) -> None:
    """
    Validate activation cache structure and content.
    
    Args:
        activation_cache: Activation cache dictionary
        model_spec: Model specification to validate against
        required_keys: Optional list of required keys in each layer's cache
        
    Raises:
        ActivationCacheError: If activation cache is invalid
    """
    if not isinstance(activation_cache, dict):
        raise ActivationCacheError(
            f"activation_cache must be dict, got {type(activation_cache)}"
        )
    
    if not activation_cache:
        raise ActivationCacheError("activation_cache cannot be empty")
    
    # Check that all layers have entries
    layer_names = {layer.name for layer in model_spec.layers}
    cache_keys = set(activation_cache.keys())
    
    missing_layers = layer_names - cache_keys
    if missing_layers:
        raise ActivationCacheError(
            f"activation_cache missing entries for layers: {sorted(missing_layers)}"
        )
    
    # Validate each layer's cache
    required = required_keys or []
    for layer_name, cache in activation_cache.items():
        if not isinstance(cache, dict):
            raise ActivationCacheError(
                f"activation_cache['{layer_name}'] must be dict, got {type(cache)}"
            )
        
        for key in required:
            if key not in cache:
                raise ActivationCacheError(
                    f"activation_cache['{layer_name}'] missing required key: {key}"
                )
        
        # Validate values are numeric and not NaN/Inf
        for key, value in cache.items():
            if not isinstance(value, (int, float)):
                raise ActivationCacheError(
                    f"activation_cache['{layer_name}']['{key}'] must be numeric, got {type(value)}"
                )
            # Check for NaN or Inf
            if isinstance(value, float):
                import math
                if math.isnan(value) or math.isinf(value):
                    raise ActivationCacheError(
                        f"activation_cache['{layer_name}']['{key}'] contains invalid value: {value}"
                    )


def validate_config(
    profiling_cfg: ProfilingConfig | None = None,
    optimization_cfg: OptimizationConfig | None = None,
) -> None:
    """
    Validate configuration objects.
    
    Args:
        profiling_cfg: Optional profiling configuration
        optimization_cfg: Optional optimization configuration
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    if profiling_cfg is not None:
        if not isinstance(profiling_cfg, ProfilingConfig):
            raise ConfigurationError(
                f"profiling_cfg must be ProfilingConfig instance, got {type(profiling_cfg)}"
            )
        
        # Validate metric weights if provided
        if profiling_cfg.metric_weights:
            if not isinstance(profiling_cfg.metric_weights, dict):
                raise ConfigurationError("metric_weights must be dict")
            
            total_weight = sum(profiling_cfg.metric_weights.values())
            if total_weight <= 0:
                raise ConfigurationError(
                    f"metric_weights must have positive total, got {total_weight}"
                )
    
    if optimization_cfg is not None:
        if not isinstance(optimization_cfg, OptimizationConfig):
            raise ConfigurationError(
                f"optimization_cfg must be OptimizationConfig instance, got {type(optimization_cfg)}"
            )
        
        if optimization_cfg.max_trainable_params <= 0:
            raise ConfigurationError(
                f"max_trainable_params must be positive, got {optimization_cfg.max_trainable_params}"
            )
        
        if optimization_cfg.max_flops <= 0:
            raise ConfigurationError(
                f"max_flops must be positive, got {optimization_cfg.max_flops}"
            )
        
        if optimization_cfg.max_vram_gb <= 0:
            raise ConfigurationError(
                f"max_vram_gb must be positive, got {optimization_cfg.max_vram_gb}"
            )
        
        if optimization_cfg.latency_target_ms <= 0:
            raise ConfigurationError(
                f"latency_target_ms must be positive, got {optimization_cfg.latency_target_ms}"
            )
        
        # Validate candidate methods
        if not optimization_cfg.candidate_methods:
            raise ConfigurationError("candidate_methods cannot be empty")
        
        valid_methods = {"lora", "adapter", "prefix", "none"}
        invalid_methods = set(optimization_cfg.candidate_methods) - valid_methods
        if invalid_methods:
            raise ConfigurationError(
                f"invalid candidate_methods: {sorted(invalid_methods)}. "
                f"Valid methods: {sorted(valid_methods)}"
            )

