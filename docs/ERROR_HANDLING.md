# Error Handling Guide

LayerLens provides comprehensive error handling with specific exception types and detailed error messages.

## Exception Hierarchy

```
LayerLensError (base)
├── ConfigurationError
├── ValidationError
├── OptimizationError
├── ProfilingError
├── ManifestError
├── ModelSpecError
└── ActivationCacheError
```

## Common Errors and Solutions

### ConfigurationError

**When it occurs:**
- Invalid configuration parameters
- Missing required configuration fields
- Invalid method names in candidate_methods

**Example:**
```python
from layerlens.config import OptimizationConfig
from layerlens.exceptions import ConfigurationError

try:
    cfg = OptimizationConfig(
        max_trainable_params=-1000,  # Invalid: negative value
        max_flops=1e9,
        max_vram_gb=8.0,
        latency_target_ms=100.0
    )
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

**Solution:**
- Ensure all numeric parameters are positive
- Use valid method names: "lora", "adapter", "prefix", "none"
- Check that metric_weights sum to a positive value

### ModelSpecError

**When it occurs:**
- Empty layer list
- Duplicate layer names
- Invalid hidden_size values
- Missing model_name

**Example:**
```python
from layerlens.models import ModelSpec, LayerSpec
from layerlens.exceptions import ModelSpecError

try:
    spec = ModelSpec(
        model_name="",  # Invalid: empty name
        total_params=1000,
        layers=[]
    )
except ModelSpecError as e:
    print(f"Model spec error: {e}")
```

**Solution:**
- Ensure model_name is not empty
- Provide at least one layer
- Ensure all layer names are unique
- Check that hidden_size > 0 for all layers

### ActivationCacheError

**When it occurs:**
- Missing layer entries in cache
- Invalid data types (non-numeric values)
- NaN or Inf values
- Missing required keys

**Example:**
```python
from layerlens.exceptions import ActivationCacheError

try:
    cache = {
        "layer.0": {"grad_norm": float('nan')}  # Invalid: NaN
    }
    # Use validation
    from layerlens.utils.validation import validate_activation_cache
    validate_activation_cache(cache, model_spec)
except ActivationCacheError as e:
    print(f"Activation cache error: {e}")
```

**Solution:**
- Ensure all layers have entries
- Use numeric values (int or float)
- Avoid NaN, Inf, or -Inf
- Include required keys: "grad_norm", "fisher_trace"

### ProfilingError

**When it occurs:**
- Profiling fails for a specific layer
- Analyzer errors during score calculation

**Example:**
```python
from layerlens.exceptions import ProfilingError

try:
    from layerlens.pipeline import run_pipeline
    manifest = run_pipeline(...)
except ProfilingError as e:
    print(f"Profiling error: {e}")
    # Check activation_cache for the problematic layer
```

**Solution:**
- Verify activation_cache contains valid data
- Check that all required metrics are present
- Ensure analyzers are properly configured

### OptimizationError

**When it occurs:**
- Empty layers list
- Empty sensitivity_scores
- Optimization solver fails
- No allocations returned

**Example:**
```python
from layerlens.exceptions import OptimizationError

try:
    solver = AllocationSolver(config, profiling_config)
    allocations = solver.solve([], {})  # Invalid: empty inputs
except OptimizationError as e:
    print(f"Optimization error: {e}")
```

**Solution:**
- Ensure layers list is not empty
- Provide sensitivity_scores for all layers
- Check configuration constraints are reasonable

### ManifestError

**When it occurs:**
- Cannot create output directory
- JSON serialization fails
- File write permissions issue
- Empty allocations list

**Example:**
```python
from layerlens.exceptions import ManifestError

try:
    writer = ManifestWriter(Path("/readonly/path"))  # Invalid: no write permission
    writer.write([], "test")  # Invalid: empty allocations
except ManifestError as e:
    print(f"Manifest error: {e}")
```

**Solution:**
- Ensure output directory is writable
- Check file permissions
- Provide non-empty allocations list
- Verify file_name is not empty

## Input Validation

LayerLens provides validation utilities to catch errors early:

```python
from layerlens.utils.validation import (
    validate_model_spec,
    validate_activation_cache,
    validate_config
)

# Validate before running pipeline
validate_model_spec(model_spec)
validate_activation_cache(activation_cache, model_spec)
validate_config(profiling_cfg, optimization_cfg)
```

## Logging

LayerLens uses structured logging for debugging:

```python
from layerlens.utils.logging import setup_logger, get_logger

# Setup logger (optional, defaults to INFO level)
setup_logger(level=logging.DEBUG)

# Get logger in your code
logger = get_logger()
logger.info("Starting pipeline...")
logger.debug("Detailed debug information")
logger.error("Error occurred")
```

## Best Practices

1. **Always validate inputs** before calling pipeline functions
2. **Catch specific exceptions** rather than generic Exception
3. **Check error messages** for actionable guidance
4. **Use logging** for debugging production issues
5. **Validate configuration** early in your code

## Example: Complete Error Handling

```python
from layerlens.pipeline import run_pipeline
from layerlens.utils.validation import (
    validate_model_spec,
    validate_activation_cache,
    validate_config
)
from layerlens.exceptions import (
    ModelSpecError,
    ActivationCacheError,
    ConfigurationError,
    ProfilingError,
    OptimizationError,
    ManifestError,
)

try:
    # Validate inputs
    validate_model_spec(model_spec)
    validate_activation_cache(activation_cache, model_spec)
    validate_config(profiling_cfg, optimization_cfg)
    
    # Run pipeline
    manifest_path = run_pipeline(
        model_spec=model_spec,
        profiling_cfg=profiling_cfg,
        optimization_cfg=optimization_cfg,
        activation_cache=activation_cache,
        output_dir=output_dir
    )
    print(f"Success! Manifest: {manifest_path}")
    
except (ModelSpecError, ActivationCacheError, ConfigurationError) as e:
    print(f"Input validation failed: {e}")
    # Fix input and retry
    
except ProfilingError as e:
    print(f"Profiling failed: {e}")
    # Check activation_cache data
    
except OptimizationError as e:
    print(f"Optimization failed: {e}")
    # Check configuration constraints
    
except ManifestError as e:
    print(f"Manifest generation failed: {e}")
    # Check output directory permissions
    
except Exception as e:
    print(f"Unexpected error: {e}")
    # Log for debugging
```

