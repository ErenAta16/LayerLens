# Troubleshooting Guide

Common issues and solutions for LayerLens.

## Installation Issues

### Cython Modules Not Compiling

**Symptoms:**
- `ModuleNotFoundError: No module named 'layerlens.profiling._batch'`
- Warning about Cython modules not being compiled

**Solution:**
LayerLens works without Cython modules (uses Python fallback). If you want Cython acceleration:

```bash
# Install build dependencies
pip install Cython>=3.0 setuptools>=65 wheel

# Reinstall
pip install -e . --force-reinstall
```

### Package Not Found After Installation

**Symptoms:**
- `ModuleNotFoundError: No module named 'layerlens'`
- Import errors in Colab

**Solution:**
```bash
# Reinstall in editable mode
pip install -e . --force-reinstall

# Verify installation
python -c "import layerlens; print(layerlens.__file__)"
```

## Configuration Errors

### Missing Required Parameter

**Error:**
```
TypeError: OptimizationConfig.__init__() missing 1 required positional argument: 'latency_target_ms'
```

**Solution:**
Always provide `latency_target_ms`:
```python
optimization_cfg = OptimizationConfig(
    max_trainable_params=50_000,
    max_flops=1e9,
    max_vram_gb=8.0,
    latency_target_ms=100.0,  # Required!
    latency_profile=latency_profile
)
```

### Invalid Method Name

**Error:**
```
ConfigurationError: invalid candidate_methods: ['invalid_method']
```

**Solution:**
Use only valid methods: `"lora"`, `"adapter"`, `"prefix"`, `"none"`

## Runtime Errors

### Empty Activation Cache

**Error:**
```
ActivationCacheError: activation_cache cannot be empty
```

**Solution:**
Ensure activation_cache contains entries for all layers:
```python
activation_cache = {
    layer.name: {
        "grad_norm": 0.5,
        "fisher_trace": 0.3,
    }
    for layer in model_spec.layers
}
```

### Missing Layer in Cache

**Error:**
```
ActivationCacheError: activation_cache missing entries for layers: ['encoder.layer.5']
```

**Solution:**
Add missing layer entries to activation_cache.

### Invalid Model Specification

**Error:**
```
ModelSpecError: duplicate layer name: encoder.layer.0
```

**Solution:**
Ensure all layer names are unique.

## Performance Issues

### Slow Pipeline Execution

**Possible Causes:**
- Cython modules not compiled (using Python fallback)
- Very large models (100+ layers)
- Complex optimization constraints

**Solutions:**
1. Compile Cython modules for better performance
2. Use smaller batch sizes in latency profile
3. Simplify optimization constraints

### Memory Issues

**Symptoms:**
- Out of memory errors
- Slow execution

**Solutions:**
1. Reduce batch_size in LatencyProfile
2. Process models in smaller chunks
3. Use CPU instead of GPU if memory limited

## Colab-Specific Issues

### Import Errors After Installation

**Solution:**
1. Restart runtime: Runtime → Restart runtime
2. Re-run installation cell
3. Verify with import test cell

### GPU Not Detected

**Solution:**
1. Runtime → Change runtime type → GPU
2. Verify: `torch.cuda.is_available()`

### Package Discovery Issues

**Solution:**
Use the updated installation cell that includes package discovery fixes.

## Getting Help

1. Check error messages - they include actionable guidance
2. Review `docs/ERROR_HANDLING.md` for exception details
3. Check logs with `layerlens.utils.logging`
4. Open an issue on GitHub with:
   - Error message
   - Code snippet
   - Environment details

