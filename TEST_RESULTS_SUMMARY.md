# LayerLens Test Results Summary

## Test Execution Date
2024-11-21

## Test Overview

7 different configuration tests were executed to validate LayerLens behavior:

1. **Default-Uniform**: Default configuration with uniform layer sensitivity
2. **High-Budget-Uniform**: Higher parameter budget (4x increase)
3. **Early-Sensitive**: Early layers (0-3) more sensitive
4. **Late-Sensitive**: Late layers (8-11) more sensitive
5. **LoRA-Cheap**: LoRA method penalty reduced (0.8 vs 1.0)
6. **Utility-Focused**: Higher weight on utility (0.8 vs 0.5)
7. **Random-Sensitivity**: Random sensitivity pattern

## Test Results

### All Tests: PASSED (7/7)

However, all tests revealed the same critical issues:

| Test Name | Status | Layers | Methods | Total Rank | Avg Utility | Non-zero Utility |
|-----------|--------|--------|----------|------------|-------------|------------------|
| Default-Uniform | Success | 12 | LoRA: 12 | 12 | 0.0000 | 0/12 |
| High-Budget-Uniform | Success | 12 | LoRA: 12 | 12 | 0.0000 | 0/12 |
| Early-Sensitive | Success | 12 | LoRA: 12 | 12 | 0.0000 | 0/12 |
| Late-Sensitive | Success | 12 | LoRA: 12 | 12 | 0.0000 | 0/12 |
| LoRA-Cheap | Success | 12 | LoRA: 12 | 12 | 0.0000 | 0/12 |
| Utility-Focused | Success | 12 | LoRA: 12 | 12 | 0.0000 | 0/12 |
| Random-Sensitivity | Success | 12 | LoRA: 12 | 12 | 0.0000 | 0/12 |

## Critical Issues Identified

### Issue #1: Zero Utility Scores (HIGH PRIORITY)

**Problem:** All utility scores are calculated as 0.0 for all layers.

**Root Causes:**
1. **Double Normalization**: 
   - `demo_real_model.py` normalizes gradient/Fisher values (lines 204-205)
   - `analyzers.py` score methods normalize again (lines 48, 83)
   - Result: Values become extremely small (~0.0006 for gradient, ~0.01 for Fisher)

2. **Weight Key Mismatch**:
   - Metrics: `{"gradient_energy": 0.0006, "fisher": 0.01, "proxy_eval": 0.2}`
   - Weights: `{"utility": 0.5, "cost": 0.2, "flop": 0.1, ...}`
   - Keys don't match! All weights default to 0.0
   - Result: `aggregate_scores` returns 0.0

**Impact:**
- Cannot differentiate layer importance
- All layers get identical treatment
- Method and rank selection cannot work properly

### Issue #2: Uniform Method Selection (MEDIUM PRIORITY)

**Problem:** All layers are assigned the same PEFT method (LoRA).

**Root Cause:** Direct consequence of Issue #1. When utility = 0.0 for all layers, method selection defaults to first method.

**Impact:**
- Cannot test different PEFT methods
- Cannot evaluate method selection logic

### Issue #3: Uniform Rank Assignment (MEDIUM PRIORITY)

**Problem:** All layers are assigned rank=1.

**Root Cause:** Direct consequence of Issue #1. When utility = 0.0, rank estimation returns minimum value.

**Impact:**
- Cannot test rank differentiation
- Cannot evaluate rank estimation logic

## Detailed Analysis

### Metric Values

When testing with realistic values:
- Gradient score: ~0.0006 (after double normalization)
- Fisher score: ~0.01 (after double normalization)
- Proxy score: ~0.2 (not normalized)

### Aggregation Process

```python
metrics = {
    "gradient_energy": 0.0006,
    "fisher": 0.01,
    "proxy_eval": 0.2
}

weights = {
    "utility": 0.5,  # Key mismatch!
    "cost": 0.2,
    "flop": 0.1,
    ...
}

# aggregate_scores looks for weights["gradient_energy"], weights["fisher"], etc.
# But these keys don't exist, so all weights default to 0.0
# Result: utility = 0.0
```

### Expected Behavior

With correct weights:
```python
weights = {
    "gradient_energy": 0.4,
    "fisher": 0.4,
    "proxy_eval": 0.2
}

# utility = 0.4 * 0.0006 + 0.4 * 0.01 + 0.2 * 0.2
# utility ≈ 0.04 (still small due to double normalization)
```

## Fixes Required

### Fix 1: Remove Double Normalization

**Option A:** Remove normalization from `analyzers.py`
```python
# hyperlora/profiling/analyzers.py
def score(self, layer: LayerSpec, activations: Any) -> float:
    gradient_norm = activations.get("grad_norm", 0.0)
    return gradient_norm  # Return as-is, assume already normalized
```

**Option B:** Remove normalization from `demo_real_model.py`
```python
# demo_real_model.py
activation_cache[layer.name] = {
    "grad_norm": layer_grad,  # Raw value, let analyzers normalize
    "fisher_trace": layer_fisher,  # Raw value
    "proxy_gain": 0.1 + np.random.random() * 0.2
}
```

**Recommendation:** Option B (keep normalization in analyzers, remove from demo)

### Fix 2: Fix Weight Key Mismatch

**Option A:** Add metric weights to `ProfilingConfig`
```python
@dataclass
class ProfilingConfig:
    ...
    metric_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "gradient_energy": 0.4,
            "fisher": 0.4,
            "proxy_eval": 0.2,
        }
    )
```

**Option B:** Use metric keys directly in `OptimizationConfig`
```python
# In solver.py
utility = aggregate_scores(metrics, {
    "gradient_energy": 0.4,
    "fisher": 0.4,
    "proxy_eval": 0.2,
})
```

**Recommendation:** Option A (add to ProfilingConfig for better modularity)

## Next Steps

1. ✅ **Fix double normalization** - Remove from demo script
2. ✅ **Fix weight key mismatch** - Add metric_weights to ProfilingConfig
3. ⏳ **Re-run all tests** - Verify utility scores are non-zero
4. ⏳ **Verify differentiation** - Check that different layers get different methods/ranks
5. ⏳ **Add unit tests** - Test individual components

## Files Modified

- `ISSUES_ANALYSIS.md` - Detailed issue analysis
- `TEST_RESULTS_SUMMARY.md` - This file
- `test_configurations.py` - Test script
- `output/test_results.json` - Detailed test results

