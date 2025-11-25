# LayerLens Test Results and Issues Analysis

## Test Summary

**Date:** 2024-11-21  
**Total Tests:** 7  
**Successful:** 7  
**Failed:** 0

## Test Configurations

1. **Default-Uniform**: Default config with uniform layer sensitivity
2. **High-Budget-Uniform**: Higher parameter budget (4x)
3. **Early-Sensitive**: Early layers (0-3) more sensitive
4. **Late-Sensitive**: Late layers (8-11) more sensitive
5. **LoRA-Cheap**: LoRA method penalty reduced (0.8 vs 1.0)
6. **Utility-Focused**: Higher weight on utility (0.8 vs 0.5)
7. **Random-Sensitivity**: Random sensitivity pattern

## Critical Issues Identified

### 1. [HIGH] All Utility Scores Are Zero

**Severity:** HIGH  
**Affected Tests:** All 7 tests  
**Description:** Utility scores are calculated as 0.0 for all layers, indicating a problem with metric aggregation or normalization.

**Root Cause Analysis:**
- `GradientEnergyAnalyzer.score()` divides `grad_norm` by `hidden_size` (line 48 in `analyzers.py`)
- `FisherInformationAnalyzer.score()` divides `fisher_trace` by `hidden_size ** 0.5` (line 83 in `analyzers.py`)
- However, `demo_real_model.py` already provides normalized values (lines 204-205)
- This causes **double normalization**, making values extremely small (~0.0006 for gradient, ~0.01 for Fisher)
- When aggregated with default weights, these tiny values result in utility ≈ 0.0

**Impact:**
- All layers receive identical utility scores (0.0)
- Method selection cannot differentiate between layers
- Rank estimation cannot prioritize important layers

**Fix Required:**
- Remove normalization from `analyzers.py` score methods, OR
- Remove normalization from `demo_real_model.py` activation cache creation
- Recommended: Keep normalization in analyzers, remove from demo script

### 2. [MEDIUM] Uniform Method Selection

**Severity:** MEDIUM  
**Affected Tests:** All 7 tests  
**Description:** All layers are assigned the same PEFT method (LoRA), suggesting utility scores are not differentiating layers properly.

**Root Cause:**
- Direct consequence of Issue #1 (zero utility scores)
- When all layers have utility = 0.0, method selection defaults to the first method in sequence

**Impact:**
- Cannot test different PEFT methods
- Cannot evaluate method selection logic
- All layers get identical treatment regardless of sensitivity

**Fix Required:**
- Fix Issue #1 first
- Verify method selection logic works with non-zero utilities

### 3. [MEDIUM] Uniform Rank Assignment

**Severity:** MEDIUM  
**Affected Tests:** All 7 tests  
**Description:** All layers are assigned rank=1, indicating the rank estimation function is not working correctly or utility scores are too low.

**Root Cause:**
- Direct consequence of Issue #1 (zero utility scores)
- Rank estimation function likely uses utility score, which is 0.0 for all layers
- Results in minimum rank (1) for all layers

**Impact:**
- Cannot test rank differentiation
- Cannot evaluate rank estimation logic
- All layers get minimum rank regardless of importance

**Fix Required:**
- Fix Issue #1 first
- Verify rank estimation logic works with non-zero utilities

## Additional Observations

### Positive Findings

1. **Pipeline Execution:** All tests complete successfully without errors
2. **Manifest Generation:** JSON manifests are correctly generated
3. **Configuration Handling:** Different configurations are accepted and processed
4. **Error Handling:** No crashes or exceptions during testing

### Areas for Improvement

1. **Metric Normalization:** Need to standardize normalization approach
2. **Utility Score Range:** Need to ensure utility scores are in a reasonable range (e.g., 0-1 or 0-100)
3. **Method Differentiation:** Need to verify method selection works with varying utilities
4. **Rank Differentiation:** Need to verify rank estimation works with varying utilities
5. **Sensitivity Patterns:** Need to verify that different sensitivity patterns (early/late/middle) produce different allocations

## Recommended Fixes

### Priority 1: Fix Double Normalization

**File:** `hyperlora/profiling/analyzers.py`

**Current Code:**
```python
def score(self, layer: LayerSpec, activations: Any) -> float:
    gradient_norm = activations.get("grad_norm", 0.0)
    return gradient_norm / max(layer.hidden_size, 1)  # Double normalization!
```

**Fix:**
```python
def score(self, layer: LayerSpec, activations: Any) -> float:
    gradient_norm = activations.get("grad_norm", 0.0)
    # Return as-is if already normalized, or normalize if raw gradient
    # For now, assume activation_cache provides normalized values
    return gradient_norm
```

**Alternative Fix (in demo_real_model.py):**
```python
# Don't normalize in demo script - let analyzers handle it
activation_cache[layer.name] = {
    "grad_norm": layer_grad,  # Raw value
    "fisher_trace": layer_fisher,  # Raw value
    "proxy_gain": 0.1 + np.random.random() * 0.2
}
```

### Priority 2: Verify Aggregate Scores

**File:** `hyperlora/optimization/solver.py`

Check that `aggregate_scores` is called with correct weights:
```python
utility = aggregate_scores(metrics, self.config.objective_weights)
```

But `objective_weights` might not match metric keys. Need to verify:
- Metrics: `["gradient_energy", "fisher", "proxy_eval"]`
- Weights: `{"utility": 0.5, "cost": 0.2, ...}`

**Fix:** Use `ProfilingConfig.metrics` weights instead of `OptimizationConfig.objective_weights`

### Priority 3: Add Debug Logging

Add logging to track:
- Raw metric values before aggregation
- Aggregated utility scores
- Method selection decisions
- Rank estimation values

## Next Steps

1. Fix double normalization issue
2. Re-run all configuration tests
3. Verify utility scores are non-zero and differentiated
4. Verify method selection varies across layers
5. Verify rank assignment varies across layers
6. Add unit tests for individual components

