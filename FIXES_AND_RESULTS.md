# LayerLens Fixes and Test Results

## Date: 2024-11-21

## Issues Fixed

### ✅ Issue #1: Double Normalization (FIXED)

**Problem:** Gradient and Fisher values were normalized twice - once in `demo_real_model.py` and again in `analyzers.py`.

**Fix Applied:**
- Removed normalization from `demo_real_model.py` (lines 203-205)
- Kept normalization in `analyzers.py` (analyzers handle normalization)
- Raw gradient/Fisher values are now passed to analyzers

**Result:** Values are no longer extremely small, utility scores are now in reasonable range (0.036 - 0.066).

### ✅ Issue #2: Weight Key Mismatch (FIXED)

**Problem:** `aggregate_scores` was called with wrong weight keys:
- Metrics: `{"gradient_energy": ..., "fisher": ..., "proxy_eval": ...}`
- Weights: `{"utility": 0.5, "cost": 0.2, ...}` (keys don't match!)

**Fix Applied:**
1. Added `metric_weights` to `ProfilingConfig`:
   ```python
   metric_weights: Dict[str, float] = field(
       default_factory=lambda: {
           "gradient_energy": 0.4,
           "fisher": 0.4,
           "proxy_eval": 0.2,
       }
   )
   ```

2. Updated `AllocationSolver` to use `profiling_config.metric_weights`:
   ```python
   if self.profiling_config is not None:
       weights = self.profiling_config.metric_weights
   else:
       weights = {key: 1.0 / len(metrics) for key in metrics.keys()}
   utility = aggregate_scores(metrics, weights)
   ```

3. Updated `pipeline.py` to pass `profiling_cfg` to solver:
   ```python
   solver = AllocationSolver(optimization_cfg, profiling_cfg)
   ```

**Result:** Utility scores are now correctly calculated using matching weight keys.

## Test Results After Fixes

### Summary

| Metric | Before Fix | After Fix | Status |
|--------|-----------|-----------|--------|
| Avg Utility | 0.0000 | 0.0361 - 0.0661 | ✅ FIXED |
| Non-zero Utility | 0/12 | 12/12 | ✅ FIXED |
| Rank Variation | 12 (all rank=1) | 28-54 (varies) | ✅ FIXED |
| Method Selection | LoRA (all) | LoRA (all) | ⚠️ Still uniform |

### Detailed Results

#### Test 1: Default-Uniform
- **Utility:** 0.0446 (was 0.0000) ✅
- **Rank:** 36 (was 12) ✅
- **Methods:** LoRA x12 ⚠️

#### Test 2: High-Budget-Uniform
- **Utility:** 0.0446 (was 0.0000) ✅
- **Rank:** 36 (was 12) ✅
- **Methods:** LoRA x12 ⚠️

#### Test 3: Early-Sensitive
- **Utility:** 0.0361 (was 0.0000) ✅
- **Rank:** 28 (was 12) ✅ - Lower rank for less sensitive layers
- **Methods:** LoRA x12 ⚠️

#### Test 4: Late-Sensitive
- **Utility:** 0.0361 (was 0.0000) ✅
- **Rank:** 28 (was 12) ✅ - Lower rank for less sensitive layers
- **Methods:** LoRA x12 ⚠️

#### Test 5: LoRA-Cheap
- **Utility:** 0.0446 (was 0.0000) ✅
- **Rank:** 36 (was 12) ✅
- **Methods:** LoRA x12 ⚠️

#### Test 6: Utility-Focused
- **Utility:** 0.0361 (was 0.0000) ✅
- **Rank:** 28 (was 12) ✅
- **Methods:** LoRA x12 ⚠️

#### Test 7: Random-Sensitivity
- **Utility:** 0.0661 (was 0.0000) ✅ - Highest utility
- **Rank:** 54 (was 12) ✅ - Highest rank (more sensitive layers)
- **Methods:** LoRA x12 ⚠️

## Remaining Issue

### ⚠️ Issue #3: Uniform Method Selection (MEDIUM PRIORITY)

**Problem:** All layers are still assigned the same PEFT method (LoRA), even though:
- Utility scores vary (0.036 - 0.066)
- Rank values vary (28 - 54)
- Different sensitivity patterns produce different ranks

**Root Cause Analysis:**

1. **Method Selection Thresholds:**
   - `_build_thresholds()` creates thresholds based on method count
   - Default methods: `["lora", "adapter", "prefix", "none"]`
   - Thresholds are likely too high for current utility range (0.036 - 0.066)

2. **Method Selection Logic:**
   ```python
   def _select_method(self, utility: float) -> str:
       thresholds = self._method_thresholds
       method_index = select_method_c(utility, thresholds, self._method_codes)
       return self._method_sequence[method_index]
   ```
   - If utility < all thresholds → returns first method (LoRA)
   - If utility > all thresholds → returns last method (none)
   - Current utility range (0.036-0.066) is likely below all thresholds

3. **Expected Behavior:**
   - Low utility (0.01-0.03) → "none" or "prefix"
   - Medium utility (0.03-0.05) → "lora" or "adapter"
   - High utility (0.05+) → "lora" with higher rank

**Investigation Needed:**
- Check `_build_thresholds()` implementation
- Verify threshold values for utility range 0.036-0.066
- Consider adjusting thresholds or utility scaling

## Positive Findings

1. ✅ **Utility Calculation:** Now working correctly with non-zero values
2. ✅ **Rank Estimation:** Varies based on utility (28-54 range)
3. ✅ **Sensitivity Differentiation:** Different sensitivity patterns produce different ranks
4. ✅ **Pipeline Execution:** All tests complete successfully
5. ✅ **No Crashes:** System is stable and reliable

## Next Steps

1. **Investigate Method Selection:**
   - Review `_build_thresholds()` implementation
   - Check threshold values
   - Consider utility scaling or threshold adjustment

2. **Add Debug Logging:**
   - Log utility scores per layer
   - Log threshold values
   - Log method selection decisions

3. **Test with Wider Utility Range:**
   - Create test cases with utility 0.0-1.0 range
   - Verify method selection works with extreme values

4. **Consider Method Selection Strategy:**
   - Current: Threshold-based (utility < threshold → method)
   - Alternative: Cost-benefit based (utility/cost ratio)
   - Alternative: Constraint-based (select method that fits budget)

## Files Modified

1. `hyperlora/config.py` - Added `metric_weights` to `ProfilingConfig`
2. `hyperlora/optimization/solver.py` - Updated to use `profiling_config.metric_weights`
3. `hyperlora/cli/pipeline.py` - Pass `profiling_cfg` to solver
4. `demo_real_model.py` - Removed double normalization

## Conclusion

**Major Progress:** Two critical issues (double normalization, weight mismatch) have been fixed. Utility scores and rank estimation are now working correctly.

**Remaining Work:** Method selection needs investigation to ensure different methods are selected based on utility scores and constraints.

