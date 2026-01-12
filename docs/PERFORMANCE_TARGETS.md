# LayerLens Performance Targets (Elevated)

## Date: 2024-11-21

## Current Performance vs Elevated Targets

### Utility Score Quality

| Metric | Current | Previous Target | **Elevated Target** | Status |
|--------|---------|-----------------|---------------------|--------|
| Gradient Correlation | 0.854 | >= 0.7 | **>= 0.90** | ⚠️ Close |
| Fisher Correlation | 1.000 | >= 0.6 | **>= 0.95** | ✅ Exceeds |
| Coefficient of Variation | 0.387 | >= 0.2 | **>= 0.35** | ✅ Exceeds |
| Ranking Accuracy | N/A | >= 80% | **>= 85%** | ❌ Not measured |

### Method Selection

| Metric | Current | Previous Target | **Elevated Target** | Status |
|--------|---------|-----------------|---------------------|--------|
| Compatibility Rate | 100% | >= 95% | **>= 98%** | ✅ Exceeds |
| Budget Utilization | 75.26% | 70-90% | **80-88%** | ⚠️ Below |
| Constraint Satisfaction | 100% | 100% | **100%** | ✅ Meets |
| Method Diversity | 4 methods | >= 2 | **>= 3** | ✅ Exceeds |

### Rank Estimation

| Metric | Current | Previous Target | **Elevated Target** | Status |
|--------|---------|-----------------|---------------------|--------|
| Rank-Utility Correlation | 0.135 | >= 0.6 | **>= 0.70** | ❌ Below |
| Budget Utilization | 75.26% | 70-90% | **80-88%** | ⚠️ Below |
| Rank-Utility Consistency | 69.7% | >= 80% | **>= 85%** | ⚠️ Below |
| Rank Range Reasonableness | 0-6 | Reasonable | **1-16 (better spread)** | ⚠️ Narrow |

## Optimization Strategy

### Priority 1: Rank-Utility Correlation (Critical)

**Current Issue:** Correlation dropped to 0.135 after budget optimization
**Target:** >= 0.70
**Strategy:**
1. Use utility-weighted budget allocation instead of uniform scaling
2. Preserve rank-utility relationship while respecting budget
3. Only assign "none" to very low utility layers (< 0.01)

### Priority 2: Budget Utilization (High)

**Current:** 75.26%
**Target:** 80-88%
**Strategy:**
1. Increase target utilization to 85% (middle of 80-88%)
2. Better rank scaling algorithm
3. Utility-weighted allocation

### Priority 3: Rank-Utility Consistency (Medium)

**Current:** 69.7%
**Target:** >= 85%
**Strategy:**
1. Improve rank estimation to better follow utility
2. Reduce cases where high utility gets lower rank
3. Better handling of budget constraints

### Priority 4: Gradient Correlation (Medium)

**Current:** 0.854
**Target:** >= 0.90
**Strategy:**
1. Improve utility score calculation
2. Better normalization
3. Weighted aggregation optimization

## Implementation Plan

### Step 1: Utility-Weighted Budget Allocation

Instead of uniform scaling, allocate budget proportionally to utility:
- High utility layers get more budget
- Low utility layers get less budget
- Maintains rank-utility correlation

### Step 2: Improved Rank Estimation

- Better base rank calculation
- Utility-weighted scaling
- Smarter budget distribution

### Step 3: Enhanced Utility Calculation

- Optimize metric weights
- Better normalization
- Improved aggregation


