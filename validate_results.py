"""
LayerLens Results Validation Script
===================================
Validates LayerLens results against academic standards and goals.
"""

import json
from pathlib import Path
from hyperlora.cli import run_pipeline, apply_manifest
from hyperlora.config import ProfilingConfig, OptimizationConfig
from hyperlora.meta import ModelSpec, LayerSpec
from hyperlora.validation import UtilityValidator, MethodValidator, RankValidator


def load_manifest_results(manifest_path: Path) -> tuple:
    """Load results from manifest file."""
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    allocations = []
    utility_scores = {}
    gradient_norms = {}
    fisher_traces = {}

    for alloc_data in manifest.get("allocations", []):
        from hyperlora.optimization import AllocationResult
        alloc = AllocationResult(
            layer_name=alloc_data["layer"],
            method=alloc_data["method"],
            rank=alloc_data["rank"],
            utility_score=alloc_data["utility"],
            cost_estimate=alloc_data["cost"],
            flop_estimate=alloc_data["flops"],
            vram_estimate=alloc_data["vram"],
            latency_estimate=alloc_data["latency"],
            total_score=alloc_data["total_score"],
        )
        allocations.append(alloc)
        utility_scores[alloc.layer_name] = alloc.utility_score

    # Extract gradient/Fisher from metadata if available
    metadata = manifest.get("metadata", {})
    gradient_norms = metadata.get("gradient_norms", {})
    fisher_traces = metadata.get("fisher_traces", {})

    return allocations, utility_scores, gradient_norms, fisher_traces


def validate_bert_results():
    """Validate BERT-base results."""
    print("=" * 60)
    print("LayerLens Results Validation")
    print("=" * 60)

    manifest_path = Path("output/bert-base-uncased_plan.json")
    if not manifest_path.exists():
        print(f"Error: Manifest file not found: {manifest_path}")
        return

    # Load results
    allocations, utility_scores, gradient_norms, fisher_traces = load_manifest_results(
        manifest_path
    )

    # Create model spec (simplified - would need to match actual model)
    layers = [
        LayerSpec(
            name=f"encoder.layer.{i}",
            hidden_size=768,
            layer_type="transformer",
            supports_attention=True,
        )
        for i in range(12)
    ]
    model_spec = ModelSpec(
        model_name="bert-base-uncased",
        total_params=109_482_240,
        layers=layers,
    )

    print(f"\nLoaded {len(allocations)} allocations from manifest")
    print(f"Utility score range: {min(utility_scores.values()):.4f} - {max(utility_scores.values()):.4f}")

    # Validation 1: Utility Scores
    print("\n" + "=" * 60)
    print("1. UTILITY SCORE VALIDATION")
    print("=" * 60)

    utility_validator = UtilityValidator(model_spec)
    utility_results = utility_validator.validate_all(
        utility_scores,
        gradient_norms=gradient_norms if gradient_norms else None,
        fisher_traces=fisher_traces if fisher_traces else None,
    )

    cv = utility_results.get("coefficient_of_variation", 0.0)
    print(f"\nCoefficient of Variation: {cv:.4f}")
    if cv >= 0.2:
        print("  [PASS] Good variance (CV >= 0.2)")
    else:
        print("  [WARN] Low variance (CV < 0.2)")

    if "gradient_correlation" in utility_results:
        corr = utility_results["gradient_correlation"]
        p_val = utility_results["gradient_p_value"]
        print(f"\nGradient Correlation: {corr:.4f} (p={p_val:.4f})")
        if corr >= 0.90:
            print("  [PASS] Excellent correlation (r >= 0.90)")
        elif corr >= 0.85:
            print("  [PASS] Strong correlation (r >= 0.85)")
        elif corr >= 0.7:
            print("  [WARN] Good correlation (0.7 <= r < 0.85)")
        else:
            print("  [FAIL] Weak correlation (r < 0.7)")

    if "fisher_correlation" in utility_results:
        corr = utility_results["fisher_correlation"]
        p_val = utility_results["fisher_p_value"]
        print(f"\nFisher Correlation: {corr:.4f} (p={p_val:.4f})")
        if corr >= 0.95:
            print("  [PASS] Excellent correlation (r >= 0.95)")
        elif corr >= 0.90:
            print("  [PASS] Strong correlation (r >= 0.90)")
        elif corr >= 0.6:
            print("  [WARN] Good correlation (0.6 <= r < 0.90)")
        else:
            print("  [FAIL] Weak correlation (r < 0.6)")

    # Validation 2: Method Selection
    print("\n" + "=" * 60)
    print("2. METHOD SELECTION VALIDATION")
    print("=" * 60)

    method_validator = MethodValidator(model_spec)
    method_results = method_validator.validate_all(
        allocations,
        max_trainable_params=50000,
        max_flops=1e9,
        max_vram_gb=8.0,
        latency_target_ms=100.0,
    )

    compat_rate = method_results.get("compatibility_rate", 0.0)
    print(f"\nCompatibility Rate: {compat_rate:.2%}")
    if compat_rate >= 0.95:
        print("  [PASS] High compatibility (>= 95%)")
    else:
        print("  [WARN] Low compatibility (< 95%)")

    budget_ok = method_results.get("budget_compliant", False)
    total_params = method_results.get("total_params", 0)
    max_params = method_results.get("max_params", 0)
    utilization = method_results.get("budget_utilization", 0.0)
    print(f"\nBudget Compliance: {budget_ok}")
    print(f"  Total params: {total_params:,} / {max_params:,}")
    print(f"  Utilization: {utilization:.2%}")
    if budget_ok:
        print("  [PASS] Within budget")
        if 0.80 <= utilization <= 0.88:
            print("  [PASS] Optimal utilization (80-88%)")
        elif 0.70 <= utilization < 0.80:
            print("  [WARN] Acceptable utilization (70-80%)")
        elif utilization < 0.70:
            print("  [WARN] Low utilization (< 70%)")
        else:
            print("  [WARN] High utilization (> 88%)")
    else:
        print("  [FAIL] Exceeds budget")

    constraints = method_results.get("constraints", {})
    all_satisfied = method_results.get("all_constraints_satisfied", False)
    print(f"\nConstraint Satisfaction: {all_satisfied}")
    for constraint_name, (satisfied, actual, max_val) in constraints.items():
        status = "[PASS]" if satisfied else "[FAIL]"
        print(f"  {status} {constraint_name}: {actual:.2e} / {max_val:.2e}")

    method_dist = method_results.get("method_distribution", {})
    method_diversity = method_results.get("method_diversity", 0)
    print(f"\nMethod Diversity: {method_diversity} different methods")
    print(f"  Distribution: {method_dist}")
    if method_diversity >= 2:
        print("  [PASS] Good diversity (>= 2 methods)")
    else:
        print("  [WARN] Low diversity (< 2 methods)")

    # Validation 3: Rank Estimation
    print("\n" + "=" * 60)
    print("3. RANK ESTIMATION VALIDATION")
    print("=" * 60)

    rank_validator = RankValidator(model_spec)
    rank_results = rank_validator.validate_all(
        allocations,
        utility_scores,
        max_trainable_params=50000,
    )

    corr = rank_results.get("rank_utility_correlation", 0.0)
    p_val = rank_results.get("rank_utility_p_value", 1.0)
    print(f"\nRank-Utility Correlation: {corr:.4f} (p={p_val:.4f})")
    if corr >= 0.70:
        print("  [PASS] Good correlation (rho >= 0.70)")
    elif corr >= 0.60:
        print("  [WARN] Moderate correlation (0.60 <= rho < 0.70)")
    elif corr >= 0.40:
        print("  [WARN] Weak correlation (0.40 <= rho < 0.60)")
    else:
        print("  [FAIL] Very weak correlation (rho < 0.40)")

    utilization = rank_results.get("budget_utilization", 0.0)
    print(f"\nBudget Utilization: {utilization:.2%}")
    if 0.80 <= utilization <= 0.88:
        print("  [PASS] Optimal utilization (80-88%)")
    elif 0.70 <= utilization < 0.80:
        print("  [WARN] Acceptable utilization (70-80%)")
    elif utilization < 0.70:
        print("  [WARN] Low utilization (< 70%)")
    else:
        print("  [WARN] High utilization (> 88%)")

    consistency = rank_results.get("rank_utility_consistency", 0.0)
    print(f"\nRank-Utility Consistency: {consistency:.2%}")
    if consistency >= 0.85:
        print("  [PASS] High consistency (>= 85%)")
    elif consistency >= 0.80:
        print("  [WARN] Good consistency (80-85%)")
    else:
        print("  [WARN] Low consistency (< 80%)")

    range_reasonable = rank_results.get("rank_range_reasonable", False)
    min_rank = rank_results.get("min_rank", 0)
    max_rank = rank_results.get("max_rank", 0)
    avg_rank = rank_results.get("avg_rank", 0.0)
    print(f"\nRank Range: {min_rank} - {max_rank} (avg: {avg_rank:.1f})")
    if range_reasonable:
        print("  [PASS] Reasonable range")
    else:
        print("  [WARN] Range may be unreasonable")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = (
        cv >= 0.2 and
        budget_ok and
        all_satisfied and
        method_diversity >= 2 and
        corr >= 0.6 and
        0.7 <= utilization <= 0.9
    )

    if all_passed:
        print("\n[SUCCESS] ALL VALIDATION CHECKS PASSED")
    else:
        print("\n[WARNING] SOME VALIDATION CHECKS NEED ATTENTION")
        print("\nRecommendations:")
        if cv < 0.2:
            print("  - Improve utility score variance (increase differentiation)")
        if not budget_ok:
            print("  - Adjust rank estimation to respect budget")
        if not all_satisfied:
            print("  - Review constraint handling")
        if method_diversity < 2:
            print("  - Review method selection thresholds")
        if corr < 0.6:
            print("  - Improve rank-utility correlation")
        if not (0.7 <= utilization <= 0.9):
            print("  - Optimize budget utilization")


if __name__ == "__main__":
    validate_bert_results()

