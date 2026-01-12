"""
results_validator.py
--------------------
Validates LayerLens results against academic standards and goals.
Contains the core validation logic that can be used programmatically.
"""

import json
from pathlib import Path
from typing import Tuple, Dict, List

from ..optimization import AllocationResult
from ..meta import ModelSpec, LayerSpec
from .utility_validator import UtilityValidator
from .method_validator import MethodValidator
from .rank_validator import RankValidator


def load_manifest_results(manifest_path: Path) -> Tuple[List[AllocationResult], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Load results from manifest file.
    
    Args:
        manifest_path: Path to manifest JSON file
        
    Returns:
        Tuple of (allocations, utility_scores, gradient_norms, fisher_traces)
    """
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    allocations = []
    utility_scores = {}
    gradient_norms = {}
    fisher_traces = {}

    for alloc_data in manifest.get("allocations", []):
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


def validate_manifest(
    manifest_path: Path,
    model_spec: ModelSpec,
    max_trainable_params: int = 50000,
    max_flops: float = 1e9,
    max_vram_gb: float = 8.0,
    latency_target_ms: float = 100.0,
    verbose: bool = True,
) -> Dict:
    """
    Validate a manifest file against academic standards and goals.
    
    Args:
        manifest_path: Path to manifest JSON file
        model_spec: Model specification
        max_trainable_params: Maximum trainable parameters constraint
        max_flops: Maximum FLOPs constraint
        max_vram_gb: Maximum VRAM constraint
        latency_target_ms: Latency target in milliseconds
        verbose: Whether to print validation results
        
    Returns:
        Dictionary containing all validation results
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    # Load results
    allocations, utility_scores, gradient_norms, fisher_traces = load_manifest_results(
        manifest_path
    )

    if verbose:
        print("=" * 60)
        print("LayerLens Results Validation")
        print("=" * 60)
        print(f"\nLoaded {len(allocations)} allocations from manifest")
        print(f"Utility score range: {min(utility_scores.values()):.4f} - {max(utility_scores.values()):.4f}")

    results = {}

    # Validation 1: Utility Scores
    if verbose:
        print("\n" + "=" * 60)
        print("1. UTILITY SCORE VALIDATION")
        print("=" * 60)

    utility_validator = UtilityValidator(model_spec)
    utility_results = utility_validator.validate_all(
        utility_scores,
        gradient_norms=gradient_norms if gradient_norms else None,
        fisher_traces=fisher_traces if fisher_traces else None,
    )
    results["utility"] = utility_results

    if verbose:
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

    # Validation 2: Method Selection
    if verbose:
        print("\n" + "=" * 60)
        print("2. METHOD SELECTION VALIDATION")
        print("=" * 60)

    method_validator = MethodValidator(model_spec)
    method_results = method_validator.validate_all(
        allocations,
        max_trainable_params=max_trainable_params,
        max_flops=max_flops,
        max_vram_gb=max_vram_gb,
        latency_target_ms=latency_target_ms,
    )
    results["method"] = method_results

    if verbose:
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

    # Validation 3: Rank Estimation
    if verbose:
        print("\n" + "=" * 60)
        print("3. RANK ESTIMATION VALIDATION")
        print("=" * 60)

    rank_validator = RankValidator(model_spec)
    rank_results = rank_validator.validate_all(
        allocations,
        utility_scores,
        max_trainable_params=max_trainable_params,
    )
    results["rank"] = rank_results

    if verbose:
        corr = rank_results.get("rank_utility_correlation", 0.0)
        p_val = rank_results.get("rank_utility_p_value", 1.0)
        print(f"\nRank-Utility Correlation: {corr:.4f} (p={p_val:.4f})")
        if corr >= 0.70:
            print("  [PASS] Good correlation (rho >= 0.70)")
        elif corr >= 0.60:
            print("  [WARN] Moderate correlation (0.60 <= rho < 0.70)")
        else:
            print("  [WARN] Weak correlation (rho < 0.60)")

        utilization = rank_results.get("budget_utilization", 0.0)
        print(f"\nBudget Utilization: {utilization:.2%}")

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        all_passed = (
            utility_results.get("coefficient_of_variation", 0.0) >= 0.2 and
            method_results.get("budget_compliant", False) and
            method_results.get("all_constraints_satisfied", False) and
            method_results.get("method_diversity", 0) >= 2 and
            rank_results.get("rank_utility_correlation", 0.0) >= 0.6 and
            0.7 <= rank_results.get("budget_utilization", 0.0) <= 0.9
        )

        if all_passed:
            print("\n[SUCCESS] ALL VALIDATION CHECKS PASSED")
        else:
            print("\n[WARNING] SOME VALIDATION CHECKS NEED ATTENTION")

    results["summary"] = {
        "all_passed": all_passed if verbose else None,
        "utility_cv": utility_results.get("coefficient_of_variation", 0.0),
        "budget_compliant": method_results.get("budget_compliant", False),
        "rank_correlation": rank_results.get("rank_utility_correlation", 0.0),
    }

    return results

