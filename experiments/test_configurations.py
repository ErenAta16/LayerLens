"""
LayerLens Configuration Tests
==============================
Tests LayerLens with different configurations to identify issues and validate behavior.
"""

from pathlib import Path
import json
import numpy as np
from layerlens.cli import run_pipeline
from layerlens.config import ProfilingConfig, OptimizationConfig
from layerlens.models import ModelSpec, LayerSpec


def create_test_model_spec(num_layers: int = 12, hidden_size: int = 768) -> ModelSpec:
    """Create a test model specification."""
    layers = []
    for i in range(num_layers):
        layers.append(LayerSpec(
            name=f"encoder.layer.{i}",
            hidden_size=hidden_size,
            layer_type="transformer",
            supports_attention=True,
            metadata={"layer_index": i}
        ))
    
    total_params = num_layers * hidden_size * 1000  # Rough estimate
    return ModelSpec(
        model_name="test-model",
        total_params=total_params,
        layers=layers
    )


def create_realistic_activation_cache(model_spec: ModelSpec, variation: str = "uniform") -> dict:
    """
    Create realistic activation cache with different variation patterns.
    
    variation options:
    - "uniform": All layers have similar sensitivity
    - "early": Early layers more sensitive
    - "late": Late layers more sensitive
    - "middle": Middle layers more sensitive
    - "random": Random variation
    """
    activation_cache = {}
    num_layers = len(model_spec.layers)
    
    for i, layer in enumerate(model_spec.layers):
        if variation == "uniform":
            base_grad = 0.5
            base_fisher = 0.3
            base_proxy = 0.2
        elif variation == "early":
            # Early layers (0-3) more sensitive
            factor = 1.0 if i < 4 else 0.3
            base_grad = 0.8 * factor
            base_fisher = 0.5 * factor
            base_proxy = 0.3 * factor
        elif variation == "late":
            # Late layers (8-11) more sensitive
            factor = 1.0 if i >= 8 else 0.3
            base_grad = 0.8 * factor
            base_fisher = 0.5 * factor
            base_proxy = 0.3 * factor
        elif variation == "middle":
            # Middle layers (4-7) more sensitive
            factor = 1.0 if 4 <= i <= 7 else 0.3
            base_grad = 0.8 * factor
            base_fisher = 0.5 * factor
            base_proxy = 0.3 * factor
        elif variation == "random":
            base_grad = np.random.random() * 0.8 + 0.1
            base_fisher = np.random.random() * 0.5 + 0.1
            base_proxy = np.random.random() * 0.3 + 0.1
        else:
            base_grad = 0.5
            base_fisher = 0.3
            base_proxy = 0.2
        
        activation_cache[layer.name] = {
            "grad_norm": base_grad,
            "fisher_trace": base_fisher,
            "proxy_gain": base_proxy,
        }
    
    return activation_cache


def run_configuration_test(
    test_name: str,
    model_spec: ModelSpec,
    profiling_cfg: ProfilingConfig,
    optimization_cfg: OptimizationConfig,
    activation_cache: dict,
    output_dir: Path,
) -> dict:
    """Run a single configuration test and return results."""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")
    
    try:
        manifest_path = run_pipeline(
            model_spec=model_spec,
            profiling_cfg=profiling_cfg,
            optimization_cfg=optimization_cfg,
            activation_cache=activation_cache,
            output_dir=output_dir,
        )
        
        # Read and analyze results
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        allocations = manifest.get("allocations", [])
        
        # Analyze results
        method_counts = {}
        rank_sum = 0
        utility_sum = 0.0
        utility_nonzero = 0
        
        for alloc in allocations:
            method = alloc.get("method", "none")
            rank = alloc.get("rank", 0)
            utility = alloc.get("utility", 0.0)
            
            method_counts[method] = method_counts.get(method, 0) + 1
            rank_sum += rank
            utility_sum += utility
            if utility > 0:
                utility_nonzero += 1
        
        results = {
            "test_name": test_name,
            "status": "success",
            "total_layers": len(allocations),
            "method_distribution": method_counts,
            "total_rank": rank_sum,
            "avg_utility": utility_sum / len(allocations) if allocations else 0.0,
            "utility_nonzero_count": utility_nonzero,
            "manifest_path": str(manifest_path),
        }
        
        print(f"  Status: Success")
        print(f"  Layers: {results['total_layers']}")
        print(f"  Methods: {method_counts}")
        print(f"  Total Rank: {rank_sum}")
        print(f"  Avg Utility: {results['avg_utility']:.4f}")
        print(f"  Non-zero Utility: {utility_nonzero}/{len(allocations)}")
        
        return results
        
    except Exception as e:
        print(f"  Status: Failed")
        print(f"  Error: {e}")
        return {
            "test_name": test_name,
            "status": "failed",
            "error": str(e),
        }


def main():
    """Run multiple configuration tests."""
    print("="*60)
    print("LayerLens Configuration Tests")
    print("="*60)
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Base model specification
    model_spec = create_test_model_spec(num_layers=12, hidden_size=768)
    
    all_results = []
    
    # Test 1: Default configuration with uniform sensitivity
    print("\n[Test 1] Default config, uniform sensitivity")
    profiling_cfg = ProfilingConfig()
    optimization_cfg = OptimizationConfig(
        max_trainable_params=50000,
        max_flops=1e9,
        max_vram_gb=8.0,
        latency_target_ms=100.0,
    )
    activation_cache = create_realistic_activation_cache(model_spec, "uniform")
    result1 = run_configuration_test(
        "Default-Uniform",
        model_spec, profiling_cfg, optimization_cfg, activation_cache, output_dir
    )
    all_results.append(result1)
    
    # Test 2: Higher parameter budget
    print("\n[Test 2] Higher parameter budget")
    optimization_cfg2 = OptimizationConfig(
        max_trainable_params=200000,  # 4x more
        max_flops=5e9,
        max_vram_gb=16.0,
        latency_target_ms=200.0,
    )
    result2 = run_configuration_test(
        "High-Budget-Uniform",
        model_spec, profiling_cfg, optimization_cfg2, activation_cache, output_dir
    )
    all_results.append(result2)
    
    # Test 3: Early layers more sensitive
    print("\n[Test 3] Early layers more sensitive")
    activation_cache3 = create_realistic_activation_cache(model_spec, "early")
    result3 = run_configuration_test(
        "Early-Sensitive",
        model_spec, profiling_cfg, optimization_cfg, activation_cache3, output_dir
    )
    all_results.append(result3)
    
    # Test 4: Late layers more sensitive
    print("\n[Test 4] Late layers more sensitive")
    activation_cache4 = create_realistic_activation_cache(model_spec, "late")
    result4 = run_configuration_test(
        "Late-Sensitive",
        model_spec, profiling_cfg, optimization_cfg, activation_cache4, output_dir
    )
    all_results.append(result4)
    
    # Test 5: Different method penalties
    print("\n[Test 5] Different method penalties (LoRA cheaper)")
    optimization_cfg5 = OptimizationConfig(
        max_trainable_params=50000,
        max_flops=1e9,
        max_vram_gb=8.0,
        latency_target_ms=100.0,
        method_penalties={"lora": 0.8, "adapter": 1.5, "prefix": 1.3, "none": 0.5},
    )
    result5 = run_configuration_test(
        "LoRA-Cheap",
        model_spec, profiling_cfg, optimization_cfg5, activation_cache, output_dir
    )
    all_results.append(result5)
    
    # Test 6: Different objective weights (utility-focused)
    print("\n[Test 6] Utility-focused objective weights")
    optimization_cfg6 = OptimizationConfig(
        max_trainable_params=50000,
        max_flops=1e9,
        max_vram_gb=8.0,
        latency_target_ms=100.0,
        objective_weights={
            "utility": 0.8,  # Higher weight on utility
            "cost": 0.1,
            "flop": 0.05,
            "vram": 0.03,
            "latency": 0.02,
        },
    )
    result6 = run_configuration_test(
        "Utility-Focused",
        model_spec, profiling_cfg, optimization_cfg6, activation_cache3, output_dir
    )
    all_results.append(result6)
    
    # Test 7: Random sensitivity pattern
    print("\n[Test 7] Random sensitivity pattern")
    activation_cache7 = create_realistic_activation_cache(model_spec, "random")
    result7 = run_configuration_test(
        "Random-Sensitivity",
        model_spec, profiling_cfg, optimization_cfg, activation_cache7, output_dir
    )
    all_results.append(result7)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    successful_tests = [r for r in all_results if r.get("status") == "success"]
    failed_tests = [r for r in all_results if r.get("status") == "failed"]
    
    print(f"\nTotal tests: {len(all_results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    
    if successful_tests:
        print("\nSuccessful Test Results:")
        for result in successful_tests:
            print(f"\n  {result['test_name']}:")
            print(f"    Methods: {result.get('method_distribution', {})}")
            print(f"    Total Rank: {result.get('total_rank', 0)}")
            print(f"    Avg Utility: {result.get('avg_utility', 0.0):.4f}")
            print(f"    Non-zero Utility: {result.get('utility_nonzero_count', 0)}/{result.get('total_layers', 0)}")
    
    if failed_tests:
        print("\nFailed Tests:")
        for result in failed_tests:
            print(f"  {result['test_name']}: {result.get('error', 'Unknown error')}")
    
    # Identify issues
    print("\n" + "="*60)
    print("IDENTIFIED ISSUES")
    print("="*60)
    
    issues = []
    
    # Check for zero utility issue
    zero_utility_tests = [r for r in successful_tests if r.get("avg_utility", 1.0) == 0.0]
    if zero_utility_tests:
        issues.append({
            "severity": "HIGH",
            "issue": "All utility scores are zero",
            "affected_tests": [r["test_name"] for r in zero_utility_tests],
            "description": "Utility scores are calculated as 0.0 for all layers, indicating a problem with metric aggregation or normalization."
        })
    
    # Check for uniform method selection
    uniform_method_tests = []
    for r in successful_tests:
        method_dist = r.get("method_distribution", {})
        if len(method_dist) == 1:
            uniform_method_tests.append(r["test_name"])
    
    if uniform_method_tests:
        issues.append({
            "severity": "MEDIUM",
            "issue": "Uniform method selection across all layers",
            "affected_tests": uniform_method_tests,
            "description": "All layers are assigned the same PEFT method, suggesting utility scores are not differentiating layers properly."
        })
    
    # Check for uniform rank
    uniform_rank_tests = []
    for r in successful_tests:
        total_rank = r.get("total_rank", 0)
        total_layers = r.get("total_layers", 1)
        avg_rank = total_rank / total_layers if total_layers > 0 else 0
        if avg_rank <= 1.1:  # All layers have rank 1
            uniform_rank_tests.append(r["test_name"])
    
    if uniform_rank_tests:
        issues.append({
            "severity": "MEDIUM",
            "issue": "Uniform rank assignment (all layers get rank=1)",
            "affected_tests": uniform_rank_tests,
            "description": "All layers are assigned rank=1, indicating the rank estimation function is not working correctly or utility scores are too low."
        })
    
    # Print issues
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"\n{i}. [{issue['severity']}] {issue['issue']}")
            print(f"   Affected tests: {', '.join(issue['affected_tests'])}")
            print(f"   Description: {issue['description']}")
    else:
        print("\nNo major issues identified!")
    
    # Save results
    results_file = output_dir / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "summary": {
                "total_tests": len(all_results),
                "successful": len(successful_tests),
                "failed": len(failed_tests),
            },
            "results": all_results,
            "issues": issues,
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    print("="*60)


if __name__ == "__main__":
    main()

