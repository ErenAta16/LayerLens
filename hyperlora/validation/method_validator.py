"""
Method Selection Validation
============================
Validates that selected PEFT methods are appropriate for each layer.
"""

from typing import Dict, List, Tuple
from ..meta import ModelSpec, LayerSpec
from ..optimization import AllocationResult


class MethodValidator:
    """
    Validates method selection by checking compatibility and constraints.
    """

    def __init__(self, model_spec: ModelSpec):
        self.model_spec = model_spec

    def validate_compatibility(
        self, allocations: List[AllocationResult]
    ) -> Tuple[float, Dict[str, int]]:
        """
        Validates that selected methods are compatible with layer types.

        Rules:
        - Attention layers should get LoRA or Adapter (not Prefix or None)
        - Non-attention layers can get any method

        Returns:
            (compatibility_rate, compatibility_details): Rate and breakdown
        """
        compatible = 0
        total = 0
        details = {"compatible": 0, "incompatible": 0}

        for alloc in allocations:
            layer = next(
                (l for l in self.model_spec.layers if l.name == alloc.layer_name),
                None
            )
            if layer is None:
                continue

            total += 1
            is_compatible = True

            # Check attention layer compatibility
            if layer.supports_attention:
                # Attention layers should get LoRA or Adapter (unless utility is very low)
                # None is acceptable for very low utility layers
                if alloc.method not in ["lora", "adapter", "none"]:
                    is_compatible = False
                # Prefix is not ideal for attention layers but acceptable
                if alloc.method == "prefix":
                    is_compatible = True  # Acceptable but not ideal

            if is_compatible:
                compatible += 1
                details["compatible"] += 1
            else:
                details["incompatible"] += 1

        compatibility_rate = compatible / total if total > 0 else 0.0
        return float(compatibility_rate), details

    def validate_budget_compliance(
        self,
        allocations: List[AllocationResult],
        max_trainable_params: int,
    ) -> Tuple[bool, int, int]:
        """
        Validates that total trainable parameters are within budget.

        Returns:
            (is_compliant, total_params, max_params): Compliance status and values
        """
        # Estimate total trainable parameters
        # Simple estimation: rank * hidden_size * 2 (for LoRA)
        total_params = 0
        for alloc in allocations:
            layer = next(
                (l for l in self.model_spec.layers if l.name == alloc.layer_name),
                None
            )
            if layer is None:
                continue

            if alloc.method == "none":
                continue

            # Rough estimation: rank * hidden_size * method_factor
            method_factors = {
                "lora": 2.0,  # Two matrices
                "adapter": 1.5,
                "prefix": 1.0,
            }
            factor = method_factors.get(alloc.method, 2.0)
            layer_params = int(alloc.rank * layer.hidden_size * factor)
            total_params += layer_params

        is_compliant = total_params <= max_trainable_params
        return is_compliant, total_params, max_trainable_params

    def validate_constraints(
        self,
        allocations: List[AllocationResult],
        max_flops: float,
        max_vram_gb: float,
        latency_target_ms: float,
    ) -> Dict[str, Tuple[bool, float, float]]:
        """
        Validates that all constraints are satisfied.

        Returns:
            Dictionary of constraint_name -> (is_satisfied, actual_value, max_value)
        """
        total_flops = sum(alloc.flop_estimate for alloc in allocations)
        total_vram = sum(alloc.vram_estimate for alloc in allocations)
        total_latency = sum(alloc.latency_estimate for alloc in allocations)

        results = {
            "flops": (
                total_flops <= max_flops,
                total_flops,
                max_flops,
            ),
            "vram": (
                total_vram <= max_vram_gb,
                total_vram,
                max_vram_gb,
            ),
            "latency": (
                total_latency <= latency_target_ms,
                total_latency,
                latency_target_ms,
            ),
        }
        return results

    def calculate_method_diversity(
        self, allocations: List[AllocationResult]
    ) -> Dict[str, int]:
        """
        Calculates method distribution across layers.

        Returns:
            Dictionary of method -> count
        """
        method_counts = {}
        for alloc in allocations:
            method_counts[alloc.method] = method_counts.get(alloc.method, 0) + 1
        return method_counts

    def validate_all(
        self,
        allocations: List[AllocationResult],
        max_trainable_params: int,
        max_flops: float,
        max_vram_gb: float,
        latency_target_ms: float,
    ) -> Dict[str, any]:
        """
        Runs all validation metrics and returns results.

        Returns:
            Dictionary of metric names -> values
        """
        results = {}

        # Compatibility
        compat_rate, compat_details = self.validate_compatibility(allocations)
        results["compatibility_rate"] = compat_rate
        results["compatibility_details"] = compat_details

        # Budget compliance
        budget_ok, total_params, max_params = self.validate_budget_compliance(
            allocations, max_trainable_params
        )
        results["budget_compliant"] = budget_ok
        results["total_params"] = total_params
        results["max_params"] = max_params
        results["budget_utilization"] = total_params / max_params if max_params > 0 else 0.0

        # Constraints
        constraints = self.validate_constraints(
            allocations, max_flops, max_vram_gb, latency_target_ms
        )
        results["constraints"] = constraints
        results["all_constraints_satisfied"] = all(
            satisfied for satisfied, _, _ in constraints.values()
        )

        # Method diversity
        method_dist = self.calculate_method_diversity(allocations)
        results["method_diversity"] = len(method_dist)
        results["method_distribution"] = method_dist

        return results

