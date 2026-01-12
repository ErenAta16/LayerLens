"""
solver.py
---------
Reference solver that performs method and rank selection on a per-layer basis.
Cython-accelerated optimization will be added in the real implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
from array import array

from typing import Optional
from ..config import OptimizationConfig, ProfilingConfig, LatencyProfile
from ..models import LayerSpec
from ..profiling import aggregate_scores

# Cython modules are optional - fallback to Python if not compiled
try:
    from ._solver import select_method as select_method_c
    from ._solver import estimate_rank as estimate_rank_c
    from ._solver import estimate_cost as estimate_cost_c
except ImportError:
    # Fallback: define Python implementations
    select_method_c = None
    estimate_rank_c = None
    estimate_cost_c = None


@dataclass
class AllocationResult:
    """
    Data class representing optimization output.
    """

    layer_name: str
    method: str
    rank: int
    utility_score: float
    cost_estimate: float
    flop_estimate: float
    vram_estimate: float
    latency_estimate: float
    total_score: float


class AllocationSolver:
    """
    Provides fast allocation using greedy + constraint checking.
    Uses Cython-accelerated helper functions.
    """
    
    # PERFORMANCE FIX: Class-level constant to avoid repeated dict creation
    _METHOD_FACTORS = {"lora": 2.0, "adapter": 1.5, "prefix": 1.0, "none": 0.0}

    def __init__(self, config: OptimizationConfig, profiling_config: Optional[ProfilingConfig] = None) -> None:
        self.config = config
        self.profiling_config = profiling_config
        self._method_sequence = config.candidate_methods
        self._method_thresholds = self._build_thresholds(len(self._method_sequence))
        self._method_codes = self._encode_methods(len(self._method_sequence))
        # Optimization: Dictionary lookup cache
        self._penalty_cache: Dict[str, float] = {}

    def solve(
        self,
        layers: List[LayerSpec],
        sensitivity_scores: Dict[str, Dict[str, float]],
    ) -> List[AllocationResult]:
        """
        Returns candidate method and rank assignment for each layer.
        The sensitivity_scores dictionary contains layer name -> metric scores.
        """

        results: List[AllocationResult] = []
        
        # Step 1: Calculate utilities and initial rank estimates
        layer_utilities = []
        layer_data = []
        
        for layer in layers:
            metrics = sensitivity_scores.get(layer.name, {})
            
            # Check if normalized utility is already computed (from pipeline)
            if "_normalized_utility" in metrics:
                utility = metrics["_normalized_utility"]
            else:
                # Use metric_weights from ProfilingConfig if available, otherwise fallback
                if self.profiling_config is not None:
                    weights = self.profiling_config.metric_weights
                else:
                    # Fallback: create weights from metrics (equal weights)
                    weights = {key: 1.0 / len(metrics) for key in metrics.keys()} if metrics else {}
                raw_utility = aggregate_scores(metrics, weights)
                # Normalize utility to 0-0.1 range for method selection
                # Simple normalization: divide by 1000 to bring large values down
                utility = min(raw_utility / 1000.0, 0.1) if raw_utility > 0 else 0.0
            
            method = self._select_method(utility)
            # Use budget-aware rank estimation
            initial_rank = self._estimate_rank_budget_aware(layer, utility)
            
            layer_utilities.append(utility)
            layer_data.append({
                "layer": layer,
                "utility": utility,
                "method": method,
                "initial_rank": initial_rank,
            })
        
        # Step 2: Scale ranks to better utilize budget (target: 70-90% utilization)
        # Preserve rank-utility correlation by using uniform scaling
        # Calculate total estimated params with initial ranks
        # PERFORMANCE FIX: Use class-level constant instead of creating dict each time
        total_estimated_params = 0
        for data in layer_data:
            if data["method"] == "none":
                continue
            factor = self._METHOD_FACTORS.get(data["method"], 2.0)
            layer_params = int(data["initial_rank"] * data["layer"].hidden_size * factor)
            total_estimated_params += layer_params
        
        # Calculate target utilization (elevated: 80-88%, target 84%)
        # Use adaptive target based on initial utilization
        initial_utilization = total_estimated_params / self.config.max_trainable_params if self.config.max_trainable_params > 0 else 0.0
        if initial_utilization < 0.3:
            # Very low utilization - target 82% (middle of 80-84%)
            target_utilization = 0.82
        elif initial_utilization < 0.5:
            # Low utilization - target 84% (middle of 80-88%)
            target_utilization = 0.84
        elif initial_utilization < 0.7:
            # Moderate utilization - target 85% (middle of 82-88%)
            target_utilization = 0.85
        else:
            # Already high utilization - target 86% (middle of 84-88%)
            target_utilization = 0.86
        
        target_params = int(self.config.max_trainable_params * target_utilization)
        
        # Use utility-weighted scaling instead of uniform scaling
        # This preserves rank-utility correlation better
        if total_estimated_params > 0 and total_estimated_params < target_params:
            # Calculate utility-weighted scale factors
            # Higher utility layers get proportionally more scaling
            total_utility = sum(data["utility"] for data in layer_data if data["method"] != "none")
            
            if total_utility > 0:
                # Utility-weighted scaling: each layer's scale factor is proportional to its utility
                for data in layer_data:
                    if data["method"] == "none":
                        data["scale_factor"] = 1.0
                    else:
                        # Base scale factor from budget
                        base_scale = target_params / total_estimated_params
                        # Utility weight: higher utility gets more scaling
                        utility_weight = data["utility"] / total_utility if total_utility > 0 else 1.0
                        # Scale factor: base * (1 + utility_weight * factor) to favor high utility
                        # Reduced factor to 0.2 to keep utilization in 80-88% range
                        # This preserves correlation while staying within target
                        data["scale_factor"] = min(base_scale * (1.0 + utility_weight * 0.2), 2.0)
            else:
                # Fallback to uniform scaling if no utility
                uniform_scale = min(target_params / total_estimated_params, 2.0)
                for data in layer_data:
                    data["scale_factor"] = uniform_scale
        else:
            # No scaling needed
            for data in layer_data:
                data["scale_factor"] = 1.0
        
        # Step 3: Apply scaled ranks with utility-weighted allocation
        # Sort by utility (descending) to allocate budget to high-utility layers first
        sorted_data = sorted(
            [d for d in layer_data if d["method"] != "none"],
            key=lambda x: x["utility"],
            reverse=True
        )
        none_layers = [d for d in layer_data if d["method"] == "none"]
        
        consumed_params = 0
        
        # Allocate budget to high-utility layers first
        for data in sorted_data:
            # Scale rank using utility-weighted scale factor
            scaled_rank = data["initial_rank"] * data["scale_factor"]
            
            # Ensure rank is within reasonable bounds
            min_rank = 1.0
            max_rank = min(data["layer"].hidden_size / 10, 64.0)  # Reasonable max
            rank = max(min_rank, min(scaled_rank, max_rank))
            rank = int(rank)
            
            # Check budget constraint
            # PERFORMANCE FIX: Use class-level constant
            factor = self._METHOD_FACTORS.get(data["method"], 2.0)
            layer_params = int(rank * data["layer"].hidden_size * factor)
            
            if consumed_params + layer_params > self.config.max_trainable_params:
                # Exceeds budget - reduce rank or set to none
                # Try to fit within budget by reducing rank
                max_rank_for_budget = int(
                    (self.config.max_trainable_params - consumed_params) / 
                    (data["layer"].hidden_size * factor)
                )
                if max_rank_for_budget >= 1:
                    rank = max_rank_for_budget
                    layer_params = int(rank * data["layer"].hidden_size * factor)
                else:
                    # Cannot fit even rank=1 - set to none
                    data["method"] = "none"
                    rank = 0
                    layer_params = 0
            
            if data["method"] != "none":
                consumed_params += layer_params
            
            data["final_rank"] = rank
            data["final_method"] = data["method"]
        
        # Process none layers
        for data in none_layers:
            data["final_rank"] = 0
            data["final_method"] = "none"
        
        # Step 4: Create results from all layers (maintain original order)
        for data in layer_data:
            
            # Use final rank and method from Step 3
            method = data.get("final_method", data["method"])
            rank = data.get("final_rank", 0)
            
            if method != "none" and rank > 0:
                cost = self._estimate_cost(rank, method)
                flop = self._estimate_flops(data["layer"], rank)
                vram = self._estimate_vram(rank)
                latency = self._estimate_latency(rank)
                total_cost = self._score_multi_objective(
                    cost=cost, flop=flop, vram=vram, latency=latency
                )
            else:
                method = "none"
                rank = 0
                cost = 0.0
                flop = 0.0
                vram = 0.0
                latency = 0.0
                total_cost = 0.0
            
            results.append(
                AllocationResult(
                    layer_name=data["layer"].name,
                    method=method,
                    rank=rank,
                    utility_score=data["utility"],
                    cost_estimate=cost,
                    flop_estimate=flop,
                    vram_estimate=vram,
                    latency_estimate=latency,
                    total_score=total_cost,
                )
            )

        return results

    def _select_method(self, utility: float) -> str:
        thresholds = self._method_thresholds
        
        # Use Cython if available, otherwise Python fallback
        if select_method_c is not None:
            method_index = select_method_c(
                utility,
                thresholds,
                self._method_codes,
            )
        else:
            # Python fallback: find first threshold that utility exceeds
            method_index = 0
            for i, threshold in enumerate(thresholds):
                if utility >= threshold:
                    method_index = i + 1
                else:
                    break
        
        return self._method_sequence[method_index]

    def _estimate_rank(self, layer: LayerSpec, utility: float) -> float:
        # Use Cython if available, otherwise Python fallback
        if estimate_rank_c is not None:
            return estimate_rank_c(layer.hidden_size, utility)
        else:
            # Python fallback: simple heuristic
            # Base rank scales with utility and hidden size
            base_rank = max(1.0, utility * 10.0 * (layer.hidden_size / 768.0))
            return min(base_rank, layer.hidden_size / 10.0)
    
    def _estimate_rank_budget_aware(self, layer: LayerSpec, utility: float) -> float:
        """
        Budget-aware rank estimation that better utilizes available budget.
        Uses a more aggressive scaling factor based on utility.
        """
        # Base rank from Cython function or Python fallback
        if estimate_rank_c is not None:
            base_rank = estimate_rank_c(layer.hidden_size, utility)
        else:
            # Python fallback
            base_rank = max(1.0, utility * 10.0 * (layer.hidden_size / 768.0))
            base_rank = min(base_rank, layer.hidden_size / 10.0)
        
        # Scale based on utility to better utilize budget
        # Higher utility gets proportionally higher rank
        # Scale factor: utility * 10 (so utility=0.1 gives 1.0x, utility=0.05 gives 0.5x)
        utility_scale = max(utility * 10.0, 0.5)  # Minimum 0.5x, max 1.0x for utility=0.1
        
        # Apply scaling: more aggressive for higher utility
        scaled_rank = base_rank * utility_scale * 2.0  # Additional 2x multiplier for budget utilization
        
        # Ensure reasonable bounds
        min_rank = 1.0
        max_rank = min(layer.hidden_size / 10, 64.0)
        return max(min_rank, min(scaled_rank, max_rank))

    def _estimate_cost(self, rank: float, method: str) -> float:
        # Optimization: Dictionary lookup cache
        if method not in self._penalty_cache:
            self._penalty_cache[method] = self.config.method_penalties.get(method, 1.0)
        penalty = self._penalty_cache[method]
        
        # Use Cython if available, otherwise Python fallback
        if estimate_cost_c is not None:
            return estimate_cost_c(rank, penalty)
        else:
            # Python fallback: simple cost calculation
            return rank * penalty * 0.1

    def _estimate_flops(self, layer: LayerSpec, rank: float) -> float:
        """
        Simple FLOP estimate: rank * hidden_size * constant.
        """

        return rank * layer.hidden_size * 2.0

    def _estimate_vram(self, rank: float) -> float:
        """
        VRAM estimate (MB): rank * constant coefficient.
        """

        return rank * 0.001

    def _estimate_latency(self, rank: float) -> float:
        """
        Latency estimate (ms).

        This uses a more realistic parametric model when a LatencyProfile is
        provided, otherwise it falls back to the original simple heuristic.
        """

        profile: Optional[LatencyProfile] = getattr(self.config, "latency_profile", None)
        if profile is None:
            # Backwards-compatible fallback: simple heuristic
            return rank * 0.01

        # Base per-layer latency in ms: base + rank * slope
        base = max(profile.base_ms_per_layer, 0.0)
        slope = max(profile.ms_per_rank_unit, 0.0)
        latency = base + slope * max(rank, 0.0)

        # Device factor: GPUs are typically faster per layer, CPUs slower.
        # PERFORMANCE FIX: Cache lower() result to avoid repeated string operations
        device_type_lower = profile.device_type.lower()
        if device_type_lower == "gpu":
            device_factor = 0.7
        else:
            # Default to CPU-like behaviour
            device_factor = 1.8

        # Workload scaling:
        # - For LLMs, latency roughly scales with sequence length and batch size.
        # - For vision models (YOLO / ViT), it scales with resolution^2 and batch.
        model_family = profile.model_family.lower()
        batch_factor = max(profile.batch_size / 8.0, 0.25)

        if model_family == "llm":
            seq_factor = max(profile.sequence_length / 2048.0, 0.25)
            workload_factor = batch_factor * seq_factor
        else:
            # Vision-style default (YOLO, ViT, etc.)
            baseline_res = 640.0
            res = float(profile.input_resolution or baseline_res)
            res_factor = max((res * res) / (baseline_res * baseline_res), 0.25)
            workload_factor = batch_factor * res_factor

        latency *= device_factor * workload_factor

        # Add any fixed I/O or orchestration overhead (e.g. inter-model hops).
        latency += max(profile.io_overhead_ms, 0.0)

        return latency

    def _score_multi_objective(
        self, cost: float, flop: float, vram: float, latency: float
    ) -> float:
        """
        Computes multi-criteria cost score.
        """

        weights = self.config.objective_weights
        # Utility weight is already in aggregate_scores. Use cost weights here.
        flop_w = weights.get("flop", 0.1)
        vram_w = weights.get("vram", 0.1)
        latency_w = weights.get("latency", 0.1)
        cost_w = weights.get("cost", 0.3)

        return (
            cost * cost_w
            + flop * flop_w
            + vram * vram_w
            + latency * latency_w
        )

    @staticmethod
    def _encode_methods(count: int):
        """
        Encodes methods as indices. Used as int on the C side.
        """

        return array("d", [float(i) for i in range(count)])

    @staticmethod
    def _build_thresholds(count: int):
        """
        Creates a dynamic threshold list based on the number of methods.
        Thresholds are scaled to work with typical utility ranges (0.0-0.1).
        For 4 methods: [0.025, 0.05, 0.075] instead of [0.25, 0.5, 0.75]
        """

        if count <= 1:
            return array("d", [])

        # Scale thresholds to work with utility range ~0.0-0.1
        # Original: step = 1.0 / count, values = [0.25, 0.5, 0.75] for 4 methods
        # Scaled: step = 0.1 / count, values = [0.025, 0.05, 0.075] for 4 methods
        step = 0.1 / count  # Scale to 0.1 range instead of 1.0
        values = [step * (i + 1) for i in range(count - 1)]
        return array("d", values)

