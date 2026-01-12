"""
Rank Estimation Validation
==========================
Validates rank estimation by measuring correlation with utility and budget utilization.
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.stats import spearmanr

from ..models import ModelSpec, LayerSpec
from ..optimization import AllocationResult


class RankValidator:
    """
    Validates rank estimation by comparing with utility scores and checking budget utilization.
    """

    def __init__(self, model_spec: ModelSpec):
        self.model_spec = model_spec

    def validate_rank_utility_correlation(
        self,
        allocations: List[AllocationResult],
        utility_scores: Dict[str, float],
    ) -> Tuple[float, float]:
        """
        Calculates Spearman correlation between ranks and utility scores.
        Excludes "none" method layers (rank=0) to get better correlation measure.

        Returns:
            (correlation, p_value): Spearman correlation coefficient and p-value
        """
        ranks = []
        utils = []

        for alloc in allocations:
            if alloc.layer_name in utility_scores and alloc.method != "none":
                # Only include layers with actual PEFT methods (rank > 0)
                ranks.append(alloc.rank)
                utils.append(utility_scores[alloc.layer_name])

        if len(ranks) < 2:
            return 0.0, 1.0

        correlation, p_value = spearmanr(ranks, utils)
        return float(correlation), float(p_value)

    def calculate_budget_utilization(
        self,
        allocations: List[AllocationResult],
        max_trainable_params: int,
    ) -> Tuple[float, int, int]:
        """
        Calculates budget utilization percentage.

        Returns:
            (utilization_rate, total_params, max_params): Utilization and values
        """
        # Estimate total trainable parameters
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
                "lora": 2.0,
                "adapter": 1.5,
                "prefix": 1.0,
            }
            factor = method_factors.get(alloc.method, 2.0)
            layer_params = int(alloc.rank * layer.hidden_size * factor)
            total_params += layer_params

        utilization = total_params / max_trainable_params if max_trainable_params > 0 else 0.0
        return float(utilization), total_params, max_trainable_params

    def validate_rank_distribution(
        self,
        allocations: List[AllocationResult],
        utility_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Validates that rank distribution follows utility distribution.

        Returns:
            Dictionary of validation metrics
        """
        # Get ranks and utilities in same order
        layer_ranks = {}
        layer_utils = {}

        for alloc in allocations:
            if alloc.layer_name in utility_scores:
                layer_ranks[alloc.layer_name] = alloc.rank
                layer_utils[alloc.layer_name] = utility_scores[alloc.layer_name]

        if len(layer_ranks) < 2:
            return {
                "rank_increases_with_utility": False,
                "rank_utility_consistency": 0.0,
            }

        # Check if rank increases with utility
        sorted_by_utility = sorted(
            layer_utils.items(),
            key=lambda x: x[1],
            reverse=True
        )

        ranks_ordered = [layer_ranks[name] for name, _ in sorted_by_utility]
        is_increasing = all(
            ranks_ordered[i] >= ranks_ordered[i + 1]
            for i in range(len(ranks_ordered) - 1)
        )

        # Calculate consistency (how often higher utility = higher rank)
        consistent = 0
        total = 0
        for i in range(len(sorted_by_utility)):
            for j in range(i + 1, len(sorted_by_utility)):
                name_i, util_i = sorted_by_utility[i]
                name_j, util_j = sorted_by_utility[j]
                if util_i > util_j:
                    if layer_ranks[name_i] >= layer_ranks[name_j]:
                        consistent += 1
                    total += 1

        consistency_rate = consistent / total if total > 0 else 0.0

        return {
            "rank_increases_with_utility": is_increasing,
            "rank_utility_consistency": float(consistency_rate),
        }

    def validate_rank_range(
        self,
        allocations: List[AllocationResult],
    ) -> Dict[str, any]:
        """
        Validates that ranks are in reasonable range.

        Returns:
            Dictionary of validation metrics
        """
        ranks = []
        hidden_sizes = []

        for alloc in allocations:
            layer = next(
                (l for l in self.model_spec.layers if l.name == alloc.layer_name),
                None
            )
            if layer is None:
                continue

            ranks.append(alloc.rank)
            hidden_sizes.append(layer.hidden_size)

        if len(ranks) == 0:
            return {
                "min_rank": 0,
                "max_rank": 0,
                "avg_rank": 0.0,
                "rank_range_reasonable": False,
            }

        min_rank = min(ranks)
        max_rank = max(ranks)
        avg_rank = np.mean(ranks)
        max_hidden = max(hidden_sizes) if hidden_sizes else 768

        # Reasonable range: 1 to hidden_size/10
        reasonable_max = max_hidden / 10
        is_reasonable = min_rank >= 1 and max_rank <= reasonable_max

        return {
            "min_rank": int(min_rank),
            "max_rank": int(max_rank),
            "avg_rank": float(avg_rank),
            "rank_range_reasonable": is_reasonable,
            "reasonable_max": int(reasonable_max),
        }

    def validate_all(
        self,
        allocations: List[AllocationResult],
        utility_scores: Dict[str, float],
        max_trainable_params: int,
    ) -> Dict[str, any]:
        """
        Runs all validation metrics and returns results.

        Returns:
            Dictionary of metric names -> values
        """
        results = {}

        # Rank-utility correlation
        corr, p_value = self.validate_rank_utility_correlation(
            allocations, utility_scores
        )
        results["rank_utility_correlation"] = corr
        results["rank_utility_p_value"] = p_value

        # Budget utilization
        utilization, total_params, max_params = self.calculate_budget_utilization(
            allocations, max_trainable_params
        )
        results["budget_utilization"] = utilization
        results["total_params"] = total_params
        results["max_params"] = max_params

        # Rank distribution
        dist_results = self.validate_rank_distribution(allocations, utility_scores)
        results.update(dist_results)

        # Rank range
        range_results = self.validate_rank_range(allocations)
        results.update(range_results)

        return results

