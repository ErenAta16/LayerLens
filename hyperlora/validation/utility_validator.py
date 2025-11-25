"""
Utility Score Validation
========================
Validates utility scores by measuring correlation with layer importance metrics.
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.stats import pearsonr, spearmanr

from ..meta import ModelSpec, LayerSpec


class UtilityValidator:
    """
    Validates utility scores by comparing with ground truth importance metrics.
    """

    def __init__(self, model_spec: ModelSpec):
        self.model_spec = model_spec

    def validate_correlation_with_gradients(
        self,
        utility_scores: Dict[str, float],
        gradient_norms: Dict[str, float],
    ) -> Tuple[float, float]:
        """
        Calculates Pearson correlation between utility scores and gradient norms.

        Returns:
            (correlation, p_value): Pearson correlation coefficient and p-value
        """
        utils = []
        grads = []

        for layer in self.model_spec.layers:
            if layer.name in utility_scores and layer.name in gradient_norms:
                utils.append(utility_scores[layer.name])
                grads.append(gradient_norms[layer.name])

        if len(utils) < 2:
            return 0.0, 1.0

        correlation, p_value = pearsonr(utils, grads)
        return float(correlation), float(p_value)

    def validate_correlation_with_fisher(
        self,
        utility_scores: Dict[str, float],
        fisher_traces: Dict[str, float],
    ) -> Tuple[float, float]:
        """
        Calculates Pearson correlation between utility scores and Fisher traces.

        Returns:
            (correlation, p_value): Pearson correlation coefficient and p-value
        """
        utils = []
        fishers = []

        for layer in self.model_spec.layers:
            if layer.name in utility_scores and layer.name in fisher_traces:
                utils.append(utility_scores[layer.name])
                fishers.append(fisher_traces[layer.name])

        if len(utils) < 2:
            return 0.0, 1.0

        correlation, p_value = pearsonr(utils, fishers)
        return float(correlation), float(p_value)

    def calculate_variance(self, utility_scores: Dict[str, float]) -> float:
        """
        Calculates coefficient of variation (std/mean) of utility scores.

        Returns:
            Coefficient of variation (CV)
        """
        scores = [utility_scores.get(layer.name, 0.0) for layer in self.model_spec.layers]
        scores = [s for s in scores if s > 0]  # Filter zeros

        if len(scores) == 0:
            return 0.0

        mean_score = np.mean(scores)
        if mean_score == 0:
            return 0.0

        std_score = np.std(scores)
        cv = std_score / mean_score
        return float(cv)

    def validate_ranking_accuracy(
        self,
        utility_scores: Dict[str, float],
        ground_truth_ranking: List[str],
        top_k: int = 3,
    ) -> float:
        """
        Calculates top-K ranking accuracy.

        Args:
            utility_scores: Dictionary of layer name -> utility score
            ground_truth_ranking: List of layer names in order of importance (most important first)
            top_k: Number of top layers to consider

        Returns:
            Accuracy: Percentage of top-K layers correctly identified
        """
        # Sort layers by utility score (descending)
        sorted_layers = sorted(
            utility_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        predicted_top_k = [name for name, _ in sorted_layers[:top_k]]

        # Get ground truth top-K
        true_top_k = ground_truth_ranking[:top_k]

        # Calculate accuracy
        correct = len(set(predicted_top_k) & set(true_top_k))
        accuracy = correct / top_k if top_k > 0 else 0.0
        return float(accuracy)

    def validate_all(
        self,
        utility_scores: Dict[str, float],
        gradient_norms: Dict[str, float] | None = None,
        fisher_traces: Dict[str, float] | None = None,
        ground_truth_ranking: List[str] | None = None,
    ) -> Dict[str, float]:
        """
        Runs all validation metrics and returns results.

        Returns:
            Dictionary of metric names -> values
        """
        results = {}

        # Variance
        cv = self.calculate_variance(utility_scores)
        results["coefficient_of_variation"] = cv

        # Correlation with gradients
        if gradient_norms:
            corr_grad, p_grad = self.validate_correlation_with_gradients(
                utility_scores, gradient_norms
            )
            results["gradient_correlation"] = corr_grad
            results["gradient_p_value"] = p_grad

        # Correlation with Fisher
        if fisher_traces:
            corr_fisher, p_fisher = self.validate_correlation_with_fisher(
                utility_scores, fisher_traces
            )
            results["fisher_correlation"] = corr_fisher
            results["fisher_p_value"] = p_fisher

        # Ranking accuracy
        if ground_truth_ranking:
            ranking_acc = self.validate_ranking_accuracy(
                utility_scores, ground_truth_ranking
            )
            results["ranking_accuracy"] = ranking_acc

        return results

