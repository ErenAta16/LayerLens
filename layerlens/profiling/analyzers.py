"""
analyzers.py
------------
Contains analysis classes used to measure layer sensitivity.
This module contains abstract interfaces and reference Python prototypes
that will be accelerated with Cython implementations in the future.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy.typing as npt

import numpy as np

from ..config import ProfilingConfig
from ..models import LayerSpec
from ..exceptions import ProfilingError
from .aggregators import aggregate_scores

# Cython modules are optional - fallback to Python if not compiled
try:
    from ._batch import gradient_energy_batch, fisher_trace_batch, hutchinson_trace_batch
except ImportError:
    # Fallback: define dummy functions that will use Python implementations
    gradient_energy_batch = None
    fisher_trace_batch = None
    hutchinson_trace_batch = None


class LayerSensitivityAnalyzer(ABC):
    """
    Common interface for all analysis strategies.
    """

    def __init__(self, config: ProfilingConfig) -> None:
        self.config = config

    @abstractmethod
    def score(self, layer: LayerSpec, activations: Any) -> float:
        """
        Produces a sensitivity score for a specific layer.
        The activations parameter is kept framework-independent.
        """


class GradientEnergyAnalyzer(LayerSensitivityAnalyzer):
    """
    Scorer based on gradient energy norms.
    Will be moved to vectorized computation on the Cython side in the future.
    """

    def score(self, layer: LayerSpec, activations: Any) -> float:
        """
        Returns a gradient-based sensitivity score.

        Important: `create_activation_cache` in the real-model demo already
        computes **raw** gradient norms. We keep this method as the single
        normalization point to avoid double-normalization issues that were
        identified in earlier experiments.
        """
        gradient_norm = activations.get("grad_norm", 0.0)
        # Normalize once by hidden size to make scores comparable across layers.
        return gradient_norm / max(layer.hidden_size, 1)

    def batch_score(self, gradient_matrix: Any) -> npt.NDArray[np.float64]:
        """
        Computes gradient energy scores for multiple layers using Cython function.
        Falls back to Python implementation if Cython is not available.
        """

        try:
            grads = np.ascontiguousarray(gradient_matrix, dtype=np.float64)
        except Exception as e:
            raise ProfilingError(f"Failed to convert gradient_matrix to array: {e}") from e
        
        # Edge case: Empty or invalid input
        if grads.size == 0:
            return np.array([], dtype=np.float64)
        if grads.ndim != 2:
            raise ProfilingError(
                f"gradient_matrix must be 2D, got {grads.ndim}D"
            )

        rows = grads.shape[0]
        if rows == 0:
            return np.array([], dtype=np.float64)

        output = np.zeros(rows, dtype=np.float64)
        
        # Use Cython if available, otherwise Python fallback
        if gradient_energy_batch is not None:
            gradient_energy_batch(grads, output)
        else:
            # Python fallback: compute gradient energy for each row
            for i in range(rows):
                output[i] = np.linalg.norm(grads[i]) ** 2
        
        return output


class FisherInformationAnalyzer(LayerSensitivityAnalyzer):
    """
    Measures layer importance using Fisher information approach.
    """

    def score(self, layer: LayerSpec, activations: Any) -> float:
        """
        Returns a Fisher-trace-based sensitivity score.

        The demo caches store raw approximated Fisher traces; here we apply a
        single normalization by sqrt(hidden_size) to stabilise magnitudes.
        """
        fisher_trace = activations.get("fisher_trace", 0.0)
        return fisher_trace / (layer.hidden_size ** 0.5)

    def batch_score(self, fisher_matrix: Any) -> npt.NDArray[np.float64]:
        """
        Computes Fisher trace values for multiple layers.
        Uses diagonal reading for 2D inputs, Hutchinson estimation for 3D inputs.
        Falls back to Python implementation if Cython is not available.
        """

        try:
            fisher = np.ascontiguousarray(fisher_matrix, dtype=np.float64)
        except Exception as e:
            raise ProfilingError(f"Failed to convert fisher_matrix to array: {e}") from e

        # Edge case: Empty or invalid input
        if fisher.size == 0:
            return np.array([], dtype=np.float64)
        if fisher.ndim not in (2, 3):
            raise ProfilingError(
                f"fisher_matrix must be 2D or 3D, got {fisher.ndim}D"
            )

        if fisher.ndim == 3:
            # Optimization: List conversion unnecessary
            return self._hutchinson_trace(fisher)

        rows = fisher.shape[0]
        if rows == 0:
            return np.array([], dtype=np.float64)

        output = np.zeros(rows, dtype=np.float64)
        
        # Use Cython if available, otherwise Python fallback
        if fisher_trace_batch is not None:
            fisher_trace_batch(fisher, output)
        else:
            # Python fallback: compute trace for each row (diagonal sum)
            for i in range(rows):
                if fisher.ndim == 2:
                    # 2D: assume each row is a flattened matrix, compute diagonal
                    output[i] = np.trace(fisher[i].reshape(int(np.sqrt(fisher.shape[1])), -1))
                else:
                    output[i] = np.trace(fisher[i])
        
        return output

    def _hutchinson_trace(self, tensor: np.ndarray) -> np.ndarray:
        """
        Layer-based trace estimation using Hutchinson estimator (Cython-accelerated).
        tensor.shape = (layers, dim, dim)
        
        Different seed is used for each layer (based on layer index).
        Falls back to Python implementation if Cython is not available.
        """

        layers, dim, _ = tensor.shape
        samples = max(self.config.fisher_trace_samples, 1)
        output = np.zeros(layers, dtype=np.float64)
        
        # Use Cython if available
        if hutchinson_trace_batch is not None:
            import time
            base_seed = int(time.time() * 1000) % (2**31)  # 32-bit int range
            hutchinson_trace_batch(tensor, output, samples, base_seed)
        else:
            # Python fallback: Hutchinson trace estimation
            np.random.seed(42)  # Fixed seed for reproducibility
            for layer_idx in range(layers):
                layer_matrix = tensor[layer_idx]
                trace_estimate = 0.0
                for _ in range(samples):
                    v = np.random.randn(dim)
                    v = v / np.linalg.norm(v)
                    trace_estimate += v.T @ layer_matrix @ v
                output[layer_idx] = trace_estimate / samples
        
        return output


class ProxyFineTuneAnalyzer(LayerSensitivityAnalyzer):
    """
    Measures performance gain obtained from short proxy fine-tune steps.
    """

    def score(self, layer: LayerSpec, activations: Any) -> float:
        proxy_gain = activations.get("proxy_gain", 0.0)
        return proxy_gain


__all__ = [
    "LayerSensitivityAnalyzer",
    "GradientEnergyAnalyzer",
    "FisherInformationAnalyzer",
    "ProxyFineTuneAnalyzer",
    "aggregate_scores",
]

