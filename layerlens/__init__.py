"""
LayerLens - Adaptive Low-Rank Adaptation Selector Package
---------------------------------------------------------
This package contains modular components for layer sensitivity profiling,
low-rank adaptation selection, and manifest generation. Each submodule provides
Python interfaces for core algorithms that will be accelerated with Cython.
"""

from .config import ProfilingConfig, OptimizationConfig, LatencyProfile

# Import submodules to make them available for type checking
from . import validation  # noqa: F401
from . import models  # noqa: F401
from . import cli  # noqa: F401
from . import pipeline  # noqa: F401

__all__ = ["ProfilingConfig", "OptimizationConfig", "LatencyProfile"]

