"""
LayerLens - Adaptive Low-Rank Adaptation Selector Package
---------------------------------------------------------
This package contains modular components for layer sensitivity profiling,
low-rank adaptation selection, and manifest generation. Each submodule provides
Python interfaces for core algorithms that will be accelerated with Cython.
"""

__version__ = "0.1.0"

from .config import ProfilingConfig, OptimizationConfig, LatencyProfile

# Import submodules to make them available for type checking
from . import validation  # noqa: F401
from . import models  # noqa: F401
from . import cli  # noqa: F401
from . import pipeline  # noqa: F401
from . import exceptions  # noqa: F401
from . import utils  # noqa: F401

# Export exceptions for user convenience
from .exceptions import (
    LayerLensError,
    ConfigurationError,
    ValidationError,
    OptimizationError,
    ProfilingError,
    ManifestError,
    ModelSpecError,
    ActivationCacheError,
)

__all__ = [
    "ProfilingConfig",
    "OptimizationConfig",
    "LatencyProfile",
    "LayerLensError",
    "ConfigurationError",
    "ValidationError",
    "OptimizationError",
    "ProfilingError",
    "ManifestError",
    "ModelSpecError",
    "ActivationCacheError",
    "__version__",
]

