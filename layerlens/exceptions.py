"""
exceptions.py
-------------
Custom exception classes for LayerLens.

Provides specific exception types for better error handling and debugging.
"""


class LayerLensError(Exception):
    """Base exception for all LayerLens errors."""
    pass


class ConfigurationError(LayerLensError):
    """Raised when configuration is invalid."""
    pass


class ValidationError(LayerLensError):
    """Raised when validation fails."""
    pass


class OptimizationError(LayerLensError):
    """Raised when optimization fails."""
    pass


class ProfilingError(LayerLensError):
    """Raised when profiling fails."""
    pass


class ManifestError(LayerLensError):
    """Raised when manifest operations fail."""
    pass


class ModelSpecError(LayerLensError):
    """Raised when model specification is invalid."""
    pass


class ActivationCacheError(LayerLensError):
    """Raised when activation cache is invalid or missing."""
    pass

