"""
cli package contains command-line interfaces that orchestrate modules.
"""

from .pipeline import run_pipeline
from .apply_manifest import apply_manifest

__all__ = ["run_pipeline", "apply_manifest"]

