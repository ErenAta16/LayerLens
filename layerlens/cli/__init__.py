"""
cli package contains command-line interfaces and utilities.
"""

# Import pipeline function from pipeline module (moved from cli/pipeline.py)
from ..pipeline import run_pipeline
from .apply_manifest import apply_manifest

__all__ = ["run_pipeline", "apply_manifest"]

