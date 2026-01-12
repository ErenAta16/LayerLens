"""
Pipeline orchestration module for LayerLens.

This module contains the main pipeline orchestration logic that coordinates
profiling, optimization, and manifest generation.
"""

from .full_pipeline import run_pipeline

__all__ = ["run_pipeline"]

