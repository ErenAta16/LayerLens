"""
manifest.py
-----------
Converts optimization results to file formats consumable by MLOps pipelines.
JSON is selected as the default format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from ..optimization import AllocationResult
from ..exceptions import ManifestError
from ..utils.logging import get_logger

logger = get_logger()


class ManifestWriter:
    """
    Helper class that manages manifest creation processes.
    """

    def __init__(self, output_dir: Path) -> None:
        """
        Initialize ManifestWriter.
        
        Args:
            output_dir: Directory to save manifest files
            
        Raises:
            ManifestError: If output directory cannot be created
        """
        try:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"ManifestWriter initialized with output_dir: {self.output_dir}")
        except Exception as e:
            raise ManifestError(f"Cannot create output directory {output_dir}: {e}") from e

    def write(
        self,
        allocations: List[AllocationResult],
        file_name: str,
        extra_metadata: Dict[str, Any] | None = None,
    ) -> Path:
        """
        Writes allocations to disk in JSON format.
        
        Args:
            allocations: List of allocation results
            file_name: Base name for the manifest file (without extension)
            extra_metadata: Optional additional metadata to include
            
        Returns:
            Path to the written manifest file
            
        Raises:
            ManifestError: If writing fails
        """
        if not allocations:
            raise ManifestError("Cannot write manifest: allocations list is empty")
        
        if not file_name:
            raise ManifestError("Cannot write manifest: file_name is empty")
        
        try:
            metadata = {"version": 1, "format": "layerlens_manifest"}
            if extra_metadata:
                metadata.update(extra_metadata)

            manifest = {
                "allocations": [self._serialize(item) for item in allocations],
                "metadata": metadata,
            }

            target_path = self.output_dir / f"{file_name}.json"
            
            # Validate JSON serialization
            try:
                json_str = json.dumps(manifest, indent=2)
            except (TypeError, ValueError) as e:
                raise ManifestError(f"Failed to serialize manifest to JSON: {e}") from e
            
            # Write to file
            try:
                target_path.write_text(json_str, encoding="utf-8")
                logger.debug(f"Manifest written to: {target_path}")
                return target_path
            except (IOError, OSError) as e:
                raise ManifestError(f"Failed to write manifest to {target_path}: {e}") from e
                
        except ManifestError:
            raise
        except Exception as e:
            raise ManifestError(f"Unexpected error writing manifest: {e}") from e

    @staticmethod
    def _serialize(result: AllocationResult) -> Dict[str, Any]:
        """
        Converts AllocationResult object to dictionary.
        """

        return {
            "layer": result.layer_name,
            "method": result.method,
            "rank": result.rank,
            "utility": result.utility_score,
            "cost": result.cost_estimate,
            "flops": result.flop_estimate,
            "vram": result.vram_estimate,
            "latency": result.latency_estimate,
            "total_score": result.total_score,
        }

