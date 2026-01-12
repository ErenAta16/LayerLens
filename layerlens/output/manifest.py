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


class ManifestWriter:
    """
    Helper class that manages manifest creation processes.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        allocations: List[AllocationResult],
        file_name: str,
        extra_metadata: Dict[str, Any] | None = None,
    ) -> Path:
        """
        Writes allocations to disk in JSON format.
        """

        metadata = {"version": 1, "format": "hyperlora_manifest"}
        if extra_metadata:
            metadata.update(extra_metadata)

        manifest = {
            "allocations": [self._serialize(item) for item in allocations],
            "metadata": metadata,
        }

        target_path = self.output_dir / f"{file_name}.json"
        target_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return target_path

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

