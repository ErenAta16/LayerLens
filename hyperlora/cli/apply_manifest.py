"""
apply_manifest.py
-----------------
Simple CLI tool that summarizes which layers will have LoRA/adapter/prefix applied
by reading the manifest file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


def apply_manifest(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Reads manifest and returns layer -> configuration mapping.
    This function can be used as a config initialization step in real MLOps integrations.
    
    Returns:
        Dictionary mapping layer names to their PEFT configuration
    """

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    mappings: Dict[str, Dict[str, Any]] = {}

    for allocation in data.get("allocations", []):
        layer_name = allocation.get("layer", allocation.get("layer_name", ""))
        mappings[layer_name] = {
            "method": allocation.get("method", "none"),
            "rank": allocation.get("rank", 0),
            "utility": allocation.get("utility", 0.0),
            "cost": allocation.get("cost", 0.0),
            "flops": allocation.get("flops", 0.0),
            "vram": allocation.get("vram", 0.0),
            "latency": allocation.get("latency", 0.0),
            "total_score": allocation.get("total_score", 0.0),
        }

    # At this point, the real application can instantiate LoRA/adapter modules.
    return mappings

