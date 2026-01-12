"""
model_description.py
--------------------
Contains lightweight definitions of layers and global properties in the model.
This structure facilitates data exchange between profiling and optimization modules.
Layer-based metadata is explained in comments.
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class LayerSpec:
    """
    Basic properties of a single layer.
    - name: Unique layer name within the model.
    - hidden_size: Dimension to be used when determining rank limits.
    - layer_type: To distinguish categories such as Attention, MLP, etc.
    """

    name: str
    hidden_size: int
    layer_type: str
    supports_attention: bool = False
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelSpec:
    """
    Data class representing the model as a whole.
    Contains layer list and global statistics.
    """

    model_name: str
    total_params: int
    layers: List[LayerSpec]

    def attention_layers(self) -> List[LayerSpec]:
        """
        Returns only attention-compatible layers.
        This function enables quick access to specific methods like LoRA.
        """

        return [layer for layer in self.layers if layer.supports_attention]

