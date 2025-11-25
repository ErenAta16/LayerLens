# LayerLens Usage Guide

## Quick Start

### Installation

```bash
# Install LayerLens
pip install -e .

# For demo with real models (optional)
pip install transformers torch
```

### Basic Usage

```python
from pathlib import Path
from hyperlora.cli import run_pipeline
from hyperlora.config import ProfilingConfig, OptimizationConfig
from hyperlora.meta import ModelSpec, LayerSpec

# Define your model
layers = [
    LayerSpec(name="layer1", hidden_size=768, layer_type="transformer", supports_attention=True),
    LayerSpec(name="layer2", hidden_size=768, layer_type="transformer", supports_attention=True),
]
model_spec = ModelSpec(model_name="my-model", total_params=110_000_000, layers=layers)

# Configure profiling
profiling_cfg = ProfilingConfig(
    calibration_batch_size=64,
    gradient_window=128,
    fisher_trace_samples=8,
)

# Configure optimization
optimization_cfg = OptimizationConfig(
    max_trainable_params=50000,
    max_flops=1e9,
    max_vram_gb=8.0,
    latency_target_ms=100.0,
)

# Provide activation data (gradients, Fisher info, etc.)
activation_cache = {
    "layer1": {
        "grad_norm": 0.5,
        "fisher_trace": 0.3,
        "proxy_gain": 0.2,
    },
    "layer2": {
        "grad_norm": 0.3,
        "fisher_trace": 0.2,
        "proxy_gain": 0.1,
    },
}

# Run pipeline
output_dir = Path("output")
manifest_path = run_pipeline(
    model_spec=model_spec,
    profiling_cfg=profiling_cfg,
    optimization_cfg=optimization_cfg,
    activation_cache=activation_cache,
    output_dir=output_dir,
)

print(f"Manifest saved to: {manifest_path}")
```

### Real Model Demo

For testing with a real transformer model (BERT-base):

```bash
# Install dependencies
pip install transformers torch

# Run demo
python demo_real_model.py
```

The demo will:
1. Load BERT-base model
2. Compute real gradients and Fisher information
3. Run LayerLens optimization
4. Generate manifest with optimal PEFT configuration

### Reading the Manifest

```python
from hyperlora.cli import apply_manifest
from pathlib import Path

manifest_path = Path("output/bert-base-uncased_plan.json")
allocations = apply_manifest(manifest_path)

for layer_name, config in allocations.items():
    print(f"Layer: {layer_name}")
    print(f"  Method: {config['method']}")
    print(f"  Rank: {config['rank']}")
    print(f"  Score: {config['total_score']:.2f}")
```

## Configuration Options

### ProfilingConfig

- `calibration_batch_size`: Batch size for calibration (default: 64)
- `gradient_window`: Number of gradient steps to collect (default: 128)
- `fisher_trace_samples`: Number of samples for Hutchinson estimator (default: 8, max: 1000)
- `proxy_steps`: Number of proxy fine-tune steps (default: 5)

### OptimizationConfig

- `max_trainable_params`: Maximum number of trainable parameters
- `max_flops`: Maximum FLOPs budget
- `max_vram_gb`: Maximum VRAM usage in GB
- `latency_target_ms`: Target latency in milliseconds
- `candidate_methods`: List of PEFT methods to consider (default: ["lora", "adapter", "prefix", "none"])
- `method_penalties`: Cost penalties for each method
- `objective_weights`: Weights for multi-objective optimization

## Output Format

The manifest is a JSON file with the following structure:

```json
{
  "allocations": [
    {
      "layer": "encoder.layer.0",
      "method": "lora",
      "rank": 16,
      "utility": 0.75,
      "cost": 12.5,
      "flops": 1.2e6,
      "vram": 0.016,
      "latency": 0.16,
      "total_score": 8.5
    }
  ],
  "metadata": {
    "version": 1,
    "format": "hyperlora_manifest",
    "layer_features": {...}
  }
}
```

