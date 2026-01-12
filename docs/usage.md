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
from layerlens.cli import run_pipeline
from layerlens.config import ProfilingConfig, OptimizationConfig
from layerlens.models import ModelSpec, LayerSpec

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

### Real Model Demos

#### LLM Demo (BERT-base)

For testing with a real transformer model:

```bash
# Install dependencies
pip install transformers torch

# Run LLM demo
python demo_real_model.py
```

The demo will:
1. Load BERT-base model
2. Compute real gradients and Fisher information
3. Run LayerLens optimization with LLM-specific latency profiling
4. Generate manifest with optimal PEFT configuration

#### YOLO Demo (Object Detection)

For testing with YOLO object detection models:

```bash
# Install dependencies
pip install torch torchvision
# Optional: for YOLOv8 support
pip install ultralytics

# Run YOLO demo
python demo_yolo_model.py
```

The YOLO demo will:
1. Load/create YOLO model (YOLOv8 if available, or simplified backbone)
2. Compute real gradients and Fisher information for convolutional layers
3. Run LayerLens optimization with YOLO-specific latency profiling (resolution-aware)
4. Generate manifest optimized for real-time object detection latency targets

#### LLM-YOLO Pipeline Demo

For end-to-end analysis combining YOLO detection with LLM textual analysis:

```bash
# Install dependencies
pip install -e .[pipeline]

# Run pipeline
python llm_yolo_pipeline.py
```

The pipeline will:
1. Load an image (e.g., X-ray)
2. Run YOLO object detection to find bounding boxes
3. Send detection results to LLM for textual analysis
4. Measure and report latency at each step
5. Save complete results with timing breakdown to JSON

Example usage in code:

```python
from llm_yolo_pipeline import LLMYOLOPipeline

# Create pipeline
pipeline = LLMYOLOPipeline(
    yolo_model="yolov8n",
    llm_model="gpt2",  # or any HuggingFace model
    device="auto",
)

# Process single image
result = pipeline.process_image(
    "path/to/image.jpg",
    image_description="X-ray medical image",
)

# Access results
print(f"YOLO latency: {result.detection.latency_ms:.2f} ms")
print(f"LLM latency: {result.llm_analysis.latency_ms:.2f} ms")
print(f"Total latency: {result.total_latency_ms:.2f} ms")
print(f"LLM analysis: {result.llm_analysis.text}")

# Process batch
results = pipeline.process_batch(
    ["img1.jpg", "img2.jpg", "img3.jpg"],
    image_description="X-ray image",
)
```

### Reading the Manifest

```python
from layerlens.cli import apply_manifest
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
- `latency_profile`: Optional `LatencyProfile` for realistic latency modeling (see below)
- `candidate_methods`: List of PEFT methods to consider (default: ["lora", "adapter", "prefix", "none"])
- `method_penalties`: Cost penalties for each method
- `objective_weights`: Weights for multi-objective optimization

### LatencyProfile

For realistic latency estimation, provide a `LatencyProfile`:

```python
from layerlens.config import LatencyProfile, OptimizationConfig

# LLM example
llm_profile = LatencyProfile(
    device_type="gpu",  # or "cpu"
    model_family="llm",
    batch_size=4,
    sequence_length=2048,
    base_ms_per_layer=0.4,
    ms_per_rank_unit=0.015,
    io_overhead_ms=5.0,
)

# YOLO example
yolo_profile = LatencyProfile(
    device_type="gpu",
    model_family="yolo",
    batch_size=2,
    input_resolution=640,  # Image resolution
    base_ms_per_layer=0.8,
    ms_per_rank_unit=0.025,
    io_overhead_ms=8.0,
)

optimization_cfg = OptimizationConfig(
    max_trainable_params=50000,
    max_flops=1e9,
    max_vram_gb=8.0,
    latency_target_ms=100.0,
    latency_profile=llm_profile,  # or yolo_profile
)
```

The latency profile enables:
- **Device-aware scaling**: GPU vs CPU performance differences
- **Model-family aware scaling**: LLM (sequence length) vs Vision (resolution²) scaling
- **Workload-aware scaling**: Batch size and input dimensions
- **I/O overhead**: Fixed communication/orchestration delays

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
    "format": "layerlens_manifest",
    "layer_features": {...}
  }
}
```

