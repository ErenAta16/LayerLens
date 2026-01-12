---
title: LayerLens - Adaptive Low-Rank Adaptation Selector
---

# LayerLens

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ErenAta16/LayerLens/blob/main/notebooks/colab_quick_start.ipynb)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-15%2F15-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LayerLens** is a Cython-accelerated selection engine that automatically determines the optimal PEFT (Parameter-Efficient Fine-Tuning) method and rank for each layer in large language and vision models.

**LayerLens** is a Cython-accelerated selection engine that automatically determines the optimal PEFT (Parameter-Efficient Fine-Tuning) method and rank for each layer in large language and vision models.

## Overview

Fine-tuning large language and vision models is expensive. Techniques like LoRA, adapters, and prefix tuning help reduce costs, but deciding which layers to adapt and what rank to use is still mostly done by hand. Recent work like AdaLoRA, AutoLoRA, and HyperLoRA (2023-2024) shows some automation, but they either stick to one method or need heavy AutoML that doesn't fit well in production pipelines.

LayerLens profiles a pretrained model, estimates layer sensitivities, and solves a constrained optimization problem to pick the best PEFT method (LoRA, adapter, prefix, or none) and rank for each layer. The output is a configuration that feeds directly into fine-tuning jobs, making iterations faster and decisions reproducible.

## How It Works

The system works in three steps:

1. **Layer Sensitivity Profiling**
   - Compute gradient energy, Fisher information, or NTK proxies using a calibration set.
   - Run short proxy fine-tune steps on selected layers to validate theoretical scores.
   - Heavy linear algebra runs in Cython using memory views and fused types for cache-friendly access.

2. **Multi-Objective Optimization**
   - For each layer, choose a method (LoRA, adapter, prefix, or none) and a rank/size.
   - Respect constraints: total trainable parameters, FLOPs, VRAM, and latency budgets.
   - Goal: maximize predicted utility under constraints. A fast greedy-Lagrangian solver (with optional metaheuristic refinement) runs entirely in Cython for deterministic performance.

3. **Configuration Output**
   - Generate a JSON/YAML manifest describing per-layer PEFT strategy and rank.
   - An optional hypernetwork plug-in can generate parameters, but the core workflow works without it.

## System Architecture

```
┌────────────────────┐
│ Model Checkpoint   │
└──────┬─────────────┘
       │
┌──────▼─────────────┐      ┌─────────────────────┐
│ Cython Profiler    │─────▶│ Sensitivity Cache   │
└──────┬─────────────┘      └─────────────────────┘
       │
┌──────▼─────────────┐
│ Cython Optimizer   │─────▶ Optimization logs / metrics
└──────┬─────────────┘
       │
┌──────▼─────────────┐
│ Config Generator   │─────▶ JSON/YAML manifest
└──────┬─────────────┘
       │
┌──────▼─────────────┐
│ Fine-Tune Runner   │ (LoRA/adapter/prefix training)
└────────────────────┘
```

## Related Work

| Work | Key Idea | What LayerLens Addresses |
|------|----------|---------------------------|
| LoRA (Hu et al., 2021) | Low-rank updates for attention projections | Manual rank and layer selection |
| AdaLoRA (Zhang et al., 2023) | Adaptive rank re-allocation during training | Single technique, GPU-focused heuristics |
| AutoLoRA / PEFT-NAS (2024) | Reinforcement learning search for LoRA placement | High search cost, limited MLOps integration |
| HyperLoRA (2024) | Hypernetworks generate LoRA parameters | Doesn't jointly reason about method and rank |

Cython 3.0 brings better pure-Python typing support and deterministic builds for Python 3.12, making it practical for high-performance kernels deployed as wheels in containers.

## LLM-YOLO Pipeline

LayerLens includes an end-to-end pipeline that orchestrates YOLO object detection and LLM textual analysis, with detailed latency measurement at each step. This is particularly useful for scenarios like X-ray analysis where object detection results are analyzed by an LLM.

### Pipeline Flow

```
Image → YOLO Detection → Bounding Boxes → LLM Analysis → Textual Report
  ↓           ↓                ↓              ↓              ↓
Load      Inference      Format Prompt    Generate      Save Results
(ms)      (ms)           (ms)             (ms)          (JSON)
```

### Usage

```bash
# Install pipeline dependencies
pip install -e .[pipeline]

# Run pipeline demo
python demos/llm_yolo_pipeline.py
```

The pipeline measures and reports:
- **Image loading time**
- **YOLO detection latency** (per detection)
- **LLM analysis latency** (per token generation)
- **Communication overhead** between models
- **Total end-to-end latency**

Results are saved as JSON with complete timing breakdown, enabling analysis of bottlenecks and optimization opportunities.

## MLOps Integration

The system integrates into MLOps pipelines as follows:

1. **Artifact Packaging**: Cython modules compile into wheels in CI, get signed, and publish to an internal package registry. Docker images pull the wheel at runtime.

2. **Pipeline Hook**: In orchestration platforms (Kubeflow, Airflow, Azure ML, etc.), a dedicated stage triggers the profiler/optimizer and writes the manifest to object storage.

3. **Fine-Tune Stage**: Training scripts read the manifest, instantiate the requested PEFT modules, and log metrics back to the tracking server.

4. **Observation**: Telemetry (Prometheus/OpenTelemetry) captures runtime statistics for auditing and future meta-learning.

## Performance

The benchmark script is at `benchmarks/profile_batch_benchmark.py`. The target was at least 10x speedup compared to pure Python loops in gradient/Fisher batch scoring.

| Layers x Hidden | Gradient Speedup | Fisher Speedup |
|-----------------|------------------|----------------|
| 256 x 1024      | 95.8x            | 9.8x           |
| 512 x 2048      | 124.3x           | 13.5x          |
| 1024 x 4096     | 119.9x           | 14.8x          |

The gradient side exceeded the target significantly. The Fisher side met the target for large configurations, with additional optimization planned for smaller ones.

## Evaluation Plan

| Aspect | Description |
|--------|-------------|
| Models | BERT-base, LLaMA-2-7B, ViT variants |
| Tasks | GLUE/SuperGLUE subsets, SQuAD, ImageNet-1k fine-grained |
| Baselines | Fixed LoRA (uniform rank), AdaLoRA, AutoLoRA, adapter-only |
| Metrics | Fine-tune wall time, GPU memory usage, task accuracy/F1, inference latency |
| Ablations | Sensitivity metric variants, optimizer heuristics, with/without hypernetwork |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ErenAta16/LayerLens.git
cd LayerLens

# Install in editable mode with all dependencies
pip install -e ".[demo,yolo,pipeline]"

# Or install minimal dependencies
pip install -e .
```

### Basic Usage

```python
from layerlens.pipeline import run_pipeline
from layerlens.config import ProfilingConfig, OptimizationConfig, LatencyProfile
from layerlens.models import ModelSpec, LayerSpec

# Define your model
model_spec = ModelSpec(
    model_name="bert-base-uncased",
    total_params=110_000_000,
    layers=[
        LayerSpec(name=f"encoder.layer.{i}", hidden_size=768, layer_type="transformer")
        for i in range(12)
    ]
)

# Configure profiling
profiling_cfg = ProfilingConfig(
    metric_weights={"gradient_energy": 0.4, "fisher": 0.4, "proxy_eval": 0.2}
)

# Configure optimization with latency profile
latency_profile = LatencyProfile(
    device_type="gpu",
    model_family="llm",
    batch_size=4,
    sequence_length=512
)

optimization_cfg = OptimizationConfig(
    max_trainable_params=50_000,
    max_flops=1e9,
    max_vram_gb=8.0,
    latency_target_ms=100.0,
    latency_profile=latency_profile
)

# Prepare activation cache (from your model profiling)
activation_cache = {
    f"encoder.layer.{i}": {
        "grad_norm": 0.5 + i * 0.1,
        "fisher_trace": 0.3 + i * 0.05
    }
    for i in range(12)
}

# Run pipeline
from pathlib import Path
output_dir = Path("./output")
manifest_path = run_pipeline(
    model_spec=model_spec,
    profiling_cfg=profiling_cfg,
    optimization_cfg=optimization_cfg,
    activation_cache=activation_cache,
    output_dir=output_dir
)

print(f"Manifest saved to: {manifest_path}")
```

### Running Demos

```bash
# BERT demo
python demos/demo_bert.py

# YOLO demo
python demos/demo_yolo.py

# LLM-YOLO pipeline
python demos/llm_yolo_pipeline.py
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=layerlens --cov-report=html
```

### Error Handling

LayerLens provides comprehensive error handling with specific exception types:

```python
from layerlens.exceptions import (
    ConfigurationError,
    ModelSpecError,
    ActivationCacheError,
    ProfilingError,
    OptimizationError,
    ManifestError,
)
from layerlens.utils.validation import (
    validate_model_spec,
    validate_activation_cache,
    validate_config,
)

# Validate inputs before running pipeline
try:
    validate_model_spec(model_spec)
    validate_activation_cache(activation_cache, model_spec)
    validate_config(profiling_cfg, optimization_cfg)
    
    manifest_path = run_pipeline(...)
except (ModelSpecError, ActivationCacheError, ConfigurationError) as e:
    print(f"Validation failed: {e}")
    # Fix inputs and retry
```

See `docs/ERROR_HANDLING.md` for detailed error handling guide.

## Google Colab Setup

For running on Google Colab with A100 GPU, use the following setup cells:

### Cell 1: Install Dependencies
```python
!git clone https://github.com/ErenAta16/LayerLens.git
%cd LayerLens
!pip install -e ".[demo,yolo,pipeline]" -q
```

### Cell 2: Verify GPU
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### Cell 3: Run BERT Demo
```python
# Copy content from demos/demo_bert.py
# Or import and run directly
from demos.demo_bert import main
main()
```

## Roadmap

1. ✅ Implement profiling kernels in Cython (gradient/Fisher + proxy training).
2. ✅ Develop the constrained optimization solver and manifest generator.
3. ✅ Build CLI/REST interface and sample pipeline integrations.
4. ✅ Execute benchmark suite and publish results with reproducible scripts.
5. 🔄 Prepare academic report and public technical documentation.

## Expected Contributions

- A reproducible methodology for joint method-and-rank selection across PEFT families.
- An open-source Cython engine that integrates into diverse MLOps environments.
- Empirical evidence on large language and vision models showing improved cost-performance trade-offs over existing adaptive approaches.

