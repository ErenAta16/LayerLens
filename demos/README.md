# LayerLens Demos

This directory contains demonstration scripts showing how to use LayerLens with different model types.

## Available Demos

### `demo_bert.py`
Demonstrates LayerLens with BERT-base transformer model:
- Loads BERT-base from HuggingFace
- Computes real gradients and Fisher information
- Runs LayerLens optimization with LLM-specific latency profiling
- Generates manifest with optimal PEFT configuration

**Usage:**
```bash
pip install transformers torch
python demos/demo_bert.py
```

### `demo_yolo.py`
Demonstrates LayerLens with YOLO object detection models:
- Loads YOLO model (YOLOv8 if available, or simplified backbone)
- Computes real gradients and Fisher information for convolutional layers
- Runs LayerLens optimization with YOLO-specific latency profiling (resolution-aware)
- Generates manifest optimized for real-time object detection latency targets

**Usage:**
```bash
pip install torch torchvision
# Optional: pip install ultralytics
python demos/demo_yolo.py
```

### `llm_yolo_pipeline.py`
End-to-end pipeline combining YOLO detection with LLM textual analysis:
- Takes an image (e.g., X-ray)
- Runs YOLO object detection to find bounding boxes
- Sends detection results to LLM for textual analysis
- Measures and reports latency at each step
- Saves complete results with timing breakdown to JSON

**Usage:**
```bash
pip install -e .[pipeline]
python demos/llm_yolo_pipeline.py
```

## Output

All demos save their results to the `output/` directory:
- `bert-base-uncased_plan.json` - BERT optimization manifest
- `yolo-demo_plan.json` - YOLO optimization manifest
- `llm_yolo_pipeline_result.json` - Pipeline results with timing breakdown

