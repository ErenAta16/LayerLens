"""
LayerLens YOLO Model Demo
==========================
This script demonstrates LayerLens with a YOLO object detection model.
It loads a YOLO model (YOLOv8 or YOLOv5), computes real gradients and Fisher information,
and generates an optimization manifest with YOLO-specific latency profiling.

Requirements:
    pip install torch torchvision numpy
    # Optional: pip install ultralytics  # for YOLOv8
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperlora.cli import run_pipeline
from hyperlora.config import ProfilingConfig, OptimizationConfig, LatencyProfile
from hyperlora.meta import ModelSpec, LayerSpec


def create_simple_yolo_backbone(input_channels: int = 3, num_layers: int = 6) -> nn.Module:
    """
    Creates a simplified YOLO-like backbone for demonstration.
    This mimics the structure of YOLO models with convolutional layers.
    
    In production, you would load actual YOLO models from ultralytics or torchvision.
    """
    layers = []
    
    # Initial conv block
    layers.append(nn.Sequential(
        nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
    ))
    
    # Progressive downsampling blocks (similar to YOLO)
    channels = [64, 128, 256, 512, 512, 1024]
    for i in range(min(num_layers, len(channels))):
        in_ch = channels[i-1] if i > 0 else 64
        out_ch = channels[i]
        layers.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ))
    
    return nn.Sequential(*layers)


def extract_yolo_layers(model: nn.Module, model_name: str, input_resolution: int = 640) -> ModelSpec:
    """
    Extract layer specifications from a YOLO-like model.
    
    Args:
        model: YOLO model (backbone or full model)
        model_name: Name identifier for the model
        input_resolution: Input image resolution (for feature size calculation)
    """
    layers = []
    total_params = sum(p.numel() for p in model.parameters())
    
    # Extract convolutional layers from the model
    layer_idx = 0
    current_resolution = input_resolution
    
    for name, module in model.named_modules():
        # Focus on Conv2d layers (main computational blocks in YOLO)
        if isinstance(module, nn.Conv2d):
            # Calculate feature map size based on stride
            if hasattr(module, 'stride') and isinstance(module.stride, tuple):
                stride = module.stride[0] if len(module.stride) > 0 else 1
            else:
                stride = 1
            
            # Estimate hidden size (output channels * feature map area)
            out_channels = module.out_channels
            # Approximate feature map size (simplified)
            if stride > 1:
                current_resolution = current_resolution // stride
            
            # Hidden size for YOLO: channels * spatial_dimension (approximate)
            # We use out_channels as the main dimension, scaled by resolution factor
            hidden_size = out_channels * (current_resolution // 8)  # Approximate spatial dim
            
            # Determine layer type
            layer_type = "conv" if stride == 1 else "conv_downsample"
            supports_attention = False  # YOLO typically doesn't use attention
            
            layers.append(LayerSpec(
                name=f"backbone.{layer_idx}",
                hidden_size=max(hidden_size, out_channels),  # At least out_channels
                layer_type=layer_type,
                supports_attention=supports_attention,
                metadata={
                    "layer_index": layer_idx,
                    "out_channels": out_channels,
                    "stride": stride,
                    "feature_resolution": current_resolution,
                }
            ))
            layer_idx += 1
    
    # If no layers found, create a default structure
    if not layers:
        # Default YOLO-like structure
        default_channels = [64, 128, 256, 512, 512, 1024]
        for i, ch in enumerate(default_channels):
            hidden_size = ch * (input_resolution // (8 * (2 ** i)))
            layers.append(LayerSpec(
                name=f"backbone.{i}",
                hidden_size=max(hidden_size, ch),
                layer_type="conv",
                supports_attention=False,
                metadata={"layer_index": i, "out_channels": ch}
            ))
    
    return ModelSpec(
        model_name=model_name,
        total_params=total_params,
        layers=layers
    )


def compute_yolo_gradients(model: nn.Module, sample_input: torch.Tensor, device: str = "cpu") -> dict:
    """
    Compute real gradients from a YOLO model forward pass.
    Returns a dictionary mapping layer names to gradient norms.
    """
    model.train()
    gradients = {}
    
    with torch.set_grad_enabled(True):
        # Forward pass
        output = model(sample_input)
        
        # For YOLO, create a dummy loss (sum of outputs)
        # In real scenarios, this would be detection loss (bbox + class)
        if isinstance(output, (list, tuple)):
            loss = sum(o.sum() if isinstance(o, torch.Tensor) else 0 for o in output)
        elif isinstance(output, dict):
            loss = sum(v.sum() if isinstance(v, torch.Tensor) else 0 for v in output.values())
        else:
            loss = output.sum()
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Extract gradient norms for each layer
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                # Extract layer name (e.g., "backbone.0" from "0.0.weight")
                parts = name.split('.')
                if len(parts) >= 2:
                    # Try to match backbone structure
                    if parts[0].isdigit():
                        layer_name = f"backbone.{parts[0]}"
                    else:
                        layer_name = f"backbone.{parts[0]}"
                else:
                    layer_name = f"backbone.{parts[0]}"
                
                if layer_name not in gradients:
                    gradients[layer_name] = []
                gradients[layer_name].append(grad_norm)
    
    model.eval()
    
    # Average gradients per layer
    layer_gradients = {}
    for layer_name, grad_list in gradients.items():
        layer_gradients[layer_name] = np.mean(grad_list) if grad_list else 0.0
    
    return layer_gradients


def compute_yolo_fisher_trace(model: nn.Module, sample_input: torch.Tensor, device: str = "cpu", num_samples: int = 8) -> dict:
    """
    Approximate Fisher Information Matrix trace using gradients for YOLO.
    """
    model.train()
    fisher_traces = {}
    
    all_gradients = {}
    
    with torch.set_grad_enabled(True):
        for _ in range(num_samples):
            output = model(sample_input)
            
            if isinstance(output, (list, tuple)):
                loss = sum(o.sum() if isinstance(o, torch.Tensor) else 0 for o in output)
            elif isinstance(output, dict):
                loss = sum(v.sum() if isinstance(v, torch.Tensor) else 0 for v in output.values())
            else:
                loss = output.sum()
            
            model.zero_grad()
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_squared = (param.grad ** 2).sum().item()
                    parts = name.split('.')
                    if len(parts) >= 2:
                        if parts[0].isdigit():
                            layer_name = f"backbone.{parts[0]}"
                        else:
                            layer_name = f"backbone.{parts[0]}"
                    else:
                        layer_name = f"backbone.{parts[0]}"
                    
                    if layer_name not in all_gradients:
                        all_gradients[layer_name] = []
                    all_gradients[layer_name].append(grad_squared)
    
    model.eval()
    
    # Average Fisher trace approximation per layer
    for layer_name, fisher_list in all_gradients.items():
        fisher_traces[layer_name] = np.mean(fisher_list) if fisher_list else 0.0
    
    return fisher_traces


def create_yolo_activation_cache(
    model_spec: ModelSpec,
    model: nn.Module,
    sample_input: torch.Tensor,
    device: str = "cpu"
) -> dict:
    """
    Create activation cache with real computed values for YOLO model.
    """
    print("Computing real gradients for YOLO model...")
    gradients = compute_yolo_gradients(model, sample_input, device)
    
    print("Computing Fisher trace approximations for YOLO model...")
    fisher_traces = compute_yolo_fisher_trace(model, sample_input, device)
    
    activation_cache = {}
    for layer in model_spec.layers:
        # Match layer names
        layer_grad = 0.0
        layer_fisher = 0.0
        
        # Try to find matching gradient
        for grad_name, grad_value in gradients.items():
            if layer.name == grad_name:
                layer_grad = grad_value
                break
            elif layer.name in grad_name or grad_name in layer.name:
                layer_grad = max(layer_grad, grad_value)
        
        # Try to find matching Fisher trace
        for fisher_name, fisher_value in fisher_traces.items():
            if layer.name == fisher_name:
                layer_fisher = fisher_value
                break
            elif layer.name in fisher_name or fisher_name in layer.name:
                layer_fisher = max(layer_fisher, fisher_value)
        
        # Store raw values - normalization will be done by analyzers
        activation_cache[layer.name] = {
            "grad_norm": layer_grad,  # Raw gradient norm
            "fisher_trace": layer_fisher,  # Raw Fisher trace
            "proxy_gain": 0.1 + np.random.random() * 0.2  # Placeholder proxy gain
        }
    
    return activation_cache


def main():
    """
    Main demo function: Load/create YOLO model, compute real metrics, and run LayerLens.
    """
    print("=" * 60)
    print("LayerLens YOLO Model Demo")
    print("=" * 60)
    
    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # YOLO model configuration
    input_resolution = 640  # Standard YOLO input size
    batch_size = 2
    num_backbone_layers = 6
    
    # Create or load YOLO model
    print("Creating YOLO-like model...")
    try:
        # Try to load from ultralytics (if available)
        try:
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")  # nano version for demo
            model = model.model  # Get the underlying PyTorch model
            print("Loaded YOLOv8n from ultralytics\n")
        except ImportError:
            # Fallback: create simple YOLO-like backbone
            model = create_simple_yolo_backbone(input_channels=3, num_layers=num_backbone_layers)
            print("Created simplified YOLO-like backbone\n")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Falling back to simplified YOLO-like backbone...")
        model = create_simple_yolo_backbone(input_channels=3, num_layers=num_backbone_layers)
        print("Using simplified YOLO-like backbone\n")
    
    model.to(device)
    model.eval()
    
    # Create sample input (batch, channels, height, width)
    print(f"Creating sample input (batch={batch_size}, resolution={input_resolution}x{input_resolution})...")
    sample_input = torch.randn(batch_size, 3, input_resolution, input_resolution).to(device)
    print("Sample input created\n")
    
    # Extract model specification
    print("Extracting YOLO model layers...")
    model_spec = extract_yolo_layers(model, "yolo-demo", input_resolution=input_resolution)
    print(f"Found {len(model_spec.layers)} layers")
    print(f"  Total parameters: {model_spec.total_params:,}\n")
    
    # Compute real activation cache
    activation_cache = create_yolo_activation_cache(model_spec, model, sample_input, device)
    print(f"Activation cache created for {len(activation_cache)} layers\n")
    
    # Configure LayerLens for YOLO
    profiling_cfg = ProfilingConfig(
        calibration_batch_size=16,  # Smaller for vision models
        gradient_window=64,
        fisher_trace_samples=8,
    )
    
    # Create YOLO-specific latency profile
    if device == "cuda":
        latency_profile = LatencyProfile(
            device_type="gpu",
            model_family="yolo",
            batch_size=batch_size,
            input_resolution=input_resolution,
            base_ms_per_layer=0.8,  # Vision models typically slower per layer
            ms_per_rank_unit=0.025,  # Rank overhead for conv layers
            io_overhead_ms=8.0,  # Image preprocessing/postprocessing overhead
        )
    else:
        latency_profile = LatencyProfile(
            device_type="cpu",
            model_family="yolo",
            batch_size=batch_size,
            input_resolution=input_resolution,
            base_ms_per_layer=2.5,  # CPU is much slower
            ms_per_rank_unit=0.08,
            io_overhead_ms=15.0,
        )
    
    optimization_cfg = OptimizationConfig(
        max_trainable_params=100000,  # YOLO can use more params
        max_flops=5e9,  # 5 GFLOPs
        max_vram_gb=8.0,  # 8 GB VRAM
        latency_target_ms=50.0,  # YOLO needs lower latency for real-time
        latency_profile=latency_profile,
    )
    
    # Run LayerLens pipeline
    print("Running LayerLens pipeline for YOLO...")
    print("-" * 60)
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    manifest_path = run_pipeline(
        model_spec=model_spec,
        profiling_cfg=profiling_cfg,
        optimization_cfg=optimization_cfg,
        activation_cache=activation_cache,
        output_dir=output_dir,
    )
    
    print("-" * 60)
    print(f"\nPipeline completed successfully!")
    print(f"  Manifest saved to: {manifest_path}\n")
    
    # Display results summary
    import json
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    print("YOLO Optimization Results Summary:")
    print("=" * 60)
    allocations = manifest.get("allocations", [])
    
    method_counts = {}
    total_rank = 0
    total_params = 0
    total_latency = 0.0
    
    for alloc in allocations:
        method = alloc.get("method", "none")
        rank = alloc.get("rank", 0)
        latency = alloc.get("latency", 0.0)
        method_counts[method] = method_counts.get(method, 0) + 1
        total_rank += rank
        # Approximate params: for conv layers, rank * channels
        avg_channels = 512  # Approximate
        total_params += rank * avg_channels
        total_latency += latency
    
    print(f"Total layers analyzed: {len(allocations)}")
    print(f"\nMethod distribution:")
    for method, count in method_counts.items():
        print(f"  {method}: {count} layers")
    
    print(f"\nTotal rank allocated: {total_rank}")
    print(f"Estimated trainable parameters: ~{total_params:,}")
    print(f"Total estimated latency: {total_latency:.2f} ms")
    print(f"Latency target: {optimization_cfg.latency_target_ms} ms")
    print(f"Latency utilization: {(total_latency / optimization_cfg.latency_target_ms * 100):.1f}%")
    print(f"\nManifest file: {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

