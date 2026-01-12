"""
LayerLens Real Model Demo
=========================
This script demonstrates LayerLens with a real transformer model.
It loads a BERT-base model, computes real gradients and Fisher information,
and generates an optimization manifest.

Requirements:
    pip install transformers torch numpy
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

from layerlens.cli import run_pipeline
from layerlens.config import ProfilingConfig, OptimizationConfig, LatencyProfile
from layerlens.models import ModelSpec, LayerSpec


def extract_model_layers(model: nn.Module, model_name: str) -> ModelSpec:
    """
    Extract layer specifications from a real transformer model.
    """
    layers = []
    total_params = sum(p.numel() for p in model.parameters())
    
    # BERT-like models have encoder layers
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        encoder_layers = model.encoder.layer
        for i, layer in enumerate(encoder_layers):
            # Get hidden size from the layer
            hidden_size = layer.attention.self.query.in_features if hasattr(layer.attention.self.query, 'in_features') else 768
            
            layers.append(LayerSpec(
                name=f"encoder.layer.{i}",
                hidden_size=hidden_size,
                layer_type="transformer",
                supports_attention=True,
                metadata={"layer_index": i}
            ))
    
    # If no encoder layers found, create a simple structure
    if not layers:
        # Fallback: create dummy layers based on model structure
        hidden_size = 768  # BERT-base default
        num_layers = 12   # BERT-base default
        
        for i in range(num_layers):
            layers.append(LayerSpec(
                name=f"layer.{i}",
                hidden_size=hidden_size,
                layer_type="transformer",
                supports_attention=True,
                metadata={"layer_index": i}
            ))
    
    return ModelSpec(
        model_name=model_name,
        total_params=total_params,
        layers=layers
    )


def compute_real_gradients(model: nn.Module, sample_input: dict, device: str = "cpu") -> dict:
    """
    Compute real gradients from a model forward pass.
    Returns a dictionary mapping layer names to gradient norms.
    """
    model.train()  # Enable gradient computation
    gradients = {}
    
    # Create a dummy loss (e.g., classification head output)
    with torch.set_grad_enabled(True):
        outputs = model(**sample_input)
        
        # For BERT, use pooler output or last hidden state
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            loss = outputs.pooler_output.sum()
        elif hasattr(outputs, 'last_hidden_state'):
            loss = outputs.last_hidden_state.sum()
        else:
            # Fallback: use first output
            loss = list(outputs.values())[0].sum() if isinstance(outputs, dict) else outputs.sum()
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Extract gradient norms for each layer
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                # Extract layer name (e.g., "encoder.layer.0" from "encoder.layer.0.attention.self.query.weight")
                parts = name.split('.')
                if len(parts) >= 3 and parts[0] == 'encoder' and parts[1] == 'layer':
                    layer_name = f"{parts[0]}.{parts[1]}.{parts[2]}"
                elif len(parts) >= 2:
                    layer_name = f"{parts[0]}.{parts[1]}"
                else:
                    layer_name = parts[0]
                
                if layer_name not in gradients:
                    gradients[layer_name] = []
                gradients[layer_name].append(grad_norm)
    
    model.eval()  # Back to eval mode
    
    # Average gradients per layer
    layer_gradients = {}
    for layer_name, grad_list in gradients.items():
        layer_gradients[layer_name] = np.mean(grad_list) if grad_list else 0.0
    
    return layer_gradients


def compute_fisher_trace_approximation(model: nn.Module, sample_input: dict, device: str = "cpu") -> dict:
    """
    Approximate Fisher Information Matrix trace using gradients.
    This is a simplified version - real Fisher would require more computation.
    """
    model.train()  # Enable gradient computation
    fisher_traces = {}
    
    # Compute gradients multiple times and average
    num_samples = 8
    all_gradients = {}
    
    with torch.set_grad_enabled(True):
        for _ in range(num_samples):
            outputs = model(**sample_input)
            
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                loss = outputs.pooler_output.sum()
            elif hasattr(outputs, 'last_hidden_state'):
                loss = outputs.last_hidden_state.sum()
            else:
                loss = list(outputs.values())[0].sum() if isinstance(outputs, dict) else outputs.sum()
            
            model.zero_grad()
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_squared = (param.grad ** 2).sum().item()
                    # Extract layer name consistently
                    parts = name.split('.')
                    if len(parts) >= 3 and parts[0] == 'encoder' and parts[1] == 'layer':
                        layer_name = f"{parts[0]}.{parts[1]}.{parts[2]}"
                    elif len(parts) >= 2:
                        layer_name = f"{parts[0]}.{parts[1]}"
                    else:
                        layer_name = parts[0]
                    
                    if layer_name not in all_gradients:
                        all_gradients[layer_name] = []
                    all_gradients[layer_name].append(grad_squared)
    
    model.eval()  # Back to eval mode
    
    # Average Fisher trace approximation per layer
    for layer_name, fisher_list in all_gradients.items():
        fisher_traces[layer_name] = np.mean(fisher_list) if fisher_list else 0.0
    
    return fisher_traces


def create_activation_cache(model_spec: ModelSpec, model: nn.Module, sample_input: dict, device: str = "cpu") -> dict:
    """
    Create activation cache with real computed values.
    """
    print("Computing real gradients...")
    gradients = compute_real_gradients(model, sample_input, device)
    
    print("Computing Fisher trace approximations...")
    fisher_traces = compute_fisher_trace_approximation(model, sample_input, device)
    
    activation_cache = {}
    for layer in model_spec.layers:
        # Match layer names (handle different naming conventions)
        layer_grad = 0.0
        layer_fisher = 0.0
        
        # Try to find matching gradient - check if layer name appears in gradient keys
        for grad_name, grad_value in gradients.items():
            # Direct match or partial match
            if layer.name == grad_name:
                layer_grad = grad_value
                break
            elif layer.name in grad_name or grad_name in layer.name:
                layer_grad = max(layer_grad, grad_value)  # Take max if multiple matches
        
        # Try to find matching Fisher trace
        for fisher_name, fisher_value in fisher_traces.items():
            if layer.name == fisher_name:
                layer_fisher = fisher_value
                break
            elif layer.name in fisher_name or fisher_name in layer.name:
                layer_fisher = max(layer_fisher, fisher_value)  # Take max if multiple matches
        
        # Store raw values - normalization will be done by analyzers
        # This prevents double normalization
        activation_cache[layer.name] = {
            "grad_norm": layer_grad,  # Raw gradient norm
            "fisher_trace": layer_fisher,  # Raw Fisher trace
            "proxy_gain": 0.1 + np.random.random() * 0.2  # Placeholder proxy gain
        }
    
    return activation_cache


def main():
    """
    Main demo function: Load BERT-base, compute real metrics, and run LayerLens.
    """
    print("=" * 60)
    print("LayerLens Real Model Demo")
    print("=" * 60)
    
    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load BERT-base model
    print("Loading BERT-base model...")
    try:
        config = BertConfig.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")
        model.to(device)
        model.eval()
        print("Model loaded successfully\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to random initialization...")
        config = BertConfig()
        model = BertModel(config)
        model.to(device)
        model.eval()
        print("Using randomly initialized model\n")
    
    # Create sample input
    print("Creating sample input...")
    batch_size = 2
    seq_length = 128
    sample_input = {
        "input_ids": torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(device),
        "attention_mask": torch.ones(batch_size, seq_length, dtype=torch.long).to(device),
    }
    print("Sample input created\n")
    
    # Extract model specification
    print("Extracting model layers...")
    model_spec = extract_model_layers(model, "bert-base-uncased")
    print(f"Found {len(model_spec.layers)} layers")
    print(f"  Total parameters: {model_spec.total_params:,}\n")
    
    # Compute real activation cache
    activation_cache = create_activation_cache(model_spec, model, sample_input, device)
    print(f"Activation cache created for {len(activation_cache)} layers\n")
    
    # Configure LayerLens
    profiling_cfg = ProfilingConfig(
        calibration_batch_size=32,
        gradient_window=64,
        fisher_trace_samples=8,
    )

    # Build a latency profile tailored for an LLM running on the detected device.
    # These values are heuristic but give the optimizer a more realistic signal
    # than the original fixed rank * constant model.
    if device == "cuda":
        latency_profile = LatencyProfile(
            device_type="gpu",
            model_family="llm",
            batch_size=batch_size,
            sequence_length=seq_length,
            base_ms_per_layer=0.4,
            ms_per_rank_unit=0.015,
            io_overhead_ms=5.0,
        )
    else:
        latency_profile = LatencyProfile(
            device_type="cpu",
            model_family="llm",
            batch_size=batch_size,
            sequence_length=seq_length,
            base_ms_per_layer=1.2,
            ms_per_rank_unit=0.03,
            io_overhead_ms=10.0,
        )

    optimization_cfg = OptimizationConfig(
        max_trainable_params=50000,  # Allow up to 50k trainable parameters
        max_flops=1e9,  # 1 GFLOPs
        max_vram_gb=8.0,  # 8 GB VRAM
        latency_target_ms=100.0,  # 100ms end-to-end latency target
        latency_profile=latency_profile,
    )
    
    # Run LayerLens pipeline
    print("Running LayerLens pipeline...")
    print("-" * 60)
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True, parents=True)
    
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
    
    print("Optimization Results Summary:")
    print("=" * 60)
    allocations = manifest.get("allocations", [])
    
    method_counts = {}
    total_rank = 0
    total_params = 0
    
    for alloc in allocations:
        method = alloc.get("method", "none")
        rank = alloc.get("rank", 0)
        method_counts[method] = method_counts.get(method, 0) + 1
        total_rank += rank
        total_params += rank * 768  # Approximate params per rank
    
    print(f"Total layers analyzed: {len(allocations)}")
    print(f"\nMethod distribution:")
    for method, count in method_counts.items():
        print(f"  {method}: {count} layers")
    
    print(f"\nTotal rank allocated: {total_rank}")
    print(f"Estimated trainable parameters: ~{total_params:,}")
    print(f"\nManifest file: {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

