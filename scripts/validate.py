"""
LayerLens Results Validation CLI
==================================
Command-line interface for validating LayerLens manifest results.
"""

import sys
import argparse
from pathlib import Path

from layerlens.validation import validate_manifest
from layerlens.models import ModelSpec, LayerSpec


def create_bert_model_spec() -> ModelSpec:
    """Create BERT-base model specification."""
    layers = [
        LayerSpec(
            name=f"encoder.layer.{i}",
            hidden_size=768,
            layer_type="transformer",
            supports_attention=True,
        )
        for i in range(12)
    ]
    return ModelSpec(
        model_name="bert-base-uncased",
        total_params=109_482_240,
        layers=layers,
    )


def main():
    """CLI entry point for validation."""
    parser = argparse.ArgumentParser(
        description="Validate LayerLens manifest results"
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to manifest JSON file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bert",
        choices=["bert"],
        help="Model type (default: bert)",
    )
    parser.add_argument(
        "--max-params",
        type=int,
        default=50000,
        help="Maximum trainable parameters (default: 50000)",
    )
    parser.add_argument(
        "--max-flops",
        type=float,
        default=1e9,
        help="Maximum FLOPs (default: 1e9)",
    )
    parser.add_argument(
        "--max-vram",
        type=float,
        default=8.0,
        help="Maximum VRAM in GB (default: 8.0)",
    )
    parser.add_argument(
        "--latency-target",
        type=float,
        default=100.0,
        help="Latency target in ms (default: 100.0)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # Create model spec based on model type
    if args.model == "bert":
        model_spec = create_bert_model_spec()
    else:
        print(f"Error: Unknown model type: {args.model}")
        sys.exit(1)

    # Validate manifest
    try:
        results = validate_manifest(
            manifest_path=args.manifest,
            model_spec=model_spec,
            max_trainable_params=args.max_params,
            max_flops=args.max_flops,
            max_vram_gb=args.max_vram,
            latency_target_ms=args.latency_target,
            verbose=not args.quiet,
        )

        # Exit code based on validation results
        if results.get("summary", {}).get("all_passed", False):
            sys.exit(0)
        else:
            sys.exit(1)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Error during validation: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
