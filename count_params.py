#!/usr/bin/env python3
"""Count model parameters for SafeVLA architectures."""

import argparse
import sys
import torch
import torch.nn as nn
from pathlib import Path


def count_parameters(model: nn.Module, trainable_only=True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def count_parameters_by_module(model: nn.Module, trainable_only=True) -> dict:
    """Count parameters by module, showing breakdown."""
    result = {}
    for name, module in model.named_modules():
        if not list(module.children()):  # Leaf modules only
            params = sum(
                p.numel()
                for p in module.parameters()
                if (p.requires_grad or not trainable_only)
            )
            if params > 0:
                result[name] = params
    return result


def load_model_from_checkpoint(ckpt_path: str) -> nn.Module:
    """Load model from checkpoint."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Create a simple wrapper model to load state dict
    model = nn.Module()
    model.load_state_dict(state_dict)
    return model


def format_number(num: int) -> str:
    """Format number with commas and millions/billions notation."""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B ({num:,})"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M ({num:,})"
    else:
        return f"{num:,}"


def main():
    parser = argparse.ArgumentParser(
        description="Count model parameters in SafeVLA checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Count parameters in a checkpoint
  python count_params.py --ckpt_path path/to/checkpoint.pt

  # Show detailed breakdown by module
  python count_params.py --ckpt_path path/to/checkpoint.pt --detailed
        """
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show parameter breakdown by module",
    )

    args = parser.parse_args()

    try:
        print("=" * 80)
        print(f"Loading checkpoint: {args.ckpt_path}")
        print("=" * 80)

        model = load_model_from_checkpoint(args.ckpt_path)

        trainable_params = count_parameters(model, trainable_only=True)
        total_params = count_parameters(model, trainable_only=False)

        print(f"\n{'Trainable Parameters:':<30} {format_number(trainable_params)}")
        print(f"{'Total Parameters:':<30} {format_number(total_params)}")

        frozen_params = total_params - trainable_params
        if frozen_params > 0:
            print(f"{'Frozen Parameters:':<30} {format_number(frozen_params)}")

        if args.detailed:
            print("\n" + "=" * 80)
            print("Parameter Breakdown by Module")
            print("=" * 80)

            params_by_module = count_parameters_by_module(model, trainable_only=False)

            # Sort by parameter count (descending)
            sorted_modules = sorted(params_by_module.items(), key=lambda x: x[1], reverse=True)

            for name, params in sorted_modules[:20]:  # Show top 20
                pct = (params / total_params) * 100
                print(f"{name:<50} {format_number(params):>20} ({pct:>5.2f}%)")

            if len(sorted_modules) > 20:
                print(f"\n... and {len(sorted_modules) - 20} more modules")

        print("\n" + "=" * 80)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
