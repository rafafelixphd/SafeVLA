#!/usr/bin/env python3
"""Count model parameters for SafeVLA architectures"""

import torch
import torch.nn as nn
from architecture.models.allenact_transformer_models.allenact_dino_transformer import (
    DinoLLAMATxNavActorCritic,
)
from architecture.models.allenact_transformer_models.separate_actor_critic import (
    SafeDinoLLAMATxNavActorCriticSeparate,
)


def count_parameters(model: nn.Module, trainable_only=True):
    """Count model parameters"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def count_parameters_by_module(model: nn.Module, trainable_only=True):
    """Count parameters by module"""
    result = {}
    for name, module in model.named_modules():
        if not list(module.children()):  # Leaf modules only
            params = sum(
                p.numel()
                for p in module.parameters()
                if p.requires_grad or not trainable_only
            )
            if params > 0:
                result[name] = params
    return result


if __name__ == "__main__":
    import gym
    from gym.spaces import Dict as SpaceDict

    # Mock observation space (minimal setup to instantiate)
    obs_space = SpaceDict({
        "rgb_dinov2": gym.spaces.Box(low=0, high=255, shape=(1, 384), dtype="float32"),
        "natural_language_spec": gym.spaces.Box(low=0, high=1000, shape=(15,), dtype="int32"),
        "time_step": gym.spaces.Box(low=0, high=1, shape=(1,), dtype="int32"),
        "traj_index": gym.spaces.Box(low=0, high=1, shape=(1,), dtype="int32"),
    })

    action_space = gym.spaces.Discrete(20)  # Approximate action space

    # Single model
    print("=" * 80)
    print("DinoLLAMATxNavActorCritic (single)")
    print("=" * 80)
    try:
        model_single = DinoLLAMATxNavActorCritic(
            action_space=action_space,
            observation_space=obs_space,
            goal_sensor_uuid="natural_language_spec",
            rgb_dino_preprocessor_uuid="rgb_dinov2",
            hidden_size=512,
            num_tx_layers=3,
            num_tx_heads=8,
        )
        trainable = count_parameters(model_single, trainable_only=True)
        total = count_parameters(model_single, trainable_only=False)
        print(f"\nTrainable Parameters: {trainable:,}")
        print(f"Total Parameters: {total:,}")
    except Exception as e:
        print(f"Error: {e}")

    # Separate actor/critic
    print("\n" + "=" * 80)
    print("SafeDinoLLAMATxNavActorCriticSeparate (actor + 2x critic)")
    print("=" * 80)
    try:
        model_safe = SafeDinoLLAMATxNavActorCriticSeparate(
            action_space=action_space,
            observation_space=obs_space,
            goal_sensor_uuid="natural_language_spec",
            rgb_dino_preprocessor_uuid="rgb_dinov2",
            hidden_size=512,
            num_tx_layers=3,
            num_tx_heads=8,
        )
        trainable = count_parameters(model_safe, trainable_only=True)
        total = count_parameters(model_safe, trainable_only=False)
        print(f"\nTrainable Parameters: {trainable:,}")
        print(f"Total Parameters: {total:,}")
    except Exception as e:
        print(f"Error: {e}")