# jax_utils.py
import gymnasium as gym
import os
import json
import random
from typing import Dict, Any
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from gym_environments.cliffwalking_wrappers import (
    CliffWalkingTerminatesOnFallWrapper,
    CliffWalkingNeverTerminatesWrapper,
)
from gym_environments import (
    SolitaireDiceEnv,
    DiscreteMCEnv,
    SingleStateBernoulliEnv,
)

def set_random_seeds(seed: int = 0):
    """Set random seeds for reproducibility (NumPy + Python)."""
    np.random.seed(seed)
    random.seed(seed)


def plot_training_loss(losses, title: str = "Training Loss",
                       save_path: str | None = None):
    """
    Plot training loss and optionally save to file.
    Does NOT show the figure (no plt.show()).
    """
    losses = np.asarray(losses)

    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    # Always close – no GUI popup
    plt.close(fig)

def create_environment(env_name: str, **kwargs):
    """Create an environment based on the name (same API as PyTorch utils). :contentReference[oaicite:7]{index=7}"""
    env_name_lower = env_name.lower()

    if env_name_lower == "solitaire":
        gamma = kwargs.get("gamma", 0.9)
        return SolitaireDiceEnv(gamma=gamma)

    elif env_name_lower == "discrete_mc":
        n = kwargs.get("n", 20)
        return DiscreteMCEnv(n=n)

    elif env_name_lower == "bernoulli":
        gamma = kwargs.get("gamma", 0.5)
        max_episode_steps = kwargs.get("max_episode_steps", 100)
        return SingleStateBernoulliEnv(gamma=gamma, max_episode_steps=max_episode_steps)
    
    elif env_name == "cliff_episodic":
        base_env = gym.make("CliffWalking-v1")
        return CliffWalkingTerminatesOnFallWrapper(base_env, fall_penalty=-1.0)

    elif env_name == "cliff_continuing":
        base_env = gym.make("CliffWalking-v1")
        return CliffWalkingNeverTerminatesWrapper(base_env)
    else:
        raise ValueError(
            f"Unknown environment: {env_name}. Must be 'solitaire', 'discrete_mc', or 'bernoulli'"
        )


def get_default_states_to_test(env_name: str, env):
    """Same helper as in PyTorch version for evaluation. :contentReference[oaicite:8]{index=8}"""
    env_name_lower = env_name.lower()

    if env_name_lower == "solitaire":
        return [0]
    elif env_name_lower == "discrete_mc":
        n_states = env.observation_space.n
        return [1, n_states // 4, n_states // 2, n_states - 2]
    elif env_name_lower == "bernoulli":
        return [0]
    else:
        return [0]

def compute_distribution_metrics(learned_samples, empirical_samples):
    """
    Compute metrics comparing learned and empirical distributions.
    
    Args:
        learned_samples: Samples from learned distribution
        empirical_samples: Samples from empirical distribution
        
    Returns:
        Dictionary of metrics
    """
    from scipy import stats
    
    # Wasserstein distance (Earth Mover's Distance)
    wasserstein = stats.wasserstein_distance(learned_samples, empirical_samples)
    
    # KL divergence (approximate using histograms)
    # Create common bins
    all_samples = np.concatenate([learned_samples, empirical_samples])
    bins = np.linspace(all_samples.min(), all_samples.max(), 50)
    
    learned_hist, _ = np.histogram(learned_samples, bins=bins, density=True)
    empirical_hist, _ = np.histogram(empirical_samples, bins=bins, density=True)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    learned_hist = learned_hist + eps
    empirical_hist = empirical_hist + eps
    
    # Normalize to sum to 1
    learned_hist = learned_hist / learned_hist.sum()
    empirical_hist = empirical_hist / empirical_hist.sum()
    
    kl_div = stats.entropy(learned_hist, empirical_hist)
    
    # Mean and std comparison
    mean_diff = abs(np.mean(learned_samples) - np.mean(empirical_samples))
    std_diff = abs(np.std(learned_samples) - np.std(empirical_samples))
    
    metrics = {
        'wasserstein_distance': wasserstein,
        'kl_divergence': kl_div,
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'learned_mean': np.mean(learned_samples),
        'learned_std': np.std(learned_samples),
        'empirical_mean': np.mean(empirical_samples),
        'empirical_std': np.std(empirical_samples),
    }
    
    return metrics