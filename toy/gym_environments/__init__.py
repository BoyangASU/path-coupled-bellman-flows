"""
Custom Gymnasium Environments for Reinforcement Learning Research

This package contains custom environments implemented following the Gymnasium API:
- DiscreteMCEnv: A discrete Markov chain environment for testing RL algorithms
- SingleStateBernoulliEnv: Single-state Bernoulli MDP from Example 2.10
- SolitaireDiceEnv: Solitaire dice game from Example 2.8
"""

# Note: shimmy (which provides MuJoCo compatibility) should be uninstalled
# if you're only using toy environments. Run: pip uninstall shimmy

from .discrete_mc_env import DiscreteMCEnv, register_discrete_mc_env
from .single_state_bernoulli_env import SingleStateBernoulliEnv, register_single_state_bernoulli_env
from .solitaire_dice_env import SolitaireDiceEnv, register_solitaire_dice_env
from .cliffwalking_wrappers import CliffWalkingTerminatesOnFallWrapper, CliffWalkingNeverTerminatesWrapper

__all__ = [
    'DiscreteMCEnv', 
    'register_discrete_mc_env',
    'SingleStateBernoulliEnv',
    'register_single_state_bernoulli_env',
    'SolitaireDiceEnv',
    'register_solitaire_dice_env',
    'CliffWalkingTerminatesOnFallWrapper',
    'CliffWalkingNeverTerminatesWrapper'
]

# Automatically register the environments
try:
    register_discrete_mc_env()
    register_single_state_bernoulli_env()
    register_solitaire_dice_env()
except:
    pass  # Ignore if already registered 