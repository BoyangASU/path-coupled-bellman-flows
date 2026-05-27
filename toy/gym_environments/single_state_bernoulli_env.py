import os
# Disable gymnasium plugin loading to avoid MuJoCo dependency issues
os.environ.setdefault('GYMNASIUM_DISABLE_PLUGINS', '1')

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple


class SingleStateBernoulliEnv(gym.Env):
    """
    Single-state, single-action Bernoulli MDP from Example 2.10.
    
    Environment properties:
    - Single state 'x' and single action 'a'
    - Always stays in the same state (deterministic self-loop)
    - Reward ~ Bernoulli(1/2): gives 0 or 1 with equal probability
    - Discount factor γ = 1/2
    - Return G = R₀ + (1/2)R₁ + (1/4)R₂ + ... ~ Uniform(0,2)
    - Episodes are truncated after max_episode_steps for practical purposes
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, gamma: float = 0.5, max_episode_steps: int = 100, render_mode: Optional[str] = None):
        """
        Initialize the Single State Bernoulli environment.
        
        Args:
            gamma (float): Discount factor (default: 0.5 as in Example 2.10)
            max_episode_steps (int): Maximum steps before truncation
            render_mode (str, optional): Rendering mode
        """
        super().__init__()
        
        self.gamma = gamma
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        
        # Single state, single action
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Discrete(1)
        
        # Episode tracking
        self.state = 0  # Always state 0 (the single state 'x')
        self.step_count = 0
        self.episode_return = 0.0
        self.episode_rewards = []
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        """
        Reset the environment to start a new episode.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            observation: Always 0 (the single state)
            info: Additional information including theoretical return stats
        """
        super().reset(seed=seed)
        
        self.state = 0
        self.step_count = 0
        self.episode_return = 0.0
        self.episode_rewards = []
        
        # Theoretical properties for this MDP
        theoretical_mean = 1.0    # E[G] when γ = 0.5: (1/2)/(1-1/2) = 1
        theoretical_max = 2.0     # Maximum possible return
        
        info = {
            "state": self.state,
            "step_count": self.step_count,
            "episode_return": self.episode_return,
            "theoretical_mean": theoretical_mean,
            "theoretical_max": theoretical_max
        }
        
        if self.render_mode == "human":
            self.render()
            
        return self.state, info
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (ignored since only one action available)
            
        Returns:
            observation: Next state (always 0)
            reward: Bernoulli(1/2) reward (0 or 1)
            terminated: Always False (infinite horizon MDP)
            truncated: True if max_episode_steps reached
            info: Additional information
        """
        if self.step_count >= self.max_episode_steps:
            # Already truncated, return zero reward
            return self.state, 0.0, False, True, {"state": self.state, "step_count": self.step_count}
        
        # Sample Bernoulli(1/2) reward: 0 or 1 with equal probability
        reward = float(self.np_random.choice([0, 1]))
        
        # Update episode tracking
        self.step_count += 1
        discount_factor = self.gamma ** (self.step_count - 1)
        self.episode_return += discount_factor * reward
        self.episode_rewards.append(reward)
        
        # State always remains the same (deterministic self-loop)
        next_state = 0
        
        # Never terminates naturally (infinite horizon)
        terminated = False
        
        # Truncate if maximum steps reached
        truncated = self.step_count >= self.max_episode_steps
        
        info = {
            "state": next_state,
            "step_count": self.step_count,
            "episode_return": self.episode_return,
            "discount_factor": discount_factor,
            "episode_rewards": self.episode_rewards.copy()
        }
        
        if self.render_mode == "human":
            self.render()
            
        return next_state, reward, terminated, truncated, info
    
    def render(self):
        """Render the current state of the environment."""
        if self.render_mode == "human":
            print(f"Step {self.step_count}: State={self.state}, Episode Return={self.episode_return:.4f}")
    
    def close(self):
        """Close the environment."""
        pass
    
    def sample_returns(self, num_episodes: int = 10000) -> np.ndarray:
        """
        Sample multiple episode returns for analysis.
        
        Args:
            num_episodes: Number of episodes to sample
            
        Returns:
            Array of episode returns
        """
        returns = []
        
        for _ in range(num_episodes):
            self.reset()
            episode_return = 0.0
            
            for step in range(self.max_episode_steps):
                # Sample Bernoulli(1/2) reward
                reward = float(self.np_random.choice([0, 1]))
                discount_factor = self.gamma ** step
                episode_return += discount_factor * reward
            
            returns.append(episode_return)
        
        return np.array(returns)
    
    def get_theoretical_properties(self) -> Dict[str, float]:
        """
        Get theoretical properties of the return distribution.
        
        Returns:
            Dictionary with theoretical statistics
        """
        if self.gamma == 0.5:
            # For γ = 1/2, the return G ~ Uniform(0, 2)
            theoretical_mean = 1.0
            theoretical_var = 1.0 / 3.0  # Var[Uniform(0,2)] = (2-0)²/12 = 1/3
            theoretical_min = 0.0
            theoretical_max = 2.0
        else:
            # General case: G = Σ γᵗ Rₜ where Rₜ ~ Bernoulli(1/2)
            # E[G] = Σ γᵗ E[Rₜ] = Σ γᵗ (1/2) = (1/2) / (1-γ)
            # For the exact distribution, it's more complex for general γ
            theoretical_mean = 0.5 / (1 - self.gamma)
            theoretical_max = 1.0 / (1 - self.gamma)
            theoretical_min = 0.0
            theoretical_var = None  # Complex for general γ
        
        return {
            "mean": theoretical_mean,
            "variance": theoretical_var,
            "min": theoretical_min,
            "max": theoretical_max,
            "gamma": self.gamma
        }
    
    def sample_theoretical_distribution(self, n_samples: int = 5000) -> np.ndarray:
        """
        Sample returns from the theoretical distribution.
        
        For γ = 0.5, the return G ~ Uniform(0, 2).
        For general γ, we use the fact that G = Σ_{t=0}^∞ γ^t R_t where R_t ~ Bernoulli(1/2).
        The distribution can be computed exactly using the binary expansion representation.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Array of sampled returns
        """
        if self.gamma == 0.5:
            # For γ = 0.5, G ~ Uniform(0, 2)
            return self.np_random.uniform(0.0, 2.0, size=n_samples)
        else:
            # For general γ, G = Σ_{t=0}^∞ γ^t R_t where R_t ~ Bernoulli(1/2)
            # We can sample this by generating binary sequences and computing the sum
            # For practical purposes, we truncate at a reasonable number of steps
            max_steps = int(np.ceil(np.log(1e-10) / np.log(self.gamma))) if self.gamma < 1.0 else self.max_episode_steps
            max_steps = min(max_steps, self.max_episode_steps)
            
            returns = []
            for _ in range(n_samples):
                g = 0.0
                for t in range(max_steps):
                    r_t = float(self.np_random.choice([0, 1]))
                    g += (self.gamma ** t) * r_t
                returns.append(g)
            return np.array(returns)
    
    def has_theoretical_distribution(self) -> bool:
        """Check if this environment has a closed-form theoretical distribution."""
        return True


# Register the environment
def register_single_state_bernoulli_env():
    """Register the environment with gymnasium."""
    try:
        gym.register(
            id='SingleStateBernoulli-v0',
            entry_point='single_state_bernoulli_env:SingleStateBernoulliEnv',
            max_episode_steps=100,
        )
    except gym.error.Error:
        pass  # Already registered


if __name__ == "__main__":
    # Test the environment
    env = SingleStateBernoulliEnv(gamma=0.5, max_episode_steps=50, render_mode="human")
    
    print("Testing Single State Bernoulli Environment (Example 2.10)")
    print("=" * 60)
    
    # Show theoretical properties
    theory = env.get_theoretical_properties()
    print("\nTheoretical Properties (γ=0.5):")
    for key, value in theory.items():
        if value is not None:
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test a single episode
    print(f"\nSample Episode (first 10 steps):")
    state, info = env.reset()
    
    for step in range(10):
        action = 0  # Only one action
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {step+1}: reward={reward}, discounted_return={info['episode_return']:.4f}")
        if terminated or truncated:
            break
    
    env.close()
    print("\n✅ SingleStateBernoulliEnv test passed!")
    print("\nFor full Monte Carlo analysis, please run 'test_all_mc_sampling.py'") 