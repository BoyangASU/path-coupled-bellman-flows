import os
# Disable gymnasium plugin loading to avoid MuJoCo dependency issues
os.environ.setdefault('GYMNASIUM_DISABLE_PLUGINS', '1')

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple


class SolitaireDiceEnv(gym.Env):
    """
    Solitaire Dice Game from Example 2.8 (generalized to n-sided dice).
    
    Game mechanics:
    - Repeatedly roll an n-sided die
    - If roll 1: game ends immediately (reward = 0)
    - If roll 2-n: get 1 point and continue (reward = 1)
    - Continue until rolling a 1
    
    Return distribution:
    - Undiscounted: G ~ Geometric(1/n), range [0, ∞)
    - P(G = k) = (1/n) * ((n-1)/n)^k for k ∈ {0, 1, 2, ...}
    - Mean return = n - 1
    
    With discount γ < 1:
    - Return values: (1 - γ^(k+1)) / (1 - γ) for k = 0, 1, 2, ...
    - Same probabilities as geometric distribution
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, gamma: float = 0.9, n_sides: int = 6, render_mode: Optional[str] = None):
        """
        Initialize the Solitaire Dice environment.
        
        Args:
            gamma (float): Discount factor (default: 0.9)
            n_sides (int): Number of sides on the die (default: 6)
            render_mode (str, optional): Rendering mode
        """
        super().__init__()
        
        if n_sides < 2:
            raise ValueError(f"n_sides must be at least 2, got {n_sides}")
        
        self.gamma = gamma
        self.n_sides = n_sides
        self.render_mode = render_mode
        
        # Single state, single action (roll the die)
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Discrete(1)
        
        # Episode tracking
        self.state = 0  # Always state 0 (single state game)
        self.step_count = 0
        self.episode_return = 0.0
        self.roll_history = []
        
        # Die probabilities: 1/n_sides for each face
        self.die_probs = np.ones(n_sides) / float(n_sides)
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        """
        Reset the environment to start a new game.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            observation: Always 0 (single state)
            info: Additional information including theoretical stats
        """
        super().reset(seed=seed)
        
        self.state = 0
        self.step_count = 0
        self.episode_return = 0.0
        self.roll_history = []
        
        # Theoretical properties
        if self.gamma == 1.0:
            # Undiscounted case: geometric distribution
            # Mean = (n-1)/n / (1/n) = n-1
            # Variance = (n-1)/n / (1/n)^2 = n*(n-1)
            theoretical_mean = float(self.n_sides - 1)
            theoretical_var = float(self.n_sides * (self.n_sides - 1))
        else:
            # Discounted case: more complex calculation
            # E[G] = Σ k * P(k) * ((1-γ^(k+1))/(1-γ))
            # This is more complex, we'll compute it numerically if needed
            theoretical_mean = None
            theoretical_var = None
        
        info = {
            "state": self.state,
            "step_count": self.step_count,
            "episode_return": self.episode_return,
            "theoretical_mean": theoretical_mean,
            "theoretical_var": theoretical_var,
            "roll_history": self.roll_history.copy()
        }
        
        if self.render_mode == "human":
            self.render()
            
        return self.state, info
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment (roll the die).
        
        Args:
            action: Action to take (ignored since only one action: roll die)
            
        Returns:
            observation: Next state (always 0)
            reward: 1 if roll 2-6, 0 if roll 1
            terminated: True if rolled 1, False otherwise
            truncated: Always False (no truncation)
            info: Additional information
        """
        # Roll the n-sided die (1 to n_sides)
        die_roll = self.np_random.integers(1, self.n_sides + 1)
        self.roll_history.append(die_roll)
        
        # Determine reward and termination
        if die_roll == 1:
            # Game ends, no points for this roll
            reward = 0.0
            terminated = True
        else:
            # Continue playing, get 1 point (for rolls 2 to n_sides)
            reward = 1.0
            terminated = False
        
        # Update episode tracking
        self.step_count += 1
        discount_factor = self.gamma ** (self.step_count - 1)
        self.episode_return += discount_factor * reward
        
        # State remains the same (single state game)
        next_state = 0
        
        info = {
            "state": next_state,
            "step_count": self.step_count,
            "episode_return": self.episode_return,
            "die_roll": die_roll,
            "roll_history": self.roll_history.copy(),
            "discount_factor": discount_factor
        }
        
        if self.render_mode == "human":
            self.render()
            
        return next_state, reward, terminated, False, info
    
    def render(self):
        """Render the current state of the game."""
        if self.render_mode == "human":
            if self.roll_history:
                last_roll = self.roll_history[-1]
                if last_roll == 1:
                    print(f"Step {self.step_count}: Rolled {last_roll} - GAME OVER! Final return: {self.episode_return:.4f}")
                else:
                    print(f"Step {self.step_count}: Rolled {last_roll} - Continue playing, return so far: {self.episode_return:.4f}")
            else:
                print(f"Game ready to start. Current return: {self.episode_return:.4f}")
    
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
            
            while True:
                next_state, reward, terminated, truncated, info = self.step(0)
                
                if terminated or truncated:
                    returns.append(self.episode_return)
                    break
        
        return np.array(returns)
    
    def sample_undiscounted_returns(self, num_episodes: int = 10000) -> np.ndarray:
        """
        Sample undiscounted returns (number of successes before first failure).
        
        Args:
            num_episodes: Number of episodes to sample
            
        Returns:
            Array of undiscounted returns (integers)
        """
        returns = []
        
        for _ in range(num_episodes):
            # Count consecutive non-1 rolls
            successes = 0
            while True:
                roll = self.np_random.integers(1, self.n_sides + 1)
                if roll == 1:
                    break
                successes += 1
            returns.append(successes)
        
        return np.array(returns)
    
    def get_theoretical_properties(self) -> Dict[str, float]:
        """
        Get theoretical properties of the return distribution.
        
        Returns:
            Dictionary with theoretical statistics
        """
        if self.gamma == 1.0:
            # Undiscounted geometric distribution
            # P(X = k) = (1/n) * ((n-1)/n)^k, k = 0, 1, 2, ...
            # E[X] = ((n-1)/n) / (1/n) = n-1
            # Var[X] = ((n-1)/n) / (1/n)^2 = n*(n-1)
            theoretical_mean = float(self.n_sides - 1)
            theoretical_var = float(self.n_sides * (self.n_sides - 1))
            theoretical_std = np.sqrt(theoretical_var)
        else:
            # Discounted case: compute numerically
            # Return values are (1 - γ^(k+1)) / (1 - γ) for k = 0, 1, 2, ...
            # Probabilities are (1/n) * ((n-1)/n)^k
            
            # Compute mean and variance numerically (truncate at reasonable k)
            max_k = 100  # Should be sufficient for convergence
            k_values = np.arange(max_k + 1)
            p_success = (self.n_sides - 1) / float(self.n_sides)
            p_fail = 1.0 / float(self.n_sides)
            probs = p_fail * (p_success ** k_values)
            returns = (1 - self.gamma ** (k_values + 1)) / (1 - self.gamma)
            
            theoretical_mean = np.sum(probs * returns)
            theoretical_var = np.sum(probs * (returns - theoretical_mean) ** 2)
            theoretical_std = np.sqrt(theoretical_var)
        
        return {
            "mean": theoretical_mean,
            "variance": theoretical_var,
            "std": theoretical_std,
            "gamma": self.gamma,
            "distribution": "geometric" if self.gamma == 1.0 else "transformed_geometric"
        }
    
    def get_theoretical_pmf(self, max_k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get theoretical probability mass function.
        
        Args:
            max_k: Maximum k value to compute
            
        Returns:
            Tuple of (return_values, probabilities)
        """
        k_values = np.arange(max_k + 1)
        p_success = (self.n_sides - 1) / float(self.n_sides)
        p_fail = 1.0 / float(self.n_sides)
        probs = p_fail * (p_success ** k_values)
        
        if self.gamma == 1.0:
            # Undiscounted: returns are just k
            return_values = k_values.astype(float)
        else:
            # Discounted: returns are (1 - γ^(k+1)) / (1 - γ)
            return_values = (1 - self.gamma ** (k_values + 1)) / (1 - self.gamma)
        
        return return_values, probs
    
    def sample_theoretical_distribution(self, n_samples: int = 5000) -> np.ndarray:
        """
        Sample returns from the theoretical distribution (geometric).
        
        This is more efficient than Monte Carlo simulation since we have
        a closed-form solution for the distribution.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Array of sampled returns
        """
        p_fail = 1.0 / float(self.n_sides)
        p_success = (self.n_sides - 1) / float(self.n_sides)
        
        # Sample number of successes before failure (geometric distribution)
        # Using inverse transform sampling
        u = self.np_random.random(n_samples)
        k_values = np.floor(np.log(1 - u) / np.log(p_success)).astype(int)
        
        if self.gamma == 1.0:
            # Undiscounted: returns are just k
            return_values = k_values.astype(float)
        else:
            # Discounted: returns are (1 - γ^(k+1)) / (1 - γ)
            return_values = (1 - self.gamma ** (k_values + 1)) / (1 - self.gamma)
        
        return return_values
    
    def has_theoretical_distribution(self) -> bool:
        """Check if this environment has a closed-form theoretical distribution."""
        return True


# Register the environment
def register_solitaire_dice_env():
    """Register the environment with gymnasium."""
    try:
        gym.register(
            id='SolitaireDice-v0',
            entry_point='solitaire_dice_env:SolitaireDiceEnv',
            max_episode_steps=None,  # No truncation, episodes end naturally
        )
    except gym.error.Error:
        pass  # Already registered


if __name__ == "__main__":
    # Test the environment
    env = SolitaireDiceEnv(gamma=1.0, n_sides=6, render_mode="human")
    
    print("Testing Solitaire Dice Environment (Example 2.8)")
    print(f"Using {env.n_sides}-sided die")
    print("=" * 55)
    
    # Show theoretical properties
    theory = env.get_theoretical_properties()
    print("\nTheoretical Properties (undiscounted):")
    for key, value in theory.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test a single episode
    print(f"\nSample Episode:")
    state, info = env.reset()
    
    step_count = 0
    while True:
        action = 0  # Roll the die
        next_state, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        if terminated or truncated:
            break
    
    print(f"Episode finished after {step_count} steps")
    print(f"Final return: {info['episode_return']:.4f}")
    print(f"Roll history: {info['roll_history']}")

    env.close()
    print("\n✅ SolitaireDiceEnv test passed!")
    print("\nFor full Monte Carlo analysis, please run 'test_all_mc_sampling.py'") 