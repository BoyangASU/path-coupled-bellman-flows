import os
# Disable gymnasium plugin loading to avoid MuJoCo dependency issues
os.environ.setdefault('GYMNASIUM_DISABLE_PLUGINS', '1')

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple

class DiscreteMCEnv(gym.Env):
    """
    A Discrete Markov Chain environment implemented as a Gymnasium environment.
    
    The environment consists of n states (0 to n-1) where:
    - States 0 and n-1 are terminal/absorbing states
    - Episodes start from random non-terminal states (1 to n-2)
    - Each transition gives reward 1.0, except transitions to terminal states give 0.0
    - Transition probabilities follow a complex pattern based on exponential weighting
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, n: int = 20, render_mode: Optional[str] = None):
        """
        Initialize the Discrete MC environment.
        
        Args:
            n (int): Number of states in the Markov chain
            render_mode (str, optional): Rendering mode
        """
        super().__init__()
        
        self.n = n
        self.render_mode = render_mode
        
        # Create transition matrix
        self.P, self.p = self._create_transition_matrix()
        
        # Define action and observation spaces
        # Single action since transitions are stochastic based on current state
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Discrete(n)
        
        # Current state
        self.state = None
        
    def _create_transition_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create the transition matrix for a 1D nearest-neighbor Markov chain.
        
        Returns:
            P: Transition matrix of shape (n, n)
            p: The array used in the probability ratio for transitions
        """
        p = np.exp((self.n - 1) / (4 * np.pi) * np.cos(4 * np.pi * (np.arange(self.n) - 1) / (self.n - 1)))
        p /= p.sum()  # Normalize to get probabilities
        
        P = np.zeros((self.n, self.n))
        for i in range(1, self.n - 1):
            P[i, i - 1] = p[i - 1] / (p[i] + p[i - 1])
            P[i, i + 1] = p[i + 1] / (p[i] + p[i + 1])
            P[i, i] = 1.0 - P[i, i - 1] - P[i, i + 1]
            P[i, i - 1] = max(P[i, i - 1], 0)
            P[i, i + 1] = max(P[i, i + 1], 0)
            P[i, i] = max(P[i, i], 0)
        
        # Terminal states transition patterns
        P[0, 1] = 1.0
        P[self.n - 1, self.n - 2] = 1.0
        
        # Normalize to ensure valid probability distributions
        P = P / P.sum(axis=1, keepdims=True)
        
        return P, p
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        """
        Reset the environment to start a new episode.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Start from a specific state if provided, otherwise random non-terminal
        if options and 'start_state' in options:
            self.state = options['start_state']
        else:
            self.state = self.np_random.integers(1, self.n - 1)
        
        info = {"state": self.state}
        
        if self.render_mode == "human":
            self.render()
            
        return self.state, info
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (ignored since transitions are stochastic)
            
        Returns:
            observation: Next state
            reward: Reward for the transition
            terminated: Whether episode has ended
            truncated: Whether episode was truncated (always False here)
            info: Additional information
        """
        if self.state is None:
            raise RuntimeError("Environment not reset. Call reset() before step().")
        
        # Check if already in terminal state
        if self._is_terminal(self.state):
            return self.state, 0.0, True, False, {"state": self.state}
        
        # Sample next state based on transition probabilities
        next_state = self.np_random.choice(self.n, p=self.P[self.state])
        
        # Calculate reward: 1.0 for non-terminal transitions, 0.0 for terminal transitions
        if self._is_terminal(next_state):
            reward = 0.0
            terminated = True
        else:
            reward = 1.0
            terminated = False
        
        self.state = next_state
        
        info = {"state": self.state}
        
        if self.render_mode == "human":
            self.render()
            
        return self.state, reward, terminated, False, info
    
    def _is_terminal(self, state: int) -> bool:
        """Check if a state is terminal."""
        return state in [0, self.n - 1]
    
    def render(self):
        """Render the current state of the environment."""
        if self.render_mode == "human":
            print(f"Current state: {self.state} {'(Terminal)' if self._is_terminal(self.state) else ''}")
    
    def close(self):
        """Close the environment."""
        pass
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get the transition probability matrix."""
        return self.P.copy()
    
    def get_state_probabilities(self) -> np.ndarray:
        """Get the state probability array used in transition computation."""
        return self.p.copy()


# Register the environment (optional, for easy access)
def register_discrete_mc_env():
    """Register the environment with gymnasium."""
    try:
        gym.register(
            id='DiscreteMC-v0',
            entry_point='discrete_mc_env:DiscreteMCEnv',
            max_episode_steps=1000,  # Prevent infinite episodes
        )
    except gym.error.Error:
        pass  # Already registered


if __name__ == "__main__":
    # Test the environment
    env = DiscreteMCEnv(n=5, render_mode="human")
    
    print("Testing Discrete MC Environment")
    print("=" * 40)
    
    # Test one episode
    print(f"\nSample Episode:")
    state, info = env.reset()
    print(f"Initial state: {state}")
    
    total_reward = 0
    step_count = 0
    
    while True:
        action = 0  # Only one action available
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        print(f"  Step {step_count}: {state} -> {next_state}, reward: {reward}")
        
        if terminated or truncated:
            print(f"  Episode ended. Total reward: {total_reward}, Steps: {step_count}")
            break
            
        state = next_state
    
    env.close()
    print("\n✅ DiscreteMCEnv test passed!")
    print("\nFor full Monte Carlo analysis, please run 'test_all_mc_sampling.py'") 