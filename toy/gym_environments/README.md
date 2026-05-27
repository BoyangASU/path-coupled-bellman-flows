# Custom Gymnasium Environments for Reinforcement Learning

This package contains custom environments implemented following the Gymnasium API for testing reinforcement learning algorithms, particularly distributional RL methods.

## Environments

### 1. DiscreteMCEnv - Discrete Markov Chain

A 1D nearest-neighbor Markov chain with reflecting boundaries.

**Environment Details:**
- **States**: 0 to N-1 (default N=10)
- **Actions**: 2 (move left or right)
- **Transitions**: Probabilistic movement with exponential weighting
- **Rewards**: 1.0 for all non-terminal transitions, 0.0 for terminal transitions
- **Terminal States**: 0 and N-1

**Mathematical Properties:**
- Exponential transition probabilities: `p ∝ exp(-β * |action_effect|)`
- Reflecting boundaries at terminal states
- Expected episode length depends on starting position and transition parameters

### 2. SingleStateBernoulliEnv - Example 2.10

Single-state Bernoulli MDP from distributional RL literature.

**Environment Details:**
- **States**: 1 (single state)
- **Actions**: 1 (single action)
- **Rewards**: Bernoulli(0.5) → {0, 1} with equal probability
- **Episodes**: Single-step episodes
- **Return Distribution**: With γ=0.5, G ~ Uniform(0, 2)

**Mathematical Properties:**
- Infinite geometric series: G = R₀ + γR₁ + γ²R₂ + ...
- Closed form: G ~ Uniform(0, 2/(1-γ)) when γ < 1
- Mean: 1/(1-γ), Variance: 1/(3(1-γ)²)

### 3. SolitaireDiceEnv - Example 2.8

Solitaire dice game with geometric return distribution.

**Environment Details:**
- **States**: 1 (single state)
- **Actions**: 1 (roll the die)
- **Game Rules**:
  - Roll a six-sided die
  - If roll 1: game ends (reward = 0)
  - If roll 2-6: get 1 point (reward = 1) and continue
- **Episodes**: Variable length, terminate on rolling 1

**Mathematical Properties:**
- **Undiscounted (γ=1)**: Return ~ Geometric(1/6)
  - P(Return = k) = (1/6) × (5/6)^k for k ∈ {0, 1, 2, ...}
  - Mean: 5, Variance: 30
- **Discounted (γ<1)**: Return values = (1-γ^(k+1))/(1-γ)
  - Same probability mass function as geometric
  - Bounded support: [0, 1/(1-γ)]

### 4. GamblersRuinEnv - Closed-form hitting probabilities

Classic gambler’s ruin with two biased coins:
- **States**: 0..N-1 (0=ruin, N-1=goal)
- **Actions**: 2 coins with win probabilities `p0`, `p1`
- **Transitions**: Win → +1 capital, Lose → -1 capital (clamped to terminals)
- **Rewards**: 1.0 when reaching goal, 0 otherwise
- **Closed form**: Probability of reaching goal before ruin is known analytically; provided for each coin plus optimal stationary policy (pick higher win-prob coin).

### 5. RiverSwimEnv - Noisy chain with small/large rewards

Osband-style RiverSwim:
- **States**: Linear chain (default 6)
- **Actions**: 0=left, 1=right with asymmetric slip/stay probabilities
- **Rewards**: Small reward at left end for action 0; large reward at right end for action 1
- **Dynamics**: P(left)=0.6/0.05, P(right)=0.05/0.35, P(stay)=remainder for left/right actions
- **Closed form**: Optimal value function/policy computed exactly via value iteration on the small tabular MDP.

## Installation

```python
# Add to your Python path or install as package
import sys
sys.path.append('path/to/gym_environments')

# Import environments
from gym_environments import DiscreteMCEnv, SingleStateBernoulliEnv, SolitaireDiceEnv
```

## Usage Examples

### Basic Usage

```python
import gymnasium as gym
from gym_environments import DiscreteMCEnv, SingleStateBernoulliEnv, SolitaireDiceEnv

# Create environments
discrete_mc = DiscreteMCEnv(num_states=10, beta=1.0)
bernoulli = SingleStateBernoulliEnv(gamma=0.5)
dice = SolitaireDiceEnv(gamma=1.0)

# Standard gym interface
state, info = env.reset()
action = env.action_space.sample()
next_state, reward, terminated, truncated, info = env.step(action)
```

### Sampling Return Distributions

```python
# Sample episode returns for analysis
returns = env.sample_returns(num_episodes=1000)

# Get theoretical properties
theory = env.get_theoretical_properties()
print(f"Theoretical mean: {theory['mean']}")
print(f"Theoretical std: {theory['std']}")
```

### Discrete MC Specific

```python
# Create with custom parameters
env = DiscreteMCEnv(
    num_states=15,
    beta=2.0,          # Higher β = more deterministic
    gamma=0.95,        # Discount factor
    render_mode="human"
)

# Get transition matrix
P = env.get_transition_matrix()
print(f"Transition matrix shape: {P.shape}")
```

### Bernoulli MDP Specific

```python
# Example 2.10 with γ=0.5
env = SingleStateBernoulliEnv(gamma=0.5)

# Theoretical: G ~ Uniform(0, 2)
returns = env.sample_returns(10000)
print(f"Empirical mean: {np.mean(returns):.3f} (should be ~1.0)")
print(f"Empirical range: [{np.min(returns):.3f}, {np.max(returns):.3f}]")
```

### Solitaire Dice Specific

```python
# Example 2.8 undiscounted
env = SolitaireDiceEnv(gamma=1.0)

# Theoretical: geometric distribution with mean=5
undiscounted_returns = env.sample_undiscounted_returns(10000)
print(f"Empirical mean: {np.mean(undiscounted_returns):.3f} (should be ~5.0)")

# Test discounted version
env_discounted = SolitaireDiceEnv(gamma=0.9)
discounted_returns = env_discounted.sample_returns(10000)

# Get theoretical PMF
return_values, probabilities = env.get_theoretical_pmf(max_k=20)
```

## Demo Scripts

Several demo scripts are provided to showcase the environments:

1. **`quantile_rl_gym.py`** - Quantile RL on Discrete MC environment
2. **`quantile_rl_bernoulli_demo.py`** - Quantile RL on Bernoulli MDP (Example 2.10)
3. **`quantile_rl_dice_demo.py`** - Quantile RL on Solitaire Dice (Example 2.8)

Run a demo:
```bash
python quantile_rl_dice_demo.py
```

## Quantile RL Integration

All environments are designed to work seamlessly with quantile RL algorithms:

```python
class QuantileRLAgent:
    def __init__(self, num_quantiles=51, learning_rate=0.01, gamma=0.99):
        self.tau = np.linspace(0.5/num_quantiles, 1-0.5/num_quantiles, num_quantiles)
        self.quantiles = # Initialize appropriately for environment
        
    def update(self, state, action, reward, next_state, terminated):
        # Quantile regression update
        # See demo scripts for full implementation
        pass
```

## Environment Registration

The environments can be registered with Gymnasium:

```python
import gymnasium as gym
from gym_environments import register_discrete_mc_env, register_single_state_bernoulli_env, register_solitaire_dice_env

# Register environments
register_discrete_mc_env()
register_single_state_bernoulli_env()
register_solitaire_dice_env()

# Use via gym.make
env = gym.make('DiscreteMC-v0')
env = gym.make('SingleStateBernoulli-v0')
env = gym.make('SolitaireDice-v0')
```

## Key Features

- **Standard Gymnasium API**: Full compatibility with modern RL frameworks
- **Theoretical Validation**: Known return distributions for algorithm validation
- **Rich Information**: Detailed `info` dictionaries with theoretical properties
- **Flexible Parameters**: Customizable environment parameters
- **Visualization Support**: Built-in rendering and plotting capabilities
- **Distribution Sampling**: Methods to sample and analyze return distributions

## Dependencies

- `numpy`
- `gymnasium`
- `matplotlib` (for visualization)
- `tqdm` (for progress bars in demos)

## References

The environments implement examples from distributional reinforcement learning literature:

- **Example 2.8**: Solitaire dice game with geometric return distribution
- **Example 2.10**: Single-state Bernoulli MDP with uniform return distribution
- **Discrete MC**: Custom environment for testing distributional RL algorithms

These environments provide ground truth for validating distributional RL methods like quantile regression, distributional DQN, and other distributional algorithms. 