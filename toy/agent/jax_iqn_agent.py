# jax_iqn_agent.py
"""
JAX IQN (Implicit Quantile Network) Agent for Discrete Toy Environments

Implements Quantile Regression DQN / IQN for distributional RL.
Learns the quantile function of the value distribution.
"""

from dataclasses import dataclass
from typing import Any, Dict

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from flax.core import FrozenDict
import optax
import numpy as np

from pathlib import Path
import json
from flax import serialization


class QuantileNetwork(nn.Module):
    """Quantile network that outputs quantile values for given tau samples."""
    n_states: int
    n_actions: int
    hidden_dims: tuple = (256, 256)
    num_cosines: int = 64

    @nn.compact
    def __call__(self, states, taus, actions=None):
        """
        Args:
            states: [B] integer states
            taus: [B, N] quantile fractions in [0, 1]
            actions: [B] integer actions (optional)
        Returns:
            quantiles: [B, N] quantile values
        """
        B = states.shape[0]
        N = taus.shape[1]

        # State embedding
        state_emb = nn.Embed(num_embeddings=self.n_states, features=self.hidden_dims[0])(states)  # [B, H]
        
        # Action embedding (if multi-action)
        if actions is not None and self.n_actions > 1:
            action_emb = nn.Embed(num_embeddings=self.n_actions, features=self.hidden_dims[0])(actions)
            state_emb = state_emb + action_emb  # [B, H]

        # Cosine embedding for tau
        # tau_emb[i] = cos(pi * i * tau) for i = 1, ..., num_cosines
        i_pi = jnp.arange(1, self.num_cosines + 1, dtype=jnp.float32) * jnp.pi  # [num_cosines]
        cos_features = jnp.cos(taus[:, :, None] * i_pi[None, None, :])  # [B, N, num_cosines]
        tau_emb = nn.Dense(self.hidden_dims[0])(cos_features)  # [B, N, H]
        tau_emb = nn.relu(tau_emb)

        # Combine state and tau embeddings
        # Expand state_emb to [B, N, H]
        state_emb_expanded = jnp.expand_dims(state_emb, axis=1)  # [B, 1, H]
        state_emb_expanded = jnp.broadcast_to(state_emb_expanded, (B, N, self.hidden_dims[0]))  # [B, N, H]
        
        x = state_emb_expanded * tau_emb  # Element-wise multiplication [B, N, H]

        # MLP
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)

        # Output quantile values
        quantiles = nn.Dense(1)(x).squeeze(-1)  # [B, N]
        return quantiles


class IQNTrainState(train_state.TrainState):
    target_params: FrozenDict


@dataclass
class JaxIQNAgent:
    """JAX IQN Agent for discrete toy environments.
    
    IQN learns the full quantile function by sampling random quantile
    fractions tau and predicting the corresponding quantile values.
    """
    env: Any
    batch_size: int
    gamma: float = 0.9
    learning_rate: float = 5e-3
    tau: float = 0.005  # Polyak averaging coefficient
    num_quantiles: int = 32  # Number of quantile samples for training
    num_quantiles_eval: int = 64  # Number of quantile samples for evaluation
    kappa: float = 1.0  # Huber loss threshold
    rng: jax.Array = None
    n_states: int = None
    n_actions: int = None
    model: QuantileNetwork = None
    state: IQNTrainState = None
    total_hist: list = None

    @classmethod
    def create(
        cls,
        env,
        batch_size: int,
        gamma: float = 0.9,
        lambda_param: float = None,  # unused, for interface compatibility
        learning_rate: float = 5e-3,
        tau: float = 0.005,
        seed: int = 0,
        num_quantiles: int = 32,
        kappa: float = 1.0,
    ):
        obs_space = env.observation_space
        act_space = env.action_space

        if not hasattr(obs_space, "n") or not hasattr(act_space, "n"):
            raise NotImplementedError("IQN agent currently supports only discrete envs.")

        n_states = int(obs_space.n)
        n_actions = int(act_space.n)

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        model = QuantileNetwork(
            n_states=n_states,
            n_actions=n_actions,
        )

        # Initialize
        dummy_states = jnp.zeros((1,), dtype=jnp.int32)
        dummy_taus = jnp.zeros((1, num_quantiles), dtype=jnp.float32)
        dummy_actions = jnp.zeros((1,), dtype=jnp.int32) if n_actions > 1 else None

        if dummy_actions is not None:
            params = model.init(init_rng, dummy_states, dummy_taus, dummy_actions)["params"]
        else:
            params = model.init(init_rng, dummy_states, dummy_taus)["params"]

        tx = optax.adam(learning_rate)
        state = IQNTrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
            target_params=params,
        )

        return cls(
            env=env,
            batch_size=batch_size,
            gamma=float(gamma),
            learning_rate=learning_rate,
            tau=tau,
            num_quantiles=num_quantiles,
            num_quantiles_eval=64,
            kappa=float(kappa),
            rng=rng,
            n_states=n_states,
            n_actions=n_actions,
            model=model,
            state=state,
            total_hist=[],
        )

    def _sample_batch(self) -> Dict[str, np.ndarray]:
        """Sample a one-step batch from the environment."""
        B = self.batch_size
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for _ in range(B):
            obs, _ = self.env.reset()
            action = np.random.randint(0, self.n_actions) if self.n_actions > 1 else 0
            next_obs, reward, terminated, truncated, _ = self.env.step(action)

            states.append(int(obs))
            actions.append(int(action))
            rewards.append(float(reward))
            next_states.append(int(next_obs))
            dones.append(float(terminated or truncated))

        batch = {
            "states": np.array(states, dtype=np.int32),
            "actions": np.array(actions, dtype=np.int32),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_states": np.array(next_states, dtype=np.int32),
            "dones": np.array(dones, dtype=np.float32),
        }
        return batch

    def _make_train_step(self):
        """Create a jitted train_step function."""
        gamma = self.gamma
        n_actions = self.n_actions
        polyak_tau = self.tau
        model = self.model
        num_quantiles = self.num_quantiles
        kappa = self.kappa

        @jax.jit
        def train_step_jit(
            state: IQNTrainState,
            batch: Dict[str, jnp.ndarray],
            rng: jax.Array,
        ):
            rng, tau_rng, tau_prime_rng = jax.random.split(rng, 3)

            B = batch["states"].shape[0]
            states = batch["states"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            next_states = batch["next_states"]
            dones = batch["dones"]

            # Sample tau for current state quantiles
            taus = jax.random.uniform(tau_rng, (B, num_quantiles))  # [B, N]
            
            # Sample tau' for next state quantiles
            taus_prime = jax.random.uniform(tau_prime_rng, (B, num_quantiles))  # [B, N']

            # Get next state quantiles from target network
            if n_actions > 1:
                next_quantiles = model.apply(
                    {"params": state.target_params},
                    next_states,
                    taus_prime,
                    actions,  # Use same action for simplicity
                )
            else:
                next_quantiles = model.apply(
                    {"params": state.target_params},
                    next_states,
                    taus_prime,
                )
            # next_quantiles: [B, N']

            # Compute target quantiles: T_tau = r + γ * Z(s', a')
            gamma_mask = gamma * (1.0 - dones)
            target_quantiles = rewards[:, None] + gamma_mask[:, None] * next_quantiles  # [B, N']

            def loss_fn(params):
                # Get current state quantiles
                if n_actions > 1:
                    quantiles = model.apply({"params": params}, states, taus, actions)
                else:
                    quantiles = model.apply({"params": params}, states, taus)
                # quantiles: [B, N]

                # Quantile Huber loss
                # TD error: [B, N, N'] where we compare each quantile with each target
                td_error = target_quantiles[:, None, :] - quantiles[:, :, None]  # [B, N, N']

                # Huber loss
                huber_loss = jnp.where(
                    jnp.abs(td_error) <= kappa,
                    0.5 * td_error ** 2,
                    kappa * (jnp.abs(td_error) - 0.5 * kappa)
                )

                # Quantile regression loss
                # tau - I(td_error < 0)
                tau_weights = jnp.abs(taus[:, :, None] - (td_error < 0).astype(jnp.float32))
                qr_loss = tau_weights * huber_loss / kappa

                # Mean over all dimensions
                loss = qr_loss.mean()
                return loss

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params)
            new_state = state.apply_gradients(grads=grads)

            # Polyak averaging for target network
            new_target_params = jax.tree_util.tree_map(
                lambda p, tp: polyak_tau * p + (1.0 - polyak_tau) * tp,
                new_state.params,
                state.target_params,
            )
            new_state = new_state.replace(target_params=new_target_params)

            return new_state, rng, loss

        return train_step_jit

    def train_step(self):
        """Sample a batch from env and perform one update."""
        batch_np = self._sample_batch()
        batch = {k: jnp.array(v) for k, v in batch_np.items()}

        train_step_jit = getattr(self, "_train_step_jit", None)
        if train_step_jit is None:
            self._train_step_jit = self._make_train_step()
            train_step_jit = self._train_step_jit

        self.state, self.rng, loss = train_step_jit(self.state, batch, self.rng)

        loss_float = float(loss)
        self.total_hist.append(loss_float)
        return loss_float

    def get_value(self, state: int, action: int = 0) -> float:
        """Get expected value for a state(-action) by averaging quantiles."""
        self.rng, tau_rng = jax.random.split(self.rng)
        
        s = jnp.array([state], dtype=jnp.int32)
        # Use uniform quantile fractions for evaluation
        taus = jnp.linspace(0.05, 0.95, self.num_quantiles_eval).reshape(1, -1)
        a = jnp.array([action], dtype=jnp.int32) if self.n_actions > 1 else None

        if a is not None:
            quantiles = self.model.apply({"params": self.state.params}, s, taus, a)
        else:
            quantiles = self.model.apply({"params": self.state.params}, s, taus)

        # Expected value is mean of quantiles
        expected_value = jnp.mean(quantiles, axis=-1)
        return float(expected_value.squeeze())

    def select_action(self, state: int, epsilon: float = 0.1) -> int:
        """ε-greedy action selection."""
        if self.n_actions == 1:
            return 0
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions)
        values = [self.get_value(state, a) for a in range(self.n_actions)]
        return int(np.argmax(values))

    def save(self, path: str | Path):
        """Save the agent."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        state_bytes = serialization.to_bytes(self.state)
        with open(path / "state.msgpack", "wb") as f:
            f.write(state_bytes)

        meta = {
            "gamma": float(self.gamma),
            "learning_rate": float(self.learning_rate),
            "tau": float(self.tau),
            "num_quantiles": int(self.num_quantiles),
            "kappa": float(self.kappa),
            "n_states": int(self.n_states),
            "n_actions": int(self.n_actions),
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        if self.total_hist is not None:
            np.save(path / "loss_history.npy", np.array(self.total_hist, dtype=np.float32))

    @classmethod
    def load(cls, path: str | Path, env: Any, batch_size: int, seed: int = 0):
        """Load agent from checkpoint."""
        path = Path(path)

        with open(path / "meta.json", "r") as f:
            meta = json.load(f)

        agent = cls.create(
            env=env,
            batch_size=batch_size,
            gamma=meta["gamma"],
            learning_rate=meta["learning_rate"],
            tau=meta["tau"],
            seed=seed,
            num_quantiles=meta["num_quantiles"],
            kappa=meta["kappa"],
        )

        with open(path / "state.msgpack", "rb") as f:
            state_bytes = f.read()
        agent.state = serialization.from_bytes(agent.state, state_bytes)

        loss_path = path / "loss_history.npy"
        if loss_path.exists():
            agent.total_hist = list(np.load(loss_path).astype(float))

        return agent