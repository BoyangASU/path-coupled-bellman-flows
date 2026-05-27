# jax_agent.py
"""
JAX Agent for Lambda-Flow Distributional RL

Implements the lambda-transform algorithm:
    v_target = r + λ*v_θ'(z_t', t|s') + (γ-λ)*x' + (λ-1)*ε
"""

from dataclasses import dataclass
from typing import Any, Dict

import jax
import jax.numpy as jnp
from flax.training import train_state
from flax.core import FrozenDict
import optax
import numpy as np

from jax_models import Velocity, ode_solve
from pathlib import Path
import json
from flax import serialization


class FlowTrainState(train_state.TrainState):
    target_params: FrozenDict


@dataclass
class JaxDistributionalFlowRL:
    """JAX version of DistributionalFlowRL with lambda-transform target.

    Implements the lambda-transform algorithm:
        v_target = r + λ*v_θ'(z_t', t|s') + (γ-λ)*x' + (λ-1)*ε
    
    When λ = 0: reduces to BCFM (pure bootstrapping)
    When λ = γ: standard lambda-flow
    """
    env: Any
    batch_size: int
    gamma: float = 0.9
    lam: float = 0.9
    learning_rate: float = 5e-3
    tau: float = 0.005
    train_ode_steps: int = 20
    eval_ode_steps: int = 50
    rng: jax.Array = None
    n_states: int = None
    n_actions: int = None
    model: Velocity = None
    state: FlowTrainState = None
    total_hist: list = None

    @classmethod
    def create(
        cls,
        env,
        batch_size: int,
        gamma: float = 0.9,
        lambda_param: float = None,
        learning_rate: float = 5e-3,
        tau: float = 0.005,
        seed: int = 0,
    ):
        obs_space = env.observation_space
        act_space = env.action_space

        if not hasattr(obs_space, "n") or not hasattr(act_space, "n"):
            raise NotImplementedError("JAX agent currently supports only discrete envs.")

        n_states = int(obs_space.n)
        n_actions = int(act_space.n)

        lam = float(lambda_param) if lambda_param is not None else float(gamma)

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        x0 = jnp.zeros((1, 1))
        t0 = jnp.zeros((1, 1))
        s0 = jnp.zeros((1,), dtype=jnp.int32)
        a0 = jnp.zeros((1,), dtype=jnp.int32) if n_actions > 1 else None

        model = Velocity(n_states=n_states, n_actions=n_actions)

        if a0 is None:
            params = model.init(init_rng, x0, t0, s0)["params"]
        else:
            params = model.init(init_rng, x0, t0, s0, a0)["params"]

        tx = optax.adam(learning_rate)

        state = FlowTrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
            target_params=params,
        )

        return cls(
            env=env,
            batch_size=batch_size,
            gamma=float(gamma),
            lam=float(lam),
            learning_rate=learning_rate,
            tau=tau,
            train_ode_steps=20,
            eval_ode_steps=50,
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
        terminal_current = []
        
        has_terminal_check = hasattr(self.env, '_is_terminal')

        for _ in range(B):
            # Occasionally sample from terminal states if available
            if np.random.random() < 0.1 and has_terminal_check:
                if self.n_states > 0:
                    terminal_states = [s for s in range(self.n_states) if self.env._is_terminal(s)]
                    if terminal_states:
                        obs = np.random.choice(terminal_states)
                        states.append(int(obs))
                        actions.append(np.random.randint(0, self.n_actions) if self.n_actions > 1 else 0)
                        rewards.append(0.0)
                        next_states.append(int(obs))
                        dones.append(1.0)
                        terminal_current.append(1.0)
                        continue
            
            obs, _ = self.env.reset()
            action = np.random.randint(0, self.n_actions) if self.n_actions > 1 else 0
            next_obs, reward, terminated, truncated, _ = self.env.step(action)

            states.append(int(obs))
            actions.append(int(action))
            rewards.append(float(reward))
            next_states.append(int(next_obs))
            dones.append(float(terminated or truncated))
            
            if has_terminal_check:
                terminal_current.append(1.0 if self.env._is_terminal(int(obs)) else 0.0)
            else:
                terminal_current.append(0.0)

        batch = {
            "states": np.array(states, dtype=np.int32),
            "actions": np.array(actions, dtype=np.int32),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_states": np.array(next_states, dtype=np.int32),
            "dones": np.array(dones, dtype=np.float32),
            "terminal_current": np.array(terminal_current, dtype=np.float32),
        }
        return batch

    def _make_train_step(self):
        """Create a jitted train_step function."""

        gamma = self.gamma
        lam = self.lam
        n_actions = self.n_actions
        tau = self.tau
        model = self.model
        ode_steps = self.train_ode_steps

        @jax.jit
        def train_step_jit(
            state: FlowTrainState, 
            batch: Dict[str, jnp.ndarray], 
            rng: jax.Array,
        ):
            rng, t_rng, eps_rng = jax.random.split(rng, 3)

            B = batch["states"].shape[0]
            states = batch["states"]
            actions = batch["actions"]
            rewards = batch["rewards"].reshape(B, 1)
            next_states = batch["next_states"]
            dones = batch["dones"].reshape(B, 1)
            terminal_current = batch.get("terminal_current", jnp.zeros((B, 1))).reshape(B, 1)

            t = jax.random.uniform(t_rng, (B, 1))
            eps = jax.random.normal(eps_rng, (B, 1))

            gamma_mask = gamma * (1.0 - dones)
            lambda_mask = lam * (1.0 - dones)
            next_actions = actions if n_actions > 1 else None

            # x' from target flow at s'
            x_prime = ode_solve(
                model.apply,
                state.target_params,
                eps,
                s=next_states,
                a=next_actions,
                steps=ode_steps,
            )
            if x_prime.ndim == 1:
                x_prime = x_prime[:, None]
            
            # Zero out x' for terminal transitions
            x_prime = jnp.where(dones, 0.0, x_prime)

            # z_t' = t * x' + (1-t) * eps
            z_t_prime = t * x_prime + (1.0 - t) * eps
            
            # z_t bridge construction
            z_t_terminal_current = (1.0 - t) * eps
            z_t_non_terminal = (
                t * rewards
                + gamma_mask * z_t_prime
                + (1.0 - t) * (1.0 - gamma_mask) * eps
            )
            z_t = jnp.where(terminal_current, z_t_terminal_current, z_t_non_terminal)

            # Compute target velocity v_θ'(z_t', t | s', a')
            if n_actions > 1:
                target_velocity = model.apply(
                    {"params": state.target_params},
                    z_t_prime,
                    t,
                    next_states,
                    next_actions,
                )
            else:
                target_velocity = model.apply(
                    {"params": state.target_params},
                    z_t_prime,
                    t,
                    next_states,
                    None,
                )
            if target_velocity.ndim == 1:
                target_velocity = target_velocity[:, None]
            
            # For terminal next states, target velocity should be -eps
            target_velocity = jnp.where(dones, -eps, target_velocity)

            # Lambda-transform target:
            # v_target = r + λ*v_θ'(z_t', t|s') + (γ-λ)*x' + (λ-1)*ε
            epsilon_term = (lam - 1.0) * eps - dones * lam * eps
            
            v_target = (
                rewards
                + lambda_mask * target_velocity
                + (gamma_mask - lambda_mask) * x_prime
                + epsilon_term
            )
            
            # For terminal current states, target is -eps (Dirac at 0)
            v_target = jnp.where(terminal_current, -eps, v_target)

            def loss_fn(params):
                if n_actions > 1:
                    v_pred = model.apply({"params": params}, z_t, t, states, actions)
                else:
                    v_pred = model.apply({"params": params}, z_t, t, states, None)
                if v_pred.ndim == 1:
                    v_pred = v_pred[:, None]
                return jnp.mean((v_pred - v_target) ** 2)

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params)
            new_state = state.apply_gradients(grads=grads)

            # Polyak averaging for target network
            new_target_params = jax.tree_util.tree_map(
                lambda p, tp: tau * p + (1.0 - tau) * tp,
                new_state.params,
                state.target_params,
            )

            new_state = new_state.replace(target_params=new_target_params)
            
            return new_state, rng, loss

        return train_step_jit

    def train_step(self):
        """Sample a batch from env and perform one JAX update."""
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
        """Estimate value using flow model."""
        rng, eps_rng = jax.random.split(self.rng)
        self.rng = rng
        eps = jax.random.normal(eps_rng, (1, 1))
        s = jnp.array([state], dtype=jnp.int32)
        a = jnp.array([action], dtype=jnp.int32) if self.n_actions > 1 else None

        x = ode_solve(
            self.model.apply,
            self.state.params,
            eps,
            s=s,
            a=a,
            steps=self.eval_ode_steps,
        )
        return float(x.squeeze())

    def select_action(self, state: int, epsilon: float = 0.1) -> int:
        """ε-greedy over actions using current value estimates."""
        if self.n_actions == 1:
            return 0
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions)
        values = [self.get_value(state, a) for a in range(self.n_actions)]
        return int(np.argmax(values))

    def save(self, path: str | Path):
        """Save the agent's train state and metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        state_bytes = serialization.to_bytes(self.state)
        with open(path / "state.msgpack", "wb") as f:
            f.write(state_bytes)

        meta = {
            "gamma": float(self.gamma),
            "lam": float(self.lam),
            "learning_rate": float(self.learning_rate),
            "tau": float(self.tau),
            "train_ode_steps": int(self.train_ode_steps),
            "eval_ode_steps": int(self.eval_ode_steps),
            "n_states": int(self.n_states),
            "n_actions": int(self.n_actions),
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        if self.total_hist is not None:
            np.save(path / "loss_history.npy", np.array(self.total_hist, dtype=np.float32))

    @classmethod
    def load(cls, path: str | Path, env: Any, batch_size: int, seed: int = 0):
        """Recreate an agent from a saved checkpoint."""
        path = Path(path)

        with open(path / "meta.json", "r") as f:
            meta = json.load(f)

        agent = cls.create(
            env=env,
            batch_size=batch_size,
            gamma=meta["gamma"],
            lambda_param=meta["lam"],
            learning_rate=meta["learning_rate"],
            tau=meta["tau"],
            seed=seed,
        )

        with open(path / "state.msgpack", "rb") as f:
            state_bytes = f.read()
        agent.state = serialization.from_bytes(agent.state, state_bytes)

        loss_path = path / "loss_history.npy"
        if loss_path.exists():
            agent.total_hist = list(np.load(loss_path).astype(float))

        return agent