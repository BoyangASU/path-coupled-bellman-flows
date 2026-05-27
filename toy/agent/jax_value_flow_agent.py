# jax_value_flow_agent.py

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
class JaxValueFlowRL:
    """
    JAX Value-Flow RL with two losses (unweighted in the sense w ≡ 1):

        L_ValueFlow(v) = L_DCFM(v) + λ * L_BCFM(v)

    where λ is a scalar hyperparameter (lambda_param in create()).
    """

    env: Any
    batch_size: int
    gamma: float = 0.9
    learning_rate: float = 5e-3
    tau: float = 0.005
    train_ode_steps: int = 20
    eval_ode_steps: int = 50
    rng: jax.Array = None  # PRNGKey
    n_states: int = None
    n_actions: int = None
    model: Velocity = None
    state: FlowTrainState = None
    total_hist: list = None  # Python-side loss history

    # λ for the BCFM term
    bcfm_lambda: float = 0.0

    # ---------- construction ----------

    @classmethod
    def create(
        cls,
        env,
        batch_size: int,
        gamma: float = 0.9,
        lambda_param: float | None = None,   # used as BCFM weight λ
        learning_rate: float = 5e-3,
        tau: float = 0.005,
        seed: int = 0,
    ):
        """
        lambda_param here is *not* TD(λ); it's the weight on L_BCFM.
        If lambda_param is None, we default to 0 (pure DCFM).
        """
        obs_space = env.observation_space
        act_space = env.action_space

        if not hasattr(obs_space, "n") or not hasattr(act_space, "n"):
            raise NotImplementedError("JAX agent currently supports only discrete envs.")

        n_states = int(obs_space.n)
        n_actions = int(act_space.n)

        # λ for BCFM; default 0 if not provided
        bcfm_lambda = 0.0 if lambda_param is None else float(lambda_param)

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        # Example inputs for network initialization
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
            bcfm_lambda=bcfm_lambda,
        )

    # ---------- environment sampling (Python, non-JAX) ----------

    def _sample_batch(self) -> Dict[str, np.ndarray]:
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

    # ---------- JAX loss and train step ----------

    def _make_train_step(self):
        """
        L_ValueFlow(v) = L_DCFM(v) + λ * L_BCFM(v)

        DCFM:
            x'    ~ target flow at (s', a')
            z_t'  = (1 - t) * ε + t * x'
            y     = r + γ (1-d) * z_t'
            L_DCFM = E[(vθ(y,t,s,a) - v̄(z_t',t,s',a'))^2]

        BCFM:
            z_TD^1 = r + γ (1-d) * x'
            z_TD^t = (1 - t) * ε + t * z_TD^1
            L_BCFM = E[(vθ(z_TD^t,t,s,a) - (z_TD^1 - ε))^2]
        """

        gamma = self.gamma
        n_actions = self.n_actions
        tau = self.tau
        model = self.model
        ode_steps = self.train_ode_steps
        bcfm_lambda = self.bcfm_lambda

        @jax.jit
        def train_step_jit(state: FlowTrainState, batch: Dict[str, jnp.ndarray], rng: jax.Array):
            rng, t_rng, eps_rng = jax.random.split(rng, 3)

            B = batch["states"].shape[0]
            states = batch["states"]
            actions = batch["actions"]
            rewards = batch["rewards"].reshape(B, 1)
            next_states = batch["next_states"]
            dones = batch["dones"].reshape(B, 1)

            t = jax.random.uniform(t_rng, (B, 1))
            eps = jax.random.normal(eps_rng, (B, 1))
            gamma_mask = gamma * (1.0 - dones)

            next_actions = actions if n_actions > 1 else None

            # ----- 1) Flow sample x' at (s', a') (t=1) -----
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

            # ----- 2) DCFM construction -----
            z_t_prime = (1.0 - t) * eps + t * x_prime
            y = rewards + gamma_mask * z_t_prime  # r + γ z_t'

            # ----- 3) BCFM construction -----
            z_TD_1 = rewards + gamma_mask * x_prime       # z_TD^1
            z_TD_t = (1.0 - t) * eps + t * z_TD_1         # z_TD^t
            v_TD_target = z_TD_1 - eps                    # z_TD^1 - ε

            def loss_fn(params):
                # ---- DCFM term ----
                if n_actions > 1:
                    v_pred_d = model.apply(
                        {"params": params},
                        y,
                        t,
                        states,
                        actions,
                    )
                    v_target_d = model.apply(
                        {"params": state.target_params},
                        z_t_prime,
                        t,
                        next_states,
                        next_actions,
                    )
                else:
                    v_pred_d = model.apply(
                        {"params": params},
                        y,
                        t,
                        states,
                        None,
                    )
                    v_target_d = model.apply(
                        {"params": state.target_params},
                        z_t_prime,
                        t,
                        next_states,
                        None,
                    )

                if v_pred_d.ndim == 1:
                    v_pred_d = v_pred_d[:, None]
                if v_target_d.ndim == 1:
                    v_target_d = v_target_d[:, None]

                diff2_d = (v_pred_d - v_target_d) ** 2  # [B, 1]
                L_d = jnp.mean(diff2_d)

                # ---- BCFM term ----
                if n_actions > 1:
                    v_pred_b = model.apply(
                        {"params": params},
                        z_TD_t,
                        t,
                        states,
                        actions,
                    )
                else:
                    v_pred_b = model.apply(
                        {"params": params},
                        z_TD_t,
                        t,
                        states,
                        None,
                    )
                if v_pred_b.ndim == 1:
                    v_pred_b = v_pred_b[:, None]

                diff2_b = (v_pred_b - v_TD_target) ** 2   # [B, 1]
                L_b = jnp.mean(diff2_b)

                loss = L_d + bcfm_lambda * L_b
                return loss

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params)
            new_state = state.apply_gradients(grads=grads)

            # Polyak target update
            new_target_params = jax.tree_util.tree_map(
                lambda p, tp: tau * p + (1.0 - tau) * tp,
                new_state.params,
                state.target_params,
            )
            new_state = new_state.replace(target_params=new_target_params)

            return new_state, rng, loss

        return train_step_jit

    # ---------- public API ----------

    def train_step(self):
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

    # --- value & greedy action (for evaluation) ---

    def get_value(self, state: int, action: int = 0) -> float:
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
        if self.n_actions == 1:
            return 0
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions)
        values = [self.get_value(state, a) for a in range(self.n_actions)]
        return int(np.argmax(values))

    # ---------- saving / loading ----------

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        state_bytes = serialization.to_bytes(self.state)
        with open(path / "state.msgpack", "wb") as f:
            f.write(state_bytes)

        meta = {
            "gamma": float(self.gamma),
            "learning_rate": float(self.learning_rate),
            "tau": float(self.tau),
            "train_ode_steps": int(self.train_ode_steps),
            "eval_ode_steps": int(self.eval_ode_steps),
            "n_states": int(self.n_states),
            "n_actions": int(self.n_actions),
            "bcfm_lambda": float(self.bcfm_lambda),
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        if self.total_hist is not None:
            np.save(path / "loss_history.npy", np.array(self.total_hist, dtype=np.float32))

    @classmethod
    def load(cls, path: str | Path, env: Any, batch_size: int, seed: int = 0):
        path = Path(path)

        with open(path / "meta.json", "r") as f:
            meta = json.load(f)

        agent = cls.create(
            env=env,
            batch_size=batch_size,
            gamma=meta["gamma"],
            lambda_param=meta.get("bcfm_lambda", 0.0),
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
