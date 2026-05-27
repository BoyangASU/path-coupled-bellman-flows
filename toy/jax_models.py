# jax_models.py
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional


class Velocity(nn.Module):
    """Velocity network for distributional flow matching with state/action conditioning.

    JAX/Flax version of the PyTorch Velocity network. :contentReference[oaicite:1]{index=1}

    Args:
        n_states: Number of discrete states in the environment.
        n_actions: Number of discrete actions (default: 1).
        hidden: Hidden layer size.
        emb_dim: Embedding dimension for states/actions.
    """
    n_states: int
    n_actions: int = 1
    hidden: int = 256
    emb_dim: int = 16

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,          # [B, 1]
        t: jnp.ndarray,          # [B, 1]
        s: Optional[jnp.ndarray] = None,  # [B], int
        a: Optional[jnp.ndarray] = None,  # [B], int
    ) -> jnp.ndarray:
        inputs = [x, t]

        if self.n_states > 0:
            if s is None:
                raise ValueError("State indices 's' must be provided when n_states > 0.")
            s = jnp.clip(s, 0, self.n_states - 1)
            state_emb = nn.Embed(
                num_embeddings=self.n_states,
                features=self.emb_dim,
                name="state_emb",
            )
            inputs.append(state_emb(s))

        if self.n_actions > 1:
            if a is None:
                raise ValueError("Action indices 'a' must be provided when n_actions > 1.")
            a = jnp.clip(a, 0, self.n_actions - 1)
            action_emb = nn.Embed(
                num_embeddings=self.n_actions,
                features=self.emb_dim,
                name="action_emb",
            )
            inputs.append(action_emb(a))

        h = jnp.concatenate(inputs, axis=-1)

        h = nn.relu(nn.Dense(self.hidden)(h))
        h = nn.relu(nn.Dense(self.hidden)(h))
        h = nn.relu(nn.Dense(self.hidden)(h))

        out_dim = 1 if self.n_actions == 1 else self.n_actions
        v = nn.Dense(out_dim)(h)
        return v


def ode_solve(
    apply_fn,
    params,
    x0: jnp.ndarray,
    s: Optional[jnp.ndarray] = None,
    a: Optional[jnp.ndarray] = None,
    steps: int = 20,
):
    """Euler ODE solver for flow matching (JAX version). :contentReference[oaicite:2]{index=2}

    Args:
        apply_fn: Flax apply function, typically `model.apply`.
        params: Parameters for the velocity network.
        x0: Initial samples [B, 1].
        s: State indices [B].
        a: Action indices [B] or None.
        steps: Number of Euler integration steps.

    Returns:
        Final samples after ODE integration [B, 1].
    """
    dt = 1.0 / steps

    def step(x, i):
        t = jnp.full_like(x, i / steps)
        v = apply_fn({"params": params}, x, t, s, a)
        # If multi-action output, average over actions.
        if v.ndim > x.ndim:
            v = v.mean(axis=-1, keepdims=True)
        x_next = x + dt * v
        return x_next, None

    x, _ = jax.lax.scan(step, x0, jnp.arange(steps))
    return x
