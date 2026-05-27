import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from flax import serialization
import optax
import numpy as np
from typing import Any, Dict, Sequence, Optional
from dataclasses import dataclass
from pathlib import Path
import json

# ==========================================
# 1. C51 Network Architecture
# ==========================================

class DistributionalNetwork(nn.Module):
    """
    Predicts the categorical distribution (logits) for each action.
    Output shape: [Batch, n_actions, num_atoms]
    """
    n_states: int
    n_actions: int
    num_atoms: int
    hidden_dims: Sequence[int] = (64, 64)
    emb_dim: int = 16

    @nn.compact
    def __call__(self, s):
        # s: [Batch] or [Batch, 1]
        s = s.squeeze() if s.ndim > 1 else s
        
        # State Embedding
        x = nn.Embed(num_embeddings=self.n_states, features=self.emb_dim)(s)
        
        # MLP Body
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        
        # Output Head: [Batch, n_actions * num_atoms]
        logits = nn.Dense(self.n_actions * self.num_atoms)(x)
        
        # Reshape to [Batch, n_actions, num_atoms]
        logits = logits.reshape((-1, self.n_actions, self.num_atoms))
        return logits

# ==========================================
# 2. Train State
# ==========================================

class C51TrainState(train_state.TrainState):
    target_params: Any

# ==========================================
# 3. The Agent Class
# ==========================================

@dataclass
class JaxC51Agent:
    """
    Categorical DQN (C51) Agent adapted to the 'JaxDistributionalFlowRL' API.
    """
    env: Any
    batch_size: int
    gamma: float
    learning_rate: float
    
    # C51 specific parameters
    v_min: float
    v_max: float
    num_atoms: int
    atoms: jnp.ndarray  # The support vector z
    delta_z: float

    # Training state
    n_states: int
    n_actions: int
    model: DistributionalNetwork
    state: C51TrainState
    rng: jax.Array
    
    # Target update rate
    tau: float = 0.005
    
    # History
    total_hist: list = None

    @classmethod
    def create(
        cls,
        env,
        batch_size: int,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        v_min: float = -10.0, # Adjust based on your env reward scale
        v_max: float = 10.0,  # Adjust based on your env reward scale
        num_atoms: int = 51,
        tau: float = 0.005,
        seed: int = 0,
        **kwargs # Ignore extra args like lambda_param
    ):
        # 1. Environment Specs
        obs_space = env.observation_space
        act_space = env.action_space
        n_states = int(obs_space.n)
        n_actions = int(act_space.n)

        # 2. C51 Support (Atoms)
        # z_i = v_min + i * delta_z
        atoms = jnp.linspace(v_min, v_max, num_atoms)
        delta_z = (v_max - v_min) / (num_atoms - 1)

        # 3. Initialize Network
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        
        model = DistributionalNetwork(
            n_states=n_states, 
            n_actions=n_actions, 
            num_atoms=num_atoms
        )
        
        # Dummy input for initialization [Batch]
        dummy_s = jnp.zeros((1,), dtype=jnp.int32)
        params = model.init(init_rng, dummy_s)["params"]

        # 4. Optimizer & State
        tx = optax.adam(learning_rate)
        
        state = C51TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
            target_params=params # Initialize target = current
        )

        return cls(
            env=env,
            batch_size=batch_size,
            gamma=gamma,
            learning_rate=learning_rate,
            v_min=v_min,
            v_max=v_max,
            num_atoms=num_atoms,
            atoms=atoms,
            delta_z=delta_z,
            n_states=n_states,
            n_actions=n_actions,
            model=model,
            state=state,
            rng=rng,
            tau=tau,
            total_hist=[]
        )

    # ---------- Internal Sampling (Matches jax_agent.py pattern) ----------
    def _sample_batch(self) -> Dict[str, np.ndarray]:
        B = self.batch_size
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for _ in range(B):
            obs, _ = self.env.reset()
            # Random policy for data collection (or you could use epsilon-greedy)
            action = np.random.randint(0, self.n_actions)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)

            states.append(int(obs))
            actions.append(int(action))
            rewards.append(float(reward))
            next_states.append(int(next_obs))
            dones.append(float(terminated or truncated))

        return {
            "states": np.array(states, dtype=np.int32),
            "actions": np.array(actions, dtype=np.int32),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_states": np.array(next_states, dtype=np.int32),
            "dones": np.array(dones, dtype=np.float32),
        }

    # ---------- The C51 Loss Logic ----------
    def _make_train_step(self):
        gamma = self.gamma
        num_atoms = self.num_atoms
        v_min = self.v_min
        v_max = self.v_max
        delta_z = self.delta_z
        atoms = self.atoms
        n_actions = self.n_actions
        tau = self.tau

        @jax.jit
        def train_step_jit(state: C51TrainState, batch: Dict[str, jnp.ndarray], rng: jax.Array):
            states = batch["states"]       # [B]
            actions = batch["actions"]     # [B]
            rewards = batch["rewards"]     # [B]
            next_states = batch["next_states"] # [B]
            dones = batch["dones"]         # [B]

            # ----------------------------------------
            # 1. Compute Target Distribution
            # ----------------------------------------
            
            # (a) Get next state logits from Target Net
            next_logits = state.apply_fn({"params": state.target_params}, next_states) # [B, A, Atoms]
            next_probs = jax.nn.softmax(next_logits, axis=-1)

            # (b) Double DQN Selection: Use Online Net to select best action
            online_next_logits = state.apply_fn({"params": state.params}, next_states)
            online_next_probs = jax.nn.softmax(online_next_logits, axis=-1)
            # Expected Q = sum(p_i * z_i)
            online_next_q = jnp.sum(online_next_probs * atoms, axis=-1) # [B, A]
            best_next_actions = jnp.argmax(online_next_q, axis=-1)      # [B]

            # (c) Select the distribution corresponding to best actions
            # Indexing: [B, Atoms]
            best_next_probs = next_probs[jnp.arange(len(next_states)), best_next_actions]

            # (d) Projected Distribution (Categorical Algorithm)
            # T_z = r + gamma * z
            # We must handle broadcasting: rewards [B, 1] + atoms [1, Atoms]
            rewards = rewards[:, None]
            dones = dones[:, None]
            
            Tz = rewards + (1.0 - dones) * gamma * atoms # [B, Atoms]
            Tz = jnp.clip(Tz, v_min, v_max)
            
            # Calculate indices in the support
            b = (Tz - v_min) / delta_z
            l = jnp.floor(b).astype(jnp.int32)
            u = jnp.ceil(b).astype(jnp.int32)
            
            # Ensure indices are valid (redundant with Tz clip but safe)
            l = jnp.clip(l, 0, num_atoms - 1)
            u = jnp.clip(u, 0, num_atoms - 1)

            # Distribute probability mass
            # m_l = probs * (u - b)
            # m_u = probs * (b - l)
            # We use jax.ops.index_add to accumulate mass into the target distribution 'm'
            
            def project_row(probs, l_row, u_row, b_row):
                m_row = jnp.zeros(num_atoms)
                # If l == u, logic handles it correctly (b-l=0 or u-b=0)
                m_row = m_row.at[l_row].add(probs * (u_row - b_row))
                m_row = m_row.at[u_row].add(probs * (b_row - l_row))
                return m_row

            # Vectorize over batch
            target_probs = jax.vmap(project_row)(best_next_probs, l, u, b)
            # Stop gradient on targets
            target_probs = jax.lax.stop_gradient(target_probs)

            # ----------------------------------------
            # 2. Compute Loss
            # ----------------------------------------
            def loss_fn(params):
                current_logits = state.apply_fn({"params": params}, states) # [B, A, Atoms]
                
                # Select logits for the actions taken
                chosen_action_logits = current_logits[jnp.arange(len(states)), actions] # [B, Atoms]
                
                # Log probabilities
                chosen_action_log_probs = jax.nn.log_softmax(chosen_action_logits, axis=-1)
                
                # Cross Entropy Loss: - sum( target * log_pred )
                loss = -jnp.sum(target_probs * chosen_action_log_probs, axis=-1)
                return jnp.mean(loss)

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params)
            new_state = state.apply_gradients(grads=grads)

            # ----------------------------------------
            # 3. Soft Update Target Network
            # ----------------------------------------
            new_target_params = jax.tree_util.tree_map(
                lambda p, tp: tau * p + (1.0 - tau) * tp,
                new_state.params,
                state.target_params
            )
            new_state = new_state.replace(target_params=new_target_params)

            return new_state, rng, loss

        return train_step_jit

    # ---------- Public Interface ----------

    def train_step(self):
        batch_np = self._sample_batch()
        batch = {k: jnp.array(v) for k, v in batch_np.items()}

        if not hasattr(self, "_train_step_jit"):
            self._train_step_jit = self._make_train_step()

        self.state, self.rng, loss = self._train_step_jit(self.state, batch, self.rng)
        
        loss_val = float(loss)
        self.total_hist.append(loss_val)
        return loss_val

    def get_value(self, state: int, action: int = 0) -> float:
        """Evaluate Expected Value for specific state-action."""
        s = jnp.array([state], dtype=jnp.int32)
        logits = self.state.apply_fn({"params": self.state.params}, s) # [1, A, Atoms]
        probs = jax.nn.softmax(logits, axis=-1)
        
        # Expected value = sum(p * z)
        q_values = jnp.sum(probs * self.atoms, axis=-1) # [1, A]
        
        return float(q_values[0, action])

    def get_distribution(self, state: int, action: int = 0):
        """Returns (support, probs) for plotting."""
        s = jnp.array([state], dtype=jnp.int32)
        logits = self.state.apply_fn({"params": self.state.params}, s)
        probs = jax.nn.softmax(logits, axis=-1)[0, action]
        return self.atoms, probs

    def select_action(self, state: int, epsilon: float = 0.1) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions)
        
        s = jnp.array([state], dtype=jnp.int32)
        logits = self.state.apply_fn({"params": self.state.params}, s)
        probs = jax.nn.softmax(logits, axis=-1)
        q_values = jnp.sum(probs * self.atoms, axis=-1)
        return int(jnp.argmax(q_values))

    # ---------- Saving / Loading ----------

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        state_bytes = serialization.to_bytes(self.state)
        with open(path / "state.msgpack", "wb") as f:
            f.write(state_bytes)
            
        meta = {
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "v_min": self.v_min,
            "v_max": self.v_max,
            "num_atoms": self.num_atoms
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str | Path, env: Any, batch_size: int, seed: int = 0):
        path = Path(path)
        with open(path / "meta.json", "r") as f:
            meta = json.load(f)
            
        agent = cls.create(
            env=env,
            batch_size=batch_size,
            gamma=meta["gamma"],
            learning_rate=meta["learning_rate"],
            v_min=meta["v_min"],
            v_max=meta["v_max"],
            num_atoms=meta["num_atoms"],
            seed=seed
        )
        
        with open(path / "state.msgpack", "rb") as f:
            state_bytes = f.read()
        agent.state = serialization.from_bytes(agent.state, state_bytes)
        
        return agent