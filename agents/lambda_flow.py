from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, ValueVectorField

class LambdaFlowAgent(flax.struct.PyTreeNode):
    """Lambda-transform flow-matching agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        batch_size = batch["actions"].shape[0]
        rng, t_rng, eps_rng = jax.random.split(rng, 3)

        observations = batch["observations"]
        actions = batch["actions"]
        rewards = jnp.expand_dims(batch["rewards"], axis=-1)
        next_observations = batch["next_observations"]
        if "dones" in batch:
            dones = jnp.expand_dims(batch["dones"], axis=-1)
        elif "terminals" in batch:
            dones = jnp.expand_dims(batch["terminals"], axis=-1)
        elif "masks" in batch:
            dones = jnp.expand_dims(1.0 - batch["masks"], axis=-1)
        else:
            dones = jnp.zeros((batch_size, 1))
        terminal_current = batch.get("terminal_current", jnp.zeros((batch_size, 1)))

        t = jax.random.uniform(t_rng, (batch_size, 1))
        eps = jax.random.normal(eps_rng, (batch_size, 1))

        gamma = self.config["discount"]
        lam = self.config["lambda_param"]
        gamma_mask = gamma * (1.0 - dones)
        lambda_mask = lam * (1.0 - dones)

        next_actions = batch.get("next_actions", actions)

        # Compute target returns using a single target critic
        x_prime = self.compute_flow_returns(
            eps, next_observations, next_actions, flow_network_name="target_critic_flow1"
        )
        x_prime = jnp.where(dones, 0.0, x_prime)

        z_t_prime = t * x_prime + (1.0 - t) * eps
        z_t_terminal = (1.0 - t) * eps
        z_t_non_terminal = t * rewards + gamma_mask * z_t_prime + (1.0 - t) * (1.0 - gamma_mask) * eps
        z_t = jnp.where(terminal_current, z_t_terminal, z_t_non_terminal)

        # Single target velocity
        target_velocity = self.network.select("target_critic_flow1")(z_t_prime, t, next_observations, next_actions)
        target_velocity = jnp.where(dones, -eps, target_velocity)

        epsilon_term = (lam - 1.0) * eps - dones * lam * eps
        v_target = rewards + lambda_mask * target_velocity + (gamma_mask - lambda_mask) * x_prime + epsilon_term
        v_target = jnp.where(terminal_current, -eps, v_target)

        # Single critic prediction and loss
        v_pred = self.network.select("critic_flow1")(z_t, t, observations, actions, params=grad_params)
        critic_loss = ((v_pred - v_target) ** 2).mean()

        return critic_loss, {"critic_loss": critic_loss}

    def actor_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch["actions"].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch["actions"]
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select("actor_flow")(
            batch["observations"], x_t, t, params=grad_params
        )
        actor_loss = jnp.mean((pred - vel) ** 2)
        return actor_loss, {"actor_loss": actor_loss}

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, critic_rng, actor_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "critic_flow1")
        self.target_update(new_network, "critic_flow2")

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=("flow_network_name",))
    def compute_flow_returns(
        self,
        noises,
        observations,
        actions,
        init_times=None,
        end_times=None,
        flow_network_name="critic_flow",
    ):
        noisy_returns = noises
        if init_times is None:
            init_times = jnp.zeros((*noisy_returns.shape[:-1], 1), dtype=noisy_returns.dtype)
        if end_times is None:
            end_times = jnp.ones((*noisy_returns.shape[:-1], 1), dtype=noisy_returns.dtype)
        step_size = (end_times - init_times) / self.config["num_flow_steps"]

        def func(carry, i):
            (noisy_returns,) = carry
            times = i * step_size + init_times
            vector_field = self.network.select(flow_network_name)(
                noisy_returns, times, observations, actions
            )
            new_noisy_returns = noisy_returns + step_size * vector_field
            if self.config["clip_flow_returns"]:
                new_noisy_returns = jnp.clip(
                    new_noisy_returns,
                    self.config["min_reward"] / (1 - self.config["discount"]),
                    self.config["max_reward"] / (1 - self.config["discount"]),
                )
            return (new_noisy_returns,), None

        (noisy_returns,), _ = jax.lax.scan(
            func, (noisy_returns,), jnp.arange(self.config["num_flow_steps"])
        )
        return noisy_returns

    @jax.jit
    def compute_flow_actions(self, noises, observations, init_times=None, end_times=None):
        noisy_actions = noises
        if init_times is None:
            init_times = jnp.zeros((*noisy_actions.shape[:-1], 1), dtype=noisy_actions.dtype)
        if end_times is None:
            end_times = jnp.ones((*noisy_actions.shape[:-1], 1), dtype=noisy_actions.dtype)
        step_size = (end_times - init_times) / self.config["num_flow_steps"]

        def func(carry, i):
            (noisy_actions,) = carry
            times = i * step_size + init_times
            vector_field = self.network.select("actor_flow")(
                observations, noisy_actions, times
            )
            new_noisy_actions = noisy_actions + vector_field * step_size
            if self.config["clip_flow_actions"]:
                new_noisy_actions = jnp.clip(new_noisy_actions, -1, 1)
            return (new_noisy_actions,), None

        (noisy_actions,), _ = jax.lax.scan(
            func, (noisy_actions,), jnp.arange(self.config["num_flow_steps"])
        )
        if not self.config["clip_flow_actions"]:
            noisy_actions = jnp.clip(noisy_actions, -1, 1)
        return noisy_actions

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        action_seed, q_seed = jax.random.split(seed, 2)
        actor_noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config["ob_dims"])],
                self.config["num_samples"],
                self.config["action_dim"],
            ),
        )
        n_observations = jnp.repeat(
            jnp.expand_dims(observations, -2),
            self.config["num_samples"],
            axis=-2,
        )
        flow_actions = self.compute_flow_actions(actor_noises, n_observations)

        q_noises = jax.random.normal(
            q_seed,
            (*observations.shape[: -len(self.config["ob_dims"])], self.config["num_samples"], 1),
        )
        q1 = (q_noises + self.network.select("critic_flow1")(
            q_noises, jnp.zeros_like(q_noises), n_observations, flow_actions
        )).squeeze(-1)
        q2 = (q_noises + self.network.select("critic_flow2")(
            q_noises, jnp.zeros_like(q_noises), n_observations, flow_actions
        )).squeeze(-1)
        if self.config["clip_flow_returns"]:
            q1 = jnp.clip(
                q1,
                self.config["min_reward"] / (1 - self.config["discount"]),
                self.config["max_reward"] / (1 - self.config["discount"]),
            )
            q2 = jnp.clip(
                q2,
                self.config["min_reward"] / (1 - self.config["discount"]),
                self.config["max_reward"] / (1 - self.config["discount"]),
            )
        if self.config["q_agg"] == "min":
            q = jnp.minimum(q1, q2)
        else:
            q = (q1 + q2) / 2
        if len(q.shape) > 1:
            actions = flow_actions[jnp.arange(q.shape[0]), jnp.argmax(q, axis=-1)]
        else:
            actions = flow_actions[jnp.argmax(q, axis=-1)]
        return actions

    @classmethod
    def create(cls, seed, example_batch, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch["observations"]
        ex_actions = example_batch["actions"]
        ex_returns = ex_actions[..., :1]
        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        min_reward = example_batch["min_reward"]
        max_reward = example_batch["max_reward"]

        encoders = {}
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["critic_flow"] = encoder_module()
            encoders["target_critic_flow"] = encoder_module()
            encoders["actor_flow"] = encoder_module()

        critic_flow1_def = ValueVectorField(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["value_layer_norm"],
            num_ensembles=1,
            encoder=encoders.get("critic_flow"),
        )
        critic_flow2_def = ValueVectorField(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["value_layer_norm"],
            num_ensembles=1,
            encoder=encoders.get("critic_flow"),
        )
        target_critic_flow1_def = ValueVectorField(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["value_layer_norm"],
            num_ensembles=1,
            encoder=encoders.get("target_critic_flow"),
        )
        target_critic_flow2_def = ValueVectorField(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["value_layer_norm"],
            num_ensembles=1,
            encoder=encoders.get("target_critic_flow"),
        )
        actor_flow_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("actor_flow"),
        )

        network_info = dict(
            critic_flow1=(critic_flow1_def, (ex_returns, ex_times, ex_observations, ex_actions)),
            critic_flow2=(critic_flow2_def, (ex_returns, ex_times, ex_observations, ex_actions)),
            target_critic_flow1=(target_critic_flow1_def, (ex_returns, ex_times, ex_observations, ex_actions)),
            target_critic_flow2=(target_critic_flow2_def, (ex_returns, ex_times, ex_observations, ex_actions)),
            actor_flow=(actor_flow_def, (ex_observations, ex_actions, ex_times)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        # network_tx = optax.adam(learning_rate=config["lr"])
        
        network_tx = optax.chain(
            optax.clip_by_global_norm(1.0),  # grad clipping
            optax.adam(learning_rate=config["lr"])
        )
        
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params["modules_target_critic_flow1"] = params["modules_critic_flow1"]
        params["modules_target_critic_flow2"] = params["modules_critic_flow2"]

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim
        config["min_reward"] = min_reward
        config["max_reward"] = max_reward
        config["lambda_param"] = float(config["lambda_param"])
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="lambda_flow",
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            min_reward=ml_collections.config_dict.placeholder(float),
            max_reward=ml_collections.config_dict.placeholder(float),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=True,
            value_layer_norm=True,
            discount=0.99,
            lambda_param=ml_collections.config_dict.placeholder(float),
            tau=0.005,
            ret_agg="mean",
            q_agg="mean",
            clip_flow_actions=True,
            clip_flow_returns=True,
            num_samples=16,
            num_flow_steps=10,
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config
