# jax_evaluation.py
"""
JAX evaluation utilities for all distributional RL agents.

Supports:
- jax_lambda_flow / jax_value_flow: Flow-based sampling
- c51: Categorical distribution sampling
- iqn: Quantile-based sampling
"""

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from scipy.stats import wasserstein_distance


def get_mc_return_distribution(
    env,
    policy,
    n_episodes: int = 5000,
    max_steps_per_episode: int = 500,
    gamma: float = 1.0,
    start_state: int = None,
    use_theoretical: bool = True,
):
    """Collect Monte Carlo return samples or use theoretical distribution."""
    if use_theoretical and hasattr(env, "has_theoretical_distribution") and env.has_theoretical_distribution():
        if hasattr(env, "sample_theoretical_distribution"):
            import inspect
            sig = inspect.signature(env.sample_theoretical_distribution)
            if 'gamma' in sig.parameters:
                return env.sample_theoretical_distribution(n_samples=n_episodes, gamma=gamma)
            else:
                env_gamma = getattr(env, 'gamma', None)
                if env_gamma is not None and abs(env_gamma - gamma) > 1e-6:
                    print(f"WARNING: Environment gamma ({env_gamma}) differs from requested gamma ({gamma})")
                    print("         Falling back to Monte Carlo sampling for accurate comparison.")
                else:
                    return env.sample_theoretical_distribution(n_samples=n_episodes)

    # Monte Carlo sampling
    returns = []
    for _ in range(n_episodes):
        if start_state is not None:
            obs, info = env.reset(options={"start_state": start_state})
        else:
            obs, info = env.reset()

        g = 0.0
        done = False
        t = 0
        while (not done) and t < max_steps_per_episode:
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            g += (gamma ** t) * reward
            t += 1
            done = terminated or truncated
        returns.append(g)

    return np.array(returns)


def get_learned_distribution_flow(trainer, state: int, n_samples: int = 1000):
    """Sample from flow-based agents (lambda_flow, value_flow)."""
    from jax_models import ode_solve
    
    rng = jax.random.PRNGKey(42)
    eps_samples = jax.random.normal(rng, (n_samples, 1))
    s = jnp.array([state] * n_samples, dtype=jnp.int32)
    
    a = None
    if trainer.n_actions > 1:
        a = jnp.zeros((n_samples,), dtype=jnp.int32)

    x_samples = ode_solve(
        trainer.model.apply,
        trainer.state.params,
        eps_samples,
        s=s,
        a=a,
        steps=trainer.eval_ode_steps,
    )
    return np.array(x_samples).flatten()


def get_learned_distribution_c51(trainer, state: int, n_samples: int = 1000, action: int = 0):
    """Sample from C51 agent's categorical distribution."""
    s = jnp.array([state], dtype=jnp.int32)
    
    # Get logits from the network - output is [B, n_actions, num_atoms]
    logits = trainer.model.apply({"params": trainer.state.params}, s)
    
    # Select the action's distribution: [num_atoms]
    logits_for_action = logits[0, action, :]
    
    # Convert to probabilities
    probs = jax.nn.softmax(logits_for_action, axis=-1)
    probs = np.array(probs)
    atoms = np.array(trainer.atoms)
    
    # Sample from categorical distribution
    rng = np.random.default_rng(42)
    samples = rng.choice(atoms, size=n_samples, p=probs)
    
    return samples


def get_learned_distribution_iqn(trainer, state: int, n_samples: int = 1000):
    """Sample from IQN agent's quantile distribution."""
    rng = jax.random.PRNGKey(42)
    
    s = jnp.array([state], dtype=jnp.int32)
    # Sample uniform quantile fractions
    taus = jax.random.uniform(rng, (1, n_samples))  # [1, n_samples]
    a = jnp.array([0], dtype=jnp.int32) if trainer.n_actions > 1 else None
    
    # Get quantile values
    if a is not None:
        quantiles = trainer.model.apply({"params": trainer.state.params}, s, taus, a)
    else:
        quantiles = trainer.model.apply({"params": trainer.state.params}, s, taus)
    
    return np.array(quantiles).flatten()


def get_learned_distribution(trainer, state: int, n_samples: int = 1000, action: int = 0):
    """Get learned distribution samples based on agent type."""
    agent_class_name = trainer.__class__.__name__
    
    if agent_class_name in ["JaxDistributionalFlowRL", "JaxValueFlowRL"]:
        return get_learned_distribution_flow(trainer, state, n_samples)
    elif agent_class_name == "JaxC51Agent":
        return get_learned_distribution_c51(trainer, state, n_samples, action)
    elif agent_class_name == "JaxIQNAgent":
        return get_learned_distribution_iqn(trainer, state, n_samples)
    else:
        # Try flow-based as default (for backwards compatibility)
        try:
            return get_learned_distribution_flow(trainer, state, n_samples)
        except AttributeError:
            raise ValueError(f"Unknown agent type: {agent_class_name}. "
                           f"Supported: JaxDistributionalFlowRL, JaxValueFlowRL, JaxC51Agent, JaxIQNAgent")


def plot_cdf(ax, data, label, style, **kwargs):
    """Plot CDF of data."""
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, cdf, style, label=label, **kwargs)


def evaluate_distributional_learning(
    trainer,
    env,
    states_to_test=None,
    gamma: float = 0.95,
    save_path: str = None,
    use_theoretical: bool = False,
):
    """Evaluate the learned return distribution vs theoretical/empirical."""
    if states_to_test is None:
        states_to_test = [0, 1, 2]

    # Determine agent type for labeling
    agent_class_name = trainer.__class__.__name__
    if agent_class_name == "JaxDistributionalFlowRL":
        agent_label = "Lambda Flow"
    elif agent_class_name == "JaxValueFlowRL":
        agent_label = "Value Flow"
    elif agent_class_name == "JaxC51Agent":
        agent_label = "C51"
    elif agent_class_name == "JaxIQNAgent":
        agent_label = "IQN"
    else:
        agent_label = agent_class_name

    fig, axes = plt.subplots(len(states_to_test), 2, figsize=(14, 5 * len(states_to_test)), squeeze=False)

    env_name = getattr(env, "spec", "Unknown Environment")
    if env_name == "Unknown Environment":
        env_name = env.__class__.__name__

    # Check if we can use theoretical distribution with correct gamma
    can_use_theoretical = False
    if use_theoretical:
        if hasattr(env, "has_theoretical_distribution") and env.has_theoretical_distribution():
            env_gamma = getattr(env, 'gamma', None)
            if env_gamma is not None and abs(env_gamma - gamma) < 1e-6:
                can_use_theoretical = True
            else:
                print(f"Warning: Requested theoretical, but env.gamma={env_gamma} != {gamma}. Fallback to MC.")
        else:
            print("Warning: Requested theoretical, but environment does not support it. Fallback to MC.")

    # Set Title
    if can_use_theoretical:
        fig.suptitle(f"{env_name}: {agent_label} vs Theoretical (γ={gamma})", fontsize=16)
    else:
        fig.suptitle(f"{env_name}: {agent_label} vs Empirical (γ={gamma})", fontsize=16)

    all_metrics = {}

    for i, state in enumerate(states_to_test):
        ax_cdf = axes[i, 0]
        ax_pdf = axes[i, 1]

        if can_use_theoretical:
            print(f"Sampling from theoretical distribution for state {state}...")
        else:
            print(f"Collecting MC samples for state {state}...")

        mc_dist = get_mc_return_distribution(
            env,
            lambda s: 0,
            n_episodes=5000,
            gamma=gamma,
            start_state=state,
            use_theoretical=can_use_theoretical,
        )

        print(f"Sampling from {agent_label} model for state {state}...")
        learned_dist = get_learned_distribution(trainer, state, n_samples=5000)

        # --- CDF ---
        plot_cdf(ax_cdf, learned_dist, f"Learned ({agent_label})", "b-", linewidth=2, alpha=0.7)
        if can_use_theoretical:
            plot_cdf(ax_cdf, mc_dist, "Theoretical", "r--", linewidth=2)
        else:
            plot_cdf(ax_cdf, mc_dist, "Empirical (MC)", "r--", linewidth=2)

        ax_cdf.set_title(f"State {state} - CDF")
        ax_cdf.set_ylabel("Quantile (CDF)")
        ax_cdf.legend()
        ax_cdf.grid(True, alpha=0.3)

        # --- PDF/PMF ---
        if can_use_theoretical:
            label_mc = "Theoretical"
        else:
            label_mc = "Empirical (MC)"

        # Determine bin range
        all_data = np.concatenate([learned_dist, mc_dist])
        bin_min, bin_max = np.percentile(all_data, [1, 99])
        bins = np.linspace(bin_min, bin_max, 30)

        ax_pdf.hist(mc_dist, bins=bins, density=True, alpha=0.5, label=label_mc, color="red")
        ax_pdf.hist(learned_dist, bins=bins, density=True, alpha=0.5, label=f"Learned ({agent_label})", color="blue")
        ax_pdf.set_title(f"State {state} - PDF/PMF")
        ax_pdf.set_xlabel("Return")
        ax_pdf.set_ylabel("Density")
        ax_pdf.legend()
        ax_pdf.grid(True, alpha=0.3)

        # Compute metrics
        w_dist = wasserstein_distance(learned_dist, mc_dist)
        learned_mean = np.mean(learned_dist)
        learned_std = np.std(learned_dist)
        mc_mean = np.mean(mc_dist)
        mc_std = np.std(mc_dist)

        metrics = {
            "wasserstein_distance": float(w_dist),
            "learned_mean": float(learned_mean),
            "learned_std": float(learned_std),
            "empirical_mean": float(mc_mean),
            "empirical_std": float(mc_std),
            "mean_error": float(abs(learned_mean - mc_mean)),
            "std_error": float(abs(learned_std - mc_std)),
        }
        all_metrics[f"state_{state}"] = metrics

        print(f"  State {state}: W-dist={w_dist:.4f}, "
              f"Learned mean={learned_mean:.2f} (true: {mc_mean:.2f}), "
              f"Learned std={learned_std:.2f} (true: {mc_std:.2f})")

    # Set common x-label on bottom row
    for ax in axes[-1]:
        ax.set_xlabel("Return")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Distribution comparison plot saved to: {save_path}")

    return fig, all_metrics