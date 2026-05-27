"""
JAX training script for Distributional RL on multiple environments.

Supports multiple agents:
- jax_lambda_flow: Lambda-Flow Distributional RL (default)
- jax_value_flow: Value Flow (λ=0)
- c51: Categorical DQN (C51)
- iqn: Implicit Quantile Network (IQN)

GPU Support:
    By default, JAX will use GPU if available. Use --cpu to force CPU mode.
"""

import os
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def to_serializable(obj):
    """Recursively convert NumPy types to Python-native types for JSON dumping."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def create_agent(agent_name, env, args, gamma, lam):
    """Create the appropriate agent based on agent_name."""
    
    if agent_name == "jax_lambda_flow":
        from agent.jax_agent import JaxDistributionalFlowRL
        return JaxDistributionalFlowRL.create(
            env=env,
            batch_size=args.batch_size,
            gamma=gamma,
            lambda_param=lam,
            learning_rate=args.learning_rate,
            tau=0.005,
            seed=args.seed,
        )
    
    elif agent_name == "jax_value_flow":
        from agent.jax_value_flow_agent import JaxValueFlowRL
        return JaxValueFlowRL.create(
            env=env,
            batch_size=args.batch_size,
            gamma=gamma,
            lambda_param=0.0,  # Value Flow uses λ=0
            learning_rate=args.learning_rate,
            tau=0.005,
            seed=args.seed,
        )
    
    elif agent_name == "c51":
        from agent.jax_c51_agent import JaxC51Agent
        return JaxC51Agent.create(
            env=env,
            batch_size=args.batch_size,
            gamma=gamma,
            learning_rate=args.learning_rate,
            tau=0.005,
            seed=args.seed,
            num_atoms=51,
        )
    
    elif agent_name == "iqn":
        from agent.jax_iqn_agent import JaxIQNAgent
        return JaxIQNAgent.create(
            env=env,
            batch_size=args.batch_size,
            gamma=gamma,
            learning_rate=args.learning_rate,
            tau=0.005,
            seed=args.seed,
            num_quantiles=32,
        )
    
    else:
        raise ValueError(f"Unknown agent: {agent_name}. "
                        f"Supported agents: jax_lambda_flow, jax_value_flow, c51, iqn")


def main():
    parser = argparse.ArgumentParser(
        description="Train JAX Distributional RL on various environments"
    )

    parser.add_argument(
        "--agent",
        type=str,
        default="jax_lambda_flow",
        choices=["jax_lambda_flow", "jax_value_flow", "c51", "iqn"],
        help="Which agent to use: jax_lambda_flow, jax_value_flow, c51, iqn",
    )

    # Environment selection
    parser.add_argument(
        "--env",
        type=str,
        default="solitaire",
        choices=["solitaire", "discrete_mc", "bernoulli", "cliff_episodic", "cliff_continuing"],
        help="Environment to use",
    )

    # Lambda parameter (only for lambda_flow)
    parser.add_argument(
        "--lambda_param",
        type=float,
        default=None,
        help="Lambda for lambda-flow; if None, defaults to gamma",
    )

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--gamma", type=float, default=None, help="Discount factor")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Env-specific
    parser.add_argument("--n_states", type=int, default=20, help="Number of states for DiscreteMCEnv")
    parser.add_argument("--max_episode_steps", type=int, default=100, help="Max episode steps")

    # Evaluation
    parser.add_argument("--eval", action="store_true", help="Run periodic evaluation")
    parser.add_argument("--eval_interval", type=int, default=0, help="Evaluation interval")

    # Logging & Saving
    parser.add_argument("--save_dir", type=str, default="./runs/jax_flow_rl", help="Save directory")
    parser.add_argument("--save_interval", type=int, default=0, help="Checkpoint save interval")
    parser.add_argument("--log_interval", type=int, default=1, help="Log interval (ignored)")

    # Device
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")

    args = parser.parse_args()

    # Set GPU/CPU BEFORE importing JAX
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        print("\n[Device] Running on CPU")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"\n[Device] Using GPU {args.gpu}")

    # Import JAX-related modules AFTER setting device
    from jax_utils import (
        set_random_seeds,
        create_environment,
        get_default_states_to_test,
        plot_training_loss,
    )

    import jax
    try:
        devices = jax.devices()
        print(f"[Device] JAX devices: {devices}")
    except Exception as e:
        print(f"[Device] Warning: Could not detect JAX devices: {e}")

    # Set random seeds
    set_random_seeds(args.seed)

    # Default gamma per env
    if args.gamma is None:
        if args.env == "solitaire":
            gamma = 0.9
        elif args.env == "bernoulli":
            gamma = 0.5
        elif args.env in ("cliff_episodic", "cliff_continuing"):
            gamma = 0.99
        else:
            gamma = 0.95
    else:
        gamma = args.gamma

    # Determine lambda based on agent type
    if args.agent == "jax_value_flow":
        lam = 0.0
    elif args.agent in ["c51", "iqn"]:
        lam = None  # Not used for C51/IQN
    else:
        lam = gamma if args.lambda_param is None else args.lambda_param

    # Prepare directories
    run_root = Path(args.save_dir)
    checkpoints_dir = run_root / "checkpoints"
    plots_dir = run_root / "plots"
    metrics_dir = run_root / "metrics"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = {
        "agent": args.agent,
        "env": args.env,
        "gamma": gamma,
        "lambda_param": lam,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "save_dir": args.save_dir,
    }
    (run_root / "config.json").write_text(json.dumps(config_dict, indent=2))

    print("\n" + "=" * 60)
    print("JAX Training Configuration:")
    print(f"  Agent: {args.agent}")
    print(f"  Environment: {args.env}")
    print(f"  Gamma: {gamma}")
    if lam is not None:
        print(f"  Lambda: {lam}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Seed: {args.seed}")
    print(f"  Save dir: {run_root}")
    print("=" * 60 + "\n")

    # Create environment
    env_kwargs = {"gamma": gamma}
    if args.env == "discrete_mc":
        env_kwargs["n"] = args.n_states
    elif args.env == "bernoulli":
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    env = create_environment(args.env, **env_kwargs)
    print(f"Created environment: {args.env}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}\n")

    # Create agent
    agent = create_agent(args.agent, env, args, gamma, lam)

    print(f"\nTraining {args.agent} on {args.env}")
    pbar = tqdm(range(1, args.epochs + 1), desc="Training")

    losses = []
    best_loss = float("inf")
    best_epoch = -1

    # Training Loop
    for epoch in pbar:
        loss = float(agent.train_step())
        losses.append(loss)

        if not np.isfinite(loss):
            pbar.write(f"\n[Warning] Loss became {loss} at epoch {epoch}. Stopping.")
            break

        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            best_ckpt_path = checkpoints_dir / "best_model"
            agent.save(best_ckpt_path)

        pbar.set_postfix({"loss": f"{loss:.4e}", "best": f"{best_loss:.4e}"})

        # Periodic checkpoint
        if args.save_interval > 0 and epoch % args.save_interval == 0:
            ckpt_path = checkpoints_dir / f"checkpoint_epoch_{epoch}"
            agent.save(ckpt_path)

    print("\nTraining complete!")
    print(f"Best training loss: {best_loss:.4e} at epoch {best_epoch}")

    # Final save
    final_ckpt_path = checkpoints_dir / f"final_model_epoch_{args.epochs}"
    final_ckpt_path.mkdir(parents=True, exist_ok=True)
    agent.save(final_ckpt_path)
    print(f"Final model saved to: {final_ckpt_path}")

    # Save loss history
    losses_arr = np.array(losses, dtype=np.float32)
    loss_history_path = final_ckpt_path / "loss_history.npy"
    np.save(loss_history_path, losses_arr)
    print(f"Loss history ({len(losses_arr)} steps) saved to: {loss_history_path}")

    # Save training summary metrics
    training_metrics = {
        "agent": args.agent,
        "env": args.env,
        "gamma": gamma,
        "lambda_param": lam,
        "epochs": args.epochs,
        "best_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "final_loss": float(losses_arr[-1]) if len(losses_arr) > 0 else None,
        "mean_loss_last_1000": float(np.mean(losses_arr[-1000:])) if len(losses_arr) >= 1000 else float(np.mean(losses_arr)),
        "std_loss_last_1000": float(np.std(losses_arr[-1000:])) if len(losses_arr) >= 1000 else float(np.std(losses_arr)),
    }
    training_metrics_path = metrics_dir / f"training_metrics_{args.agent}.json"
    with training_metrics_path.open("w") as f:
        json.dump(to_serializable(training_metrics), f, indent=2)
    print(f"Training metrics saved to: {training_metrics_path}")

    # Plot training loss
    if args.agent in ["jax_lambda_flow", "jax_value_flow"]:
        plot_title = f"Training Loss - {args.env} ({args.agent}, λ={lam})"
    else:
        plot_title = f"Training Loss - {args.env} ({args.agent})"
    
    loss_plot_path = plots_dir / f"training_loss_{args.env}_{args.agent}.png"

    valid_losses = losses_arr[np.isfinite(losses_arr)]
    if len(valid_losses) > 0:
        plot_training_loss(valid_losses, title=plot_title, save_path=str(loss_plot_path))
        print(f"Training plot saved to: {loss_plot_path}")

    # Final evaluation with PDF/CDF plots
    if args.eval:
        print("\n[Eval] Running final evaluation with PDF/CDF plots...")
        try:
            from jax_evaluation import evaluate_distributional_learning
            states_to_test = get_default_states_to_test(args.env, env)
            
            eval_plot_path = plots_dir / f"eval_final_{args.agent}.png"
            metrics_path = metrics_dir / f"metrics_final_{args.agent}.json"
            
            fig, metrics = evaluate_distributional_learning(
                agent,
                env,
                states_to_test=states_to_test,
                gamma=gamma,
                save_path=str(eval_plot_path),
            )
            plt.close(fig)
            
            with metrics_path.open("w") as f:
                json.dump(to_serializable(metrics), f, indent=2)
            
            print(f"[Eval] PDF/CDF plot saved to: {eval_plot_path}")
            print(f"[Eval] Metrics saved to: {metrics_path}")
        except Exception as e:
            print(f"[Eval] Error during evaluation: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone!")


if __name__ == "__main__":
    main()