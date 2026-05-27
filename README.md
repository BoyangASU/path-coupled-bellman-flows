# Path-Coupled Bellman Flows (PCBF)

Official implementation of **Path-Coupled Bellman Flows for Distributional Reinforcement Learning** (ICML 2026).

[[Paper]](https://arxiv.org/abs/2605.08253)

## Overview

PCBF is a continuous-time distributional RL method that couples current and successor return flows through shared base noise, enforcing Bellman consistency along entire flow trajectories. A λ-parameterized control variate reduces training variance while preserving the Bellman endpoint geometry.

**Key idea:** Instead of enforcing Bellman consistency only at endpoint samples, PCBF encodes it along the full probability path:

```
Successor interpolant:  Z'_t = (1-t)ε + tX'
Current interpolant:    Z_t  = tR + γ̃Z'_t + (1-t)(1-γ̃)ε

BCFM target:    Y = R + γ̃X' − ε
Control variate: C = v_θ⁻(t, Z'_t | s',a') − (X' − ε)
PCBF λ-target:  u = Y + λC
```

- `λ = 0` → unbiased BCFM (high variance)
- `λ = γ` → eliminates noisy X' via velocity prediction (low variance, small bias with shared-noise coupling)

## Installation

```bash
conda create -n pcbf python=3.10
conda activate pcbf

pip install jax[cuda12] jaxlib
pip install flax optax ml-collections
pip install ogbench gymnasium tqdm wandb
pip install d4rl  # optional, for D4RL Adroit tasks
```

## Quick Start

```bash
# OGBench single task
python main.py \
    --env_name=cube-double-play-singletask-task2-v0 \
    --agent=agents/lambda_flow.py \
    --agent.discount=0.995 \
    --agent.lambda_param=0.4 \
    --seed=0

# Toy environments
cd toy
python run_training_jax.py --env=solitaire --gamma=0.9 --lambda_param=0.5
```

## Reproduce Paper Results

### OGBench (Table 1, 8 seeds)

```bash
bash scripts/run_all_seeds.sh scripts/run_ogbench.sh 4
```

### D4RL Adroit (Table 1, 8 seeds)

```bash
bash scripts/run_all_seeds.sh scripts/run_d4rl.sh 4
```

### Visual OGBench (Table 1, 4 seeds)

```bash
bash scripts/run_all_seeds.sh scripts/run_visual.sh 4
```

### Single experiment

```bash
# cube-triple-play, γ=0.995, λ=0.995
python main.py \
    --env_name=cube-triple-play-singletask-task1-v0 \
    --agent=agents/lambda_flow.py \
    --agent.discount=0.995 \
    --agent.lambda_param=0.995 \
    --train_steps=1000000 \
    --seed=0
```

## Hyperparameters

Domain-level hyperparameters from the paper (Table 5). λ is tuned per domain on the task marked with *.

| Domain | γ | λ |
|---|---|---|
| cube-double-play | 0.995 | 0.4 |
| cube-triple-play | 0.995 | 0.995 |
| puzzle-4x4-play | 0.99 | 0.2 |
| scene-play | 0.99 | 0.2 |
| visual-antmaze-teleport | 0.99 | 0.0 |
| visual-cube-double-play | 0.995 | 0.9 |

D4RL Adroit uses per-task λ; see `scripts/run_d4rl.sh` for details.

All methods share the following defaults:

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 3×10⁻⁴ |
| Batch size | 256 |
| MLP hidden dims | (512, 512, 512, 512) |
| Activation | GELU |
| Layer norm | Yes |
| Flow steps (Euler) | 10 |
| Rejection sampling candidates | 16 |
| Target network τ | 0.005 |
| Q ensembles | 2 |

## Repository Structure

```
path-coupled-bellman-flows/
├── main.py                       # Training entry point (OGBench / D4RL)
├── agents/
│   ├── __init__.py               # Agent registry
│   └── lambda_flow.py            # PCBF agent (Algorithm 1)
├── envs/
│   ├── env_utils.py              # OGBench environment wrapper
│   └── d4rl_utils.py             # D4RL dataset loading
├── utils/
│   ├── datasets.py               # Dataset and replay buffer
│   ├── encoders.py               # IMPALA visual encoder
│   ├── evaluation.py             # Evaluation loop
│   ├── flax_utils.py             # TrainState, ModuleDict, save/restore
│   ├── log_utils.py              # CSV and W&B logging
│   └── networks.py               # MLP, ValueVectorField, ActorVectorField
├── toy/                          # Toy environment experiments
│   ├── agent/                    # PCBF agent for discrete envs
│   ├── gym_environments/         # Solitaire, Bernoulli, Discrete MC
│   ├── jax_models.py             # Velocity network
│   ├── jax_evaluation.py         # Evaluation utilities
│   ├── jax_utils.py              # JAX helper functions
│   └── run_training_jax.py       # Toy training script
└── scripts/
    ├── run_ogbench.sh            # All OGBench domains
    ├── run_d4rl.sh               # D4RL Adroit tasks
    ├── run_visual.sh             # Visual OGBench tasks
    └── run_all_seeds.sh          # Multi-seed launcher
```

## Results

### Offline RL (Table 1)

| Domain | IQN | CODAC | FQL | IQL | Value Flows | **PCBF** |
|---|---|---|---|---|---|---|
| cube-double-play | 42±8 | 61±6 | 29±6 | 7±1 | 69±4 | **71±5** |
| scene-play | 40±1 | 55±1 | 56±2 | 28±3 | **59±4** | 54±4 |
| puzzle-4x4-play | 27±4 | 20±18 | 17±5 | 7±2 | 27±4 | **30±4** |
| cube-triple-play | 6±0 | 2±1 | 4±2 | 1±1 | **14±3** | 4±1 |
| D4RL adroit | 66±5 | 69±0 | **71±4** | 70 | 65±2 | 69±2 |

Bold = within 95% of best. Results averaged over 8 seeds.

## Citation

```bibtex
@inproceedings{xu2026pathcoupled,
  title={Path-Coupled Bellman Flows for Distributional Reinforcement Learning},
  author={Xu, Boyang and Zou, Qing and Yang, Siqin and Yan, Hao},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
```

## Acknowledgements

This codebase is built on [FQL](https://github.com/seohongpark/fql) and [Value Flows](https://github.com/chongyi-zheng/value-flows). We thank Research Computing at Arizona State University for providing A100 GPU resources on the Sol supercomputer.

## License

MIT
