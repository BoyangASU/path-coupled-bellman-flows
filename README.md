<h1 align="center">Path-Coupled Bellman Flows for Distributional Reinforcement Learning</h1>

<p align="center">
  Official implementation of <b>Path-Coupled Bellman Flows for Distributional Reinforcement Learning</b>
  <br>
  <i>ICML 2026</i>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.08253"><img src="https://img.shields.io/badge/arXiv-2605.08253-b31b1b.svg" alt="Paper"></a>
  <a href="https://github.com/BoyangASU/path-coupled-bellman-flows"><img src="https://img.shields.io/badge/GitHub-Code-blue.svg" alt="Code"></a>
  <a href="https://icml.cc"><img src="https://img.shields.io/badge/ICML-2026-4b44ce.svg" alt="ICML 2026"></a>
</p>

<p align="center">
  <img src="figures/comparison.png" width="700">
  <br>
  <em><b>Figure 1.</b> Architecture of Path-Coupled Bellman Flows (PCBF). A shared noise variable is propagated along the Bellman path, producing a path-consistent flow-matching objective for return distributions.</em>
</p>

<p align="center">
  <img src="https://github.com/BoyangASU/path-coupled-bellman-flows/raw/main/figures/demo.gif" width="700">
  <br>
  <em><b>Figure 2.</b> Demonstration of the trained agent using Path-Coupled Bellman Flows on Discrete MC Environment.</em>
</p>

## Overview

Path-Coupled Bellman Flows (PCBF) introduces a flow-based perspective for distributional reinforcement learning. Rather than treating each return as an independent sample, PCBF couples the noise along a Bellman trajectory, yielding a path-consistent flow-matching objective for the return distribution. The method was accepted as a regular-track paper at ICML 2026.

**Core idea.** Standard flow matching learns a velocity field that transports Gaussian noise $\epsilon$ into return samples. The distributional Bellman equation says the current return equals reward plus a discounted successor return: $Z(s,a) \stackrel{d}{=} R + \gamma Z(s',a')$. PCBF exploits this by using the *same* noise $\epsilon$ to generate both current and successor returns, so their flow paths are geometrically coupled:

$$Z_t^{s'} = (1-t)\cdot \epsilon + t\cdot X', \qquad Z_t^{s} = t\cdot R + \tilde{\gamma}\cdot Z_t^{s'} + (1-t)(1-\tilde{\gamma})\cdot \epsilon$$

Both paths start from the same $\epsilon$ at $t=0$ and reach their Bellman-related endpoints at $t=1$. Differentiating gives the BCFM velocity target $Y = R + \tilde{\gamma}X' - \epsilon$, which is unbiased but high-variance because it depends on the noisy sample $X'$.

PCBF reduces this variance with a control variate built from the successor velocity field, yielding a $\lambda$-parameterized family of targets:

$$u = Y + \lambda\cdot C, \qquad C = v_{\theta^-}(t,\, Z_t^{s'} \mid s', a') - (X' - \epsilon)$$

- $\lambda = 0$: pure BCFM — unbiased, high variance
- $\lambda > 0$: replaces noisy $X'$ with smoother velocity predictions, reducing variance with small controlled bias
- $\lambda = \gamma$: fully eliminates $X'$ from the target

The shared-noise coupling ensures the bias from $\lambda > 0$ is small — scaling as $O((1-\gamma)(1-t))$ under Gaussian analysis — making the bias–variance trade-off favorable in practice.
## Installation

1. Create an Anaconda environment: `conda create -n PCBF python=3.10.13 -y`
2. Activate the environment: `conda activate PCBF`
3. Install the dependencies:

```bash
conda install -c conda-forge glew -y
conda install -c conda-forge mesalib -y
pip install -r requirements.txt
```

## Usage

Every experiment is a single call to `main.py`. A run is fully specified by the environment, the agent config file, and the two PCBF hyperparameters — `--agent.discount` (γ) and `--agent.lambda_param` (λ):

```bash
python main.py \
  --env_name=cube-double-play-singletask-task1-v0 \
  --agent=agents/lambda_flow.py \
  --agent.discount=0.995 \
  --agent.lambda_param=0.4 \
  --seed=0
```

Substitute `task1` → `task2` … `task5` for the other tasks in a domain. Output goes to `exp/<wandb_run_group>/<run_name>/`, containing `flags.json`, `train.csv`, and `eval.csv`; the `evaluation/success` column of `eval.csv` is the success rate reported throughout this README.

### Per-domain commands

γ and λ below are the paper's Table 5 values.

```bash
# OGBench, state-based
python main.py --env_name=cube-double-play-singletask-task1-v0 --agent=agents/lambda_flow.py --agent.discount=0.995 --agent.lambda_param=0.4
python main.py --env_name=cube-triple-play-singletask-task1-v0 --agent=agents/lambda_flow.py --agent.discount=0.995 --agent.lambda_param=0.995
python main.py --env_name=puzzle-4x4-play-singletask-task1-v0  --agent=agents/lambda_flow.py --agent.discount=0.99  --agent.lambda_param=0.2
python main.py --env_name=scene-play-singletask-task1-v0       --agent=agents/lambda_flow.py --agent.discount=0.99  --agent.lambda_param=0.2

# OGBench, pixel-based (image augmentation + frame stacking + IMPALA encoder)
python main.py --env_name=visual-antmaze-teleport-navigate-singletask-task1-v0 --agent=agents/lambda_flow.py --agent.discount=0.99  --agent.lambda_param=0.0 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3
python main.py --env_name=visual-cube-double-play-singletask-task1-v0          --agent=agents/lambda_flow.py --agent.discount=0.995 --agent.lambda_param=0.9 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3

# D4RL Adroit (λ is per task — see the Hyperparameters section)
python main.py --env_name=hammer-cloned-v1 --agent=agents/lambda_flow.py --agent.discount=0.99 --agent.lambda_param=0.8
python main.py --env_name=hammer-expert-v1 --agent=agents/lambda_flow.py --agent.discount=0.99 --agent.lambda_param=0.9
```

### Sweeping tasks and seeds

The results below are 8 seeds × 5 tasks per domain. There is no launcher script in the repo — a plain shell loop (or your cluster's array-job equivalent) is enough:

```bash
for task in task1 task2 task3 task4 task5; do
  for seed in 0 1 2 3 4 5 6 7; do
    python main.py \
      --env_name=cube-double-play-singletask-${task}-v0 \
      --agent=agents/lambda_flow.py \
      --agent.discount=0.995 --agent.lambda_param=0.6 \
      --agent.ret_agg=max --agent.q_agg=max \
      --seed=${seed} \
      --wandb_run_group=cube_double_${task}
  done
done
```

### Frequently used flags

| Flag | Default | Meaning |
|---|---|---|
| `--env_name` | `antmaze-large-navigate-v0` | OGBench (`*-singletask-*`) or D4RL dataset name |
| `--agent` | `agents/value_flows.py` | Agent config file; use `agents/lambda_flow.py` for PCBF |
| `--agent.discount` | `0.99` | Discount γ |
| `--agent.lambda_param` | `0.0` | Control-variate weight λ; `0` = BCFM, valid range `[0, γ]` |
| `--agent.ret_agg` | `max` | Combines the two target return samples in the λ-target (`max` / `mean` / `min`) |
| `--agent.q_agg` | `max` | Combines Q₁/Q₂ when `sample_actions` picks among its candidates (`max` / `mean` / `min`) |
| `--agent.num_samples` | `16` | Candidate actions per rejection-sampling step |
| `--agent.num_flow_steps` | `10` | Euler steps used to integrate the flow |
| `--agent.encoder` | `None` | Visual encoder, e.g. `impala_small` (pixel-based tasks only) |
| `--seed` | `0` | Random seed |
| `--train_steps` | `1000000` | Gradient steps |
| `--eval_interval` | `100000` | Steps between evaluations (11 checkpoints over a 1M-step run) |
| `--eval_episodes` | `50` | Episodes per evaluation |
| `--save_dir` | `exp/` | Output root |
| `--wandb_run_group` | `debug` | Subdirectory under `--save_dir`, also the W&B group name |
| `--enable_wandb` | `0` | Set to `1` to log to Weights & Biases |

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

D4RL Adroit tunes λ per task instead of per domain (γ = 0.99 throughout):

| Task | λ | | Task | λ |
|---|---:|---|---|---:|
| pen-cloned-v1 | 0.7 | | door-cloned-v1 | 0.5 |
| pen-expert-v1 | 0.7 | | door-expert-v1 | 0.99 |
| hammer-cloned-v1 | 0.8 | | relocate-cloned-v1 | 0.99 |
| hammer-expert-v1 | 0.9 | | relocate-expert-v1 | 0.99 |

<details>
<summary><b>Common hyperparameters (click to expand)</b></summary>

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

</details>

## Results

### OGBench Environments
<p align="center">
  <img src="figures/ogbench.png" width="700">
  <figcaption align="center">
    <b>Figure 3.</b> OGBench Tasks.
  </figcaption>
</p>

### Offline RL (Table 1)

| Domain | IQN | CODAC | FQL | IQL | Value Flows | **PCBF** |
|---|---|---|---|---|---|---|
| cube-double-play | 42±8 | 61±6 | 29±6 | 7±1 | 69±4 | **71±5** |
| scene-play | 40±1 | 55±1 | 56±2 | 28±3 | **59±4** | 54±4 |
| puzzle-4x4-play | 27±4 | 20±18 | 17±5 | 7±2 | 27±4 | **30±4** |
| cube-triple-play | 6±0 | 2±1 | 4±2 | 1±1 | **14±3** | 4±1 |
| D4RL adroit | 66±5 | 69±0 | **71±4** | 70 | 65±2 | 69±2 |

Bold = within 95% of best. Results averaged over 8 seeds.

### Reproduction with this codebase

160 runs on a single A100 — 4 OGBench state-based domains × 5 tasks × 8 seeds, 1M steps each, evaluated on 50 episodes every 100K steps (11 checkpoints per run). Task names are abbreviated; the full environment id is `<domain>-singletask-<task>-v0`.

The critic-ensemble aggregation is a free choice the paper does not pin down, and `agents/lambda_flow.py` exposes it as two flags: `--agent.ret_agg` combines the two return samples inside the λ-target during critic training, and `--agent.q_agg` combines Q₁/Q₂ when `sample_actions` picks the best of its candidates. Both were set to the same value in every run; the per-domain configuration is listed at the end of this section.

#### Per-task results

Per seed, the score is the **best single evaluation checkpoint** of the run. Each task reports the **max over its 8 seeds**, the seed that attained it, and the checkpoint it came from, alongside the mean ± std over all 8 seeds. (∗) marks the per-domain tuning task.

| Task | Max over 8 seeds | Best seed | Ckpt | Mean ± std | Paper (Table 3) |
|---|---:|:---:|---:|---:|---:|
| cube-double-play-task1 | **100** | s4 | 500K | 93 ± 5 | 92 ± 3 |
| cube-double-play-task2 (∗) | **84** | s5 | 900K | 77 ± 5 | 74 ± 7 |
| cube-double-play-task3 | **76** | s1 | 700K | 70 ± 5 | 81 ± 8 |
| cube-double-play-task4 | **40** | s0 | 800K | 30 ± 7 | 22 ± 5 |
| cube-double-play-task5 | **66** | s3 | 1000K | 56 ± 9 | 84 ± 3 |
| scene-play-task1 | **100** | s0 | 300K | 100 ± 0 | 100 ± 0 |
| scene-play-task2 (∗) | **100** | s7 | 600K | 86 ± 11 | 57 ± 13 |
| scene-play-task3 | **100** | s2 | 800K | 99 ± 1 | 98 ± 2 |
| scene-play-task4 | **8** | s1 | 300K | 3 ± 3 | 12 ± 3 |
| scene-play-task5 | **0** | — | — | 0 ± 0 | 2 ± 1 |
| puzzle-4x4-play-task1 | **50** | s5 | 300K | 39 ± 6 | 38 ± 6 |
| puzzle-4x4-play-task2 | **36** | s1 | 800K | 28 ± 4 | 23 ± 5 |
| puzzle-4x4-play-task3 | **52** | s3 | 400K | 38 ± 6 | 40 ± 4 |
| puzzle-4x4-play-task4 (∗) | **40** | s2 | 400K | 35 ± 5 | 28 ± 4 |
| puzzle-4x4-play-task5 | **30** | s0 | 900K | 18 ± 6 | 23 ± 3 |
| cube-triple-play-task1 | **46** | s5 | 400K | 25 ± 9 | 18 ± 4 |
| cube-triple-play-task2 (∗) | **2** | s0 | 1000K | 0 ± 1 | 0 ± 1 |
| cube-triple-play-task3 | **6** | s6 | 1000K | 3 ± 2 | 1 ± 1 |
| cube-triple-play-task4 | **2** | s1 | 400K | 0 ± 1 | 0 ± 0 |
| cube-triple-play-task5 | **2** | s3 | 900K | 0 ± 1 | 1 ± 1 |

> **Read the last two columns with care.** The paper reports the average over the *final three* checkpoints, whereas the first four columns select the best checkpoint per seed and then the best seed. That selection is optimistic by construction, so it is a statement about what the codebase can reach, not a like-for-like comparison. The paper-protocol numbers are in the second collapsible below, and they are consistently lower.

#### Domain summary

| Domain | Max over seeds | Mean ± std | Paper (Table 1) |
|---|---:|---:|---:|
| cube-double-play (5 tasks) | 73.2 | 65 ± 6 | 71 |
| scene-play (5 tasks) | 61.6 | 58 ± 3 | 54 |
| puzzle-4x4-play (5 tasks) | 41.6 | 32 ± 5 | 30 |
| cube-triple-play (5 tasks) | 11.6 | 6 ± 3 | 4 |
| **all 20 tasks** | **47.0** | **40 ± 4** | **40** |

<details>
<summary><b>Per-seed scores (click to expand)</b></summary>

Best single evaluation checkpoint per run, in %.

| Domain | Task | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cube-double-play | task1 | 96 | 96 | 82 | 88 | **100** | 92 | 94 | 96 | **100** |
| | task2 | 76 | 80 | 80 | 72 | 76 | **84** | 82 | 68 | **84** |
| | task3 | 70 | **76** | 70 | 72 | 76 | 68 | 58 | 74 | **76** |
| | task4 | **40** | 34 | 28 | 22 | 20 | 24 | 30 | 38 | **40** |
| | task5 | 60 | 54 | 60 | **66** | 34 | 56 | 62 | 54 | **66** |
| scene-play | task1 | **100** | 100 | 100 | 100 | 100 | 100 | 100 | 100 | **100** |
| | task2 | 84 | 92 | 92 | 64 | 94 | 74 | 90 | **100** | **100** |
| | task3 | 98 | 96 | **100** | 100 | 100 | 100 | 100 | 98 | **100** |
| | task4 | 6 | **8** | 2 | 0 | 2 | 2 | 0 | 4 | **8** |
| | task5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| puzzle-4x4-play | task1 | 28 | 38 | 38 | 38 | 40 | **50** | 42 | 40 | **50** |
| | task2 | 30 | **36** | 22 | 26 | 30 | 30 | 26 | 24 | **36** |
| | task3 | 38 | 34 | 36 | **52** | 42 | 34 | 36 | 34 | **52** |
| | task4 | 26 | 38 | **40** | 36 | 40 | 28 | 30 | 40 | **40** |
| | task5 | **30** | 12 | 18 | 16 | 16 | 16 | 12 | 24 | **30** |
| cube-triple-play | task1 | 18 | 16 | 24 | 18 | 26 | **46** | 26 | 26 | **46** |
| | task2 | **2** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **2** |
| | task3 | 2 | 2 | 0 | 4 | 2 | 4 | **6** | 6 | **6** |
| | task4 | 0 | **2** | 0 | 0 | 0 | 0 | 0 | 0 | **2** |
| | task5 | 0 | 0 | 0 | **2** | 0 | 0 | 0 | 0 | **2** |

</details>

<details>
<summary><b>Same runs under the paper's protocol (click to expand)</b></summary>

Per-run score is the average of `evaluation/success` over the final three checkpoints (800K / 900K / 1M), which is what the paper reports — so these columns are directly comparable.

| Domain | Mean ± std | Paper (Table 3) |
|---|---:|---:|
| cube-double-play (5 tasks) | 53 ± 7 | **71 ± 5** |
| scene-play (5 tasks) | 48 ± 4 | **54 ± 4** |
| puzzle-4x4-play (5 tasks) | 22 ± 5 | **30 ± 4** |
| cube-triple-play (5 tasks) | 3 ± 1 | **4 ± 1** |
| **all 20 tasks** | 31 ± 4 | **40 ± 4** |

| Task | Mean ± std | Paper |
|---|---:|---:|
| cube-double-play-task1 | 82 ± 6 | 92 ± 3 |
| cube-double-play-task2 | 60 ± 9 | 74 ± 7 |
| cube-double-play-task3 | 56 ± 6 | 81 ± 8 |
| cube-double-play-task4 | 20 ± 6 | 22 ± 5 |
| cube-double-play-task5 | 44 ± 9 | 84 ± 3 |
| scene-play-task1 | 100 ± 1 | 100 ± 0 |
| scene-play-task2 | 43 ± 15 | 57 ± 13 |
| scene-play-task3 | 95 ± 2 | 98 ± 2 |
| scene-play-task4 | 0 ± 1 | 12 ± 3 |
| scene-play-task5 | 0 ± 0 | 2 ± 1 |
| puzzle-4x4-play-task1 | 28 ± 5 | 38 ± 6 |
| puzzle-4x4-play-task2 | 20 ± 5 | 23 ± 5 |
| puzzle-4x4-play-task3 | 25 ± 4 | 40 ± 4 |
| puzzle-4x4-play-task4 | 26 ± 5 | 28 ± 4 |
| puzzle-4x4-play-task5 | 11 ± 4 | 23 ± 3 |
| cube-triple-play-task1 | 12 ± 5 | 18 ± 4 |
| cube-triple-play-task2 | 0 ± 0 | 0 ± 1 |
| cube-triple-play-task3 | 2 ± 1 | 1 ± 1 |
| cube-triple-play-task4 | 0 ± 0 | 0 ± 0 |
| cube-triple-play-task5 | 0 ± 0 | 1 ± 1 |

Under the paper's protocol the reproduction falls short on every domain, with the largest gap on cube-double-play.

</details>

Configuration used for these runs:

| Domain | λ | γ | `ret_agg` / `q_agg` |
|---|---:|---:|:---:|
| cube-double-play | 0.6 | 0.995 | `max` |
| scene-play | 0.99 | 0.99 | `mean` |
| puzzle-4x4-play | 0.3 | 0.99 | `mean` |
| cube-triple-play | 0.995 | 0.995 | `mean` |

λ here is the value picked by the sweep below, which is not always the paper's Table 5 value.

### λ sweep

<p align="center">
  <img src="figures/lambda_sweep_aggs.png" width="700">
  <br>
  <em><b>Figure 4.</b> Success rate against λ on task 2 of each domain, for the three critic aggregations.</em>
</p>

λ swept from 0 to γ in steps of 0.1 (with γ as the final point) on task 2 of each domain, `ret_agg = q_agg`, seed 0, 1M steps. Each cell is the **best** `evaluation/success` (%) over the 11 evaluation checkpoints, written as `max / mean / min`.

| λ | cube-double-play<br>(γ=0.995) | cube-triple-play<br>(γ=0.995) | puzzle-4x4-play<br>(γ=0.99) | scene-play<br>(γ=0.99) |
|---|---:|---:|---:|---:|
| 0 | 64 / 66 / 4 | 0 / 0 / 0 | 18 / 32 / 32 | 38 / 16 / 2 |
| 0.1 | 64 / 72 / 2 | 0 / 0 / 0 | 18 / 18 / 20 | 44 / 28 / 6 |
| 0.2 | 68 / 62 / 4 | 0 / 0 / 0 | 28 / 24 / 20 | 48 / 48 / 4 |
| 0.3 | 62 / 76 / 0 | 0 / 0 / 0 | 12 / 36 / 32 | 42 / 40 / 2 |
| 0.4 | 70 / 64 / 4 | 0 / 0 / 0 | 18 / 20 / 24 | 62 / 42 / 2 |
| 0.5 | 78 / 72 / 0 | 0 / 0 / 0 | 22 / 34 / 22 | 46 / 68 / 2 |
| 0.6 | **82** / **78** / 0 | 0 / 0 / 0 | 6 / 28 / 16 | 46 / 48 / 2 |
| 0.7 | 72 / 66 / 0 | 0 / 0 / 0 | 18 / 30 / 20 | 56 / 38 / 2 |
| 0.8 | 80 / 64 / 0 | 0 / 0 / 0 | 14 / 30 / 22 | **70** / 46 / 4 |
| 0.9 | 80 / 54 / 0 | 0 / 0 / 0 | 8 / 32 / 18 | 44 / 60 / 4 |
| 0.99 | – | – | 12 / 26 / 12 | 68 / **86** / 2 |
| 0.995 | 76 / 66 / 0 | 0 / **2** / 0 | – | – |

Best λ per domain, and the λ the paper selected:

| Domain | best λ (`max`) | best λ (`mean`) | best λ (`min`) | paper λ |
|---|---:|---:|---:|---:|
| cube-double-play | 0.6 (82%) | 0.6 (78%) | 0 (4%) | 0.4 |
| cube-triple-play | 0.1 (0%) | 0.995 (2%) | 0.1 (0%) | 0.995 |
| puzzle-4x4-play | 0.2 (28%) | 0.3 (36%) | 0 (32%) | 0.2 |
| scene-play | 0.8 (70%) | 0.99 (86%) | 0.1 (6%) | 0.2 |

Averaged over the whole λ grid, `max` and `mean` are close (34.9% vs 35.7% across domains) while `min` collapses (6.5%) — the pessimistic aggregation that works for scalar Q-ensembles is harmful here, since both flow samples share the same base noise and the minimum systematically truncates the return distribution. Note that this sweep is a single seed at one task per domain, so individual λ cells are noisy; only the broad shape (a usable 0.3–0.6 band on cube-double, the `min` collapse) is well supported.

### Distributional Accuracy (Toy Environments)

<p align="center">
  <img src="figures/toy.png" width="700">
  <figcaption align="center">
    <b>Figure 5.</b> Learned PCBF Maps on Toy Environments. Left Top
(Solitaire); Right Top (Bernoulli); and Bottom (Discrete MC).
  </figcaption>
</p>

### Comparison on Learned Distribution (Toy Environments)

<p align="center">
  <img src="figures/vs.png" width="700">
  <figcaption align="center">
    <b>Figure 6.</b> Distributional accuracy comparison on toy environments.
  </figcaption>
</p>

### Visualization of Learned Velocity Field (Toy Environments)

<p align="center">
  <img src="figures/vis.png" width="700">
  <figcaption align="center">
    <b>Figure 7.</b> Distributional Flow Analysis on the Discrete MC Environment. We visualize the learned PCBF return distributions across states s = 1 to s = 20. The estimated probability density of the flow-transported samples (blue filled) is compared against Ground Truth Monte Carlo rollouts(black dashed lines). Characteristic flow trajectories transporting random noise samples (t = 0) to the target return distribution (t = 1) over flow time. Trajectory colors distinguish individual particles sampled from the base distribution p(x0), illustrating how the model maps stochastic noise to specific return outcomes.
  </figcaption>
</p>

## Repository Structure

```
path-coupled-bellman-flows/
├── main.py                       # Training entry point (OGBench / D4RL)
├── agents/
│   ├── __init__.py               # Agent registry
│   ├── lambda_flow.py            # PCBF agent (Algorithm 1)
│   └── ...                       # Baselines: c51, codac, fbrac, fql, ifql,
│                                 #            iql, iqn, rebrac, value_flows
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
├── figures/                      # Figures and GIFs
└── requirements.txt
```

## Citation

```bibtex
@article{xu2026path,
  title={Path-Coupled Bellman Flows for Distributional Reinforcement Learning},
  author={Xu, Boyang and Zou, Qing and Yang, Siqin and Yan, Hao},
  journal={arXiv preprint arXiv:2605.08253},
  year={2026}
}
```

## Acknowledgements

This codebase is built on [FQL](https://github.com/seohongpark/fql) and [Value Flows](https://github.com/chongyi-zheng/value-flows). We thank Research Computing at Arizona State University for providing A100 GPU resources on the Sol supercomputer.

