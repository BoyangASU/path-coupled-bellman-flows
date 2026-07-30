"""Render demo GIFs from a saved PCBF checkpoint.

Rolls out episodes with a restored agent, keeps the ones that succeed, and
writes them out as a GIF. Environment and agent hyperparameters are read from
the run's own ``flags.json``, so only the run directory has to be given.

Example:
    MUJOCO_GL=egl python render_demo.py \
      --run_dir=exp/my_group/sd000_.../ \
      --restore_epoch=1000000 \
      --num_episodes=20 --num_keep=3 --out=figures/demo_cube_double.gif
"""

import json
import os

import imageio
import jax
import numpy as np
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset
from utils.flax_utils import restore_agent

FLAGS = flags.FLAGS

flags.DEFINE_string('run_dir', None, 'Run directory containing flags.json and params_*.pkl.')
flags.DEFINE_integer('restore_epoch', 1000000, 'Checkpoint epoch to restore.')
flags.DEFINE_string('out', 'demo.gif', 'Output GIF path.')
flags.DEFINE_integer('num_episodes', 20, 'Episodes to roll out.')
flags.DEFINE_integer('num_keep', 3, 'Successful episodes to keep in the GIF.')
flags.DEFINE_integer('frame_skip', 3, 'Render every Nth environment step.')
flags.DEFINE_integer('fps', 30, 'GIF frame rate.')
flags.DEFINE_integer('scale', 2, 'Integer upscaling factor applied to the rendered frames.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_bool('keep_failures', False, 'Fall back to the longest-surviving episodes if too few succeed.')

# Only used to build the agent skeleton; the values are overwritten from flags.json.
config_flags.DEFINE_config_file('agent', 'agents/lambda_flow.py', lock_config=False)


def _upscale(frame, scale):
    """Nearest-neighbour upscale, so no extra image dependency is needed."""
    if scale <= 1:
        return frame
    return frame.repeat(scale, axis=0).repeat(scale, axis=1)


def main(_):
    assert FLAGS.run_dir is not None, '--run_dir is required.'
    run_dir = FLAGS.run_dir.rstrip('/')

    with open(os.path.join(run_dir, 'flags.json')) as f:
        saved_flags = json.load(f)
    env_name = saved_flags['env_name']
    saved_agent_config = saved_flags['agent']

    config = FLAGS.agent
    for key, value in saved_agent_config.items():
        if key in config and value is not None:
            config[key] = value
    print(f'[demo] env={env_name} agent={config["agent_name"]} '
          f'gamma={config["discount"]} lambda={config["lambda_param"]}')

    frame_stack = saved_flags.get('frame_stack')
    env, eval_env, train_dataset, _ = make_env_and_datasets(env_name, frame_stack=frame_stack)
    train_dataset = Dataset.create(**train_dataset)

    # OGBench manipulation envs render at 200x200; their MJCF caps the offscreen
    # framebuffer at that size, so anything larger has to be upscaled afterwards.

    np.random.seed(FLAGS.seed)
    example_batch = train_dataset.sample(1)
    example_batch['min_reward'] = float(train_dataset['rewards'].min())
    example_batch['max_reward'] = float(train_dataset['rewards'].max())

    agent = agents[config['agent_name']].create(FLAGS.seed, example_batch, config)
    agent = restore_agent(agent, run_dir, FLAGS.restore_epoch)

    rng = jax.random.PRNGKey(FLAGS.seed)
    episodes = []  # (success, num_steps, frames)
    for episode in range(FLAGS.num_episodes):
        observation, _ = eval_env.reset()
        done, step, frames, success = False, 0, [], 0.0
        while not done:
            rng, action_rng = jax.random.split(rng)
            action = np.clip(np.array(agent.sample_actions(observations=observation, seed=action_rng)), -1, 1)
            observation, _, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            step += 1
            if step % FLAGS.frame_skip == 0 or done:
                frames.append(_upscale(eval_env.render(), FLAGS.scale))
            success = max(success, float(info.get('success', 0.0)))
        episodes.append((success, step, frames))
        print(f'[demo] episode {episode}: success={success:.0f} steps={step} frames={len(frames)}', flush=True)
        if not FLAGS.keep_failures and sum(e[0] > 0 for e in episodes) >= FLAGS.num_keep:
            break

    # Shortest successes first: they are the cleanest to watch.
    kept = sorted((e for e in episodes if e[0] > 0), key=lambda e: e[1])[: FLAGS.num_keep]
    if len(kept) < FLAGS.num_keep and FLAGS.keep_failures:
        rest = sorted((e for e in episodes if e[0] == 0), key=lambda e: -e[1])
        kept += rest[: FLAGS.num_keep - len(kept)]
    assert kept, 'No episode was recorded; try more --num_episodes or --keep_failures.'

    frames = [frame for _, _, episode_frames in kept for frame in episode_frames]
    os.makedirs(os.path.dirname(os.path.abspath(FLAGS.out)), exist_ok=True)
    imageio.mimsave(FLAGS.out, frames, fps=FLAGS.fps, loop=0)
    print(f'[demo] wrote {FLAGS.out} ({len(kept)} episodes, {len(frames)} frames, '
          f'{sum(e[0] > 0 for e in episodes)}/{len(episodes)} rollouts succeeded)')


if __name__ == '__main__':
    app.run(main)
