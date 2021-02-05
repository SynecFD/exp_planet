import gym
import pybullet_envs
import torch
import numpy as np

from dataclasses import dataclass
from model import VariationalEncoder, RecurrentStateSpaceModel
from util.im_utils import preprocess_observation_
from pathlib import Path
from typing import Tuple

EPISODE_PATH = Path.cwd() / "data" / "episode.npz"
EPISODE_PATH.parent.mkdir(exist_ok=True)


@dataclass
class Args:
    """
    env: environment ID
    seed: RNG seed
    render: OpenGL Visualizer
    rgb: rgb_array gym rendering
    steps: Number of steps
    """
    env: str = "AntBulletEnv-v0"
    seed: int = 0
    render: bool = False
    rgb: bool = False
    steps: int = 1
    episodes: int = 1


def get_data(args: Args = None, recreate: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if recreate and args is None:
        raise ValueError("Cannot recreate episode without args")
    path = EPISODE_PATH
    if path.exists() and not recreate:
        return load_data_from_disk()
    elif args is not None:
        return gen_data(args)
    else:
        raise ValueError("Cannot recreate episode without args")


def load_data_from_disk() -> Tuple[np.ndarray, np.ndarray]:
    path = EPISODE_PATH
    dict = np.load(path, allow_pickle=False, fix_imports=False)
    print(f"Episode loaded from {path}")
    return dict["episode"], dict["actions"]


def gen_data(args: Args) -> Tuple[np.ndarray, np.ndarray]:
    env = gym.make(args.env)
    env.env.configure(args)
    env.env._render_width = 64
    env.env._render_height = 64
    print(f"args.render = {args.render}")
    if args.render:
        env.render(mode="human")
    env.reset()
    print(f"action space: {env.action_space.shape}")
    episode_actions = []
    episodes = []
    for _ in range(args.episodes):
        actions = []
        episode = []
        for _ in range(args.steps):
            action = env.action_space.sample()
            actions.append(action)
            obs, rewards, done, _ = env.step(action)
            if args.rgb:
                rgb = env.render(mode="rgb_array")
                # print(f"RGB dims = {rgb.shape}")
                episode.append(rgb)
            if done:
                break
            # print("obs =")
            # print(obs)
            # print(f"rewards = {rewards}")
            # print(f"done = {done}")
        episode_actions.append(actions)
        episodes.append(episode)
    episode_actions = torch.stack(list(map(torch.from_numpy, episode_actions)))
    episodes = torch.stack(list(map(torch.from_numpy, episodes)))
    print(f"stacked obs = {episodes.shape}")
    path = EPISODE_PATH
    np.savez_compressed(path, episode=episodes, actions=episode_actions)
    print(f"Episode saved to {path}")
    return episodes, episode_actions


def test(episode: np.ndarray, actions: np.ndarray) -> None:
    episode = torch.from_numpy(episode)
    actions = torch.from_numpy(actions)
    latent = test_vae(episode)
    test_rssm(latent, actions)


def test_vae(episode: torch.Tensor) -> torch.Tensor:
    enc = VariationalEncoder()
    assert len(episode) > 0
    # FIXME Do it in batch not in loop! Concatenate from [Batch, Seq, img^3] to [Batch * Seq, img^3]
    for frame in episode:
        frame = preprocess_observation_(frame)
        z = enc.forward(frame)
    print(frame)
    print(f"latent z dims: {z.shape}")
    return z


def test_rssm(latent: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
    action_dim = sum(prev_actions.shape[1:])
    rssm = RecurrentStateSpaceModel(action_dim)
    state_prior, state_posterior, recurrent_hidden_state = rssm.forward(prev_state=None, prev_action=prev_actions,
                                                                        recurrent_hidden_state=None,
                                                                        latent_observation=latent)
    print(f"state_prior.shape = {state_prior.shape}")
    print(f"state_posterior.shape = {state_posterior.shape}")
    print(f"recurrent_hidden_state.shape = {recurrent_hidden_state.shape}")
    print(f"state_prior = {state_prior}")
    print(f"state_posterior = {state_posterior}")
    print(f"recurrent_hidden_state = {recurrent_hidden_state}")


if __name__ == "__main__":
    args = Args(env="HalfCheetahBulletEnv-v0", render=False, rgb=True, steps=10, episodes=2)
    episode, actions = get_data(args, recreate=True)
    test(episode, actions)
