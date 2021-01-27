import gym
import pybullet_envs
import torch
import numpy as np

from dataclasses import dataclass
from model.variational_encoder import VariationalEncoder
from util.im_utils import preprocess_observation_
from pathlib import Path

EPISODE_PATH = Path.cwd() / "data" / "episode.npy"
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


def get_data(args: Args = None, recreate: bool = False) -> np.ndarray:
    if recreate and args is None:
        raise ValueError("Cannot recreate episode without args")
    path = EPISODE_PATH
    if path.exists() and not recreate:
        return load_data_from_disk()
    elif args is not None:
        return gen_data(args)
    else:
        raise ValueError("Cannot recreate episode without args")


def load_data_from_disk() -> np.ndarray:
    path = EPISODE_PATH
    episode = np.load(path, allow_pickle=False, fix_imports=False)
    print(f"Episode loaded from {path}")
    return episode


def gen_data(args: Args) -> np.ndarray:
    env = gym.make(args.env)
    env.env.configure(args)
    env.env._render_width = 64
    env.env._render_height = 64
    print(f"args.render = {args.render}")
    if args.render:
        env.render(mode="human")
    env.reset()
    print("action space:")
    sample = env.action_space.sample()
    action = sample * 0.0
    print(f"action = {action}")

    episode = []
    for i in range(args.steps):
        obs, rewards, done, _ = env.step(action)
        if args.rgb:
            rgb = env.render(mode="rgb_array")
            # print(f"RGB dims = {rgb.shape}")
            episode.append(rgb)
        # print("obs =")
        # print(obs)
        # print(f"rewards = {rewards}")
        # print(f"done = {done}")
    episode = np.stack(episode)
    print(f"stacked obs = {episode.shape}")
    path = EPISODE_PATH
    np.save(path, episode, allow_pickle=False, fix_imports=False)
    print(f"Episode saved to {path}")
    return episode


def test(episode: np.ndarray) -> None:
    episode = torch.from_numpy(episode)
    latent = test_vae(episode)


def test_vae(episode: torch.Tensor) -> torch.Tensor:
    enc = VariationalEncoder()
    assert len(episode) > 0
    for frame in episode:
        frame = preprocess_observation_(frame)
        z = enc.forward(frame)
    print(frame)
    print(f"latent z dims: {z.shape}")
    return z


def test_rssm(latent: torch.Tensor) -> torch.Tensor:
    pass


if __name__ == "__main__":
    args = Args(env="HalfCheetahBulletEnv-v0", render=False, rgb=True, steps=10)
    episode = get_data(args)
    test(episode)
