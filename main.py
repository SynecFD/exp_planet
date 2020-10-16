import gym
import pybullet_envs
import torch
import numpy as np

from dataclasses import dataclass
from model.variational_encoder import VariationalEncoder
from util.im_utils import preprocess_observation_


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


def test(args: Args) -> None:
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

    enc = VariationalEncoder()
    episode = torch.from_numpy(episode)
    episode = preprocess_observation_(episode)
    print(episode)
    z = enc.forward(episode)
    print(f"latent z dims: {z.size()}")


if __name__ == "__main__":
    args = Args(env="HalfCheetahBulletEnv-v0", render=False, rgb=True, steps=10)
    test(args)
