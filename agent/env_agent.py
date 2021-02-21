from dataclasses import dataclass, asdict
import torch
import gym
from pathlib import Path


@dataclass
class Batch:
    episode_actions: list[torch.Tensor]
    episodes: list[torch.Tensor]


class EnvAgent:
    def __init__(self, episode_path, args):
        self.episode_path = episode_path
        self.env = args.env
        self.args = args
        env = gym.make(self.env)
        env.env.configure(args)
        env.env._render_width = 64
        env.env._render_height = 64

    def train(self) -> Batch:
        print(f"args.render = {self.args.render}")
        if self.args.render:
            self.env.render(mode="human")
        self.env.reset()
        print(f"action space: {self.env.action_space.shape}")
        episode_actions = []
        episodes = []
        for _ in range(self.args.episodes):
            seq_actions = []
            sequence = []
            for _ in range(self.args.steps):
                action = self.env.action_space.sample()
                seq_actions.append(action)
                obs, rewards, done, _ = self.env.step(action)
                if self.args.rgb:
                    rgb = self.env.render(mode="rgb_array")
                    # print(f"RGB dims = {rgb.shape}")
                    sequence.append(rgb)
                if done:
                    break
                # print("obs =")
                # print(obs)
                # print(f"rewards = {rewards}")
                # print(f"done = {done}")
            episode_actions.append(torch.stack(list(map(torch.from_numpy, seq_actions))))
            episodes.append(torch.stack(list(map(torch.from_numpy, sequence))))

        return self.to_batch(episodes, episode_actions)

    def to_batch(self, episodes: list[torch.Tensor], episode_actions: list[torch.Tensor]) -> Batch:
        path = self.episode_path
        batch = Batch(episode_actions, episodes)
        torch.save(asdict(batch), path)
        print(f"Batch saved to {path}")
        return batch