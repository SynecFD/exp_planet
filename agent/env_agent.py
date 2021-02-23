from dataclasses import dataclass, asdict

import gym
import torch


@dataclass
class Batch:
    episode_actions: list[torch.Tensor]
    rewards: list[torch.Tensor]
    episodes: list[torch.Tensor]


class EnvAgent:
    def __init__(self, episode_path, args):
        self.episode_path = episode_path
        self.args = args
        self.env = gym.make(self.args.env)

        self.env.env.configure(args)
        self.env.env._render_width = 64
        self.env.env._render_height = 64

    def train(self) -> Batch:
        print(f"args.render = {self.args.render}")
        if self.args.render:
            self.env.render(mode="human")
        self.env.reset()
        print(f"action space: {self.env.action_space.shape}")
        episode_actions = []
        episode_rewards = []
        episodes = []
        for _ in range(self.args.episodes):
            seq_actions = []
            rewards = []
            sequence = []
            for _ in range(self.args.steps):
                action = self.env.action_space.sample()
                seq_actions.append(action)
                obs, reward, done, _ = self.env.step(action)
                rewards.append(reward)
                if self.args.rgb:
                    rgb = self.env.render(mode="rgb_array")
                    # print(f"RGB dims = {rgb.shape}")
                    sequence.append(rgb)
                if done:
                    break
                # print("{obs=}")
                # print(f"{reward=}")
                # print(f"{done=}")
            episode_actions.append(torch.stack(list(map(torch.from_numpy, seq_actions))))
            episode_rewards.append(torch.as_tensor(rewards))
            episodes.append(torch.stack(list(map(torch.from_numpy, sequence))))

        return self.to_batch(episodes, episode_actions, episode_rewards)

    def to_batch(self, episodes: list[torch.Tensor], episode_actions: list[torch.Tensor],
                 episode_rewards: list[torch.Tensor]) -> Batch:
        path = self.episode_path
        batch = Batch(episode_actions, episode_rewards, episodes)
        torch.save(asdict(batch), path)
        print(f"Batch saved to {path}")
        return batch
