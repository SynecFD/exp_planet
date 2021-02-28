from dataclasses import dataclass
from pathlib import Path

import gym
import torch
from torch.distributions import Normal

from agent import PlanningAgent
from model import ExperienceReplay


@dataclass
class Batch:
    episode_actions: list[torch.Tensor]
    rewards: list[torch.Tensor]
    episodes: list[torch.Tensor]


class Agent:
    def __init__(self,
                 planner: PlanningAgent,
                 explore_noise: float,
                 replay_buffer: ExperienceReplay,
                 episode_path: Path,
                 env: gym.Env,
                 render: bool) -> None:
        self.planner = planner
        self.explore_noise = explore_noise
        self.replay_buffer = replay_buffer
        self.render = render
        self.episode_path = episode_path

        self.env = env

        self.state = None
        self.action_dim = self.action_space.shape.sum()

    def reset(self) -> torch.Tensor:
        self.env.reset()
        self.state = self.env.render(mode="rgb_array")
        return self.state

    @torch.no_grad()
    def action(self, obs: torch.Tensor) -> torch.Tensor:
        action_mean = self.planner(obs)
        return Normal(action_mean, self.explore_noise).sample()

    def step(self):
        action = self.action(self.state).numpy()
        _, reward, done, _ = self.env.step(action)
        self.state = self.env.render(mode="rgb_array")
        self.replay_buffer.add_step_data(self.state, action, reward)
        if done:
            self.replay_buffer.stack_episode()

    @property
    def action_space(self):
        return self.env.action_space
