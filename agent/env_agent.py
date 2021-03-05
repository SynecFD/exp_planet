from pathlib import Path
from typing import Optional, Union

import gym
import numpy as np
import torch
from torch.distributions import Normal

from agent import PlanningAgent
from model import ExperienceReplay
from util import preprocess_observation_


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

        self.obs = None
        self.action_dim = sum(self.action_space.shape)

    def reset(self) -> torch.Tensor:
        self.replay_buffer.stack_episode()
        self.env.reset()
        self.obs = self.env.render(mode="rgb_array")
        if (self.obs == 255.0).all():
            self.obs = np.zeros_like(self.obs)
        self.obs = preprocess_observation_(self.obs)
        self.replay_buffer.add_step_data(self.obs, np.zeros(self.action_space.shape), 0.0)
        return self.obs

    @torch.no_grad()
    def action(self, obs: torch.Tensor) -> torch.Tensor:
        action_mean = self.planner(obs)
        return Normal(action_mean, self.explore_noise).sample()

    def step(self, action: Optional[Union[np.ndarray, torch.Tensor]] = None) -> tuple[np.ndarray, int, bool]:
        if action is None:
            action = self.action(self.obs).cpu().numpy()
        elif isinstance(action, torch.Tensor):
            action = action.numpy()
        _, reward, done, _ = self.env.step(action)
        self.obs = self.env.render(mode="rgb_array")
        self.obs = preprocess_observation_(self.obs)
        self.replay_buffer.add_step_data(self.obs, action, reward)
        if done:
            self.reset()
        return self.obs, reward, done

    @property
    def action_space(self):
        return self.env.action_space
