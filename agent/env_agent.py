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
        if self.render:
            self.env.render(mode="human")

        self.obs = None
        self.action_dim = sum(self.action_space.shape)

    def reset(self) -> torch.Tensor:
        self.replay_buffer.stack_episode()
        self.env.reset()
        self.current_obs = self.env.render(mode="rgb_array")
        if (self.current_obs == 255.0).all():
            self.current_obs = np.zeros_like(self.current_obs)
        self.current_obs = preprocess_observation_(self.current_obs)
        return self.current_obs

    @torch.no_grad()
    def action(self, obs: torch.Tensor, device: Optional[torch.device] = torch.device("cpu")) -> torch.Tensor:
        action_mean = self.planner(obs, device)
        return Normal(action_mean, self.explore_noise).sample()

    def step(self, action: Optional[Union[np.ndarray, torch.Tensor]] = None,
             device: Optional[torch.device] = torch.device("cpu")) -> tuple[np.ndarray, int, bool]:
        if action is None:
            action = self.action(self.obs, device).cpu().numpy()
        elif isinstance(action, torch.Tensor):
            action = action.numpy()
        _, reward, done, _ = self.env.step(action)
        next_obs = self.env.render(mode="rgb_array")
        next_obs = preprocess_observation_(next_obs)
        self.replay_buffer.add_step_data(self.current_obs, action, reward)
        self.current_obs = next_obs
        if done:
            self.reset()
        return self.obs, reward, done

    @property
    def action_space(self):
        return self.env.action_space
