from pathlib import Path
from typing import Optional, Union

import gym
import numpy as np
import torch
from torch.distributions import Normal

from agent import PlanningAgent
from envs import PyBulletGym
from model import ExperienceReplay


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
        # Gym Classic Control tasks will always render.
        if self.render and isinstance(self.env, PyBulletGym):
            self.env.render(mode="human")

        self.current_obs = None
        self.action_dim = sum(self.action_space.shape)

    def reset(self) -> torch.Tensor:
        self.replay_buffer.stack_episode()
        self.current_obs = self.env.reset()
        return self.current_obs

    @torch.no_grad()
    def action(self, obs: torch.Tensor, device: Optional[torch.device] = torch.device("cpu")) -> torch.Tensor:
        action_mean = self.planner(obs, device)
        if self.explore_noise > 0:
            return Normal(action_mean, self.explore_noise).sample()
        else:
            return action_mean

    def step(self, action: Optional[Union[np.ndarray, torch.Tensor]] = None,
             device: Optional[torch.device] = torch.device("cpu")) -> tuple[torch.Tensor, int, bool, torch.Tensor]:
        if action is None:
            action = self.action(self.current_obs, device).cpu().numpy()
        elif isinstance(action, torch.Tensor):
            action = action.numpy()
        next_obs, reward, done, _ = self.env.step(action)
        self.replay_buffer.add_step_data(self.current_obs, action, reward)
        current_obs = self.current_obs
        self.current_obs = next_obs
        if done:
            self.reset()
        return current_obs, reward, done, next_obs

    @property
    def action_space(self):
        return self.env.action_space
