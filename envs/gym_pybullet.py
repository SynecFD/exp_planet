from typing import Optional

from gym import Wrapper, Env
from numpy import ndarray


class PyBulletGym(Wrapper):

    def __init__(self, env: Env, action_repeat: Optional[int] = 1) -> None:
        super().__init__(env)
        assert action_repeat >= 1, f"Action Repeat cannot be less than 1, was {action_repeat}"
        self.times = action_repeat

    def step(self, action: ndarray) -> tuple[ndarray, float, bool, dict]:
        total_reward = 0
        for _ in range(self.times):
            _, reward, done, info = super().step(action)
            total_reward += reward
            if done:
                break
        obs = super().render(mode="rgb_array")
        return obs, total_reward, done, info

    def reset(self, **kwargs) -> ndarray:
        super().reset(**kwargs)
        return super().render(mode="rgb_array")
