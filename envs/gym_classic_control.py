from typing import Optional

from gym import Wrapper, Env
from numpy import ndarray


class GymClassicControl(Wrapper):

    def __init__(self, env: Env, action_repeat: Optional[int] = 1, height_slice: Optional[slice] = None,
                 width_slice: Optional[slice] = None) -> None:
        super().__init__(env)
        self.height_slice = height_slice
        self.width_slice = width_slice
        assert action_repeat >= 1, f"Action Repeat cannot be less than 1, was {action_repeat}"
        self.times = action_repeat

    def step(self, action: ndarray) -> tuple[ndarray, float, bool, dict]:
        total_reward = 0
        for _ in range(self.times):
            _, reward, done, info = super().step(action)
            total_reward += reward
            if done:
                break
        obs = self.resize(super().render(mode="rgb_array"))
        return obs, total_reward, done, info

    def reset(self, **kwargs) -> ndarray:
        super().reset(**kwargs)
        return self.resize(super().render(mode="rgb_array"))

    def resize(self, obs: ndarray):
        if (self.height_slice is not None and (self.height_slice.start is not None
                                               or self.height_slice.stop is not None
                                               or self.height_slice.step is not None)) \
                or (self.width_slice is not None and (self.width_slice.start is not None
                                                      or self.width_slice.stop is not None
                                                      or self.width_slice.step is not None)):
            return obs[self.height_slice, self.width_slice].copy()
        else:
            return obs
