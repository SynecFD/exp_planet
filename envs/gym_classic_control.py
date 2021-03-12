from collections import Sequence
from functools import partial
from typing import Optional

import torchvision.transforms.functional as FT
from gym import Wrapper, Env
from numpy import ndarray
from torch import Tensor

from util import to_tensor, preprocess_observation_


class GymClassicControl(Wrapper):

    def __init__(self, env: Env, action_repeat: Optional[int] = 1, top: Optional[int] = None,
                 left: Optional[int] = None, height: Optional[int] = None, width: Optional[int] = None,
                 size: Optional[Sequence[int]] = None) -> None:
        super().__init__(env)
        resize = size is not None
        crop = top is not None and left is not None and height is not None and width is not None
        if crop and resize:
            self.transform = partial(FT.resized_crop, top=top, left=left, height=height, width=width, size=size)
        elif crop:
            self.transform = partial(FT.crop, top=top, left=left, height=height, width=width)
        elif resize:
            self.transform = partial(FT.resize, size=size)
        else:
            self.transform = lambda img: img
        assert action_repeat >= 1, f"Action Repeat cannot be less than 1, was {action_repeat}"
        self.times = action_repeat

    def step(self, action: ndarray) -> tuple[Tensor, float, bool, dict]:
        total_reward = 0
        for _ in range(self.times):
            _, reward, done, info = super().step(action)
            total_reward += reward
            if done:
                break
        obs = self._transform(super().render(mode="rgb_array"))
        return obs, total_reward, done, info

    def reset(self, **kwargs) -> Tensor:
        super().reset(**kwargs)
        return self._transform(super().render(mode="rgb_array"))

    def _transform(self, img: ndarray) -> Tensor:
        return preprocess_observation_(self.transform(to_tensor(img)))
