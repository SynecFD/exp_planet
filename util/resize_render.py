from gym import Wrapper, Env


class ResizeRender(Wrapper):

    def __init__(self, env: Env, height_slice: slice, width_slice: slice) -> None:
        super().__init__(env)
        self.height_slice = height_slice
        self.width_slice = width_slice

    def render(self, mode: str = "human", **kwargs):
        obs = super().render(mode, **kwargs)
        return obs[self.height_slice, self.width_slice].copy() if mode == "rgb_array" else obs
