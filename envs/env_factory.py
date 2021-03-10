from typing import Optional

import gym

from util import ActionRepeat, ResizeRender

HALF_CHEETAH = "HalfCheetahBulletEnv-v0"
PENDULUM = "Pendulum-v0"
CART_POLE = "CartPole-v1"


def init_env(env: str, action_repeat: Optional[int] = None) -> tuple[gym.Env, int, int, int]:
    factory = FACTORY.get(env, None)
    if factory is not None:
        func, default_action_repeat = factory
        single_step_env, h, w = func()
        # Uses default value per env if None or 0, no Action Repeat if 1, else the given value
        repeat_times = action_repeat or default_action_repeat
        return ActionRepeat(single_step_env, repeat_times) if repeat_times > 1 else single_step_env, h, w, repeat_times
    else:
        raise NotImplementedError(f"No factory method for {env}")


def _half_cheetah_bullet_env() -> tuple[gym.Env, int, int]:
    # noinspection PyUnresolvedReferences
    import pybullet_envs
    env = gym.make(HALF_CHEETAH)
    env.env._render_width = 64
    env.env._render_height = 64
    return env, env.env._render_height, env.env._render_width


def _pendulum_env() -> tuple[gym.Env, int, int]:
    env = gym.make(PENDULUM)
    env = ResizeRender(env, slice(90, 410, 5), slice(90, 410, 5))
    return env, 64, 64


def _cart_pole_env() -> tuple[gym.Env, int, int]:
    env = gym.make(CART_POLE)
    env = ResizeRender(env, slice(80, None, 5), slice(140, 460, 5))
    return env, 64, 64


FACTORY = {
    # name-str: (factory-function, default action-repeat)
    HALF_CHEETAH: (_half_cheetah_bullet_env, 4),
    PENDULUM: (_pendulum_env, 2),
    CART_POLE: (_cart_pole_env, 8)
}
