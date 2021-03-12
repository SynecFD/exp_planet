from typing import Optional

import gym

from envs.gym_classic_control import GymClassicControl
from envs.gym_pybullet import PyBulletGym

HALF_CHEETAH = "HalfCheetahBulletEnv-v0"
PENDULUM = "Pendulum-v0"
CART_POLE = "InvertedPendulumSwingupBulletEnv-v0"


def init_env(env: str, action_repeat: Optional[int] = None) -> tuple[gym.Env, int, int, int]:
    factory = FACTORY.get(env, None)
    if factory is not None:
        func, default_action_repeat = factory
        # Uses default value per env if None or 0, else the given value
        repeat_times = action_repeat or default_action_repeat
        return func(repeat_times)
    else:
        raise NotImplementedError(f"No factory method for {env}")


def _half_cheetah_bullet_env(action_repeat: int) -> tuple[gym.Env, int, int]:
    # noinspection PyUnresolvedReferences
    import pybullet_envs
    env = gym.make(HALF_CHEETAH)
    env.env._render_width = 64
    env.env._render_height = 64
    return PyBulletGym(env, action_repeat), env.env._render_height, env.env._render_width


def _pendulum_env(action_repeat: int) -> tuple[gym.Env, int, int]:
    env = gym.make(PENDULUM)
    env = GymClassicControl(env, action_repeat, slice(90, 410, 5), slice(90, 410, 5))
    return env, 64, 64


def _cart_pole_env(action_repeat: int) -> tuple[gym.Env, int, int]:
    # noinspection PyUnresolvedReferences
    import pybullet_envs
    env = gym.make(CART_POLE)
    env.env._render_width = 64
    env.env._render_height = 64
    return PyBulletGym(env, action_repeat), env.env._render_height, env.env._render_width


FACTORY = {
    # name-str: (factory-function, default action-repeat)
    HALF_CHEETAH: (_half_cheetah_bullet_env, 4),
    PENDULUM: (_pendulum_env, 2),
    CART_POLE: (_cart_pole_env, 8)
}
