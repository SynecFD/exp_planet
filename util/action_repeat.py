# https://github.com/tensorflow/agents/blob/master/tf_agents/environments/wrappers.py

from gym import Env, Wrapper


class ActionRepeat(Wrapper):
    """Repeates actions over n-steps while acummulating the received reward."""

    def __init__(self, env: Env, times: int) -> None:
        """Creates an action repeat wrapper.

        Args:
            env: Environment to wrap.
            times: Number of times the action should be repeated.

        Raises:
            ValueError: If the times parameter is not greater than 1.
        """
        super().__init__(env)
        if times <= 1:
            raise ValueError(f'Times parameter ({times}) should be greater than 1')
        self._times = times

    def step(self, action):
        total_reward = 0

        for _ in range(self._times):
            observe, reward, done, info = super().step(action)
            total_reward += reward
            if done:
                break
        return observe, total_reward, done, info
