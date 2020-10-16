import numpy as np
import torch

class ExperienceReplay:

    def __init__(self, size, observation_shape, action_space):
        self.size = size
        self.observations = np.empty((size, *observation_shape), dtype=np.uint8)
        self.actions = np.empty((size, *action_space), dtype=np.float32)
        self.rewards = np.empty((size, 1), dtype=np.float32)
        self.terminal_state = np.empty((size, 1), dtype=np.bool)
        self.index = 0

        self.full = False

    def append(self, observation, action, reward, done):
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.terminal_state[self.index] = done

        if self.index == self.size-1:
            self.full = True
        self.index = (self.index + 1) % self.size

    def sample(self, batch_size, length):
        np.where(np.terminal_state)[0]



