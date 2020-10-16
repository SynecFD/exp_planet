import torch
from torch import functional as F
from torch import nn


class RewardModel(nn.Module):
    """Reward model to predict reward from state and rnn hidden state

    p(r_t | s_t, h_t)
    """

    def __init__(self, state_dim: int, hidden_input_dim: int, hidden_units_dim: int = 300,
                 activation_func: str = 'relu'):
        super(RewardModel, self).__init__()
        self.activation_func = getattr(F, activation_func)
        self.fc1 = nn.Linear(state_dim + hidden_input_dim, hidden_units_dim)
        self.fc2 = nn.Linear(hidden_units_dim, hidden_units_dim)
        self.fc3 = nn.Linear(hidden_units_dim, hidden_units_dim)
        self.fc4 = nn.Linear(hidden_units_dim, 1)

    def forward(self, state, hidden_state):
        hidden = self.activation_func(self.fc1(torch.cat([state, hidden_state], dim=1)))
        hidden = self.activation_func(self.fc2(hidden))
        hidden = self.activation_func(self.fc3(hidden))
        mean_reward = self.fc4(hidden).squeeze()  # FIXME: maybe only for dim=1
        # Note that the log-likelihood under a Gaussian distribution with unit variance equals
        # the mean squared error up to a constant.
        return mean_reward
