import torch
from torch import nn


class RewardModel(nn.Module):
    """Reward model to predict reward from state and rnn hidden state

    p(r_t | s_t, h_t)
    """

    def __init__(self,
                 state_dim: int,
                 hidden_input_dim: int,
                 hidden_units_dim: int = 300,
                 activation_func: str = 'ReLU'
                 ) -> None:
        super().__init__()
        activation_function = getattr(nn, activation_func)
        self.net = nn.Sequential(nn.Linear(state_dim + hidden_input_dim, hidden_units_dim), activation_function(),
                                 nn.Linear(hidden_units_dim, hidden_units_dim), activation_function(),
                                 nn.Linear(hidden_units_dim, hidden_units_dim), activation_function(),
                                 nn.Linear(hidden_units_dim, 1), activation_function())

    def forward(self, state, hidden_state):
        hidden = self.net.forward(torch.cat([state, hidden_state], dim=2))
        mean_reward = hidden.squeeze(dim=2)
        # Note that the log-likelihood under a Gaussian distribution with unit variance equals
        # the mean squared error up to a constant.
        return mean_reward
