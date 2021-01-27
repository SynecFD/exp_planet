"""Learning Latent Dynamics for Planning from Pixels: Sec. 3"""

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


class RecurrentStateSpaceModel(nn.Module):

    def __init__(self, action_dim: int, state_dim: int = 30, hidden_dim: int = 200, latent_dim: int = 1024,
                 min_std_dev: float = 1e-1, activation_function: str = 'relu') -> None:
        super().__init__()
        self.activation_func = getattr(F, activation_function)
        self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim)

        # fc = fully connected
        # prior
        self.fc_latent_state_action = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc_latent_state_prior = nn.Linear(hidden_dim, hidden_dim)
        self.fc_state_mean_prior = nn.Linear(hidden_dim, state_dim)
        self.fc_state_std_dev_prior = nn.Linear(hidden_dim, state_dim)
        # posterior
        self.fc_hidden_latent_observation = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.fc_state_mean_posterior = nn.Linear(hidden_dim, state_dim)
        self.fc_state_std_dev_posterior = nn.Linear(hidden_dim, state_dim)

    def forward(self, prev_state: torch.Tensor, prev_action: torch.Tensor, recurrent_hidden_state: torch.Tensor,
                latent_observation: torch.Tensor) -> (Normal, Normal, torch.Tensor):
        """Compute environment state prior & state filtering posterior

        Note: Latent observation must be one time-step ahead of state and action.
        h_t = f(h_t-1, s_t-1, a_t-1)
        => p(s_t | h_t) & q(s_t | h_t, o_t)
        """
        state_prior, recurrent_hidden_state = self.prior(prev_state, prev_action, recurrent_hidden_state)
        state_posterior = self.posterior(recurrent_hidden_state, latent_observation)
        return state_prior, state_posterior, recurrent_hidden_state

    def _prior(self, prev_state, prev_action, recurrent_hidden_state) -> (Normal, torch.Tensor):
        """Compute environment state prior

        h_t = f(h_t-1, s_t-1, a_t-1)
        => p(s_t | h_t)
        """
        input = torch.cat([prev_state, prev_action], dim=1)

        hidden_state = self.activation_func(self.fc_latent_state_action(input))
        recurrent_hidden_state = self.rnn(hidden_state, recurrent_hidden_state)
        hidden_state = self.activation_func(self.fc_latent_state_prior(recurrent_hidden_state))

        mean = self.fc_state_mean_prior(hidden_state)
        std_dev = F.softplus(self.fc_state_std_dev_prior(hidden_state)) + self.min_std_dev
        return Normal(loc=mean, scale=std_dev), recurrent_hidden_state

    def _posterior(self, recurrent_hidden_state, latent_observation) -> Normal:
        """Compute environment state filtering posterior

        q(s_t | h_t, o_t)
        """
        input = torch.cat([recurrent_hidden_state, latent_observation], dim=1)
        hidden_state = self.activation_func(self.fc_hidden_latent_observation(input))

        # Kai Arulkumaran unifies the following independent linear layers into one
        # layer of double the size which is then chunked into two output tensors
        mean = self.fc_state_mean_posterior(hidden_state)
        std_dev = self.min_std_dev + F.softplus(self.fc_state_std_dev_posterior(hidden_state))
        return Normal(loc=mean, scale=std_dev)
