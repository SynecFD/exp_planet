"""Learning Latent Dynamics for Planning from Pixels: Sec. 3"""

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from typing import Optional


# belief = torch.zeros(1, args.belief_size, device=args.device)
# posterior_state = torch.zeros(1, args.state_size, device=args.device)

# state = torch.zeros(args.batch_size, args.state_dim, device=device)
# rnn_hidden = torch.zeros(args.batch_size, args.rnn_hidden_dim, device=device)

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

    def forward(self, prev_state: Optional[torch.Tensor], prev_action: torch.Tensor,
                recurrent_hidden_state: Optional[torch.Tensor], latent_observation: torch.Tensor) -> (
            Normal, Normal, torch.Tensor):
        """Compute environment state prior & state filtering posterior

        Note: Latent observation must be one time-step ahead of state and action.
        h_t = f(h_t-1, s_t-1, a_t-1)
        => p(s_t | h_t) & q(s_t | h_t, o_t)
        """
        state_prior, recurrent_hidden_state = self._prior(prev_state, prev_action, recurrent_hidden_state)
        state_posterior = self._posterior(recurrent_hidden_state, latent_observation)
        return state_prior, state_posterior, recurrent_hidden_state

    def _prior(self, prev_state, prev_action, recurrent_hidden_state) -> (Normal, torch.Tensor):
        """Compute environment state prior

        h_t = f(h_t-1, s_t-1, a_t-1)
        => p(s_t | h_t)
        """
        if prev_state is None:
            batch_size = prev_action.shape[0]
            action_dim = sum(prev_action.shape[1:])
            state_dim = self.fc_latent_state_action.in_features - action_dim
            prev_state = torch.zeros(batch_size, state_dim)
        input = torch.cat([prev_state, prev_action], dim=1)

        hidden_state = self.activation_func(self.fc_latent_state_action(input))
        recurrent_hidden_state, last_recurrent_hidden_state = self.rnn(hidden_state, recurrent_hidden_state)
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
