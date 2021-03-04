"""Learning Latent Dynamics for Planning from Pixels: Sec. 3"""

from typing import Optional

import torch
from more_itertools import all_equal
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pack_padded_sequence


# state = torch.zeros(args.batch_size, args.state_dim, device=device)
# rnn_hidden = torch.zeros(args.batch_size, args.rnn_hidden_dim, device=device)

# belief = torch.zeros(1, args.belief_size, device=args.device)
# posterior_state = torch.zeros(1, args.state_size, device=args.device)


class RecurrentStateSpaceModel(nn.Module):

    def __init__(self, action_dim: int, state_dim: int = 30, hidden_dim: int = 200, latent_dim: int = 1024,
                 min_std_dev: float = 1e-1, activation_function: str = 'relu') -> None:
        super().__init__()
        self.activation_func = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

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

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

    def forward(self,
                prev_state: Optional[torch.Tensor],
                prev_action: torch.Tensor,
                action_lengths: torch.Tensor,
                recurrent_hidden_state: Optional[torch.Tensor],
                latent_observation: torch.Tensor,
                ) -> tuple[Normal, Normal, torch.Tensor, torch.Tensor]:
        """Compute environment state prior & state filtering posterior

        Note: Latent observation must be one time-step ahead of state and action.
        h_t = f(h_t-1, s_t-1, a_t-1)
        => p(s_t | h_t) & q(s_t | h_t, o_t)
        """
        # FIXME: https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism
        state_prior, recurrent_hidden_states, next_recurrent_hidden_state = self._prior(prev_state, prev_action,
                                                                                        action_lengths,
                                                                                        recurrent_hidden_state)
        state_posterior = self._posterior(recurrent_hidden_states, latent_observation)
        return state_prior, state_posterior, recurrent_hidden_states, next_recurrent_hidden_state

    def _prior(self,
               prev_state: Optional[torch.Tensor],
               prev_action: torch.Tensor,
               action_lengths: torch.Tensor,
               recurrent_hidden_state: Optional[torch.Tensor]
               ) -> tuple[Normal, torch.Tensor, torch.Tensor]:
        """Compute environment state prior

        h_t = f(h_t-1, s_t-1, a_t-1)
        => p(s_t | h_t)
        """
        total_action_seq_length = prev_action.size(1)
        if prev_state is None:
            prev_state = torch.zeros(prev_action.size(0), prev_action.size(1), self.state_dim)
        input = torch.cat([prev_state, prev_action], dim=2)

        # Q: Padded tensor unpacked in linear layer?
        # A: Write custom masked loss function (see chatbot tutorial)
        hidden_state = self.activation_func(self.fc_latent_state_action(input))

        # Start of recurrent layer (GRU)
        if not all_equal(action_lengths) or (action_lengths != prev_action.size(1)).any():
            hidden_state = pack_padded_sequence(hidden_state, action_lengths, batch_first=True, enforce_sorted=False)

        # output is: (recurrent_hidden_state, last_recurrent_hidden_state)
        recurrent_hidden_states, next_recurrent_hidden_state = self.rnn(hidden_state, recurrent_hidden_state)

        if isinstance(recurrent_hidden_states, PackedSequence):
            recurrent_hidden_states, _ = pad_packed_sequence(recurrent_hidden_states, batch_first=True,
                                                             total_length=total_action_seq_length)
        # End of recurrent layer (GRU)

        hidden_state = self.activation_func(self.fc_latent_state_prior(recurrent_hidden_states))

        mean = self.fc_state_mean_prior(hidden_state)
        std_dev = F.softplus(self.fc_state_std_dev_prior(hidden_state)) + self.min_std_dev
        # Q: Don't parameterize Normal with (batch x sequence x output)-Tensor. Maybe list?
        # A: KL Div is the same not matter the parameterization dimensions
        return Normal(loc=mean, scale=std_dev), recurrent_hidden_states, next_recurrent_hidden_state

    def _posterior(self, recurrent_hidden_states: torch.Tensor, latent_observation: torch.Tensor) -> Normal:
        """Compute environment state filtering posterior

        q(s_t | h_t, o_t)
        """
        input = torch.cat([recurrent_hidden_states, latent_observation], dim=2)
        hidden_state = self.activation_func(self.fc_hidden_latent_observation(input))

        # Kai Arulkumaran unifies the following independent linear layers into one
        # layer of double the size which is then chunked into two output tensors
        mean = self.fc_state_mean_posterior(hidden_state)
        std_dev = self.min_std_dev + F.softplus(self.fc_state_std_dev_posterior(hidden_state))

        # Q: Don't parameterize Normal with (batch x sequence x output)-Tensor. Maybe list?
        # A: Doesn't make a difference for KL-Loss
        return Normal(loc=mean, scale=std_dev)
