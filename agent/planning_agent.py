from typing import Optional

import torch
from gym import Space
from torch import Tensor
from torch.distributions import Normal

from model import RewardModel, RecurrentStateSpaceModel, VariationalEncoder


class PlanningAgent:
    """Model-predictive control (MPC) planner with cross-entropy method (CEM)"""

    def __init__(self,
                 observation_model: VariationalEncoder,
                 transition_model: RecurrentStateSpaceModel,
                 reward_model: RewardModel,
                 action_space: Space,
                 planning_horizon: int = 12,
                 num_opt_iterations: int = 10,
                 num_candidates: int = 1000,
                 num_top_candidates: int = 100,
                 ) -> None:
        # Models
        self.observation_model = observation_model
        self.transition_model = transition_model
        self.reward_model = reward_model

        # Hyper-Parameters
        self.planning_horizon = planning_horizon
        self.num_opt_iterations = num_opt_iterations
        self.num_candidates = num_candidates
        self.num_top_candidates = num_top_candidates
        self.action_space = action_space

        self.init_action = torch.zeros(1, 1, *self.action_space.shape)
        self.init_action_len = torch.ones(1)
        self.candidate_actions_lengths = torch.full((self.num_candidates,), fill_value=self.planning_horizon,
                                                    dtype=torch.int64)

    @torch.no_grad()
    def __call__(self, observation: Tensor, device: Optional[torch.device] = torch.device("cpu")) -> Tensor:
        assert observation.max() <= 0.5 and observation.min() >= -0.5 and observation.shape[-3:] == (3, 64, 64), \
            "Input obs has not been preprocessed yet"

        observation = observation.to(device)
        self.init_action = self.init_action.to(device)
        self.init_action_len = self.init_action_len.to(device)
        self.candidate_actions_lengths = self.candidate_actions_lengths.to(device)

        observation = observation[None, None]
        latent_observation = self.observation_model(observation)
        _, posterior_belief, _, next_recurrent_hidden_state = self.transition_model(None, self.init_action,
                                                                                    self.init_action_len, None,
                                                                                    latent_observation)
        posterior_sample = posterior_belief.sample((self.num_candidates, self.planning_horizon)).squeeze_()
        next_recurrent_hidden_state = next_recurrent_hidden_state.repeat(1, self.num_candidates, 1)

        action_belief = Normal(torch.zeros(self.planning_horizon, *self.action_space.shape, device=device),
                               torch.ones(self.planning_horizon, *self.action_space.shape, device=device))
        for _ in range(self.num_opt_iterations):
            candidate_actions = action_belief.sample((self.num_candidates,))

            # FIXME: Clamping assumes action_space.bounded_above & action_space.bounded_below to be all true and
            #  action_space.high & action_space.low with equal values in each (see:
            #  https://github.com/pytorch/pytorch/issues/2793)
            candidate_actions = candidate_actions.clamp_(min=self.action_space.low.max(),
                                                         max=self.action_space.high.min())
            prior_belief, recurrent_hidden_states, next_recurrent_hidden_state = self.transition_model._prior(
                posterior_sample, candidate_actions, self.candidate_actions_lengths, next_recurrent_hidden_state)

            prior_belief_sample = prior_belief.sample()
            _, top_reward_idx = self.reward_model(prior_belief_sample, recurrent_hidden_states) \
                .sum(dim=1) \
                .topk(self.num_top_candidates, sorted=False)
            top_candidate_actions = candidate_actions[top_reward_idx]
            mean = top_candidate_actions.mean(dim=0)
            std_dev = top_candidate_actions.std(dim=0, unbiased=False)
            # DEBUG
            # print(f"Refitting action distribution to device: {mean.device=} and {std_dev.device=}")
            action_belief = Normal(mean, std_dev)

        return mean[0]
