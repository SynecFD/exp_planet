from torch import Tensor
from ..model import RewardModel, RecurrentStateSpaceModel, VariationalEncoder
from ..util.im_utils import preprocess_observation_


class PlanningAgent:
    """Model-predictive control (MPC) planner with cross-entropy method (CEM)"""

    def __init__(self, observation_model: VariationalEncoder, transition_model: RecurrentStateSpaceModel,
                 reward_model: RewardModel, planning_horizon: int, num_opt_iterations: int, num_candidates: int,
                 num_top_candidates: int):
        self.observation_model = observation_model
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.planning_horizon = planning_horizon
        self.num_opt_iterations = num_opt_iterations
        self.num_candidates = num_candidates
        self.num_top_candidates = num_top_candidates

    def __call__(self, observation: Tensor):
        if not (observation.max() <= 0.5 and observation.min() >= -0.5 and observation.size() == (1, 3, 64, 64)):
            observation = preprocess_observation_(observation)
