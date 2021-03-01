import argparse
from argparse import Namespace
from collections import Iterable
from itertools import chain
from pathlib import Path

import gym
# noinspection PyUnresolvedReferences
import pybullet_envs
import pytorch_lightning as pl
import torch
from torch.nn import Parameter
from torch.utils.data import DataLoader

from agent import Agent, PlanningAgent
from model import RecurrentStateSpaceModel, VariationalEncoder, ObservationModelDecoder, RewardModel, ExperienceReplay, \
    ExperienceReplaySampler, experience_replay_collate
from util import ActionRepeat
from util.data_loader import ReplayBufferSet


class PlaNet(pl.LightningModule):

    def __init__(self,
                 env: str,
                 lr: float,
                 eps: float,
                 save_path: Path,
                 replay_cap: int,
                 batch_size: int,
                 seq_len: int,
                 grad_clip: int,
                 free_nats: int,
                 hor_len: int,
                 opt_iter: int,
                 num_candidates: int,
                 top_candidates: int,
                 seed_epi: int,
                 update_interval: int,
                 eps_noise: float,
                 action_rep: int,
                 render: bool,
                 replay_size: int,
                 episode_length: int,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.lr = lr
        self.eps = eps
        self.save_path = save_path
        self.replay_cap = replay_cap
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.grad_clip = grad_clip
        self.free_nats = free_nats
        self.hor_len = hor_len
        self.opt_iter = opt_iter
        self.num_candidates = num_candidates
        self.top_candidates = top_candidates
        self.seed_epi = seed_epi
        self.update_interval = update_interval
        self.eps_noise = eps_noise
        self.render = render
        self.replay_size = replay_size
        self.episode_length = episode_length

        self.env, height, width = self._init_gym(env, action_rep)
        self.action_dim = sum(self.env.action_space.shape)
        self.encoder = VariationalEncoder(height, width)
        self.transition_model = RecurrentStateSpaceModel(self.action_dim)
        self.decoder = ObservationModelDecoder(self.transition_model.state_dim, self.transition_model.hidden_dim,
                                               height, width)
        self.reward_model = RewardModel(self.transition_model.state_dim, self.transition_model.hidden_dim)
        self.planner = PlanningAgent(self.encoder, self.transition_model, self.reward_model, self.env.action_space,
                                     self.hor_len, self.opt_iter, self.num_candidates, self.top_candidates)
        self.replay_buffer = ExperienceReplay(self.replay_cap)
        self.agent = Agent(self.planner, self.eps_noise, self.replay_buffer, self.save_path, self.env, self.render)
        self.populate_memory(self.seed_epi, self.episode_length)

    @staticmethod
    def _init_gym(env: str, action_repeat: int) -> tuple[gym.Env, int, int]:
        env = gym.make(env)
        env.env._render_width = 64
        env.env._render_height = 64
        return ActionRepeat(env, action_repeat), env.env._render_height, env.env._render_width

    def populate_memory(self, episodes: int = 5, length: int = 200) -> None:
        if self.save_path.exists():
            self.replay_buffer.load(self.save_path)
        else:
            self.agent.reset()
            for _ in range(episodes):
                for _ in range(length):
                    action = self.env.action_space.sample()
                    _, done = self.agent.step(action)
                    if done:
                        break
                self.agent.reset()
            # FIXME: DEBUG
            self.replay_buffer.persist(self.save_path)

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = ReplayBufferSet(self.replay_buffer, self.seq_len)
        return DataLoader(dataset=dataset,
                          batch_size=self.batch_size,
                          collate_fn=experience_replay_collate,
                          sampler=ExperienceReplaySampler(self.replay_buffer.replay, self.seq_len))

    def train_dataloader(self):
        return self.__dataloader()

    def forward(self, *args, **kwargs):
        pass

    def training_step(self, batch: list[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int, *args,
                      **kwargs):
        states_batch, actions_batch, rewards_batch, length = batch
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self._get_params(), lr=self.lr, eps=self.eps)

    def _get_params(self) -> Iterable[Parameter]:
        return chain.from_iterable([self.encoder.parameters(), self.decoder.parameters(),
                                    self.transition_model.parameters(), self.reward_model.parameters()])

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument("--env", type=str, default="HalfCheetahBulletEnv-v0", help="gym environment tag")
        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam")
        parser.add_argument("--eps", type=float, default=1e-4, help="epsilon for Adam")
        parser.add_argument("--save_path", type=Path, default=Path.cwd() / "data" / "episode.npz",
                            help="epsilon for Adam")
        parser.add_argument("--replay_cap", type=int, default=1000, help="capacity of the replay buffer")
        parser.add_argument("-B", "--batch_size", type=int, default=50, help="size of the batches")
        parser.add_argument("-L", "--seq_len", type=int, default=50, help="length of sequence chunks")
        parser.add_argument("--grad_clip", type=int, default=1000, help="gradient clipping norm")
        parser.add_argument("--free_nats", type=int, default=3, help="clipping the divergence loss below this value")

        parser.add_argument("-H", "--hor_len", type=int, default=12, help="horizon length for CEM")
        parser.add_argument("-I", "--opt_iter", type=int, default=10, help="optimization iterations for CEM")
        parser.add_argument("-J", "--num_candidates", type=int, default=1000,
                            help="number of candidate samples for CEM")
        parser.add_argument("-K", "--top_candidates", type=int, default=100,
                            help="number of top candidate samples for refitting CEM")
        parser.add_argument("-S", "--seed_epi", type=int, default=5, help="number of random seed episodes")
        parser.add_argument("-C", "--update_interval", type=int, default=100,
                            help="number of update steps before episode collection")
        parser.add_argument("--eps_noise", type=float, default=0.3, help="std dev of exploration noise")
        parser.add_argument("--action_rep", type=int, default=4, help="Action repeats for the environment")

        parser.add_argument("--render", type=bool, default=False, help="display the environment")
        parser.add_argument("--replay_size", type=int, default=1000, help="capacity of the replay buffer")
        parser.add_argument("--episode_length", type=int, default=200, help="max length of an episode")

        return parser


def main(args: Namespace) -> None:
    model = PlaNet(**vars(args))
    trainer = pl.Trainer()
    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser = PlaNet.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)
