import argparse
from argparse import Namespace
from collections import Iterable
from itertools import chain
from math import ceil
from os import cpu_count
from pathlib import Path
from typing import Optional, Union

import gym
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from torch.nn import Parameter
from torch.utils.data import DataLoader

import envs
from agent import Agent, PlanningAgent
from model import RecurrentStateSpaceModel, VariationalEncoder, ObservationModelDecoder, RewardModel, ExperienceReplay, \
    ExperienceReplaySampler, experience_replay_collate, StubExperienceReplay
from util.data_loader import ReplayBufferSet


class PlaNet(pl.LightningModule):

    def __init__(self,
                 env: str,
                 lr: float,
                 epsi: float,
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
                 epsi_noise: float,
                 action_rep: int,
                 render: bool,
                 replay_size: int,
                 episode_max_len: int,
                 play: bool,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.lr = lr
        self.epsi = epsi
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
        self.epsi_noise = epsi_noise
        self.render = play or render
        self.replay_size = replay_size

        self.env, height, width, action_rep = self._init_gym(env, action_rep)
        self.action_dim = sum(self.env.action_space.shape)
        self.episode_max_len = ceil(episode_max_len / action_rep)

        # Nets
        self.encoder = VariationalEncoder(height, width)
        self.rssm = RecurrentStateSpaceModel(self.action_dim)
        self.decoder = ObservationModelDecoder(self.rssm.state_dim, self.rssm.hidden_dim,
                                               height, width)
        self.reward_model = RewardModel(self.rssm.state_dim, self.rssm.hidden_dim)

        # Model states
        self.belief = None
        self.recurrent_states = None
        self.last_recurrent_state = None

        # Planner
        self.planner = PlanningAgent(self.encoder, self.rssm, self.reward_model, self.env.action_space,
                                     self.hor_len, self.opt_iter, self.num_candidates, self.top_candidates)

        # Environment interaction
        self.replay_buffer = ExperienceReplay(self.replay_cap) if not play else StubExperienceReplay()
        self.agent = Agent(self.planner, self.epsi_noise, self.replay_buffer, self.save_path, self.env, self.render)
        self.agent.reset()
        if not play:
            self.populate_memory(self.seed_epi, self.episode_max_len)
        self.save_hyperparameters()

    @staticmethod
    def _init_gym(env: str, action_repeat: int) -> tuple[gym.Env, int, int, int]:
        return envs.init_env(env, action_repeat)

    def populate_memory(self, episodes: int = 5, length: int = 200) -> None:
        if self.save_path.exists():
            self.replay_buffer.load(self.save_path)
        else:
            for _ in range(episodes):
                for _ in range(length):
                    action = self.env.action_space.sample()
                    _, _, done, _ = self.agent.step(action)
                    if done:
                        break
                self.agent.reset()
            # FIXME: DEBUG
            self.replay_buffer.persist(self.save_path)

    def play(self) -> None:
        while True:
            _, _, done, _ = self.agent.step(device=self.device)
            if done:
                self.agent.reset()

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = ReplayBufferSet(self.replay_buffer, self.seq_len)
        return DataLoader(dataset=dataset,
                          batch_size=self.batch_size,
                          collate_fn=experience_replay_collate,
                          sampler=ExperienceReplaySampler(self.replay_buffer.replay, self.seq_len, allow_padding=False,
                                                          replacement=True,
                                                          num_samples=self.update_interval * self.batch_size),
                          num_workers=cpu_count() or 1,
                          pin_memory=True)

    def train_dataloader(self):
        return self.__dataloader()

    def forward(self,
                cur_belief: torch.Tensor, actions: torch.Tensor, lengths: torch.Tensor, recurrent_state: torch.Tensor,
                latent_obs: torch.Tensor, *args, **kwargs
                ) -> tuple[Normal, Normal, torch.Tensor, torch.Tensor]:
        return self.rssm(cur_belief, actions, lengths, recurrent_state, latent_obs)

    def on_train_epoch_start(self) -> None:
        self.belief = None
        self.last_recurrent_state = None

    def on_train_batch_start(self, batch: list[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int,
                             dataloader_idx: int) -> None:
        if self.belief is not None:
            self.belief = self.belief.detach()
        if self.last_recurrent_state is not None:
            self.last_recurrent_state = self.last_recurrent_state.detach()

    def training_step(self, batch: list[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int,
                      *args, **kwargs) -> torch.Tensor:
        obs_batch, actions_batch, rewards_batch, length = batch
        # FIXME: Do we really not reuse last posterior-belief sample and recurrent state between batches?
        # self.belief = None
        # self.last_recurrent_state = None

        latent_obs = self.encoder(obs_batch)
        recurrent_step = self(self.belief, actions_batch, length, self.last_recurrent_state, latent_obs)
        prior_belief, posterior_belief, self.recurrent_states, self.last_recurrent_state = recurrent_step
        self.belief = posterior_belief.rsample()

        expected_reward = self.reward_model(self.belief, self.recurrent_states)
        reconstructed_obs = self.decoder(self.belief, self.recurrent_states)

        loss = self.single_step_loss(prior_belief, posterior_belief, obs_batch, reconstructed_obs, rewards_batch,
                                     expected_reward, length, self.free_nats)
        self.log("loss", loss)
        return loss

    def training_epoch_end(self, training_step_outputs):
        # Data collection
        for _ in range(self.episode_max_len):
            self.agent.step(device=self.device)
        self.agent.reset()
        reward_sum = self.replay_buffer.replay[-1].rewards.sum()
        self.log("Episode reward", reward_sum)

    # @torch.no_grad()
    def single_step_loss(self, prior: Normal,
                         posterior: Normal,
                         obs: torch.Tensor,
                         decode_obs: torch.Tensor,
                         expected_reward: torch.Tensor,
                         reward: torch.Tensor,
                         lengths: torch.Tensor,
                         free_nats: int = 3
                         ) -> torch.Tensor:
        # https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397
        # https://juditacs.github.io/2018/12/27/masked-attention.html
        total_seq_length = obs.size(1)
        mask = torch.arange(total_seq_length, device=self.device)[None, :] < lengths[:, None]

        kl_loss = kl_divergence(prior, posterior).sum(dim=2).masked_select(mask).clamp(min=free_nats).mean()
        obs_loss = F.mse_loss(decode_obs, obs, reduction="none").sum(dim=(2, 3, 4)).masked_select(mask).mean()
        reward_loss = F.mse_loss(expected_reward, reward, reduction="none").masked_select(mask).mean()

        return kl_loss + obs_loss + reward_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self._get_params(), lr=self.lr, eps=self.epsi)

    def _get_params(self) -> Iterable[Parameter]:
        return chain.from_iterable([self.encoder.parameters(), self.decoder.parameters(),
                                    self.rssm.parameters(), self.reward_model.parameters()])

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument("--env", type=str, default="CartPole-v1", help="gym environment tag")
        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam")
        parser.add_argument("--epsi", type=float, default=1e-4, help="epsilon for Adam")
        parser.add_argument("--save-path", type=Path, default=Path.cwd() / "data" / "episode",
                            help="epsilon for Adam")
        parser.add_argument("--replay-cap", type=int, default=1000, help="capacity of the replay buffer")
        parser.add_argument("-B", "--batch-size", type=int, default=50, help="size of the batches")
        parser.add_argument("-L", "--seq-len", type=int, default=50, help="length of sequence chunks")
        parser.add_argument("--grad-clip", type=int, default=1000, help="gradient clipping norm")
        parser.add_argument("--free-nats", type=int, default=3, help="clipping the divergence loss below this value")

        parser.add_argument("-H", "--hor-len", type=int, default=12, help="horizon length for CEM")
        parser.add_argument("-I", "--opt-iter", type=int, default=10, help="optimization iterations for CEM")
        parser.add_argument("-J", "--num-candidates", type=int, default=1000,
                            help="number of candidate samples for CEM")
        parser.add_argument("-K", "--top-candidates", type=int, default=100,
                            help="number of top candidate samples for refitting CEM")
        parser.add_argument("-S", "--seed-epi", type=int, default=5, help="number of random seed episodes")
        parser.add_argument("-C", "--update-interval", type=int, default=100,
                            help="number of update steps before episode collection")
        parser.add_argument("--epsi-noise", type=float, default=0.3, help="std dev of exploration noise")
        parser.add_argument("--action-rep", type=int, help="Action repeats for the environment")
        parser.add_argument("--episode-max-len", type=int, default=1000, help="Max episode length")

        parser.add_argument("--render", type=bool, default=False, help="display the environment")
        parser.add_argument("--replay_size", type=int, default=1000, help="capacity of the replay buffer")
        parser.add_argument("--play", type=bool, default=False, help="Play environment with latest agent's weights")

        return parser


def latest_ckpt() -> Optional[Path]:
    ckpts = sorted((Path.cwd() / "lightning_logs").glob("version_*/checkpoints/*.ckpt"), key=natural_keys)
    return ckpts[-1] if ckpts else None


def natural_keys(key: Path) -> list[Union[str, int]]:
    from re import split
    return [int(part) if part.isdigit() else part for part in split(r"(\d+)", str(key))]


def main(args: Namespace) -> None:
    if not args.play:
        model = PlaNet(**vars(args))
        trainer = pl.Trainer(gradient_clip_val=1000, gpus=1)
        trainer.fit(model)
    else:
        ckpt = latest_ckpt()
        if ckpt is not None:
            model = PlaNet.load_from_checkpoint(ckpt, epsi_noise=0.0, render=True, play=True)
            model = model.cuda() if torch.cuda.is_available() else model.cpu()
            model.play()
        else:
            raise FileNotFoundError("No weights found for model, cannot play environment")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser = PlaNet.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)
