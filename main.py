from dataclasses import dataclass, asdict
from pathlib import Path

import gym
# noinspection PyUnresolvedReferences
import pybullet_envs
import torch

from agent import PlanningAgent
from model import VariationalEncoder, RecurrentStateSpaceModel, RewardModel
from util import preprocess_observation_, concatenate_batch_sequences, split_into_batch_sequences, pad_sequence

EPISODE_PATH = Path.cwd() / "data" / "episode.pt"
EPISODE_PATH.parent.mkdir(exist_ok=True)


@dataclass
class Args:
    """
    env: environment ID
    seed: RNG seed
    render: OpenGL Visualizer
    rgb: rgb_array gym rendering
    steps: Number of steps
    """
    env: str = "AntBulletEnv-v0"
    seed: int = 0
    render: bool = False
    rgb: bool = False
    steps: int = 1
    episodes: int = 1


@dataclass
class Batch:
    episode_actions: list[torch.Tensor]
    episodes: list[torch.Tensor]


def get_data(args: Args = None, recreate: bool = False) -> Batch:
    if recreate and args is None:
        raise ValueError("Cannot recreate episode without args")
    path = EPISODE_PATH
    if path.exists() and not recreate:
        return load_data_from_disk()
    elif args is not None:
        return gen_data(args)
    else:
        raise ValueError("Cannot recreate episode without args")


def load_data_from_disk() -> Batch:
    path = EPISODE_PATH
    batch_dict = torch.load(path)
    print(f"Episode loaded from {path}")
    return Batch(**batch_dict)


def init_gym(args: Args) -> gym.Env:
    env = gym.make(args.env)
    env.env.configure(args)
    env.env._render_width = 64
    env.env._render_height = 64
    print(f"args.render = {args.render}")
    if args.render:
        env.render(mode="human")
    env.reset()
    return env


def gen_data(args: Args) -> Batch:
    env = init_gym(args)
    print(f"action space: {env.action_space.shape}")
    print(f"action space min/max: {env.action_space}")
    episode_actions = []
    episodes = []
    for _ in range(args.episodes):
        seq_actions = []
        sequence = []
        for _ in range(args.steps):
            action = env.action_space.sample()
            seq_actions.append(action)
            obs, rewards, done, _ = env.step(action)
            if args.rgb:
                rgb = env.render(mode="rgb_array")
                # print(f"RGB dims = {rgb.shape}")
                sequence.append(rgb)
            if done:
                break
            # print("obs =")
            # print(obs)
            # print(f"rewards = {rewards}")
            # print(f"done = {done}")
        episode_actions.append(torch.stack(list(map(torch.from_numpy, seq_actions))))
        episodes.append(torch.stack(list(map(torch.from_numpy, sequence))))
    path = EPISODE_PATH
    batch = Batch(episode_actions, episodes)
    torch.save(asdict(batch), path)
    print(f"Batch saved to {path}")
    return batch


def test(batch: Batch) -> None:
    latent_batch = test_vae(batch.episodes)
    test_rssm(latent_batch, batch.episode_actions)


def test_vae(episode: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    enc = VariationalEncoder()
    batch, lengths = concatenate_batch_sequences(episode)
    batch = preprocess_observation_(batch)
    z = enc.forward(batch)
    print(f"latent z dims: {z.shape}")
    return split_into_batch_sequences(z, lengths)


def test_rssm(latent: tuple[torch.Tensor, ...], prev_actions: list[torch.Tensor]) -> None:
    action_dim = sum(prev_actions[0].shape[1:])
    rssm = RecurrentStateSpaceModel(action_dim)
    # FIXME: DEBUG only
    latent = (latent[0], latent[1][:5])
    prev_actions = (prev_actions[0], prev_actions[1][:5])
    latent, _ = pad_sequence(latent)
    prev_actions, action_lengths = pad_sequence(prev_actions)
    assert latent.shape[:2] == prev_actions.shape[:2], "mismatch between latent dims and actions dims"
    recurrent_step = rssm.forward(prev_state=None, prev_action=prev_actions, action_lengths=action_lengths,
                                  recurrent_hidden_states=None, latent_observation=latent)
    state_prior, state_posterior, recurrent_hidden_states, next_hidden_state = recurrent_step
    print(f"state_prior.shape = {state_prior}")
    print(f"state_posterior.shape = {state_posterior}")

    three_dim_kl = torch.distributions.kl.kl_divergence(state_prior, state_posterior).flatten(end_dim=1)
    p = torch.distributions.normal.Normal(state_prior.mean.flatten(end_dim=1), state_prior.stddev.flatten(end_dim=1))
    q = torch.distributions.normal.Normal(state_posterior.mean.flatten(end_dim=1),
                                          state_posterior.stddev.flatten(end_dim=1))
    two_dim_kl = torch.distributions.kl.kl_divergence(p, q)
    print(f"2D == flat 3D?: {(three_dim_kl == two_dim_kl).all()}")
    # print(f"{recurrent_hidden_states.shape = {}recurrent_hidden_states.shape=}")
    # print(f"{state_prior=}")
    # print(f"{state_posterior=}")
    # print(f"{recurrent_hidden_states=}")


def test_planner(args: Args) -> None:
    env = init_gym(args)
    action_space = env.action_space
    planner = PlanningAgent(VariationalEncoder(),
                            RecurrentStateSpaceModel(sum(action_space.shape)),
                            RewardModel(30, 200),
                            action_space)
    initial_obs = env.render(mode="rgb_array")
    initial_obs = torch.from_numpy(initial_obs)
    initial_action_mean = planner(initial_obs)


if __name__ == "__main__":
    args = Args(env="HalfCheetahBulletEnv-v0", render=False, rgb=True, steps=10, episodes=2)
    # batch = get_data(args, recreate=True)
    # test(batch)
    test_planner(args)
