from dataclasses import dataclass, astuple
from pathlib import Path

# noinspection PyUnresolvedReferences
import pybullet_envs
import torch

from agent import EnvAgent, Batch
from model import VariationalEncoder, RecurrentStateSpaceModel, ExperienceReplay
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


def gen_data(args: Args) -> Batch:
    env_agent = EnvAgent(episode_path=EPISODE_PATH, args=args)
    batch = env_agent.train()
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


def test_rssm(latent: tuple[torch.Tensor, ...], prev_actions: list[torch.Tensor]) -> torch.Tensor:
    action_dim = sum(prev_actions[0].shape[1:])
    rssm = RecurrentStateSpaceModel(action_dim)
    latent, latent_length = pad_sequence(latent)
    prev_actions, action_lengths = pad_sequence(prev_actions)
    assert latent.shape[:2] == prev_actions.shape[:2], "mismatch between latent dims and actions dims"
    state_prior, state_posterior, recurrent_hidden_state = rssm.forward(prev_state=None,
                                                                        prev_action=prev_actions,
                                                                        action_lengths=action_lengths,
                                                                        recurrent_hidden_state=None,
                                                                        latent_observation=latent,
                                                                        latent_seq_lengths=latent_length)
    print(f"state_prior.shape = {state_prior}")
    print(f"state_posterior.shape = {state_posterior}")
    # print(f"recurrent_hidden_state.shape = {recurrent_hidden_state.shape}")
    # print(f"state_prior = {state_prior}")
    # print(f"state_posterior = {state_posterior}")
    # print(f"recurrent_hidden_state = {recurrent_hidden_state}")


def test_dataloader(batch: Batch) -> None:
    replay = ExperienceReplay(2)
    for eps in zip(*astuple(batch)):
        for action, reward, state in zip(*eps):
            replay.add_step_data(state, action, reward)
        replay.stack_episode()
    states, actions, rewards = replay.sample(batch_size=1, length=5)
    print(f"{states=}")
    print(f"{actions=}")
    print(f"{rewards=}")


if __name__ == "__main__":
    args = Args(env="HalfCheetahBulletEnv-v0", render=False, rgb=True, steps=10, episodes=2)
    batch = get_data(args, recreate=False)
    test_dataloader(batch)
    # test(batch)
