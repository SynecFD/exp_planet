import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal


def loss(prior: Normal,
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
    mask = torch.arange(total_seq_length)[None, :] < lengths[:, None]

    kl_loss = kl_divergence(prior, posterior).masked_select(mask).sum(dim=2).clamp(min=free_nats).mean(dim=(0, 1))
    obs_loss = F.mse_loss(decode_obs, obs, reduction="none").sum(dim=(2, 3, 4)).mean(dim=(0, 1))
    reward_loss = F.mse_loss(expected_reward, reward)

    return kl_loss + obs_loss + reward_loss
