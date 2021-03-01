import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal


def loss(prior: Normal,
         posterior: Normal,
         obs: torch.Tensor,
         decode_obs: torch.Tensor,
         expected_loss: torch.Tensor,
         loss: torch.Tensor,
         free_nats: int = 3) -> torch.Tensor:
    kl_loss = kl_divergence(prior, posterior).sum(dim=2).clamp(min=free_nats).mean(dim=(0, 1))
    obs_loss = F.mse_loss(decode_obs, obs, reduction="none").sum(dim=(2, 3, 4)).mean(dim=(0, 1))
    # FIXME: OR? # obs_loss = F.mse_loss(decode_obs, obs, reduction="none").mean(dim=(0, 1).sum()
    reward_loss = F.mse_loss(expected_loss, loss, reduction="none").mean(dim=(0, 1))
    # FIXME: OR? # reward_loss = F.mse_loss(expected_loss, loss) # reduction="mean" is default

    return kl_loss + obs_loss + reward_loss
