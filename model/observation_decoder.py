from torch.nn import functional as F
import torch
from torch.nn import Module, ConvTranspose2d, Linear
from torch import nn


class ObservationModelDecoder(Module):

    def __init__(self, state_dim: int, hidden_dim: int, activation_func: str = 'ReLU') -> None:
        super(ObservationModelDecoder, self).__init__()
        activation_function = getattr(nn, activation_func)
        self.lin = Linear(state_dim + hidden_dim, 1024)
        self.net = nn.Sequential(ConvTranspose2d(1024, 128, kernel_size=5, stride=2), activation_function(),
                                 ConvTranspose2d(128, 64, kernel_size=5, stride=2), activation_function(),
                                 ConvTranspose2d(64, 32, kernel_size=6, stride=2), activation_function(),
                                 ConvTranspose2d(32, 3, kernel_size=6, stride=2), activation_function())

    def forward(self, state, hidden_state) -> torch.Tensor:
        latent = self.lin(torch.cat([state, hidden_state], dim=1))
        obs = self.net.forward(latent)
        return obs
