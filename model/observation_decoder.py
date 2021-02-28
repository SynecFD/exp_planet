import torch
from torch import nn


class ObservationModelDecoder(nn.Module):

    def __init__(self, state_dim: int, hidden_dim: int, height: int, width: int, activation_func: str = 'ReLU') -> None:
        super().__init__()
        activation_function = getattr(nn, activation_func)
        self.net = nn.Sequential(nn.Linear(state_dim + hidden_dim, 1024),
                                 nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2), activation_function(),
                                 nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2), activation_function(),
                                 nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2), activation_function(),
                                 nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2), activation_function(),
                                 nn.Unflatten(1, (3, height, width)))

    def forward(self, state: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.net.forward(torch.cat([state, hidden_state], dim=1))
