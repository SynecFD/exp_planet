from torch.nn import functional as F
import torch
from torch.nn import Module, ConvTranspose2d, Linear


class ObservationModelDecoder(Module):

    def __init__(self, state_dim, hidden_dim) -> None:
        super(ObservationModelDecoder, self).__init__()
        self.lin = Linear(state_dim + hidden_dim, 1024)
        self.convt1 = ConvTranspose2d(1024, 128, kernel_size=5, stride=2)
        self.convt2 = ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.convt3 = ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.convt4 = ConvTranspose2d(32, 3, kernel_size=6, stride=2)

    def forward(self, state, hidden_state) -> torch.Tensor:
        latent = self.lin(torch.cat([state, hidden_state], dim=1))
        latent = F.relu(self.cvt1(latent))
        latent = F.relu(self.cvt2(latent))
        latent = F.relu(self.cvt3(latent))
        obs = self.cvt4(latent)
        return obs
