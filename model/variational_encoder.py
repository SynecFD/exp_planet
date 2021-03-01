import torch.nn as nn


class VariationalEncoder(nn.Module):

    def __init__(self, height: int, width: int, activation_function: str = 'ReLU') -> None:
        super().__init__()
        activation_function = getattr(nn, activation_function)
        self.net = nn.Sequential(nn.Conv2d(3, 32, kernel_size=4, stride=2), activation_function(),
                                 nn.Conv2d(32, 64, kernel_size=4, stride=2), activation_function(),
                                 nn.Conv2d(64, 128, kernel_size=4, stride=2), activation_function(),
                                 nn.Conv2d(128, 256, kernel_size=4, stride=2), activation_function(),
                                 nn.Flatten())
        self._latent_size = self._latent_size(height, width)

    def forward(self, obs):
        return self.net.forward(obs)

    def _latent_size(self, height: int, width: int) -> int:
        latent_dim_x = height
        latent_dim_y = width
        out_channels = 3
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                latent_dim_x = (latent_dim_x + 2 * module.padding[0] - module.dilation[0] * (
                            module.kernel_size[0] - 1) - 1) // module.stride[0] + 1
                latent_dim_y = (latent_dim_y + 2 * module.padding[1] - module.dilation[1] * (
                            module.kernel_size[1] - 1) - 1) // module.stride[1] + 1
                out_channels = module.out_channels
        return latent_dim_x * latent_dim_y * out_channels

    @property
    def latent_size(self) -> int:
        return self._latent_size
