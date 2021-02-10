import torch.nn as nn


class VariationalEncoder(nn.Module):

    def __init__(self, activation_function: str = 'ReLU') -> None:
        super().__init__()
        activation_function = getattr(nn, activation_function)
        self.net = nn.Sequential(nn.Conv2d(3, 32, kernel_size=4, stride=2), activation_function(),
                                 nn.Conv2d(32, 64, kernel_size=4, stride=2), activation_function(),
                                 nn.Conv2d(64, 128, kernel_size=4, stride=2), activation_function(),
                                 nn.Conv2d(128, 256, kernel_size=4, stride=2), activation_function(),
                                 nn.Flatten())

    def forward(self, obs):
        return self.net.forward(obs)
