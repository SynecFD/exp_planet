from torch.nn import functional as F
from torch.nn import Module, Conv2d


class VariationalEncoder(Module):

    def __init__(self) -> None:
        super(VariationalEncoder, self).__init__()
        self.cv1 = Conv2d(3, 32, kernel_size=4, stride=2)
        self.cv2 = Conv2d(32, 64, kernel_size=4, stride=2)
        self.cv3 = Conv2d(64, 128, kernel_size=4, stride=2)
        self.cv4 = Conv2d(128, 256, kernel_size=4, stride=2)

    def forward(self, obs):
        latent = F.relu(self.cv1(obs))
        latent = F.relu(self.cv2(latent))
        latent = F.relu(self.cv3(latent))
        latent = F.relu(self.cv4(latent))
        latent = latent.view(-1, 1024)
        return latent
