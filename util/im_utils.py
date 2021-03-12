import torch
import torchvision.transforms.functional as TF
from numpy import ndarray, moveaxis


@torch.no_grad()
def preprocess_observation_(observation: torch.Tensor, bit_depth: int = 5) -> torch.Tensor:
    """Reduction from float32-Tensor [0, 255] to [-0.5, 0.5]

    Reduces to the given bit depth and centers to [-0.5, 0.5]
    In addition, adds uniform random noise to the image
    """
    assert isinstance(observation, torch.Tensor), "Input obs must be Tensor"
    observation.floor_divide_(2 ** (8 - bit_depth)).div_(2 ** bit_depth).sub_(0.5)
    observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))
    # If unsqueezing multiple dimensions at once:
    # observation = observation[(None,)*(5 - observation.ndim)] # https://github.com/pytorch/pytorch/issues/9410
    return observation


def to_tensor(observation: ndarray):
    # shift color-dim in front of h x w
    pic = torch.from_numpy(moveaxis(observation, -1, -3).copy())
    pic = TF.convert_image_dtype(pic, torch.float32)
    return pic
