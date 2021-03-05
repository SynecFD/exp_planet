from numpy import ndarray
from torch import Tensor, rand_like, float32, no_grad, from_numpy


@no_grad()
def preprocess_observation_(observation: ndarray, bit_depth: int = 5) -> Tensor:
    """Reduction from float32-Tensor [0, 255] to [-0.5, 0.5]

    Reduces to the given bit depth and centers to [-0.5, 0.5]
    In addition, adds uniform random noise to the image
    """
    assert isinstance(observation, ndarray), "Input obs must be numpy array"
    observation = from_numpy(observation).float()
    observation.floor_divide_(2 ** (8 - bit_depth)).div_(2 ** bit_depth).sub_(0.5)
    observation.add_(rand_like(observation).div_(2 ** bit_depth))
    # shift color-dim in front of h x w
    dims = list(range(observation.ndim - 1))
    dims.insert(len(dims) - 2, observation.ndim - 1)
    observation = observation.permute(dims)
    # If unsqueezing multiple dimensions at once:
    # observation = observation[(None,)*(5 - observation.ndim)] # https://github.com/pytorch/pytorch/issues/9410
    return observation
