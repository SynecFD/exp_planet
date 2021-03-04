from torch import Tensor, rand_like, float32, no_grad


@no_grad()
def preprocess_observation_(observation: Tensor, bit_depth: int = 5) -> Tensor:
    """Reduction from float32-Tensor [0, 255] to [-0.5, 0.5]

    Reduces to the given bit depth and centers to [-0.5, 0.5]
    In addition, adds uniform random noise to the image
    """
    assert observation.shape[-3:] == (64, 64, 3), 'Input obs of wrong shape'
    assert observation.dtype != float32 \
           or (observation.min() >= 0.0 and observation.max() > 0.5), "Obs has already been processed"
    observation = observation.to(dtype=float32, copy=True)
    observation.floor_divide_(2 ** (8 - bit_depth)).div_(2 ** bit_depth).sub_(0.5)
    observation.add_(rand_like(observation).div_(2 ** bit_depth))
    # shift color-dim in front of h x w
    dims = list(range(observation.ndim - 1))
    dims.insert(len(dims) - 2, observation.ndim - 1)
    observation = observation.permute(dims)
    if observation.ndim < 4:
        observation = observation.unsqueeze_(dim=0)
    # If unsqueezing multiple dimensions at once:
    # observation = observation[(None,)*(5 - observation.ndim)] # https://github.com/pytorch/pytorch/issues/9410
    return observation
