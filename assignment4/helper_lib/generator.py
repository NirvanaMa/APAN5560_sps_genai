# assignment4/helper_lib/generator.py

import torch

from .model import EnergyCNN, SimpleDiffusion


def langevin_step(
    model: EnergyCNN,
    x: torch.Tensor,
    step_size: float = 0.05,
    noise_std: float = 0.01,
) -> torch.Tensor:
    """
    Perform ONE Langevin step:
      x_{k+1} = x_k - step_size * dE/dx + noise

    Args:
        model: EnergyCNN, maps x -> scalar energy per sample.
        x: input images in [-1, 1], shape (B, 3, H, W).
    """
    # detach from previous graph and enable grad on x
    x = x.detach().requires_grad_(True)

    # compute energy and its gradient w.r.t. x
    energy = model(x)               # (B,) or (B,1)
    energy_sum = energy.sum()       # scalar
    grad_x, = torch.autograd.grad(energy_sum, x)

    # gradient descent on energy (lower E)
    x_next = x - step_size * grad_x

    # add Gaussian noise
    if noise_std > 0.0:
        x_next = x_next + noise_std * torch.randn_like(x_next)

    # clamp back to valid image range
    x_next = torch.clamp(x_next, -1.0, 1.0)

    # return detached tensor
    return x_next.detach()


def generate_ebm_samples(
    energy_model: EnergyCNN,
    device: str,
    n_samples: int = 16,
    steps: int = 60,
    step_size: float = 0.05,
    noise_std: float = 0.01,
    img_size: int = 32,
) -> torch.Tensor:
    """
    Generate samples by running Langevin dynamics from random noise.

    Returns:
        Tensor in [-1, 1] of shape (n_samples, 3, img_size, img_size).
    """
    energy_model.eval()
    # start from uniform noise in [-1, 1]
    x = torch.rand(n_samples, 3, img_size, img_size, device=device) * 2 - 1

    for _ in range(steps):
        x = langevin_step(
            energy_model,
            x,
            step_size=step_size,
            noise_std=noise_std,
        )

    return x.detach()


def generate_diffusion_samples(
    diffusion_model: SimpleDiffusion,
    device: str,
    n_samples: int = 16,
    img_size: int = 32,
) -> torch.Tensor:
    """
    Generate images by running reverse diffusion from Gaussian noise.

    Returns:
        Tensor in [-1, 1] of shape (n_samples, 3, img_size, img_size).
    """
    diffusion_model.eval()
    return diffusion_model.sample(n_samples, device=device, img_size=img_size)