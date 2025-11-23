# test_diffusion_samples.py

import math
from pathlib import Path

import torch
import torchvision.utils as vutils

from helper_lib import (
    get_model,
    load_model,
    generate_diffusion_samples,
    get_default_device,
)


def main():
    device = get_default_device()
    print("Using device:", device)

    ckpt = Path("checkpoints/diffusion_cifar10.pth")
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found. Run train_energy_diffusion.py first.")

    # 1) Load diffusion model
    diff = get_model("Diffusion")
    diff = load_model(diff, str(ckpt), device)

    # 2) Generate samples
    n_samples = 25   # 5x5 grid
    torch.manual_seed(0)

    print(f"Generating {n_samples} diffusion samples...")
    imgs = generate_diffusion_samples(
        diff,
        device=device,
        n_samples=n_samples,
        img_size=32,
    )  # (N,3,32,32) in [-1,1]

    # 3) Save grid
    imgs_vis = (imgs + 1.0) / 2.0
    imgs_vis = imgs_vis.clamp(0.0, 1.0)

    out_path = Path("diffusion_samples.png")
    nrow = int(math.sqrt(n_samples))
    vutils.save_image(imgs_vis, out_path, nrow=nrow)

    print(f"Saved diffusion sample grid to {out_path.resolve()}")


if __name__ == "__main__":
    main()