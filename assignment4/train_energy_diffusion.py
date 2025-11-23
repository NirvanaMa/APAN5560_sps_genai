# assignment4/train_energy_diffusion.py

from helper_lib import (
    get_cifar10_loader,
    get_model,
    train_energy_model,
    train_diffusion_model,
    save_model,
    get_default_device,
)


def main():
    device = get_default_device()
    print("Using device:", device)

    # CIFAR-10 train loader (normalized to [-1, 1] in data_loader.py)
    train_loader = get_cifar10_loader(
        data_dir="./data",
        batch_size=128,
        train=True,
    )

    # ---- Train EBM ----
    ebm = get_model("EBM")
    ebm = train_energy_model(
        ebm,
        train_loader,
        device=device,
        epochs=5,
        steps_langevin=20,
        step_size=0.05,
        noise_std=0.01,
    )
    save_model(ebm, "checkpoints/ebm_cifar10.pth")

    # ---- Train Diffusion ----
    diffusion = get_model("Diffusion")
    diffusion = train_diffusion_model(
        diffusion,
        train_loader,
        device=device,
        epochs=5,
    )
    save_model(diffusion, "checkpoints/diffusion_cifar10.pth")


if __name__ == "__main__":
    main()