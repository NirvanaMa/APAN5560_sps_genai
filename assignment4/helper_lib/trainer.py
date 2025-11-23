# assignment4/helper_lib/trainer.py

import torch
import torch.optim as optim
import torch.nn.functional as F

from .generator import langevin_step
from .model import EnergyCNN, SimpleDiffusion


def train_energy_model(
    model: EnergyCNN,
    train_loader,
    device: str = "mps",
    epochs: int = 5,
    steps_langevin: int = 10,
    step_size: float = 0.03,
    noise_std: float = 0.01,
    lr: float = 1e-4,
):
    """
    Lightweight EBM training with a stable, non-negative objective:

      loss = softplus(E_real - E_fake)

    This encourages E_real < E_fake, but the scalar loss stays >= 0
    and is easier to interpret and debug than a raw energy difference.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0

        for batch_idx, (x_real, _) in enumerate(train_loader):
            x_real = x_real.to(device)  # in [-1, 1]
            b = x_real.size(0)

            # ---- Positive energies ----
            E_pos = model(x_real).mean()

            # ---- Negative samples: start from noise + Langevin ----
            x_neg = torch.rand_like(x_real) * 2 - 1  # uniform in [-1,1]
            for _ in range(steps_langevin):
                x_neg = langevin_step(
                    model,
                    x_neg,
                    step_size=step_size,
                    noise_std=noise_std,
                )

            E_neg = model(x_neg).mean()

            # ---- EBM loss ----
            # We want E_pos << E_neg, so (E_pos - E_neg) should be negative.
            # softplus(E_pos - E_neg) is small when E_pos - E_neg << 0,
            # and always >= 0.
            energy_gap = E_pos - E_neg
            loss = F.softplus(energy_gap)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * b
            total_n += b

            if batch_idx % 100 == 0:
                print(
                    f"[EBM] Epoch {epoch:02d} "
                    f"Batch {batch_idx:04d}/{len(train_loader):04d} "
                    f"loss={loss.item():.4f} gap={energy_gap.item():.4f}"
                )

        avg_loss = total_loss / total_n
        print(f"[EBM] Epoch {epoch:02d} - avg loss: {avg_loss:.4f}")

    return model


def train_diffusion_model(
    model: SimpleDiffusion,
    train_loader,
    device: str = "mps",
    epochs: int = 5,
):
    """
    Standard DDPM-style training for the diffusion model:
      - Sample t
      - Add noise to x0 to get x_t
      - Train network to predict that noise
    """
    model.to(device)
    optimizer = optim.Adam(model.network.parameters(), lr=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0

        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            b = imgs.size(0)

            # sample time steps for each image
            t = torch.randint(0, model.num_steps, (b,), device=device)
            loss = model.p_losses(imgs, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * b
            total_n += b

        avg_loss = total_loss / total_n
        print(f"[Diffusion] Epoch {epoch:02d} - loss: {avg_loss:.4f}")

    return model