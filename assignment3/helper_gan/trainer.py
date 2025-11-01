import os
import torch
import torch.nn as nn
import torch.optim as optim
from .models import Discriminator, make_generator, LATENT_DIM
from .utils import save_image_grid

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def train_gan(
    train_loader,
    epochs: int = 10,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    beta1: float = 0.5,
    beta2: float = 0.999,
    outdir: str = "checkpoints",
    sample_every: int = 1,
    device: str | None = None,
):
    device = torch.device(device or pick_device())
    os.makedirs(outdir, exist_ok=True)

    G = make_generator().to(device)
    D = Discriminator().to(device)
    criterion = nn.BCEWithLogitsLoss()
    opt_g = optim.Adam(G.parameters(), lr=lr_g, betas=(beta1, beta2))
    opt_d = optim.Adam(D.parameters(), lr=lr_d, betas=(beta1, beta2))

    fixed_z = torch.randn(64, LATENT_DIM, device=device)

    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        d_running, g_running, n = 0.0, 0.0, 0

        for real, _ in train_loader:
            real = real.to(device)
            bs = real.size(0)
            valid = torch.ones(bs, device=device)
            fake  = torch.zeros(bs, device=device)

            # ---- Train D ----
            z = torch.randn(bs, LATENT_DIM, device=device)
            with torch.no_grad():
                fake_imgs = G(z)

            D.zero_grad(set_to_none=True)
            d_real = D(real)
            d_fake = D(fake_imgs)
            d_loss = criterion(d_real, valid) + criterion(d_fake, fake)
            d_loss.backward()
            opt_d.step()

            # ---- Train G ----
            G.zero_grad(set_to_none=True)
            z = torch.randn(bs, LATENT_DIM, device=device)
            gen_imgs = G(z)
            d_gen = D(gen_imgs)
            g_loss = criterion(d_gen, valid)  # fool D
            g_loss.backward()
            opt_g.step()

            d_running += d_loss.item() * bs
            g_running += g_loss.item() * bs
            n += bs

        print(f"Epoch {epoch:02d}/{epochs} - D_loss: {d_running/max(n,1):.4f} - G_loss: {g_running/max(n,1):.4f}")

        if epoch % sample_every == 0:
            with torch.no_grad():
                samples = G(fixed_z).cpu()
            save_image_grid(samples, f"{outdir}/gan_samples_epoch_{epoch:02d}.png")

    torch.save(G.state_dict(), f"{outdir}/gan_generator.pth")
    print(f"Saved generator to {outdir}/gan_generator.pth")