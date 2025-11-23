
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------- Energy-Based Model (EBM) ---------


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)



class EnergyCNN(nn.Module):
    """
    Stronger EBM backbone for CIFAR-10.
    Input:  (B, 3, 32, 32) in [-1, 1]
    Output: (B,) scalar energy per image
    """
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            # 32x32 -> 32x32
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # keep 8x8, deepen a bit
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Linear(256 * 8 * 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)                     # (B, 256, 8, 8)
        h = h.view(x.size(0), -1)            # (B, 256*8*8)
        energy = self.fc(h).squeeze(1)       # (B,)
        return energy

# --------- Diffusion: UNet + wrapper ---------


class SinusoidalEmbedding(nn.Module):
    """
    Simple sinusoidal embedding for time / noise level.
    """

    def __init__(self, num_frequencies: int = 16):
        super().__init__()
        self.num_frequencies = num_frequencies

    def forward(self, t: torch.Tensor):
        """
        t: shape (B,) in [0, 1]
        returns (B, 2 * num_frequencies)
        """
        device = t.device
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0), math.log(1000.0), self.num_frequencies, device=device
            )
        )  # (F,)
        freqs = freqs.view(1, -1)  # (1,F)

        # (B,1) * (1,F) -> (B,F)
        angles = t.view(-1, 1) * freqs
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.act = nn.ReLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.act(h)
        h = self.conv2(h)
        return self.act(h + self.skip(x))


class SimpleUNet(nn.Module):
    """
    Small UNet for CIFAR-10 sized images (3 x 32 x 32).
    Predicts noise epsilon(x_t, t).
    """

    def __init__(self, img_channels=3, base_ch=64, time_dim=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(num_frequencies=time_dim // 4),
            nn.Linear(time_dim // 2, time_dim),
            nn.ReLU(),
        )

        # down
        self.rb1 = ResidualBlock(img_channels, base_ch, time_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, 2, 1)  # 16x16

        self.rb2 = ResidualBlock(base_ch, base_ch * 2, time_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, 2, 1)  # 8x8

        # bottleneck
        self.rb_mid = ResidualBlock(base_ch * 2, base_ch * 2, time_dim)

        # up
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, 2, 1)  # 16x16
        self.rb3 = ResidualBlock(base_ch * 4, base_ch, time_dim)

        self.up2 = nn.ConvTranspose2d(base_ch, base_ch, 4, 2, 1)  # 32x32
        self.rb4 = ResidualBlock(base_ch * 2, base_ch, time_dim)

        self.out_conv = nn.Conv2d(base_ch, img_channels, 1)

    def forward(self, x, t: torch.Tensor):
        # t in [0,1], shape (B,)
        t_emb = self.time_embed(t)  # (B, time_dim)

        # down path
        h1 = self.rb1(x, t_emb)
        d1 = self.down1(h1)

        h2 = self.rb2(d1, t_emb)
        d2 = self.down2(h2)

        # middle
        h_mid = self.rb_mid(d2, t_emb)

        # up path
        u1 = self.up1(h_mid)
        u1 = torch.cat([u1, h2], dim=1)
        u1 = self.rb3(u1, t_emb)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, h1], dim=1)
        u2 = self.rb4(u2, t_emb)

        out = self.out_conv(u2)
        return out  # predicted noise


class SimpleDiffusion(nn.Module):
    """
    DDPM-style diffusion wrapper for training & sampling.
    """

    def __init__(
        self,
        network: nn.Module,
        num_steps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.network = network
        self.num_steps = num_steps

        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def _extract(self, buf, t, x_shape):
        out = buf[t].float()
        while out.ndim < len(x_shape):
            out = out.unsqueeze(-1)
        return out.expand(x_shape)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self._extract(self.alphas_cumprod.sqrt(), t, x0.shape)
        sqrt_om = self._extract((1.0 - self.alphas_cumprod).sqrt(), t, x0.shape)
        return sqrt_ac * x0 + sqrt_om * noise

    def p_losses(self, x0, t):
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise=noise)

        # normalize t to [0,1]
        t_norm = (t.float() + 1) / self.num_steps
        noise_pred = self.network(x_noisy, t_norm)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def p_sample(self, x, t):
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_ac = self._extract(
            (1.0 - self.alphas_cumprod).sqrt(), t, x.shape
        )
        sqrt_recip_alpha = self._extract((1.0 / self.alphas).sqrt(), t, x.shape)

        t_norm = (t.float() + 1) / self.num_steps
        eps_theta = self.network(x, t_norm)

        model_mean = sqrt_recip_alpha * (x - betas_t / sqrt_one_minus_ac * eps_theta)

        if (t == 0).all():
            return model_mean

        noise = torch.randn_like(x)
        sigma = torch.sqrt(betas_t)
        return model_mean + sigma * noise

    @torch.no_grad()
    def sample(self, n_samples: int, device: str, img_size: int = 32):
        self.eval()
        x = torch.randn(n_samples, 3, img_size, img_size, device=device)
        for i in reversed(range(self.num_steps)):
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t)
        return x.clamp(-1.0, 1.0).detach()


# --------- Factory ---------


def get_model(model_name: str):
    """
    In Assignment 4 helper_lib we support:
      - "EBM"       : Energy-based model
      - "Diffusion" : Diffusion model (UNet + wrapper)
    """
    name = model_name.lower()
    if name == "ebm":
        return EnergyCNN()
    elif name == "diffusion":
        unet = SimpleUNet(img_channels=3, base_ch=64, time_dim=64)
        diffusion = SimpleDiffusion(unet, num_steps=200)
        return diffusion
    else:
        raise ValueError("model_name must be one of: 'EBM', 'Diffusion'")