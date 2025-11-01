import torch
import torch.nn as nn

LATENT_DIM = 100
IMG_C = 1
IMG_H = IMG_W = 28

# -------------------
# Generator (spec)
#  z -> FC -> 7x7x128 -> TConv(128->64, k4,s2,p1) + BN + ReLU (->14x14)
#  -> TConv(64->1, k4,s2,p1) + Tanh (->28x28)
# -------------------
class Generator(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)
        self.deconv = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)                                # (B, 7*7*128)
        x = x.view(z.size(0), 128, 7, 7)             # (B,128,7,7)
        return self.deconv(x)                        # (B,1,28,28)

# -------------------
# Discriminator (spec)
#  x -> Conv(1->64,k4,s2,p1)+LReLU -> Conv(64->128,k4,s2,p1)+BN+LReLU
#  -> Flatten -> Linear(128*7*7 -> 1)    (BCEWithLogitsLoss)
# -------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 14x14 -> 7x7
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(x.size(0), -1)
        out = self.fc(h).view(-1)  # logits
        return out

def make_generator(latent_dim: int = LATENT_DIM) -> Generator:
    return Generator(latent_dim)