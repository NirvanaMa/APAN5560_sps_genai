import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 224x224 classifiers ----------
class SimpleFCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(224 * 224 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class EnhancedCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512), nn.ReLU(),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ---------- VAE (224x224 flattened) ----------
class VAE(nn.Module):
    def __init__(self, latent_dim: int = 20):
        super().__init__()
        self.enc_fc1 = nn.Linear(3 * 224 * 224, 512)
        self.enc_mu = nn.Linear(512, latent_dim)
        self.enc_logvar = nn.Linear(512, latent_dim)
        self.dec_fc1 = nn.Linear(latent_dim, 512)
        self.dec_fc2 = nn.Linear(512, 3 * 224 * 224)
        self.latent_dim = latent_dim
    def encode(self, x_flat):
        h = F.relu(self.enc_fc1(x_flat))
        return self.enc_mu(h), self.enc_logvar(h)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar); eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        return torch.sigmoid(self.dec_fc2(h))
    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ---------- CNN (64x64) ----------
class CNN64x64(nn.Module):

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        # After two pools: 64 -> 32 -> 16; channels=32 => 32*16*16 = 8192
        self.fc1   = nn.Linear(32 * 16 * 16, 100)
        self.fc2   = nn.Linear(100, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 64x64 -> 64x64 (16 ch)
        x = self.pool(x)           # 64 -> 32
        x = F.relu(self.conv2(x))  # 32x32 (32 ch)
        x = self.pool(x)           # 32 -> 16
        x = torch.flatten(x, 1)    # (B, 8192)
        x = F.relu(self.fc1(x))    # 100
        x = self.fc2(x)            # num_classes
        return x


# ---------- factory ----------
def get_model(model_name: str, num_classes: int = 10) -> nn.Module:
    name = model_name.strip().lower()
    if name == "fcnn":
        return SimpleFCNN(num_classes=num_classes)
    if name == "cnn":
        return SimpleCNN(num_classes=num_classes)
    if name in ("enhancedcnn", "enhanced_cnn"):
        return EnhancedCNN(num_classes=num_classes)
    if name == "vae":
        return VAE()
    if name in ("cnn64", "cnn_64", "cnn64x64"):
        return CNN64x64(num_classes=num_classes)
    raise ValueError("model_name must be one of: 'FCNN','CNN','EnhancedCNN','CIFARCNN','CNN64','VAE'")