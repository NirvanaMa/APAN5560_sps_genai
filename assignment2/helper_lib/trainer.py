import torch
import torch.nn.functional as F

def train_model(model, data_loader, criterion, optimizer, device: str = "cpu", epochs: int = 10):
    device = torch.device(device); model.to(device)
    for epoch in range(1, epochs + 1):
        model.train(); running_loss = 0.0; correct = 0; total = 0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward(); optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        print(f"Epoch {epoch:02d}/{epochs} - loss: {running_loss/max(total,1):.4f} - acc: {correct/max(total,1):.4f}")
    return model

# VAE bits (optional)
def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(x.size(0), -1), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (BCE + KLD) / x.size(0)

def train_vae_model(model, data_loader, optimizer, device="cpu", epochs: int = 10):
    device = torch.device(device); model.to(device)
    for epoch in range(1, epochs + 1):
        model.train(); total_loss = 0.0
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(inputs)
            loss = vae_loss(recon, inputs, mu, logvar)
            loss.backward(); optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        print(f"Epoch {epoch:02d}/{epochs} - VAE loss: {total_loss/len(data_loader.dataset):.4f}")
    return model