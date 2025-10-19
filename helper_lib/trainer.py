# helper_lib/trainer.py
import torch

def train_model(model, data_loader, criterion, optimizer, device: str = "cpu", epochs: int = 10):
    """
    Basic training loop. Prints simple progress each epoch and returns the trained model.
    """
    device = torch.device(device)
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        print(f"Epoch {epoch:02d}/{epochs} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}")

    return model