import torch

def evaluate_model(model, data_loader, criterion, device: str = "cpu"):
    device = torch.device(device); model.to(device); model.eval()
    total, correct, running_loss = 0, 0, 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    print(f"Test - loss: {avg_loss:.4f} - acc: {acc:.4f}")
    return avg_loss, acc