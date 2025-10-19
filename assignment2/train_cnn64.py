# train_cnn64.py
from helper_lib.data_loader import get_cifar10_loaders_64
from helper_lib.model import get_model
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model
from helper_lib.utils import save_model

import torch, torch.nn as nn, torch.optim as optim

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    device = pick_device()
    print("Using device:", device)

    train_loader, test_loader = get_cifar10_loaders_64("./data", batch_size=128)

    model = get_model("cnn64", num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_model(model, train_loader, criterion, optimizer, device=device, epochs=15)
    evaluate_model(model, test_loader, criterion, device=device)
    save_model(model, "checkpoints/cnn64_cifar10.pth")

if __name__ == "__main__":
    main()