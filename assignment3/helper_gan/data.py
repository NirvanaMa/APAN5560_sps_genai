import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_mnist_loader(data_dir: str = "./data", batch_size: int = 128) -> DataLoader:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # scale to [-1, 1]
    ])
    train = datasets.MNIST(root=data_dir, train=True, download=True, transform=tfm)
    # MPS/GPU pin_memory is fine; PyTorch ignores when unsupported
    return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
