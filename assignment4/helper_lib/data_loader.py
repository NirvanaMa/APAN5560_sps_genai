
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_loader(
    data_dir: str = "./data",
    batch_size: int = 128,
    train: bool = True,
    num_workers: int = 2,
):
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            # normalize to [-1, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = datasets.CIFAR10(
        root=data_dir, train=train, transform=transform, download=True
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader