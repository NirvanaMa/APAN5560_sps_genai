# helper_lib/data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data_loader(data_dir: str, batch_size: int = 32, train: bool = True) -> DataLoader:
    """
    Creates a DataLoader using an ImageFolder directory.
    If you use MNIST-like images, adjust Resize/Normalize as needed.
    """
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # ImageNet-style normalization works fine for generic RGB datasets.
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=tfm)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2, pin_memory=True)
    return loader