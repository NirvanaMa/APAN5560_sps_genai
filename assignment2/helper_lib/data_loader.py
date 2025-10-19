import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data_loader(data_dir: str, batch_size: int = 32, train: bool = True) -> DataLoader:
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(root=data_dir, transform=tfm)
    use_cuda = torch.cuda.is_available()
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=2, pin_memory=use_cuda)

def get_cifar10_loaders(data_dir: str = "./data", batch_size: int = 128):
    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2023, 0.1994, 0.2010]
    train_tfms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tfms)
    test_set  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tfms)
    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=use_cuda)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=use_cuda)
    return train_loader, test_loader

def get_imagefolder_64(data_dir: str, batch_size: int = 64, train: bool = True):
    tfm = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(root=data_dir, transform=tfm)
    use_cuda = torch.cuda.is_available()
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=2, pin_memory=use_cuda)

def get_cifar10_loaders_64(data_dir: str = "./data", batch_size: int = 128):
    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2023, 0.1994, 0.2010]
    train_tfms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tfms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=train_tfms)
    test_set  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tfms)
    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=use_cuda)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=use_cuda)
    return train_loader, test_loader