# test_cnn64.py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from helper_lib.model import get_model

CLASSES = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def unnormalize(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)
    return (img.detach().cpu() * std + mean).clamp(0,1)

def show_batch(images, labels, preds, probs=None, rows=2, cols=4):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6), constrained_layout=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes[:len(images)]):
        img = unnormalize(images[i]).permute(1,2,0).numpy()
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])

        pred_txt = f"Pred: {CLASSES[preds[i]]}"
        if probs is not None:
            pred_txt += f" ({probs[i, preds[i]].item():.2f})"
        true_txt = f"True: {CLASSES[labels[i]]}"

        combined = f"{pred_txt} | {true_txt}"

        ax.set_title(combined, fontsize=10, pad=8)

    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    plt.show()

def main():
    device = pick_device()
    print("Using device:", device)

    # ✅ 64×64 to match CNN64x64 architecture
    tfm = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True)

    model = get_model("cnn64", num_classes=10)
    state = torch.load("checkpoints/cnn64_cifar10.pth", map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()

    images, labels = next(iter(testloader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        _, preds = probs.max(1)

    show_batch(images, labels, preds, probs)

if __name__ == "__main__":
    main()