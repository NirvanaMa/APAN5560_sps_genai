from helper_gan.data import get_mnist_loader
from helper_gan.trainer import train_gan, pick_device

def main():
    device = pick_device()
    print("Using device:", device)
    loader = get_mnist_loader("./data", batch_size=128)
    train_gan(loader, epochs=15, outdir="checkpoints", device=device)

if __name__ == "__main__":
    main()