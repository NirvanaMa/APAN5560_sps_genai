import torch, matplotlib.pyplot as plt
def generate_samples(model, device="cpu", num_samples=9):
    model.eval(); device = torch.device(device)
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        samples = model.decode(z).cpu().view(num_samples, 3, 224, 224)
        n = int(num_samples ** 0.5) or 1
        fig, axes = plt.subplots(n, n, figsize=(2*n, 2*n))
        axes = axes if isinstance(axes, (list, tuple)) else [[axes]]
        for i in range(n):
            for j in range(n):
                k = i*n + j
                if k < num_samples:
                    axes[i][j].imshow(samples[k].permute(1,2,0).numpy()); axes[i][j].axis("off")
        plt.show()