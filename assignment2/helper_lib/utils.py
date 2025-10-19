import os, torch
def save_model(model, path: str = "checkpoints/model.pth"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")