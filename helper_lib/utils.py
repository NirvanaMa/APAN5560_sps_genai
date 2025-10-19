# helper_lib/utils.py
import torch
import os

def save_model(model, path: str = "model.pth"):
    """
    Saves model state dict to the given path, creating parent folders if needed.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")