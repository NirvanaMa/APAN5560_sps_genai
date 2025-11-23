
import os
import io
import base64
import torch
import torchvision.utils as vutils
from PIL import Image


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")


def load_model(model, path: str, device: str):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Loaded model from {path} on {device}")
    return model


def tensor_grid_to_base64(imgs, nrow: int = 8) -> str:
    """
    imgs: (N, C, H, W) in [-1, 1]
    returns base64-encoded PNG string
    """
    # scale from [-1,1] to [0,1]
    imgs = (imgs + 1.0) / 2.0
    imgs = imgs.clamp(0.0, 1.0)

    grid = vutils.make_grid(imgs, nrow=nrow)
    ndarr = (
        grid.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    pil_img = Image.fromarray(ndarr)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    img_bytes = buf.read()
    return base64.b64encode(img_bytes).decode("utf-8")