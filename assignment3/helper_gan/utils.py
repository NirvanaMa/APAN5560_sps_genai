import os, io, base64
import torch
import torchvision.utils as vutils
from PIL import Image

def save_image_grid(tensor, path: str, nrow: int = 8):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    grid = vutils.make_grid(tensor, nrow=nrow, normalize=True, value_range=(-1,1))
    vutils.save_image(grid, path)

def tensor_grid_base64(tensor, nrow: int = 8, fmt: str = "PNG") -> str:
    grid = vutils.make_grid(tensor, nrow=nrow, normalize=True, value_range=(-1,1))
    img = (grid.clamp(0,1) * 255).byte().permute(1,2,0).cpu().numpy()
    mode = "L" if img.shape[2] == 1 else "RGB"
    pil = Image.fromarray(img.squeeze(), mode=mode)
    buf = io.BytesIO(); pil.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")
