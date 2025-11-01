# app.py
import os, io, math
import torch
import torchvision.utils as vutils
from PIL import Image
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, Response

from helper_gan.models import make_generator, LATENT_DIM

APP_TITLE = "Module 6 API + MNIST GAN"
DESCRIPTION = "Serve a trained MNIST GAN with both base64 JSON and direct PNG endpoints."

app = FastAPI(title=APP_TITLE, description=DESCRIPTION)

# Model loading (serve on CPU)
CHECKPOINT_PATH = os.environ.get("GAN_CHECKPOINT", "checkpoints/gan_generator.pth")
DEVICE = "cpu"

GEN = make_generator().to(DEVICE)
try:
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    GEN.load_state_dict(state)
    GEN.eval()
    MODEL_READY = True
except Exception as e:
    print(f"[WARN] Could not load generator from {CHECKPOINT_PATH}: {e}")
    MODEL_READY = False

# Helpers
def make_grid_pil(tensor: torch.Tensor, nrow: int = 8) -> Image.Image:
    """
    Convert a batch of images in [-1,1] to a PIL grid image.
    tensor: (B, 1, 28, 28)
    """
    grid = vutils.make_grid(tensor, nrow=nrow, normalize=True, value_range=(-1, 1))  # (C,H,W) in [0,1]
    arr = (grid.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    # MNIST is single-channel; handle both cases safely
    if arr.shape[2] == 1:
        return Image.fromarray(arr.squeeze(axis=2), mode="L")
    return Image.fromarray(arr, mode="RGB")

def png_bytes_from_tensor(tensor: torch.Tensor, nrow: int) -> bytes:
    img = make_grid_pil(tensor, nrow=nrow)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def base64_from_tensor(tensor: torch.Tensor, nrow: int) -> str:
    import base64
    png = png_bytes_from_tensor(tensor, nrow)
    return base64.b64encode(png).decode("ascii")

# ----------------------------
# Endpoints
# ----------------------------
@app.get("/")
def root():
    return {
        "message": "API is running. See /docs.",
        "gan_ready": MODEL_READY,
        "endpoints": [
            "/gan/generate?n=64&seed=15",
            "/gan/generate.png?n=64&seed=15",
            "/health",
        ],
    }
@app.get("/gan/generate", tags=["GAN"])
def gan_generate(n: int = Query(64, ge=1, le=100), seed: int | None = None):
    """
    Returns a base64-encoded PNG grid of generated digits.
    Handy for Swagger/JSON clients.
    """
    if not MODEL_READY:
        return JSONResponse(
            status_code=503,
            content={"error": f"Generator not loaded. Expected checkpoint at {CHECKPOINT_PATH}."},
        )

    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        z = torch.randn(n, LATENT_DIM, device=DEVICE)
        samples = GEN(z).cpu()

    nrow = max(1, int(math.sqrt(n)))
    b64_png = base64_from_tensor(samples, nrow)
    return {"count": n, "seed": seed, "image_base64_png": b64_png}

@app.get("/gan/generate.png", tags=["GAN"])
def gan_generate_png(n: int = Query(64, ge=1, le=100), seed: int | None = None):
    """
    Returns a real PNG image so you can see it directly in the browser.
    """
    if not MODEL_READY:
        return JSONResponse(
            status_code=503,
            content={"error": f"Generator not loaded. Expected checkpoint at {CHECKPOINT_PATH}."},
        )

    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        z = torch.randn(n, LATENT_DIM, device=DEVICE)
        samples = GEN(z).cpu()

    nrow = max(1, int(math.sqrt(n)))
    png_bytes = png_bytes_from_tensor(samples, nrow)
    return Response(content=png_bytes, media_type="image/png")