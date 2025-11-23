# assignment4/app.py

import math
import io
from typing import Optional

import torch
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from helper_lib import (
    get_model,
    generate_ebm_samples,
    generate_diffusion_samples,
    tensor_grid_to_base64,
    get_default_device,
    load_model,
)

# -------------------------------------------------------
# FastAPI app + CORS
# -------------------------------------------------------

app = FastAPI(title="Assignment 4: CIFAR10 EBM + Diffusion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# Global config & lazy-loaded models
# -------------------------------------------------------

DEVICE = get_default_device()
EBM_CHECKPOINT = "checkpoints/ebm_cifar10.pth"
DIFF_CHECKPOINT = "checkpoints/diffusion_cifar10.pth"

_ebm_model = None
_diff_model = None


def get_ebm():
    """Lazy-load the EBM from checkpoint."""
    global _ebm_model
    if _ebm_model is None:
        model = get_model("EBM")
        _ebm_model = load_model(model, EBM_CHECKPOINT, DEVICE)
    return _ebm_model


def get_diffusion():
    """Lazy-load the diffusion model from checkpoint."""
    global _diff_model
    if _diff_model is None:
        model = get_model("Diffusion")
        _diff_model = load_model(model, DIFF_CHECKPOINT, DEVICE)
    return _diff_model


# -------------------------------------------------------
# Root / health
# -------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Assignment 4 API running.",
        "device": DEVICE,
        "endpoints": [
            "/ebm/generate?n=16&steps=60",
            "/ebm/image?n=16&steps=60",
            "/diffusion/generate?n=16",
            "/diffusion/image?n=16",
            "/docs",
        ],
    }


# -------------------------------------------------------
# EBM JSON endpoint (base64 PNG grid)
# -------------------------------------------------------

@app.get("/ebm/generate")
def ebm_generate(
    n: int = Query(16, ge=1, le=64),
    steps: int = Query(60, ge=1, le=256),
    seed: Optional[int] = Query(None),
):
    """
    Generate CIFAR-like images from the EBM with Langevin dynamics.
    Returns a JSON with a base64-encoded PNG grid.
    """
    try:
        model = get_ebm()
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"error": f"Could not load EBM: {e}"},
        )

    if seed is not None:
        torch.manual_seed(seed)

    imgs = generate_ebm_samples(
        model,
        device=DEVICE,
        n_samples=n,
        steps=steps,
        step_size=0.1,
        noise_std=0.01,
        img_size=32,
    )
    nrow = max(1, int(math.sqrt(n)))
    img_b64 = tensor_grid_to_base64(imgs, nrow=nrow)

    return {
        "model": "EBM",
        "count": n,
        "steps": steps,
        "seed": seed,
        "image_base64_png": img_b64,
    }


# -------------------------------------------------------
# EBM IMAGE endpoint (returns PNG directly)
# -------------------------------------------------------

@app.get("/ebm/image")
def ebm_image(
    n: int = Query(16, ge=1, le=64),
    steps: int = Query(60, ge=1, le=256),
    seed: Optional[int] = Query(None),
):
    """
    Generate CIFAR-like images from the EBM and return a PNG image grid.
    This is convenient to view directly in the browser.
    """
    model = get_ebm()

    if seed is not None:
        torch.manual_seed(seed)

    imgs = generate_ebm_samples(
        model,
        device=DEVICE,
        n_samples=n,
        steps=steps,
        step_size=0.1,
        noise_std=0.01,
        img_size=32,
    )

    # Convert batch in [-1,1] -> grid PNG bytes
    imgs = ((imgs + 1.0) / 2.0).clamp(0.0, 1.0).cpu()
    grid = make_grid(imgs, nrow=max(1, int(math.sqrt(n))))
    pil_img = to_pil_image(grid)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    return Response(buf.getvalue(), media_type="image/png")


# -------------------------------------------------------
# Diffusion JSON endpoint (base64 PNG grid)
# -------------------------------------------------------

@app.get("/diffusion/generate")
def diffusion_generate(
    n: int = Query(16, ge=1, le=64),
    seed: Optional[int] = Query(None),
):
    """
    Generate CIFAR-like images from the diffusion model.
    Returns a JSON with a base64-encoded PNG grid.
    """
    try:
        model = get_diffusion()
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"error": f"Could not load Diffusion model: {e}"},
        )

    if seed is not None:
        torch.manual_seed(seed)

    imgs = generate_diffusion_samples(
        model,
        device=DEVICE,
        n_samples=n,
        img_size=32,
    )
    nrow = max(1, int(math.sqrt(n)))
    img_b64 = tensor_grid_to_base64(imgs, nrow=nrow)

    return {
        "model": "Diffusion",
        "count": n,
        "seed": seed,
        "image_base64_png": img_b64,
    }


# -------------------------------------------------------
# Diffusion IMAGE endpoint (returns PNG directly)
# -------------------------------------------------------

@app.get("/diffusion/image")
def diffusion_image(
    n: int = Query(16, ge=1, le=64),
    seed: Optional[int] = Query(None),
):
    """
    Generate CIFAR-like images from the diffusion model and return a PNG grid.
    """
    model = get_diffusion()

    if seed is not None:
        torch.manual_seed(seed)

    imgs = generate_diffusion_samples(
        model,
        device=DEVICE,
        n_samples=n,
        img_size=32,
    )

    imgs = ((imgs + 1.0) / 2.0).clamp(0.0, 1.0).cpu()
    grid = make_grid(imgs, nrow=max(1, int(math.sqrt(n))))
    pil_img = to_pil_image(grid)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    return Response(buf.getvalue(), media_type="image/png")