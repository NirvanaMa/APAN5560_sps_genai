# ===== Base image =====
FROM python:3.12-slim

# Workdir
WORKDIR /app

# System deps (small set; removes apt cache afterward)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy only what's needed to serve
COPY app.py /app/app.py
COPY helper_gan /app/helper_gan
COPY checkpoints /app/checkpoints
COPY pyproject.toml /app/pyproject.toml

# Python deps (CPU)
# Note: torch/torchvision wheels exist for linux/amd64 and linux/arm64.
RUN pip install --no-cache-dir \
    torch torchvision fastapi uvicorn pillow

# Expose FastAPI port
EXPOSE 8000

# Start the server
# Set GAN_CHECKPOINT env to override default if needed
ENV GAN_CHECKPOINT=checkpoints/gan_generator.pth
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]