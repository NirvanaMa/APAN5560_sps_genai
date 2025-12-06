# Dockerfile for Assignment 5 - GPT-2 LLM FastAPI service

FROM python:3.12-slim

# Avoid interactive prompts, speed up installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Workdir inside container
WORKDIR /app

# System deps (git/curl are handy but optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files into container
# Make sure your folder has:
# - app.py
# - checkpoints/gpt2_finetuned/ (fine-tuned model)
# - any helper files you use
COPY . /app

# Install Python deps
# datasets is only needed if you plan to train inside container; for serving only, you can drop it.
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    transformers \
    torch

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app
# If your file isn't named app.py or the app object isn't named "app", adjust this.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]