FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git wget curl build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app

RUN pip install --no-cache-dir \
    torch torchvision fastapi uvicorn pillow python-multipart

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]