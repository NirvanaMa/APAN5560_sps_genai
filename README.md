# Assignment 3

This project implements, trains, and deploys a **Generative Adversarial Network (GAN)** using **PyTorch** and **FastAPI**.  
The model is trained on the **MNIST** dataset to generate hand-written digits, and then deployed in a Docker container for API-based inference.

Code for clone:
```bash
git clone -b Assignment3 --single-branch https://github.com/NirvanaMa/APAN5560_sps_genai.git
```

## Author: Siliang Ma

---

###  Code to Run Assignment 3 (Locally Without Docker)

Train the GAN model on MNIST and save the generator checkpoint:

```bash
cd ~/Desktop/sps_genai/assignment3
uv sync
uv run python train_gan.py
``` 
  
Run the FastAPI server locally after training:

```bash
uv run uvicorn app:app --reload --port 8000
```

###   Code to run Assignment 3:  (By Docker)
Since the model have saved in checkpoints/gan_generator.pth

Build and Run the container and expose port 8000:  
```bash
docker build -t mnist-gan-api .
docker run --rm -p 8000:8000 mnist-gan-api
```

The result will show up in link below:
http://localhost:8000/docs




