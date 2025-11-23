# Assignment 4

Goal:
This project implements Energy-Based Models (EBM) and Diffusion Models using PyTorch and deploys both as image generation services via FastAPI and Docker.
  
Code for clone:  
```bash
git clone -b Assignment4 --single-branch https://github.com/NirvanaMa/APAN5560_sps_genai.git
```


## Author Siliang Ma
   
- Code to run Assignment 4:  (Locally Without Docker)  

Training is slow — this step is optional if checkpoints are already included.
```bash
cd ~/Desktop/sps_genai/assignment4
uv sync
uv run python train_energy_diffusion.py
``` 
  
Locally test the output：
```bash
uv run python test_ebm_samples.py           
uv run python test_diffusion_samples.py
```
  
locally run fastapi:
```bash
uv run uvicorn app:app --reload --port 8000
```
  

- Code to run Assignment 4:  (By Docker)
Since the model have saved in checkpoints/cnn64_cifar10.pth  

Build and Run the container and expose port 8000:  
```bash
docker build -t assignment4-api .
docker run --rm -p 8000:8000 assignment4-api
```

The result will show up in link below:
http://localhost:8000/docs




  

 
