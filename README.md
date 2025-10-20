# Assignment 2

This project trains and deploys a Convolutional Neural Network (CNN) on the CIFAR-10 dataset.  
The model is first trained locally using PyTorch, then packaged with FastAPI in a Docker container for inference.

Get clone code:  
```bash
git clone --branch Assignment2 --single-branch https://github.com/NirvanaMa/APAN5560_sps_genai.git
```

## Author: Siliang Ma
   
Code to run Assignment 2:  (By Docker)  

The model have uploaded and saved in checkpoints/cnn64_cifar10.pth, model are ready to run without training.
  
* For manully train the model:
```bash
cd ~/assignment2
uv sync
uv run python train_cnn64.py
``` 

* Use Docker to Run the container and expose port 8000:  
```bash
docker build -t cifar10-fastapi .
docker run --rm -p 8000:8000 cifar10-fastapi
```
  
* The interactive result is showned below link:  
http://localhost:8000/docs


Code to run Assignment 2:  (Locally Without Docker)  
```bash
cd ~/assignment2
uv sync
uv run python train_cnn64.py
uv run python test_cnn64.py
``` 


  
