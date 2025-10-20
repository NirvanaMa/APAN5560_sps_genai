# Assignment 2

This project trains and deploys a Convolutional Neural Network (CNN) on the CIFAR-10 dataset.  
The model is first trained locally using PyTorch, then packaged with FastAPI in a Docker container for inference.

## Author Siliang Ma
   
- Code to run Assignment 2:  (Locally Without Docker)  
```bash
cd ~/Desktop/sps_genai/assignment2
uv sync
uv run python train_cnn64.py
uv run python test_cnn64.py
``` 
- Code to run Assignment 2:  (By Docker)
Since the model have saved in checkpoints/cnn64_cifar10.pth  

## The final version of assignment2 is in the Assignment2 branch linked below:  
https://github.com/NirvanaMa/APAN5560_sps_genai/tree/Assignment2




  

 
