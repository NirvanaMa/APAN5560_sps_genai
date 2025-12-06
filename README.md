# Assignment 5 — Fine-Tuning GPT-2 & Deploying via FastAPI

Goal:

This project fine-tunes a GPT-2 language model (openai-community/gpt2) using the SQuAD question–answer dataset.
The fine-tuned model is then deployed as an API using FastAPI, allowing custom-formatted text generation.

  
You may choose any answer format; in this implementation, responses will showed as below format:   

" Answer: That is a great question. "  < Answer >  " Let me know if you have any other questions."  

Code for clone:  
```bash
git clone -b Assignment5 --single-branch https://github.com/NirvanaMa/APAN5560_sps_genai.git
```

## Author Siliang Ma
   
- Code to run Assignment 5:  (Locally Without Docker)  

Training is slow — this step is optional if checkpoints are already included.
```bash
cd ~/Desktop/sps_genai/assignment5
uv sync
uv run python test_llm.py
``` 
  
locally run fastapi:
```bash
uv run uvicorn app:app --reload --port 8000
```
  

- Code to run Assignment 5:  (By Docker)
Since the model have saved in checkpoints folder

Build and Run the container and expose port 8000:  
```bash
docker build -t assignment5-llm-api .
docker run --rm -p 8000:8000 assignment5-llm-api
```

The result will show up in link below:
http://localhost:8000/docs




  

 
