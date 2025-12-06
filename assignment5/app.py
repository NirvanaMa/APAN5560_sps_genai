# assignment5/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

MODEL_PATH = "checkpoints/gpt2_finetuned"

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(DEVICE)

app = FastAPI(title="Assignment 5 LLM API")


class Query(BaseModel):
    question: str
    max_new_tokens: int = 256  # allow long answers


@app.post("/generate_with_llm")
def generate_with_llm(body: Query):
    prompt = f"Question: {body.question}\nAnswer:"

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=body.max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return {"generated_text": text}