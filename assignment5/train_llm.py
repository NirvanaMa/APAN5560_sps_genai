# assignment5/train_llm.py

import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm.auto import tqdm

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class SQuADDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])


def format_squad(example):
    q = example["question"]
    a = example["answers"]["text"][0]
    return (
        f"Question: {q}\n"
        f"Answer: That is a great question. {a} Let me know if you have any other questions."
    )


def main():
    print("Using device:", DEVICE)

    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading SQuAD dataset...")
    squad = load_dataset("rajpurkar/squad")

    # SAMPLE DATA
    train_samples = 5000
    val_samples = 1000

    #
    train_subset = squad["train"].select(range(train_samples))
    val_subset = squad["validation"].select(range(val_samples))

    train_texts = [format_squad(ex) for ex in train_subset]
    val_texts = [format_squad(ex) for ex in val_subset]

    print("Tokenizing...")
    train_enc = tokenizer(train_texts, truncation=True, padding="max_length", max_length=128)
    val_enc = tokenizer(val_texts, truncation=True, padding="max_length", max_length=128)

    train_dataset = SQuADDataset(train_enc)
    val_dataset = SQuADDataset(val_enc)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)

    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    total_steps = len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=50, num_training_steps=total_steps
    )

    epochs = 1  # Keep training fast
    print("Starting training...")

    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch in loop:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            loop.set_postfix(loss=loss.item())

    model.save_pretrained("checkpoints/gpt2_finetuned")
    tokenizer.save_pretrained("checkpoints/gpt2_finetuned")

    print("\nModel saved to checkpoints/gpt2_finetuned")


if __name__ == "__main__":
    main()