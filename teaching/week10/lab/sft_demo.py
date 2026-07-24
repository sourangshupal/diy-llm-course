"""Week 10 Lab 1: Supervised fine-tuning (SFT) demo.

Fine-tunes a small causal LM (default: gpt2) on a tiny instruction dataset.
Uses the standard transformers Trainer for simplicity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


INSTRUCTION_TEMPLATE = "### Instruction:\n{}\n\n### Response:\n{}"


class InstructionDataset(Dataset):
    """Tiny instruction dataset for demonstration."""

    def __init__(self, tokenizer, examples: list[dict], max_length: int = 128) -> None:
        self.tokenizer = tokenizer
        self.examples = examples
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        text = INSTRUCTION_TEMPLATE.format(ex["instruction"], ex["response"])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        # Labels same as input_ids for causal LM loss
        return {"input_ids": input_ids, "labels": input_ids.clone(), "attention_mask": encoding["attention_mask"].squeeze(0)}


def make_toy_data() -> list[dict]:
    """Return a small but diverse instruction dataset for demonstration.

    Note: real SFT requires thousands to millions of examples. This is a toy
    dataset to demonstrate the training loop and prompt formatting.
    """
    capitals = [
        ("France", "Paris"),
        ("Japan", "Tokyo"),
        ("Germany", "Berlin"),
        ("Italy", "Rome"),
        ("Spain", "Madrid"),
        ("Canada", "Ottawa"),
        ("Australia", "Canberra"),
        ("Brazil", "Brasília"),
        ("India", "New Delhi"),
        ("China", "Beijing"),
    ]
    facts = [
        ("the largest planet", "Jupiter"),
        ("the smallest planet", "Mercury"),
        ("the hottest planet", "Venus"),
        ("the planet with rings", "Saturn"),
        ("the red planet", "Mars"),
    ]
    colors = ["Red", "Blue", "Yellow", "Green", "Orange", "Purple"]
    math = [
        ("2 + 3", "5"),
        ("7 - 4", "3"),
        ("3 * 3", "9"),
        ("8 / 2", "4"),
        ("5 + 6", "11"),
        ("10 - 2", "8"),
        ("4 * 5", "20"),
        ("9 / 3", "3"),
        ("12 + 7", "19"),
        ("15 - 6", "9"),
    ]

    data: list[dict] = []
    for country, city in capitals:
        data.append({"instruction": f"What is the capital of {country}?", "response": f"{city}."})
    for item, answer in facts:
        data.append({"instruction": f"What is {item}?", "response": f"{answer}."})
    for color in colors:
        data.append({"instruction": f"Name a primary color.", "response": f"{color}."})
    for expr, answer in math:
        data.append({"instruction": f"Solve: {expr}", "response": answer})

    return data


def generate_sample(model, tokenizer, instruction: str, device: torch.device, max_new_tokens: int = 20) -> str:
    """Generate a response for an instruction."""
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="SFT demo")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=Path, default=Path("week10/outputs/sft_model"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    examples = make_toy_data()
    dataset = InstructionDataset(tokenizer, examples, max_length=args.max_length)

    print("\nBefore SFT:")
    for instruction in ["What is the capital of France?", "Solve: 2 + 3"]:
        print(f"  {instruction!r} -> {generate_sample(model, tokenizer, instruction, device)!r}")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=5e-5,
        logging_steps=1,
        save_strategy="epoch",
        use_cpu=device.type == "cpu",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("\nFine-tuning...")
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")

    print("\nAfter SFT:")
    for instruction in ["What is the capital of France?", "Solve: 2 + 3"]:
        print(f"  {instruction!r} -> {generate_sample(model, tokenizer, instruction, device)!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
