"""Week 10 Lab 3: Simple SFT evaluation harness.

Evaluates a fine-tuned model on the tiny math prompts.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft_demo import generate_sample


def extract_number(text: str) -> int | None:
    numbers = re.findall(r"-?\d+", text)
    if numbers:
        return int(numbers[-1])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate SFT model")
    parser.add_argument("--model_dir", type=Path, default=Path("week10/outputs/sft_model"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)

    test_prompts = [
        ("Solve: 2 + 3", 5),
        ("Solve: 7 - 4", 3),
        ("Solve: 3 * 3", 9),
        ("What is the capital of France?", None),  # open-ended
    ]

    correct = 0
    total_math = 0
    for instruction, answer in test_prompts:
        response = generate_sample(model, tokenizer, instruction, device, max_new_tokens=15)
        print(f"Q: {instruction}")
        print(f"A: {response.strip()}\n")
        if answer is not None:
            total_math += 1
            predicted = extract_number(response)
            if predicted == answer:
                correct += 1

    if total_math:
        print(f"Math accuracy: {correct}/{total_math} = {correct / total_math * 100:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
