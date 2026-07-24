"""Week 10 Lab 2: Expert iteration demo.

1. Generate candidate answers for math prompts using the current model.
2. Keep only correct answers.
3. Fine-tune the model on the filtered correct answers.

This is a simplified version of expert iteration / STaR.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft_demo import InstructionDataset, INSTRUCTION_TEMPLATE, generate_sample


def extract_number(text: str) -> int | None:
    """Extract the last integer from a response."""
    numbers = re.findall(r"-?\d+", text)
    if numbers:
        return int(numbers[-1])
    return None


def make_math_prompts() -> list[dict]:
    """Return simple math prompts with ground-truth answers."""
    return [
        {"instruction": "Solve: 1 + 2", "answer": 3},
        {"instruction": "Solve: 5 + 4", "answer": 9},
        {"instruction": "Solve: 10 - 3", "answer": 7},
        {"instruction": "Solve: 6 * 2", "answer": 12},
        {"instruction": "Solve: 8 / 2", "answer": 4},
        {"instruction": "Solve: 3 + 7", "answer": 10},
        {"instruction": "Solve: 9 - 5", "answer": 4},
        {"instruction": "Solve: 4 * 3", "answer": 12},
    ]


def generate_candidates(
    model,
    tokenizer,
    prompts: list[dict],
    device: torch.device,
    n_candidates: int = 5,
) -> list[dict]:
    """Generate candidate responses and keep correct ones."""
    correct_examples = []
    for prompt in prompts:
        for _ in range(n_candidates):
            response = generate_sample(model, tokenizer, prompt["instruction"], device, max_new_tokens=15)
            # Extract the response part after the template
            marker = "### Response:\n"
            if marker in response:
                response_text = response.split(marker, 1)[1]
            else:
                response_text = response

            predicted = extract_number(response_text)
            if predicted == prompt["answer"]:
                correct_examples.append({
                    "instruction": prompt["instruction"],
                    "response": response_text.strip(),
                })
                break
    return correct_examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Expert iteration demo")
    parser.add_argument("--model_dir", type=Path, default=Path("week10/outputs/sft_model"))
    parser.add_argument("--output_dir", type=Path, default=Path("week10/outputs/ei_model"))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--epochs_per_round", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if not args.model_dir.exists():
        print(f"Model directory not found: {args.model_dir}. Run sft_demo.py first.")
        return 1

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)

    prompts = make_math_prompts()

    for round_idx in range(1, args.rounds + 1):
        print(f"\n=== Expert Iteration Round {round_idx}/{args.rounds} ===")
        candidates = generate_candidates(model, tokenizer, prompts, device, n_candidates=5)
        print(f"Collected {len(candidates)}/{len(prompts)} correct responses")

        if not candidates:
            print("No correct candidates found; stopping.")
            break

        # Fine-tune on correct candidates
        dataset = InstructionDataset(tokenizer, candidates, max_length=64)
        from transformers import Trainer, TrainingArguments
        training_args = TrainingArguments(
            output_dir=str(args.output_dir / f"round_{round_idx}"),
            num_train_epochs=args.epochs_per_round,
            per_device_train_batch_size=2,
            learning_rate=5e-5,
            logging_steps=1,
            use_cpu=device.type == "cpu",
            report_to="none",
        )
        trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
        trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nFinal model saved to {args.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
