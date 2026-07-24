"""Week 12 Lab 2: Custom evaluation for a small model.

Evaluates a causal LM on a custom JSONL dataset of (prompt, reference) pairs.
Uses exact-match after extracting the final answer.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_number(text: str) -> int | None:
    numbers = re.findall(r"-?\d+", text)
    if numbers:
        return int(numbers[-1])
    return None


def build_sample_eval_data(path: Path) -> None:
    """Create a tiny evaluation dataset for demonstration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = [
        {"prompt": "Solve: 2 + 3", "answer": 5},
        {"prompt": "Solve: 10 - 4", "answer": 6},
        {"prompt": "Solve: 3 * 4", "answer": 12},
        {"prompt": "Solve: 8 / 2", "answer": 4},
        {"prompt": "What is the capital of France?", "answer": "paris"},
        {"prompt": "What is the capital of Japan?", "answer": "tokyo"},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def evaluate(
    model,
    tokenizer,
    dataset: list[dict],
    device: torch.device,
    max_new_tokens: int = 20,
) -> dict:
    """Evaluate model on the custom dataset."""
    correct = 0
    total = 0
    results = []

    for item in dataset:
        prompt = item["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        reference = item["answer"]
        if isinstance(reference, int):
            predicted = extract_number(generated)
            match = predicted == reference
        else:
            predicted = generated.lower()
            match = reference in predicted

        if match:
            correct += 1
        total += 1

        results.append({
            "prompt": prompt,
            "reference": reference,
            "generated": generated,
            "predicted": predicted,
            "correct": match,
        })

    accuracy = correct / total if total else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Custom evaluation demo")
    parser.add_argument("--model_dir", type=Path, default=Path("gpt2"))
    parser.add_argument("--data", type=Path, default=Path("week12/data/custom_eval.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("week12/outputs/custom_eval_results.json"))
    parser.add_argument("--max_new_tokens", type=int, default=20)
    args = parser.parse_args()

    if not args.data.exists():
        build_sample_eval_data(args.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)

    with open(args.data, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    metrics = evaluate(model, tokenizer, dataset, device, args.max_new_tokens)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    print(f"Detailed results saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
