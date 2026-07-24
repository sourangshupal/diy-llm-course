"""Week 11 Lab 3: Analyze reward components and detect reward hacking.

Generates responses from a model and prints the reward breakdown for each.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_demo import generate_group, make_math_data
from reward_functions import combined_reward


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze reward components")
    parser.add_argument("--model_dir", type=Path, default=Path("week11/outputs/grpo_model"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model_name = str(args.model_dir) if args.model_dir.exists() else "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    data = make_math_data()
    for item in data[:3]:
        print(f"\nQuestion: {item['question']} (answer: {item['answer']})")
        responses = generate_group(model, tokenizer, item["question"], device, group_size=4)
        for r in responses:
            rewards = combined_reward(r, item["answer"])
            print(f"  {r!r}")
            print(f"    correctness={rewards['correctness']:.0f}, format={rewards['format']:.0f}, length={rewards['length']:.3f}, total={rewards['total']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
