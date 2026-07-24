"""Week 11 Lab 1: Reward functions for verifiable rewards.

Provides correctness and format rewards for simple math tasks.
"""

from __future__ import annotations

import argparse
import re


def extract_final_number(text: str) -> int | None:
    """Extract the last integer from a response."""
    numbers = re.findall(r"-?\d+", text)
    if numbers:
        return int(numbers[-1])
    return None


def correctness_reward(response: str, answer: int) -> float:
    """Return 1.0 if the final number in response equals the answer."""
    predicted = extract_final_number(response)
    return 1.0 if predicted == answer else 0.0


def format_reward(response: str, marker: str = "Answer:") -> float:
    """Return 1.0 if the response contains the expected marker."""
    return 1.0 if marker in response else 0.0


def length_penalty(response: str, target_len: int = 20) -> float:
    """Penalize outputs much longer than a target length."""
    length = len(response.split())
    if length <= target_len:
        return 0.0
    return -0.01 * (length - target_len)


def combined_reward(response: str, answer: int, weights: dict[str, float] | None = None) -> dict[str, float]:
    """Compute multiple reward components."""
    if weights is None:
        weights = {"correctness": 1.0, "format": 0.2, "length": 1.0}

    components = {
        "correctness": correctness_reward(response, answer),
        "format": format_reward(response),
        "length": length_penalty(response),
    }
    total = sum(weights.get(k, 1.0) * components[k] for k in components)
    components["total"] = total
    return components


def main() -> int:
    parser = argparse.ArgumentParser(description="Reward function demo")
    parser.add_argument("--response", type=str, default="The answer is Answer: 5")
    parser.add_argument("--answer", type=int, default=5)
    args = parser.parse_args()

    rewards = combined_reward(args.response, args.answer)
    print(f"Response: {args.response!r}")
    print(f"Expected answer: {args.answer}")
    for k, v in rewards.items():
        print(f"  {k}: {v:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
