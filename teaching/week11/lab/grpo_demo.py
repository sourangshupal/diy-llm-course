"""Week 11 Lab 2: Simplified GRPO training loop.

This demo uses a small causal LM (gpt2) and simple math questions. It is a
teaching implementation, not a production RL trainer.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from reward_functions import combined_reward, extract_final_number


def make_math_data() -> list[dict]:
    """Simple math questions with verifiable answers."""
    return [
        {"question": "Solve: 1 + 2", "answer": 3},
        {"question": "Solve: 5 + 4", "answer": 9},
        {"question": "Solve: 10 - 3", "answer": 7},
        {"question": "Solve: 6 * 2", "answer": 12},
        {"question": "Solve: 8 / 2", "answer": 4},
        {"question": "Solve: 3 + 7", "answer": 10},
        {"question": "Solve: 9 - 5", "answer": 4},
        {"question": "Solve: 4 * 3", "answer": 12},
    ]


def generate_group(
    model,
    tokenizer,
    question: str,
    device: torch.device,
    group_size: int = 4,
    max_new_tokens: int = 15,
) -> list[str]:
    """Generate a group of responses for one question."""
    prompt = f"### Question:\n{question}\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    responses = []
    for _ in range(group_size):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Keep only the response part
        marker = "### Answer:\n"
        if marker in text:
            text = text.split(marker, 1)[1]
        responses.append(text.strip())
    return responses


def compute_grpo_loss(
    model,
    ref_model,
    tokenizer,
    question: str,
    responses: list[str],
    answer: int,
    device: torch.device,
    eps: float = 0.2,
    beta: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute GRPO loss for one question and its response group."""
    # Compute rewards
    rewards = [combined_reward(r, answer)["total"] for r in responses]
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    mean_r = rewards_t.mean()
    std_r = rewards_t.std(unbiased=False) + 1e-8
    advantages = (rewards_t - mean_r) / std_r

    # Tokenize prompt + responses
    prompt = f"### Question:\n{question}\n\n### Answer:\n"
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.size(1)

    total_loss = torch.tensor(0.0, device=device)
    total_tokens = 0

    for response, adv in zip(responses, advantages):
        full_text = prompt + response
        encoding = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = encoding.input_ids.to(device)
        seq_len = input_ids.size(1)
        if seq_len <= prompt_len:
            continue

        # Forward current and reference policy
        with torch.no_grad():
            ref_logits = ref_model(input_ids).logits
        logits = model(input_ids).logits

        # Compute log-probs for response tokens
        log_probs = F.log_softmax(logits[:, prompt_len - 1 : -1, :], dim=-1)
        ref_log_probs = F.log_softmax(ref_logits[:, prompt_len - 1 : -1, :], dim=-1)

        target_ids = input_ids[:, prompt_len:]
        token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        ref_token_log_probs = ref_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        ratio = torch.exp(token_log_probs - ref_token_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - eps, 1 + eps)
        policy_loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()

        # KL penalty (ref - current) averaged over response tokens
        kl = (ref_token_log_probs - token_log_probs).mean()
        loss = policy_loss + beta * kl

        total_loss += loss * (seq_len - prompt_len)
        total_tokens += seq_len - prompt_len

    avg_loss = total_loss / max(1, total_tokens)
    metrics = {
        "mean_reward": mean_r.item(),
        "std_reward": std_r.item(),
        "correct_rate": sum(1 for r in responses if extract_final_number(r) == answer) / len(responses),
    }
    return avg_loss, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Simplified GRPO demo")
    parser.add_argument("--model_dir", type=Path, default=Path("week10/outputs/sft_model"))
    parser.add_argument("--output_dir", type=Path, default=Path("week11/outputs/grpo_model"))
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_new_tokens", type=int, default=15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use SFT model if available, otherwise gpt2
    model_name = str(args.model_dir) if args.model_dir.exists() else "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    ref_model.eval()

    optimizer = AdamW(model.parameters(), lr=args.lr)
    data = make_math_data()

    print("\nRunning GRPO...")
    for step in range(1, args.steps + 1):
        step_metrics: list[dict] = []
        losses: list[torch.Tensor] = []

        for item in tqdm(data, desc=f"Step {step}/{args.steps}"):
            responses = generate_group(
                model, tokenizer, item["question"], device,
                group_size=args.group_size, max_new_tokens=args.max_new_tokens,
            )
            loss, metrics = compute_grpo_loss(
                model, ref_model, tokenizer, item["question"], responses, item["answer"], device,
            )
            losses.append(loss)
            step_metrics.append(metrics)

        avg_loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        mean_reward = sum(m["mean_reward"] for m in step_metrics) / len(step_metrics)
        correct_rate = sum(m["correct_rate"] for m in step_metrics) / len(step_metrics)
        print(f"Step {step}: loss={avg_loss.item():.4f}, mean_reward={mean_reward:.3f}, correct_rate={correct_rate:.2f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
