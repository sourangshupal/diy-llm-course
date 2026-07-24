# Week 11: Alignment — GRPO & Reinforcement Learning with Verifiable Rewards

## Learning Objectives

By the end of this week, students should be able to:

- Explain the difference between RLHF and RLVR (verifiable rewards).
- Implement a simplified GRPO training loop.
- Design reward functions for math reasoning tasks.
- Diagnose reward hacking and length bias.

## Pre-Reading

- `docs/en/chapter14/chapter14_可验证奖励的强化学习.md`
- DeepSeek-R1 paper (optional)

## Lab Files

Run these in order:

1. [`lab/reward_functions.py`](./lab/reward_functions.py) — reward functions and scoring.
2. [`lab/grpo_demo.py`](./lab/grpo_demo.py) — simplified GRPO training loop.
3. [`lab/analyze_rewards.py`](./lab/analyze_rewards.py) — inspect reward components.

## Exercises

See [`exercises/README.md`](./exercises/README.md).

## Deliverable

A short report containing:

- Reward function definition.
- Training curve of mean reward vs. GRPO step.
- Example generations showing improvement (or reward hacking).
- Ablation: effect of reward normalization or clipping.

## Key Concepts

| Concept | Description |
|---------|-------------|
| RLHF | Reinforcement learning from human feedback (uses a learned reward model). |
| RLVR | Reinforcement learning with verifiable rewards (uses executable checkers). |
| GRPO | Group Relative Policy Optimization; uses group baselines instead of a critic. |
| Advantage | `(r_i - mean(r)) / std(r)` within a group. |
| Clipping | Limits policy update size to prevent collapse. |
| Reward hacking | Model optimizes proxy reward instead of true objective (e.g., format over correctness). |
| Length normalization | Prevents rewards from favoring longer outputs. |

## Common Pitfalls

- Reward functions that are too sparse.
- Reward hacking on format or length.
- KL divergence exploding if the policy updates too aggressively.
- Group size too small for stable advantage estimates.

## Platform Note

The demo uses a small model (`gpt2`) and simple math tasks so it can run on CPU/MPS. Real GRPO experiments (e.g., Qwen2.5-Math) require large GPUs and packages like `trl`, `vllm`, and `math-verify`, which can be installed separately when needed.
