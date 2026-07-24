# Week 11 Exercises

## Exercise 11.1: Reward Function Design

Choose a task other than math (e.g., JSON formatting, date conversion, simple word problems). Write `exercises/custom_reward.py` with at least two reward components.

**Deliverable**: `exercises/custom_reward.py` + example outputs.

## Exercise 11.2: Group Size Ablation

Run `lab/grpo_demo.py` with group sizes [2, 4, 8]. Plot mean reward and correct rate vs. group size.

**Deliverable**: `exercises/group_size_ablation.py` + plot.

## Exercise 11.3: Reward Hacking Analysis

Use `lab/analyze_rewards.py` after training. Find at least one example where the model gets a high format reward but low correctness reward.

**Deliverable**: `exercises/reward_hacking_examples.md`.

## Exercise 11.4: Length Normalization Experiment

Modify `lab/grpo_demo.py` to include a strong length penalty. Observe how output length and correctness change. Document trade-offs.

**Deliverable**: `exercises/length_norm_experiment.md`.

## Exercise 11.5: Compare to SFT

Train two models: one with SFT only (Week 10) and one with SFT + GRPO. Evaluate both on the same test prompts and compare.

**Deliverable**: `exercises/sft_vs_grpo.md` with accuracy table and sample outputs.

## Lab Files

- [`analyze_rewards.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week11/lab/analyze_rewards.py)
- [`grpo_demo.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week11/lab/grpo_demo.py)
- [`reward_functions.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week11/lab/reward_functions.py)
