# Week 7 Exercises

## Exercise 7.1: Launch DDP on Your Machine

Run `lab/ddp_demo.py` with `--world_size 2` on your machine.

- On CPU: `python -m torchrun --nproc_per_node=2 week07/lab/ddp_demo.py`
- On GPU: `python -m torchrun --nproc_per_node=2 week07/lab/ddp_demo.py --backend nccl`

**Deliverable**: Terminal output showing both ranks training and losses.

## Exercise 7.2: Scaling Efficiency Table

Use `lab/communication_cost.py` to fill a table for a 1B-parameter model, world sizes [2, 4, 8], and bandwidths [100, 400, 800] Gbps. Assume 100 ms compute time per step.

**Deliverable**: `exercises/scaling_table.md`.

## Exercise 7.3: Distributed Sampler

Modify `lab/ddp_demo.py` to use `torch.utils.data.distributed.DistributedSampler` so that ranks do not see overlapping data. Verify that each rank processes a different shard.

**Deliverable**: `exercises/ddp_with_sampler.py`.

## Exercise 7.4: FSDP on GPU (if available)

Run `lab/fsdp_demo.py` on a multi-GPU machine. Compare peak memory per GPU with FSDP vs. DDP for the same model.

**Deliverable**: `exercises/fsdp_memory_report.md`.

## Exercise 7.5: Data vs. Tensor vs. Pipeline Parallelism

Write a one-page comparison (`exercises/parallelism_comparison.md`) covering:

- What is split across devices.
- Communication pattern.
- Best use case.
- A modern model that uses it (e.g., GPT-3, Llama, DeepSeek-V3).
