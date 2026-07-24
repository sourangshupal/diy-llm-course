# Week 7: Distributed Training

## Learning Objectives

By the end of this week, students should be able to:

- Explain data parallelism, tensor parallelism, and pipeline parallelism.
- Launch a PyTorch DistributedDataParallel (DDP) training job.
- Understand gradient synchronization and the communication overhead.
- Describe when to use FSDP vs. DDP.

## Pre-Reading

- Distributed Training chapter of the course book (English translation pending)

## Lab Files

Run these in order:

1. [`lab/ddp_demo.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week07/lab/ddp_demo.py) — DDP training on a simple model.
2. [`lab/fsdp_demo.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week07/lab/fsdp_demo.py) — FSDP overview (GPU recommended).
3. [`lab/communication_cost.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week07/lab/communication_cost.py) — estimate all-reduce bandwidth.

## Exercises

See [`exercises/README.md`](exercises.md).

## Deliverable

A short report showing:

- DDP training launched on 2+ processes.
- Training loss curves from each rank.
- Estimated communication overhead vs. compute time.

## Key Concepts

| Concept | Description |
|---------|-------------|
| Data parallelism | Each GPU holds a full model copy; processes different data shards; synchronizes gradients. |
| DDP | PyTorch's DistributedDataParallel wrapper. |
| All-reduce | Collective operation that sums gradients across ranks. |
| FSDP | Fully Sharded Data Parallel; shards model parameters, gradients, and optimizer states across ranks. |
| Tensor parallelism | Split individual layers across GPUs (e.g., column/row parallel linear). |
| Pipeline parallelism | Split layers across GPUs; feed micro-batches through the pipeline. |
| World size | Total number of processes/ranks. |
| Rank | Unique ID of a process in the distributed group. |

## Common Pitfalls

- Forgetting to set `MASTER_ADDR` and `MASTER_PORT`.
- Using the wrong backend (`gloo` for CPU, `nccl` for CUDA).
- Not calling `dist.destroy_process_group()` at exit.
- DDP on notebook / interactive environments can be tricky.

## Platform Note

DDP with the `gloo` backend works on CPU and is runnable on a laptop. FSDP and `nccl` require CUDA GPUs. Multi-GPU machines or cloud instances are needed for realistic scaling experiments.
