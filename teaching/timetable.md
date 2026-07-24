# Diy-LLM Weekly Teaching Timetable

**Course:** Diy-LLM — Build a Large Language Model from Scratch  
**Format:** 12 weekly sessions, 3 hours each (lecture + live coding + lab)  
**Prerequisites:** Python, PyTorch basics, linear algebra, probability, machine-learning fundamentals  
**Environment:** Python 3.12+, uv-managed virtual environment  

---

## Weekly Overview

| Week | Topic | Agenda (3 hours) | Learning Outcomes | Deliverable |
|------|-------|------------------|-------------------|-------------|
| **1** | **Course Introduction & Experiment Tracking** | - Welcome & course roadmap (20 min) <br> - Reproducible ML workflows (30 min) <br> - Weights & Biases demo: logging, dashboards, sweeps (50 min) <br> - Lab: instrument a training run (60 min) <br> - Wrap-up & homework (20 min) | - Explain why experiment tracking is essential for LLM training. <br> - Log metrics, hyperparameters, and artifacts with W&B. <br> - Compare runs and reproduce results from a config. | Working W&B run |
| **2** | **Tokenization** | - Why tokenization matters (20 min) <br> - Byte-Pair Encoding algorithm (40 min) <br> - Live coding: build BPE from scratch (60 min) <br> - Lab: train a tokenizer, inspect merges/vocabulary (50 min) <br> - Wrap-up & comparison with SentencePiece (10 min) | - Implement byte-pair encoding training and encoding/decoding. <br> - Analyze trade-offs between vocabulary size, sequence length, and multilingual support. <br> - Prepare text for a Transformer model. | Trained BPE tokenizer |
| **3** | **Transformer Architecture I** | - Attention intuition and self-attention math (40 min) <br> - Multi-head attention, masks, RoPE (50 min) <br> - Live coding: embeddings + attention block (60 min) <br> - Lab: forward pass of a reference Transformer (40 min) <br> - Debug shapes and attention patterns (30 min) | - Derive scaled dot-product attention and multi-head attention. <br> - Implement RoPE and causal masking. <br> - Build and run a forward pass of a Transformer block. | Forward-pass reference model |
| **4** | **Transformer Architecture II & Training** | - RMSNorm, SwiGLU, residual connections (30 min) <br> - Optimizers: AdamW, learning-rate schedules (30 min) <br> - Live coding: full Transformer + training loop (60 min) <br> - Lab: train an Assignment 1 baseline (60 min) <br> - Inspect loss curves and generated samples (30 min) | - Assemble a complete decoder-only Transformer with modern choices. <br> - Train a small LM with AdamW and cosine LR. <br> - Generate text and diagnose training issues. | A1 baseline trained |
| **5** | **GPU Architecture & Optimization** | - GPU memory hierarchy and throughput (30 min) <br> - Roofline model and arithmetic intensity (40 min) <br> - Profiling tools: PyTorch profiler, Nsight (40 min) <br> - Lab: profile a training step and identify bottlenecks (60 min) <br> - Discussion: kernel fusion and memory bandwidth (30 min) | - Describe GPU execution model and memory levels. <br> - Use profiling to find compute vs. memory bottlenecks. <br> - Apply roofline analysis to guide optimization. | Profiling report |
| **6** | **High-Performance Kernels** | - Why custom kernels matter (20 min) <br> - Triton basics and tile-based programming (40 min) <br> - FlashAttention-2 algorithm and memory complexity (40 min) <br> - Live coding: attention kernel in Triton (60 min) <br> - Lab: benchmark correctness and speed (30 min) | - Explain the memory-complexity advantage of FlashAttention. <br> - Write a simple Triton kernel. <br> - Validate a custom attention implementation against PyTorch. | Correct + fast attention kernel |
| **7** | **Distributed Training** | - Data parallelism and gradient synchronization (30 min) <br> - DDP vs. FSDP trade-offs (30 min) <br> - Mixed precision, gradient accumulation, checkpointing (30 min) <br> - Live coding: DDP training script (60 min) <br> - Lab: run multi-GPU training (30 min) | - Set up DDP and FSDP training. <br> - Understand communication patterns (all-reduce, all-gather). <br> - Scale batch size and learning rate across GPUs. | Multi-GPU training run |
| **8** | **Scaling Laws** | - Historical scaling laws (Kaplan, Chinchilla) (30 min) <br> - IsoFLOP analysis and compute-optimal training (40 min) <br> - Power-law fitting (30 min) <br> - Lab: run small-scale experiments and fit curves (70 min) <br> - Predict loss for new model/data sizes (20 min) | - Fit power-law relationships between loss, compute, parameters, and data. <br> - Design compute-optimal training runs. <br> - Interpret scaling-law predictions and limitations. | Scaling-law report |
| **9** | **Data Engineering** | - Web-scale data pipeline overview (20 min) <br> - Language identification and quality filtering (40 min) <br> - Deduplication: MinHash, LSH (40 min) <br> - Lab: clean a Common Crawl sample (70 min) <br> - Contamination and privacy considerations (20 min) | - Build a pre-training data cleaning pipeline. <br> - Implement near-duplicate detection with MinHash. <br> - Audit data for benchmark contamination. | Cleaned dataset |
| **10** | **Alignment: SFT & Expert Iteration** | - From pre-training to instruction following (20 min) <br> - SFT data curation and formatting (30 min) <br> - Live coding: SFT training loop (60 min) <br> - Expert iteration / self-improvement loop (30 min) <br> - Lab: fine-tune a model and evaluate outputs (40 min) | - Prepare instruction-following datasets. <br> - Fine-tune a base model with supervised learning. <br> - Implement an expert-iteration loop for iterative improvement. | SFT/EI results |
| **11** | **Alignment: GRPO & RLVR** | - RLHF vs. RLVR (20 min) <br> - GRPO objective and group-relative baselines (40 min) <br> - Reward design: correctness, format, length (30 min) <br> - Live coding: GRPO training loop (60 min) <br> - Lab: reward ablations and diagnose reward hacking (40 min) | - Contrast learned and verifiable reward sources. <br> - Implement GRPO with clipped policy gradients and KL penalty. <br> - Design reward functions and detect reward hacking. | GRPO results |
| **12** | **Evaluation & Inference** | - Why evaluation is hard (20 min) <br> - Perplexity, benchmarks, zero/few-shot, CoT (40 min) <br> - Fair comparison and contamination (30 min) <br> - Live demo: lm-evaluation-harness + custom eval (50 min) <br> - Lab: evaluate two models and write a report (50 min) | - Evaluate models with standard and custom benchmarks. <br> - Control prompts, shots, and decoding for fair comparison. <br> - Interpret scores and write a defensible evaluation report. | Evaluation report |

---

## Legend

- **Lecture:** Conceptual presentation with slides / whiteboard.
- **Live coding:** Instructor demonstrates code in real time; students follow along.
- **Lab:** Hands-on time for students to run code, experiment, and fill out a weekly report.
- **Wrap-up:** Review key takeaways, collect questions, and preview homework.

---

## Cross-Week Dependencies

| Week | Depends on |
|------|------------|
| 2 (Tokenization) | — |
| 3 (Transformer I) | Week 2 tokenizer output |
| 4 (Transformer II & Training) | Week 3 architecture code |
| 5 (GPU Optimization) | Week 4 training loop |
| 6 (Kernels) | Weeks 4–5 |
| 7 (Distributed Training) | Weeks 4–6 |
| 8 (Scaling Laws) | Weeks 4–7 training results |
| 9 (Data Engineering) | — (parallel to scaling track) |
| 10 (SFT) | Week 4 trained base model |
| 11 (GRPO) | Week 10 SFT model |
| 12 (Evaluation) | Weeks 10–11 aligned models; Week 4 base model |

---

## Suggested Weekly Homework

| Week | Homework |
|------|----------|
| 1 | Connect W&B to a personal project; log a hyperparameter sweep. |
| 2 | Train a BPE tokenizer on a new domain corpus; compare vocabulary overlap with GPT-2. |
| 3 | Implement multi-head attention without `nn.MultiheadAttention`; verify shapes. |
| 4 | Train to a target perplexity on a tiny dataset; ablate one hyperparameter. |
| 5 | Profile your Week 4 training loop and list the top three bottlenecks. |
| 6 | Implement a fused kernel for an elementwise operation; benchmark vs. eager PyTorch. |
| 7 | Run DDP on two GPUs (or two CPU processes); compare throughput vs. single GPU. |
| 8 | Fit a scaling law using at least three model sizes; predict loss for a 2× model. |
| 9 | Clean 100 MB of raw crawl; report deduplication rate and quality-filter statistics. |
| 10 | Fine-tune a base model on a small instruction dataset; evaluate before and after. |
| 11 | Add a new reward component to GRPO; observe reward-hacking behavior. |
| 12 | Design a 20-sample custom benchmark; compare two models and write a one-page memo. |
