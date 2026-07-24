# Diy-LLM Teaching Syllabus

**Course**: Diy-LLM — Build a Large Language Model from Scratch  
**Duration**: 12 weeks  
**Format**: Weekly 3-hour session (lecture + live coding + lab)  
**Prerequisites**: Python, PyTorch basics, linear algebra, probability, machine-learning fundamentals  
**Environment**: Python 3.12+, uv-managed virtual environment, GPU access required for Weeks 3–11

---

## Learning Outcomes

By the end of this course, students will be able to:

1. Implement a Transformer language model from scratch in PyTorch.
2. Train, debug, and monitor a small LLM using modern optimizers and learning-rate schedules.
3. Profile and optimize GPU kernels, including a custom FlashAttention-style attention implementation.
4. Scale training across multiple GPUs with data / tensor / pipeline parallelism.
5. Fit and interpret scaling laws to predict model performance.
6. Build a pre-training data pipeline: language identification, quality filtering, deduplication.
7. Align a model with supervised fine-tuning (SFT), expert iteration, and GRPO.
8. Evaluate models with standard academic frameworks and custom benchmarks.

---

## Weekly Schedule

| Week | Theme | Theory Reading | Practical Focus | Deliverable |
|------|-------|----------------|-----------------|-------------|
| 1 | Course Introduction & Experiment Tracking | Preface, Ch 1 | Environment setup; Weights & Biases intro | Working W&B run |
| 2 | Tokenization | Ch 2 | Byte-Pair Encoding (BPE) from scratch | Trained BPE tokenizer |
| 3 | Transformer Architecture I | Ch 3–4 | Embeddings, RoPE, attention, MLP blocks | Forward-pass reference model |
| 4 | Transformer Architecture II & Training | Ch 4 | RMSNorm, SwiGLU, AdamW, cosine LR, training loop | A1 baseline trained |
| 5 | GPU Architecture & Optimization | Ch 5–6 | GPU memory hierarchy; profiling; roofline model | Profiling report |
| 6 | High-Performance Kernels | Ch 7 | Triton; FlashAttention-2 implementation | Correct + fast attention kernel |
| 7 | Distributed Training | Ch 8 | DDP / FSDP; multi-GPU training script | Multi-GPU training run |
| 8 | Scaling Laws | Ch 9 | IsoFLOPs experiments; power-law fitting | Scaling-law report |
| 9 | Data Engineering | Ch 11 | Common Crawl filtering; MinHash dedup | Cleaned dataset |
| 10 | Alignment: SFT & Expert Iteration | Ch 13 | SFT data sweeps; iterative self-improvement | SFT/EI results |
| 11 | Alignment: GRPO & RLVR | Ch 14 | Reward functions; GRPO ablations | GRPO results |
| 12 | Evaluation & Inference | Ch 10, Ch 12 | lm-evaluation-harness; evalscope; inference optimization | Evaluation report |

---

## Assessment

| Component | Weight |
|-----------|--------|
| Weekly lab submissions | 30% |
| Assignment reports (A1–A6) | 40% |
| In-class participation / demos | 15% |
| Final mini-project | 15% |

### Mini-Project Ideas

- Scale A1 to a larger model and evaluate it on more tasks.
- Implement a missing kernel variant (e.g., block-sparse attention) in A2.
- Collect a domain-specific dataset and reproduce A4 for that domain.
- Extend A5 with a new reward function or reasoning domain.
- Build a new benchmark task and add it to lm-evaluation-harness in A6.

---

## Recommended Reading

- **Original course**: [Stanford CS336 Spring 2025](https://stanford-cs336.github.io/spring2025/)
- **Diy-LLM docs**: `docs/en/` in this repository
- **Core papers**:
  - Vaswani et al., "Attention Is All You Need"
  - Brown et al., "Language Models are Few-Shot Learners"
  - Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla)
  - Kaplan et al., "Scaling Laws for Neural Language Models"
  - Ouyang et al., "Training language models to follow instructions with human feedback"
  - DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"

---

## Hardware & Environment

- **Minimum for Weeks 1–2, 12**: any laptop (CPU okay).
- **Minimum for Weeks 3–4**: single GPU with ≥12 GB VRAM (e.g., RTX 3090/4090, A10).
- **Recommended for Weeks 5–11**: single A100 (80 GB) or multiple GPUs.
- **Cloud options**: Lambda Labs, RunPod, Google Colab, AutoDL, Alibaba Cloud PAI.

Use the shared uv environment:

```bash
uv sync
source .venv/bin/activate
```

For assignment-specific heavy dependencies:

```bash
# Systems track (Weeks 5–7)
uv sync --extra systems

# Alignment track (Weeks 10–11)
uv sync --extra alignment

# Evaluation track (Week 12)
uv sync --extra evaluation
```
