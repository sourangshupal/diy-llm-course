# Week 4 Exercises

## Exercise 4.1: Model Diagram

Draw (by hand or with a tool) the full architecture of `lab/model.py`, labeling:

- embedding layer,
- each Transformer block (attention + FFN),
- residual connections,
- normalization layers,
- output projection.

**Deliverable**: `exercises/model_diagram.png` or `exercises/model_diagram.md`.

## Exercise 4.2: Ablation Study

Create `exercises/ablation.py` that trains three small models and compares final loss:

1. Baseline (RMSNorm + SwiGLU + RoPE).
2. No RoPE (use learned positional embeddings instead).
3. Post-norm instead of pre-norm.

Use the same tiny corpus and config otherwise.

**Deliverable**: `exercises/ablation.py` + table of final losses.

## Exercise 4.3: Integrate Week 2 Tokenizer

Modify `lab/data.py` to load the BPE tokenizer trained in Week 2 (`week02/data/my_tokenizer`). Retrain the model on the same corpus but with subword tokens instead of characters.

**Deliverable**: `exercises/data_with_bpe.py` + generated text sample.

## Exercise 4.4: Learning Rate Schedule Plot

Write `exercises/plot_lr.py` that instantiates the cosine-with-warmup schedule from `lab/train.py` and plots learning rate vs. step using matplotlib.

**Deliverable**: `exercises/plot_lr.py` + saved `lr_schedule.png`.

## Exercise 4.5: Generation Sampling Comparison

Use `lab/generate.py` with the same prompt and checkpoint but three different sampling strategies:

1. Greedy (`temperature=0.01`, no top-k/top-p).
2. Temperature sampling (`temperature=0.8`).
3. Nucleus sampling (`top_p=0.9`).

**Questions**:

- Which produces the most diverse output?
- Which is most repetitive?
- How does temperature affect punctuation and spacing?

**Deliverable**: `exercises/sampling_comparison.md` with generated samples.

## Exercise 4.6: Resume from Checkpoint

Modify `lab/train.py` to support `--resume` from `outputs/last.pt`. Verify that restarting training continues the epoch/step counters and LR schedule correctly.

**Deliverable**: `exercises/train_with_resume.py` or a patch file.

## Lab Files

- [`data.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week04/lab/data.py)
- [`generate.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week04/lab/generate.py)
- [`model.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week04/lab/model.py)
- [`train.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week04/lab/train.py)
