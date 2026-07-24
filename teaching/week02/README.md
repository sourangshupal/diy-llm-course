# Week 2: Tokenization

## Learning Objectives

By the end of this week, students should be able to:

- Explain why tokenization is necessary and what trade-offs it involves.
- Implement Byte-Pair Encoding (BPE) from scratch.
- Train a BPE tokenizer on a small corpus using the Hugging Face `tokenizers` library.
- Encode and decode text, inspect merges, and analyze vocabulary.

## Pre-Reading (Optional)

- `docs/en/chapter2/chapter2_分词器.md` — Tokenizer chapter

## Lab Files

Run these in order:

1. [`lab/bpe_from_scratch.py`](./lab/bpe_from_scratch.py) — implement BPE by hand on a tiny corpus.
2. [`lab/train_hf_tokenizer.py`](./lab/train_hf_tokenizer.py) — train a real BPE tokenizer with `tokenizers`.
3. [`lab/inspect_tokenizer.py`](./lab/inspect_tokenizer.py) — analyze the learned vocabulary and merges.

## Exercises

See [`exercises/README.md`](./exercises/README.md).

## Deliverable

A trained BPE tokenizer saved to `week02/exercises/my_tokenizer/` that can:

- Encode English text.
- Decode token IDs back to text.
- Have a vocabulary size between 1,000 and 10,000.

Submit the tokenizer directory plus a short analysis (≤1 page) of:

- Vocabulary size chosen.
- 3 interesting merges learned.
- Number of tokens required for a sample sentence.

## Key Concepts

| Concept | Description |
|---------|-------------|
| Token | The atomic unit processed by a language model. |
| Vocabulary | The set of all tokens the tokenizer knows. |
| BPE | Greedy merge algorithm that starts from characters/bytes and repeatedly merges the most frequent adjacent pair. |
| Pre-tokenizer | Splits raw text into "words" before BPE (e.g., whitespace + punctuation). |
| Special tokens | `<|endoftext|>`, `<|pad|>`, `<|unk|>` used for control and padding. |
| Encoding | Text → list of token IDs. |
| Decoding | List of token IDs → text (may not be perfectly reversible). |

## Common Pitfalls

- Forgetting to add special tokens before training.
- Pre-tokenizer mismatches between training and inference.
- BPE is greedy and not globally optimal.
- Tokenization is language-dependent; English-heavy vocabularies produce long sequences for other languages.
