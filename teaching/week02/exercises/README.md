# Week 2 Exercises

## Exercise 2.1: Trace BPE by Hand

Using the output of `lab/bpe_from_scratch.py`, write down the first 5 merges for the corpus:

```text
low lower lowest
new newer newest
wide wider widest
```

**Deliverable**: `exercises/bpe_trace.md` with the merge sequence and a brief explanation of why each pair was chosen.

## Exercise 2.2: Train Your Own Tokenizer

Run `lab/train_hf_tokenizer.py`. Then modify the script in `exercises/train_my_tokenizer.py` to:

1. Use a larger corpus (e.g., download a public domain book from Project Gutenberg or use `datasets` to load `tiny_shakespeare`).
2. Try at least two different vocabulary sizes (e.g., 1,000 and 8,000).
3. Save both tokenizers under `exercises/tokenizer_1k/` and `exercises/tokenizer_8k/`.

**Deliverable**: Two trained tokenizers + a short comparison of average tokens per sentence.

## Exercise 2.3: Multilingual Tokenization

Use `lab/inspect_tokenizer.py` (or write `exercises/multilingual_check.py`) to tokenize the same sentence in three languages, e.g.:

- English: "Hello world"
- Chinese: "你好世界"
- French: "Bonjour le monde"

Count tokens per language.

**Questions**:

1. Which language uses the most tokens? Why?
2. What would happen if you trained the tokenizer on a balanced multilingual corpus?

**Deliverable**: `exercises/multilingual_check.py` + 3–4 sentence analysis.

## Exercise 2.4: Special Tokens

Write `exercises/special_tokens.py` that:

1. Loads a trained tokenizer.
2. Adds a new special token `<|mask|>`.
3. Encodes a sentence containing `<|mask|>` and verifies the token is recognized.
4. Decodes the IDs and verifies the special token reappears.

**Hint**: Use `tokenizer.add_special_tokens(["<|mask|>"])` and `tokenizer.save()`.

**Deliverable**: `exercises/special_tokens.py` + output showing the mask token round-trips.

## Exercise 2.5: Tokenizer Gotchas

Find or invent three strings that are **not** perfectly round-tripped by your tokenizer (encode → decode gives a slightly different string). Explain why each happens.

**Deliverable**: `exercises/gotchas.md` with examples and explanations.
