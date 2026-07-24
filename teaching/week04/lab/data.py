"""Week 4 Lab: tiny dataset utilities for the mini language model.

For teaching purposes we use a simple character-level corpus so training is
fast and requires no external downloads. Students can swap in their Week 2
tokenizer and real text data for the exercises.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import torch
from torch.utils.data import Dataset


TINY_CORPUS = """The quick brown fox jumps over the lazy dog.
A journey of a thousand miles begins with a single step.
To be or not to be, that is the question.
All that glitters is not gold.
The only way to do great work is to love what you do.
"""


def build_corpus_file(path: Path, repeats: int = 100) -> Path:
    """Write the tiny corpus to disk, repeated for a larger dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(repeats):
            f.write(TINY_CORPUS)
    return path


class CharTokenizer:
    """Simple character-level tokenizer for fast prototyping."""

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for i, c in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[c] for c in text if c in self.char_to_id]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char[i] for i in ids)


class CharDataset(Dataset):
    """Character-level dataset that returns (input, target) slices."""

    def __init__(self, text: str, seq_len: int) -> None:
        self.tokenizer = CharTokenizer(text)
        self.seq_len = seq_len
        self.ids = self.tokenizer.encode(text)

    def __len__(self) -> int:
        return max(0, len(self.ids) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.ids[idx : idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def main() -> int:
    """Build a corpus and inspect a sample batch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = Path(tmpdir) / "corpus.txt"
        build_corpus_file(corpus_path, repeats=10)

        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()

        dataset = CharDataset(text, seq_len=32)
        print(f"Corpus length: {len(text)} characters")
        print(f"Vocab size: {dataset.tokenizer.vocab_size}")
        print(f"Number of samples: {len(dataset)}")

        x, y = dataset[0]
        print(f"Input IDs:  {x.tolist()}")
        print(f"Target IDs: {y.tolist()}")
        print(f"Input text:  {dataset.tokenizer.decode(x.tolist())!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
