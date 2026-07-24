"""Week 9 Lab 2: Quality filtering.

Implements rule-based filters and a tiny perplexity-based filter using a
pre-trained tokenizer and a small reference model. For the demo, the perplexity
filter uses a random model; in practice, use a model trained on high-quality text.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
import torch.nn as nn


class TinyQualityModel(nn.Module):
    """Random tiny model used as a stand-in for a perplexity classifier."""

    def __init__(self, vocab_size: int = 256, d_model: int = 32) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.emb(x))


def char_tokenize(text: str) -> list[int]:
    """Simple character-level tokenization for the demo."""
    return [ord(c) % 256 for c in text]


def perplexity(text: str, model: TinyQualityModel, device: torch.device) -> float:
    """Compute a dummy perplexity score for the text."""
    ids = char_tokenize(text)
    if len(ids) < 2:
        return float("inf")
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    return math.exp(loss.item())


def rule_based_score(text: str) -> dict[str, float]:
    """Return several heuristic quality signals."""
    words = text.split()
    n_chars = max(1, len(text))
    n_words = max(1, len(words))

    # Fraction of alphabetic characters
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / n_chars

    # Repetition: max fraction of text covered by a single repeated substring
    repeats = re.findall(r"(.)\1{4,}", text)
    repeat_ratio = sum(len(r) + 4 for r in repeats) / n_chars

    # Average word length
    avg_word_len = sum(len(w) for w in words) / n_words

    return {
        "alpha_ratio": alpha_ratio,
        "repeat_ratio": repeat_ratio,
        "avg_word_len": avg_word_len,
    }


def is_high_quality(text: str, metrics: dict[str, float], ppl: float, ppl_threshold: float = 100.0) -> bool:
    """Return True if the document passes all filters."""
    if len(text) < 20:
        return False
    if metrics["alpha_ratio"] < 0.5:
        return False
    if metrics["repeat_ratio"] > 0.2:
        return False
    if metrics["avg_word_len"] > 15 or metrics["avg_word_len"] < 2:
        return False
    if ppl > ppl_threshold:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Quality filtering demo")
    parser.add_argument("--input", type=Path, default=Path("week09/data/en_only.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("week09/data/high_quality.jsonl"))
    parser.add_argument("--ppl_threshold", type=float, default=100.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyQualityModel().to(device).eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    total = 0
    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            record = json.loads(line)
            text = record["text"]
            metrics = rule_based_score(text)
            ppl = perplexity(text, model, device)
            if is_high_quality(text, metrics, ppl, args.ppl_threshold):
                record["quality_metrics"] = metrics
                record["perplexity"] = ppl
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
            else:
                print(f"REJECTED: {text[:60]}... | metrics={metrics}, ppl={ppl:.1f}")

    print(f"\nKept {kept}/{total} high-quality documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
