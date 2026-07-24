"""Week 9 Lab 1: Language identification.

Uses a simple heuristic (character-set detection) for demonstration.
For production, replace with fastText lid.176 or a Hugging Face language-id model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def detect_language_simple(text: str) -> str:
    """Very naive language detector for teaching purposes.

    Uses character ranges. Not suitable for production.
    """
    text = text.strip()
    if not text:
        return "empty"

    # Count characters from common scripts
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    cyrillic = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
    arabic = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    total = len(text)

    if cjk / total > 0.1:
        return "zh"
    if cyrillic / total > 0.1:
        return "ru"
    if arabic / total > 0.1:
        return "ar"
    return "en"


def build_sample_corpus(path: Path) -> None:
    """Create a small synthetic multilingual corpus."""
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = [
        {"text": "The quick brown fox jumps over the lazy dog.", "expected": "en"},
        {"text": "快速的棕色狐狸跳过了懒狗。", "expected": "zh"},
        {"text": "Быстрая коричневая лисица прыгает через ленивую собаку.", "expected": "ru"},
        {"text": "الثعلب البني السريع يقفز فوق الكلب الكسول.", "expected": "ar"},
        {"text": "asdf qwer zxcv 1234 !!!", "expected": "en"},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Language identification demo")
    parser.add_argument("--input", type=Path, default=Path("week09/data/raw.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("week09/data/en_only.jsonl"))
    parser.add_argument("--target_lang", type=str, default="en")
    args = parser.parse_args()

    if not args.input.exists():
        build_sample_corpus(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    total = 0
    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            record = json.loads(line)
            text = record["text"]
            lang = detect_language_simple(text)
            print(f"[{lang}] {text[:60]}...")
            if lang == args.target_lang:
                record["detected_lang"] = lang
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

    print(f"\nKept {kept}/{total} documents in language '{args.target_lang}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
