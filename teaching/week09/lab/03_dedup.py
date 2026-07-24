"""Week 9 Lab 3: MinHash near-duplicate detection.

Implements a minimal MinHash + LSH deduplication pipeline for teaching.
For production, use datasketch or a dedicated dedup library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable


def shingle(text: str, k: int = 5) -> set[str]:
    """Return a set of k-shingles (overlapping substrings)."""
    return {text[i : i + k] for i in range(max(0, len(text) - k + 1))}


def minhash_signature(shingles: set[str], num_perm: int = 128, seed: int = 42) -> list[int]:
    """Compute a MinHash signature for a set of shingles."""
    rng = random.Random(seed)
    signature = []
    for _ in range(num_perm):
        a = rng.randint(1, 2**32)
        b = rng.randint(0, 2**32)
        min_hash = min(
            ((hash_string(sh) * a + b) % (2**32)) for sh in shingles
        ) if shingles else 2**32
        signature.append(min_hash)
    return signature


def hash_string(s: str) -> int:
    """Return a deterministic integer hash for a string."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Compute Jaccard similarity of two sets."""
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union


def lsh_buckets(signature: list[int], bands: int = 16) -> list[str]:
    """Assign a signature to LSH buckets."""
    rows = len(signature) // bands
    buckets = []
    for b in range(bands):
        band = signature[b * rows : (b + 1) * rows]
        buckets.append(f"band_{b}:" + "_".join(str(x) for x in band))
    return buckets


def find_duplicates(
    records: Iterable[dict],
    k: int = 5,
    num_perm: int = 128,
    bands: int = 16,
    threshold: float = 0.8,
) -> set[int]:
    """Return indices of documents to remove as near-duplicates."""
    docs = list(records)
    buckets: dict[str, list[int]] = {}
    for idx, record in enumerate(docs):
        shingles = shingle(record["text"], k)
        sig = minhash_signature(shingles, num_perm)
        for bucket in lsh_buckets(sig, bands):
            buckets.setdefault(bucket, []).append(idx)

    removed: set[int] = set()
    for candidates in buckets.values():
        if len(candidates) < 2:
            continue
        for i in range(len(candidates)):
            if candidates[i] in removed:
                continue
            sh_i = shingle(docs[candidates[i]]["text"], k)
            for j in range(i + 1, len(candidates)):
                if candidates[j] in removed:
                    continue
                sh_j = shingle(docs[candidates[j]]["text"], k)
                if jaccard_similarity(sh_i, sh_j) >= threshold:
                    removed.add(candidates[j])

    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="MinHash deduplication demo")
    parser.add_argument("--input", type=Path, default=Path("week09/data/high_quality.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("week09/data/deduped.jsonl"))
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}. Run 01_language_id.py and 02_quality_filter.py first.")
        return 1

    with open(args.input, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    removed = find_duplicates(records, threshold=args.threshold)
    print(f"Found {len(removed)} near-duplicate documents to remove")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for idx, record in enumerate(records):
            if idx not in removed:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records) - len(removed)} deduplicated documents to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
