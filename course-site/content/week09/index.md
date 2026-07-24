# Week 9: Data Engineering

## Learning Objectives

By the end of this week, students should be able to:

- Build a pre-training data pipeline: language ID, quality filtering, deduplication.
- Implement a rule-based and a tiny neural perplexity-based quality classifier.
- Use MinHash for approximate near-duplicate detection.
- Evaluate the impact of filtering on downstream perplexity.

## Pre-Reading (Optional)

- Data Engineering chapter of the course book (optional — not yet available in English)

## Lab Files

Run these in order:

1. [`lab/01_language_id.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week09/lab/01_language_id.py) — identify document languages.
2. [`lab/02_quality_filter.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week09/lab/02_quality_filter.py) — rule-based and model-based quality filtering.
3. [`lab/03_dedup.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week09/lab/03_dedup.py) — MinHash near-duplicate detection.

## Exercises

See [`exercises/README.md`](exercises.md).

## Deliverable

A cleaned dataset plus a report containing:

- Fraction of documents removed at each stage.
- Example documents removed and why.
- Perplexity of a small model trained on raw vs. cleaned data.

## Key Concepts

| Concept | Description |
|---------|-------------|
| Language identification | Classify documents by language (e.g., fastText lid.176). |
| Quality filtering | Remove low-quality documents (gibberish, boilerplate, spam). |
| Perplexity filter | Train a small model on high-quality data; reject documents with high perplexity. |
| Deduplication | Remove exact or near-duplicate documents. |
| MinHash | Locality-sensitive hashing for estimating Jaccard similarity. |
| LSH | Locality-sensitive hashing to bucket similar documents. |

## Common Pitfalls

- Filtering too aggressively removes valuable data.
- Language ID models are slow on CPU; consider batching.
- Near-dedup has false positives; tune similarity threshold.
- Quality heuristics can introduce demographic or linguistic bias.

## Data Note

The labs use small synthetic JSONL files. For real use, substitute Common Crawl WET files or a sample from the Dolma / C4 / RefinedWeb datasets.
