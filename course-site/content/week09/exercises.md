# Week 9 Exercises

## Exercise 9.1: Real Language ID

Replace the naive `detect_language_simple` in `lab/01_language_id.py` with a real language-ID model. Options:

- fastText `lid.176` (download from https://fasttext.cc/docs/en/language-identification.html).
- Hugging Face `papluca/xlm-roberta-base-language-detection`.

**Deliverable**: `exercises/real_language_id.py` + accuracy on the sample corpus.

## Exercise 9.2: Design Quality Rules

Create `exercises/quality_rules.md` listing 5 rule-based filters you would apply to English web text. For each rule, explain the intended failure mode.

**Deliverable**: `exercises/quality_rules.md`.

## Exercise 9.3: Perplexity Filter Calibration

Collect two small corpora: one high-quality (e.g., Wikipedia) and one low-quality (e.g., randomly generated text). Train or use a small reference model to compute perplexity distributions. Choose a threshold that separates them.

**Deliverable**: `exercises/ppl_calibration.py` + histogram plot.

## Exercise 9.4: Tune Deduplication Threshold

Run `lab/03_dedup.py` with thresholds [0.5, 0.7, 0.8, 0.9, 0.95]. Plot number of duplicates removed vs. threshold. Choose a threshold and justify it.

**Deliverable**: `exercises/dedup_threshold_sweep.py` + plot.

## Exercise 9.5: End-to-End Pipeline

Write `exercises/full_pipeline.py` that chains language ID → quality filter → dedup on a JSONL file of your choice. Report document counts and bytes at each stage.

**Deliverable**: `exercises/full_pipeline.py` + summary report.

## Lab Files

- [`01_language_id.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week09/lab/01_language_id.py)
- [`02_quality_filter.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week09/lab/02_quality_filter.py)
- [`03_dedup.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week09/lab/03_dedup.py)
