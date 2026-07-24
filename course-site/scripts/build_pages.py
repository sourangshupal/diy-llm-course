#!/usr/bin/env python3
"""One-time migration: teaching/weekNN/* -> course-site/content/weekNN/*.

Not part of the mkdocs build pipeline - run by hand whenever teaching/
content changes and needs re-syncing into the site.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEACHING = REPO_ROOT / "teaching"
CONTENT = REPO_ROOT / "course-site" / "content"
GITHUB_BLOB = "https://github.com/datawhalechina/diy-llm/blob/main"

# Exact-match replacements: every teaching/* line that references the
# still-untranslated docs/en/chapterNN/*.md book chapters (Chinese content
# despite the "en" path) gets swapped for an honest plain-English pointer
# instead of a link into Chinese text or a broken path.
LINK_FIXES: dict[str, list[tuple[str, str]]] = {
    "syllabus.md": [
        (
            "- **Diy-LLM docs**: `docs/en/` in this repository",
            "- **Diy-LLM docs**: the course book under `docs/` in this repository (English translation in progress)",
        ),
    ],
    "week01/README.md": [
        (
            "- `docs/en/前言.md` — Preface",
            "- Preface chapter of the course book (English translation pending)",
        ),
        (
            "- `docs/en/chapter1/wandb使用介绍.md` — W&B usage (if available in English)",
            "- W&B usage chapter of the course book (English translation pending)",
        ),
    ],
    "week01/instructor_notes.md": [
        (
            "3. Read [`docs/en/前言.md`](../../docs/en/前言.md) (Preface) and one W&B documentation page on artifacts.",
            "3. Read the Preface chapter of the course book (English translation pending) and one W&B documentation page on artifacts.",
        ),
        (
            "- Suggested prep: review string operations in Python and read `docs/en/chapter2/` if available.",
            "- Suggested prep: review string operations in Python and read the Tokenizer chapter of the course book (English translation pending).",
        ),
    ],
    "week02/README.md": [
        (
            "- `docs/en/chapter2/chapter2_分词器.md` — Tokenizer chapter",
            "- Tokenizer chapter of the course book (English translation pending)",
        ),
    ],
    "week02/instructor_notes.md": [
        (
            "- `docs/en/chapter3/chapter3_模型架构.md` — Transformer architecture overview.",
            "- Transformer architecture overview chapter of the course book (English translation pending).",
        ),
    ],
    "week03/README.md": [
        (
            "- `docs/en/chapter3/chapter3_pytorch与资源核算.md`",
            "- PyTorch & Resource Accounting chapter of the course book (English translation pending)",
        ),
        (
            "- `docs/en/chapter4/chapter4_第四章语言模型架构和训练的技术细节.md`",
            "- Language Model Architecture & Training chapter of the course book (English translation pending)",
        ),
    ],
    "week03/instructor_notes.md": [
        (
            "   - `docs/en/chapter3/chapter3_pytorch与资源核算.md`",
            "   - PyTorch & Resource Accounting chapter of the course book (English translation pending)",
        ),
        (
            "   - `docs/en/chapter4/chapter4_第四章语言模型架构和训练的技术细节.md`",
            "   - Language Model Architecture & Training chapter of the course book (English translation pending)",
        ),
    ],
    "week04/README.md": [
        (
            "- `docs/en/chapter4/chapter4_第四章语言模型架构和训练的技术细节.md`",
            "- Language Model Architecture & Training chapter of the course book (English translation pending)",
        ),
    ],
    "week05/README.md": [
        (
            "- `docs/en/chapter5/chapter5_混合专家模型.md` (MoE overview)",
            "- MoE overview chapter of the course book (English translation pending)",
        ),
        (
            "- `docs/en/chapter6/chapter6_第六章GPU和GPU相关的优化.md`",
            "- GPU Optimization chapter of the course book (English translation pending)",
        ),
    ],
    "week05/instructor_notes.md": [
        (
            "   Skim `docs/en/chapter6/chapter6_第六章GPU和GPU相关的优化.md` and note one optimization technique you want to understand better.",
            "   Skim the GPU Optimization chapter of the course book (English translation pending) and note one optimization technique you want to understand better.",
        ),
    ],
    "week06/README.md": [
        (
            "- `docs/en/chapter7/chapter7_第七章GPU高性能编程.md`",
            "- GPU High-Performance Programming chapter of the course book (English translation pending)",
        ),
    ],
    "week07/README.md": [
        (
            "- `docs/en/chapter8/chapter8_第八章分布式训练.md`",
            "- Distributed Training chapter of the course book (English translation pending)",
        ),
    ],
    "week07/instructor_notes.md": [
        (
            "- Read `docs/en/chapter8/chapter8_第八章分布式训练.md` and identify one additional detail not covered in lecture.",
            "- Read the Distributed Training chapter of the course book (English translation pending) and identify one additional detail not covered in lecture.",
        ),
        (
            "- Read `docs/en/chapter8/chapter8_第八章分布式训练.md` in full.",
            "- Read the Distributed Training chapter of the course book (English translation pending) in full.",
        ),
    ],
    "week08/README.md": [
        (
            "- `docs/en/chapter9/chapter9_Scaling_Laws.md`",
            "- Scaling Laws chapter of the course book (English translation pending)",
        ),
    ],
    "week09/README.md": [
        (
            "- `docs/en/chapter11/chapter11_数据工程.md`",
            "- Data Engineering chapter of the course book (English translation pending)",
        ),
    ],
    "week09/instructor_notes.md": [
        (
            "4. Read the course documentation chapter `docs/en/chapter11/chapter11_数据工程.md` and note one concept not covered in class.",
            "4. Read the Data Engineering chapter of the course book (English translation pending) and note one concept not covered in class.",
        ),
    ],
    "week10/README.md": [
        (
            "- `docs/en/chapter13/chapter13_第十三章大模型的基本训练流程.md`",
            "- Large Model Training Pipeline chapter of the course book (English translation pending)",
        ),
    ],
    "week11/README.md": [
        (
            "- `docs/en/chapter14/chapter14_可验证奖励的强化学习.md`",
            "- RLVR (Reinforcement Learning with Verifiable Rewards) chapter of the course book (English translation pending)",
        ),
    ],
    "week11/instructor_notes.md": [
        (
            "- `docs/en/chapter14/chapter14_可验证奖励的强化学习.md` for the course's own framing.",
            "- The RLVR chapter of the course book (English translation pending) for the course's own framing.",
        ),
    ],
    "week12/README.md": [
        (
            "- `docs/en/chapter10/推理.md` (inference)",
            "- Inference chapter of the course book (English translation pending)",
        ),
        (
            "- `docs/en/chapter12/chapter12_评估与基准测试.md`",
            "- Evaluation & Benchmarks chapter of the course book (English translation pending)",
        ),
    ],
    "week12/instructor_notes.md": [
        (
            "1. **Read the docs.** Review `docs/en/chapter12/chapter12_评估与基准测试.md` and note three benchmarks not covered in class.",
            "1. **Read the docs.** Review the Evaluation & Benchmarks chapter of the course book (English translation pending) and note three benchmarks not covered in class.",
        ),
    ],
}

WEEK_TITLES = {
    1: "Course Introduction & Experiment Tracking",
    2: "Tokenization",
    3: "Transformer Architecture I - Building Blocks",
    4: "Transformer Architecture II - Full Model & Training Loop",
    5: "GPU Architecture & Optimization",
    6: "High-Performance Kernels with Triton",
    7: "Distributed Training",
    8: "Scaling Laws",
    9: "Data Engineering",
    10: "Alignment - SFT & Expert Iteration",
    11: "Alignment - GRPO & Reinforcement Learning with Verifiable Rewards",
    12: "Evaluation Frameworks",
}


LAB_LINK_RE = re.compile(r"\]\(\./lab/([^)]+\.py)\)")
EXERCISES_LINK_RE = re.compile(r"\]\(\./exercises/README\.md\)")


def rewrite_relative_links(text: str, week: str) -> str:
    """README.md source files link to ./lab/*.py and ./exercises/README.md,
    valid in the original teaching/weekNN/ layout but not in the site (lab
    scripts aren't rendered as pages, and exercises live at a sibling
    exercises.md page). Rewrite both to resolve correctly in the built site.
    """
    text = LAB_LINK_RE.sub(
        lambda m: f"]({GITHUB_BLOB}/teaching/{week}/lab/{m.group(1)})", text
    )
    text = EXERCISES_LINK_RE.sub("](exercises.md)", text)
    return text


def apply_link_fixes(text: str, key: str) -> tuple[str, int]:
    count = 0
    for old, new in LINK_FIXES.get(key, []):
        if old not in text:
            raise ValueError(f"LINK_FIXES entry for {key!r} not found verbatim:\n  {old!r}")
        text = text.replace(old, new)
        count += 1
    return text, count


def copy_with_fix(src: Path, dst: Path, key: str, week: str | None = None) -> int:
    text = src.read_text(encoding="utf-8")
    text, count = apply_link_fixes(text, key)
    if week is not None:
        text = rewrite_relative_links(text, week)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    return count


def build_exercises_page(week_num: int, week_dir: Path, dst: Path) -> int:
    src = week_dir / "exercises" / "README.md"
    key = f"week{week_num:02d}/exercises/README.md"
    text = src.read_text(encoding="utf-8")
    text, count = apply_link_fixes(text, key)

    lab_dir = week_dir / "lab"
    lab_files = sorted(
        p for p in lab_dir.glob("*.py") if p.is_file()
    ) if lab_dir.is_dir() else []

    if lab_files:
        lines = ["", "## Lab Files", ""]
        for p in lab_files:
            rel = p.relative_to(REPO_ROOT).as_posix()
            url = f"{GITHUB_BLOB}/{rel}"
            lines.append(f"- [`{p.name}`]({url})")
        text = text.rstrip("\n") + "\n" + "\n".join(lines) + "\n"

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    return count


def main() -> None:
    total_fixes = 0
    changed_files: list[str] = []

    for name, rel_src in (
        ("index.md", "README.md"),
        ("syllabus.md", "syllabus.md"),
        ("timetable.md", "timetable.md"),
    ):
        src = TEACHING / rel_src
        dst = CONTENT / name
        count = copy_with_fix(src, dst, rel_src)
        if count:
            total_fixes += count
            changed_files.append(f"{rel_src} ({count} fix{'es' if count != 1 else ''})")
        print(f"wrote {dst.relative_to(REPO_ROOT)}  <- {src.relative_to(REPO_ROOT)}  [{count} link fix(es)]")

    for week_num in range(1, 13):
        wk = f"week{week_num:02d}"
        week_dir = TEACHING / wk
        content_week_dir = CONTENT / wk

        for src_name, dst_name in (
            ("README.md", "index.md"),
            ("instructor_notes.md", "lecture.md"),
        ):
            src = week_dir / src_name
            dst = content_week_dir / dst_name
            key = f"{wk}/{src_name}"
            week_for_links = wk if src_name == "README.md" else None
            count = copy_with_fix(src, dst, key, week=week_for_links)
            if count:
                total_fixes += count
                changed_files.append(f"{key} ({count} fix{'es' if count != 1 else ''})")
            print(f"wrote {dst.relative_to(REPO_ROOT)}  <- {src.relative_to(REPO_ROOT)}  [{count} link fix(es)]")

        dst = content_week_dir / "exercises.md"
        count = build_exercises_page(week_num, week_dir, dst)
        if count:
            total_fixes += count
            changed_files.append(f"{wk}/exercises/README.md ({count} fix{'es' if count != 1 else ''})")
        print(f"wrote {dst.relative_to(REPO_ROOT)}  <- {(week_dir / 'exercises/README.md').relative_to(REPO_ROOT)}  [{count} link fix(es), lab files appended]")

    print()
    print(f"Total link fixes applied: {total_fixes} across {len(changed_files)} file(s):")
    for f in changed_files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
