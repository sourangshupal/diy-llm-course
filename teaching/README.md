# Diy-LLM Teaching Modules

This folder contains all instructor-facing and student-facing materials for teaching the **Diy-LLM** course (a hands-on, build-from-scratch LLM course based on Stanford CS336 Spring 2025).

## What's Inside

```text
teaching/
├── README.md                 # This file
├── pyproject.toml            # Shared Python dependencies (managed with uv)
├── uv.lock                   # Locked dependency graph
├── syllabus.md               # Full 12-week syllabus
├── week01/                   # One folder per week
│   ├── README.md             # Week-at-a-glance for students
│   ├── instructor_notes.md   # Detailed lecture + lab guide
│   ├── lab/                  # Runnable .py lab scripts
│   └── exercises/            # Practice problems and mini-assignments
├── week02/
│   └── ...
└── week12/
    └── ...
```

## Quick Start for Instructors

1. **Set up the environment** (already done in the repo root):
   ```bash
   uv sync
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   ```

2. **Read the syllabus**: [`syllabus.md`](./syllabus.md)

3. **Pick a week** and open `weekXX/instructor_notes.md`.

## Quick Start for Students

1. Activate the shared virtual environment.
2. Open the week's `README.md` to see learning objectives and deliverables.
3. Run the lab scripts in order.
4. Complete the exercises.

## Design Principles

- **One folder per week**: Each week is self-contained.
- **Markdown-first**: Lectures, notes, and exercises are plain Markdown for easy editing and version control.
- **Runnable Python labs**: Every concept is accompanied by a small, standalone `.py` script.
- **Latest dependencies**: Versions are resolved by `uv` from the current package index and pinned in `uv.lock`.
- **Incremental**: Later weeks reuse code from earlier weeks, mirroring how real LLM pipelines are built.

## Dependency Notes

Core packages (resolved June 2026):

| Package | Resolved Version |
|---------|------------------|
| torch | 2.12.1 |
| transformers | 5.12.1 |
| tokenizers | 0.22.2 |
| wandb | 0.28.0 |
| numpy | 2.5.0 |
| matplotlib | 3.11.0 |
| scipy | 1.18.0 |
| tqdm | 4.68.3 |

See `pyproject.toml` for the full dependency specification and `uv.lock` for exact pins.

## License

Same as the parent repository: CC BY-NC-SA 4.0.
