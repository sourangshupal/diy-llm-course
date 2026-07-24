#!/usr/bin/env python3
"""Verify every internal markdown link in content/**/*.md resolves to a real
file. External (http/https) and mailto links are skipped.
"""
import re
import sys
from pathlib import Path

CONTENT = Path(__file__).resolve().parents[1] / "content"
LINK_RE = re.compile(r"\]\(([^)]+)\)")


def is_external(target: str) -> bool:
    return target.startswith(("http://", "https://", "mailto:", "#"))


def main() -> int:
    broken = []
    for md_file in sorted(CONTENT.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        for match in LINK_RE.finditer(text):
            target = match.group(1).split(" ", 1)[0]  # strip optional "title"
            if is_external(target):
                continue
            target_path = target.split("#", 1)[0]
            if not target_path:
                continue
            resolved = (md_file.parent / target_path).resolve()
            if not resolved.exists():
                broken.append((md_file.relative_to(CONTENT), target))

    if broken:
        print(f"Found {len(broken)} broken internal link(s):")
        for src, target in broken:
            print(f"  {src}: {target}")
        return 1

    print("All internal links resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
