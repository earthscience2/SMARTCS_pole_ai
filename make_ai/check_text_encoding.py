#!/usr/bin/env python3
"""
Detect common mojibake patterns in text files.

Usage:
  python make_ai/check_text_encoding.py
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import os


ROOT = Path(__file__).resolve().parents[1]
TEXT_EXTS = {
    ".py",
    ".pyw",
    ".sh",
    ".bash",
    ".md",
    ".txt",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".csv",
    ".tsv",
}

# Typical mojibake markers seen from broken UTF-8/Korean conversions.
SUSPECT_RE = re.compile(r"[\uFFFD\uF900-\uFAFF]")


SKIP_DIRS = {".git", "__pycache__", "venv", "venv_wsl2", ".venv"}


def iter_text_files(base: Path):
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            p = Path(root) / name
            if p.suffix.lower() in TEXT_EXTS or p.name in {".gitattributes", ".editorconfig"}:
                yield p


def main() -> int:
    issues = []
    for path in iter_text_files(ROOT):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            issues.append((path, 0, "not valid UTF-8"))
            continue

        for idx, line in enumerate(text.splitlines(), start=1):
            if SUSPECT_RE.search(line):
                issues.append((path, idx, line.strip()))

    if not issues:
        print("OK: no mojibake markers found.")
        return 0

    print("Found possible encoding issues:")
    for path, line_no, snippet in issues:
        rel = path.relative_to(ROOT)
        if line_no == 0:
            print(f"  - {rel}: not valid UTF-8")
        else:
            print(f"  - {rel}:{line_no}: {snippet[:120]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
