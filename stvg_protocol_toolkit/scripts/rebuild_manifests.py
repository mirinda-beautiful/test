#!/usr/bin/env python3
"""Rebuild FILE_MANIFEST.csv and SHA256SUMS.txt without cyclic self-hashes.

FILE_MANIFEST.csv:
  lists package files excluding itself and SHA256SUMS.txt

SHA256SUMS.txt:
  lists package files including final FILE_MANIFEST.csv, excluding itself
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv"}
SKIP_NAMES = {"FILE_MANIFEST.csv", "SHA256SUMS.txt"}


def iter_files() -> list[Path]:
    files: list[Path] = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        rel = p.relative_to(ROOT).as_posix()
        if p.name in SKIP_NAMES and rel in SKIP_NAMES:
            continue
        if rel in SKIP_NAMES:
            continue
        files.append(p)
    return files


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    files = iter_files()
    manifest_path = ROOT / "FILE_MANIFEST.csv"
    sums_path = ROOT / "SHA256SUMS.txt"

    # 1) FILE_MANIFEST without itself / SHA256SUMS
    lines = ["path,bytes,sha256\n"]
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        digest = sha256(p)
        lines.append(f"{rel},{p.stat().st_size},{digest}\n")
    manifest_path.write_bytes("".join(lines).encode("utf-8"))

    # 2) SHA256SUMS includes FILE_MANIFEST, excludes itself
    sum_lines = []
    for p in files + [manifest_path]:
        rel = p.relative_to(ROOT).as_posix()
        sum_lines.append(f"{sha256(p)}  {rel}\n")
    sums_path.write_bytes("".join(sum_lines).encode("utf-8"))

    print(f"FILE_MANIFEST entries: {len(files)}")
    print(f"SHA256SUMS entries: {len(sum_lines)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
