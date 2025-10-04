#!/usr/bin/env python3
"""
Scan data/source_vids/already_dled_vids for .mp4 files and add corresponding
YouTube Shorts links to data/candidate_source_vids copy.txt.

Usage:
  python3 scripts/add_already_dled_to_candidates.py [--dry-run] [--backup]

By default does a dry-run; use --no-dry-run to actually write changes.
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import List


WORK_DIR = Path(__file__).resolve().parents[1]
ALREADY_DIR = WORK_DIR / "data" / "source_vids" / "already_dled_vids"
CANDIDATE_FILE = WORK_DIR / "data" / "candidate_source_vids copy.txt"


def find_mp4_ids(src_dir: Path) -> List[str]:
    ids: List[str] = []
    if not src_dir.exists():
        return ids
    for p in sorted(src_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() != ".mp4":
            continue
        stem = p.stem.strip()
        if not stem:
            continue
        ids.append(stem)
    return ids


def read_existing_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f.readlines()]


def make_short_urls(ids: List[str]) -> List[str]:
    return [f"https://www.youtube.com/shorts/{i}" for i in ids]


def append_urls(file_path: Path, urls: List[str], backup: bool = True) -> None:
    if backup and file_path.exists():
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        bak = file_path.with_name(file_path.name + ".bak." + stamp)
        shutil.copy(file_path, bak)
        print(f"Backup written to: {bak}")

    with file_path.open("a", encoding="utf-8") as f:
        for u in urls:
            f.write(u + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="Apply changes (default is dry-run)")
    parser.add_argument("--backup", dest="backup", action="store_true", help="Create a timestamped backup before writing")
    args = parser.parse_args()

    ids = find_mp4_ids(ALREADY_DIR)
    if not ids:
        print(f"No mp4 files found in: {ALREADY_DIR}")
        return 0

    urls = make_short_urls(ids)

    existing = set(line.strip() for line in read_existing_lines(CANDIDATE_FILE) if line.strip())

    to_add = [u for u in urls if u not in existing]

    print(f"Found {len(ids)} mp4 file(s) in {ALREADY_DIR}")
    print(f"{len(to_add)} new URL(s) to add to {CANDIDATE_FILE}")

    if not to_add:
        return 0

    for u in to_add:
        print(u)

    if args.dry_run:
        print("Dry-run: no changes written. Rerun with --no-dry-run to append the URLs.")
        return 0

    # Write changes
    CANDIDATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    append_urls(CANDIDATE_FILE, to_add, backup=args.backup)
    print(f"Appended {len(to_add)} URLs to {CANDIDATE_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
