#!/usr/bin/env python3
"""
Rename mp4 files in a directory using 11 characters immediately before the second-last underscore.

Behavior:
  - No command-line arguments. Configure the folder and commit behavior via the variables below.
  - For each .mp4 file, find the second-last '_' position in the stem and take the 11 characters
    immediately before that underscore as the new basename. If fewer than 11 characters are
    available, take as many as exist (but skip if none).
  - Avoid overwriting by appending _1, _2... when needed.
  - By default runs as a dry-run; set COMMIT = True to perform renames.
"""

from pathlib import Path

# === Configuration ===
# Absolute path to the folder containing the mp4 files to rename
FOLDER = Path("/workspaces/shortgen-try-2/data/source_vids/already_dled_vids")
# When False the script will only print planned renames. Set to True to actually rename files.
COMMIT = True
# Number of characters to take before the second-last underscore
NUM_CHARS = 11


def plan_renames(folder: Path):
    plans = []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() != ".mp4":
            continue
        name_no_ext = p.stem
        # find underscore positions
        underscores = [i for i, ch in enumerate(name_no_ext) if ch == '_']
        if len(underscores) < 2:
            plans.append((p, None, f"skipped: fewer than 2 underscores ({len(underscores)})"))
            continue
        second_last_pos = underscores[-2]
        start = max(0, second_last_pos - NUM_CHARS)
        new_base = name_no_ext[start:second_last_pos]
        if not new_base:
            plans.append((p, None, "skipped: no chars before second-last underscore"))
            continue
        # sanitize new_base: remove leading/trailing underscores or spaces
        new_base = new_base.strip('_ ').strip()
        if not new_base:
            plans.append((p, None, "skipped: resulting new base is empty after strip"))
            continue

        target = folder / (new_base + p.suffix)
        # avoid overwriting existing files: create a unique name if needed
        if target.exists():
            i = 1
            while True:
                candidate = folder / f"{new_base}_{i}{p.suffix}"
                if not candidate.exists():
                    target = candidate
                    break
                i += 1
        plans.append((p, target, "ok"))
    return plans


def apply_renames(plans, do_commit: bool):
    renamed = 0
    skipped = 0
    for src, dst, status in plans:
        if dst is None:
            print(f"SKIP: {src.name} -> {status}")
            skipped += 1
            continue
        print(f"RENAME: {src.name} -> {dst.name}")
        if do_commit:
            try:
                src.rename(dst)
                renamed += 1
            except Exception as e:
                print(f"ERROR renaming {src} -> {dst}: {e}")
    return renamed, skipped


def main():
    folder = FOLDER
    if not folder.exists() or not folder.is_dir():
        print(f"Folder does not exist or is not a directory: {folder}")
        return

    plans = plan_renames(folder)
    if not plans:
        print("No mp4 files found.")
        return

    print(f"Planned actions for folder: {folder}")
    for src, dst, status in plans:
        if dst is None:
            print(f"  SKIP: {src.name} -> {status}")
        else:
            print(f"  {src.name}  ->  {dst.name}")

    if COMMIT:
        renamed, skipped = apply_renames(plans, do_commit=True)
        print(f"Done. Renamed: {renamed}. Skipped: {skipped}.")
    else:
        print("Dry-run only. Set COMMIT = True in the script to apply the renames.")


if __name__ == '__main__':
    main()
