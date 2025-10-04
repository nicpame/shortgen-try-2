#!/usr/bin/env python3
"""Convert a nocodb-style JSON (root is object with numeric string keys)
into a JSON array where each value becomes an element of the root array.

Usage examples:
  # write output to infile.array.json (default)
  python scripts/convert_nocodb_to_array.py data/db/nocodb.json

  # overwrite original (makes a timestamped backup)
  python scripts/convert_nocodb_to_array.py data/db/nocodb.json --inplace

  # include the original numeric id into each record under the `id` key
  python scripts/convert_nocodb_to_array.py data/db/nocodb.json --add-id
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime


def transform_root_obj_to_array(data: object, add_id: bool = False, sort_keys: bool = True):
    if not isinstance(data, dict):
        raise ValueError("input JSON root is not an object/dict")

    # detect numeric-string keys
    keys = list(data.keys())
    if not keys:
        return []

    if all(isinstance(k, str) and k.isdigit() for k in keys):
        ordered_keys = sorted(keys, key=lambda k: int(k)) if sort_keys else keys
        out = []
        for k in ordered_keys:
            v = data[k]
            if add_id and isinstance(v, dict):
                # copy to avoid mutating original
                v = dict(v)
                v["id"] = k
            out.append(v)
        return out

    raise ValueError("input JSON root does not appear to be numeric-keyed strings")


def _backup_file(path: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    bak = f"{path}.bak.{ts}"
    os.replace(path, bak)
    return bak


def main(argv=None):
    p = argparse.ArgumentParser(description="Convert numeric-keyed root JSON object into an array")
    p.add_argument("infile", help="path to input JSON file")
    p.add_argument("-o", "--outfile", help="path to write output JSON (default: infile + .array.json)")
    p.add_argument("--inplace", action="store_true", help="overwrite the input file (makes a timestamped backup)")
    p.add_argument("--add-id", action="store_true", help="add the original numeric key into each record under 'id'")
    p.add_argument("--no-sort", dest="sort", action="store_false", help="do not sort numeric keys; preserve insertion order")
    args = p.parse_args(argv)

    infile = args.infile
    if not os.path.exists(infile):
        print(f"Error: infile does not exist: {infile}", file=sys.stderr)
        return 2

    with open(infile, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    try:
        out = transform_root_obj_to_array(data, add_id=args.add_id, sort_keys=args.sort)
    except Exception as e:
        print(f"Error transforming JSON: {e}", file=sys.stderr)
        return 3

    if args.inplace:
        # make a backup and write to infile
        bak = _backup_file(infile)
        outpath = infile
        print(f"Backed up original to: {bak}")
    else:
        outpath = args.outfile or infile + ".array.json"

    with open(outpath, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print(f"Wrote {len(out)} top-level entries to: {outpath}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
