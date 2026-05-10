"""Build a stratified sample of pi-bench scenarios for a fast eval.

Selects up to N scenarios trying to cover every leaderboard column at
least once and every (column, label) bucket where possible. Copies them
into a flat target directory under domain subfolders so the pi-bench
loader can resolve them.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


def _domain_subdir(scenario_path: Path) -> str:
    return scenario_path.parent.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    # Group scenarios by (column, label)
    by_bucket: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in sorted(args.src.glob("*/scen_*.json")):
        try:
            j = json.loads(path.read_text())
        except Exception:
            continue
        col = j.get("leaderboard", {}).get("primary", "")
        label = j.get("label", "")
        by_bucket[(col, label)].append(path)

    import random
    random.seed(args.seed)
    for paths in by_bucket.values():
        random.shuffle(paths)

    selected: list[Path] = []
    used: set[Path] = set()

    # Pass 1: one per column (first available bucket for that column)
    by_column: dict[str, list[Path]] = defaultdict(list)
    for (col, _label), paths in by_bucket.items():
        by_column[col].extend(paths)
    for col, paths in sorted(by_column.items()):
        for p in paths:
            if p not in used:
                selected.append(p)
                used.add(p)
                break
            if len(selected) >= args.n:
                break

    # Pass 2: round-robin across (column,label) buckets to fill out N
    bucket_keys = sorted(by_bucket.keys())
    cursor = {k: 0 for k in bucket_keys}
    while len(selected) < args.n:
        progressed = False
        for key in bucket_keys:
            paths = by_bucket[key]
            while cursor[key] < len(paths) and paths[cursor[key]] in used:
                cursor[key] += 1
            if cursor[key] < len(paths):
                p = paths[cursor[key]]
                cursor[key] += 1
                selected.append(p)
                used.add(p)
                progressed = True
                if len(selected) >= args.n:
                    break
        if not progressed:
            break

    args.dst.mkdir(parents=True, exist_ok=True)
    for sub in {p.parent.name for p in selected}:
        (args.dst / sub).mkdir(parents=True, exist_ok=True)
    for sub in args.dst.iterdir():
        if sub.is_dir():
            for old in sub.glob("*.json"):
                old.unlink()

    for path in selected:
        target = args.dst / _domain_subdir(path) / path.name
        shutil.copy2(path, target)

    print(f"Selected {len(selected)} scenarios across {len({_domain_subdir(p) for p in selected})} domains")
    by_col = defaultdict(int)
    for path in selected:
        j = json.loads(path.read_text())
        by_col[j.get("leaderboard", {}).get("primary", "?")] += 1
    for col, n in sorted(by_col.items()):
        print(f"  {col}: {n}")


if __name__ == "__main__":
    main()
