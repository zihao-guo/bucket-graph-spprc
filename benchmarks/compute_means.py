#!/usr/bin/env python3
"""Compute the benchmark tables published in the README.

Each row reports `sgm (s)` (shifted geometric mean,
`exp(mean(log(t + s))) − s`, with `s` = 1 s), `mean (s)` (arithmetic
mean), and `solved` (count of instances completed within the timeout).
Timeouts substitute as 120 s in both means (never dropped). Each table
is printed as github-flavored markdown so the output can be pasted
directly into `benchmarks/README.md`:

  pathwyse         comparison_pathwyse.csv per (set, ng) × {bgspprc,
                   pathwyse}; sgm shift = 1 s.
  rcspp            comparison_rcspp.csv per ng × {bgspprc, paper}; sgm
                   shift = 1 s.
  modes            bgspprc.csv per (set, ng, mode) — the 6-mode/SIMD
                   ablation, bgspprc-only.

Usage:
    python3 benchmarks/compute_means.py [pathwyse|rcspp|modes|all]

Defaults to `all`. The shifted geometric mean for a set of times t_i with
shift s is `exp(mean(log(t_i + s))) − s`.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SET_ORDER = ["spprclib", "roberti", "rcspp"]
NG_ORDER = ["8", "16", "24"]
MODE_ORDER = [
    "mono_base",
    "mono_vec",
    "bidir_base",
    "bidir_vec",
    "para_bidir_base",
    "para_bidir",
]

# Map bgspprc.csv `set` field to display name. rcspp is binned per-ng under
# directory names ng8/ng16/ng24.
SET_DISPLAY = {"ng8": "rcspp", "ng16": "rcspp", "ng24": "rcspp"}


def shifted_geomean(values: list[float], shift: float) -> float:
    if not values:
        return float("nan")
    return math.exp(sum(math.log(v + shift) for v in values) / len(values)) - shift


def arithmetic_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def modes_table(csv_path: Path, shift: float = 1.0, tl: float = 120.0) -> str:
    """bgspprc-only ablation across (set, ng, mode). Empty cost = timeout
    substituted with `tl` seconds in both the sgm and the arithmetic mean.
    `solved` is the count of instances that finished without TL."""
    rows = list(csv.DictReader(open(csv_path)))
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        s = SET_DISPLAY.get(r["set"], r["set"])
        grouped[(s, r["ng"], r["mode"])].append(r)

    lines = [
        "| set       | ng | mode             | sgm (s) | mean (s) | solved |",
        "|-----------|---:|------------------|--------:|---------:|-------:|",
    ]
    for s in SET_ORDER:
        for ng in NG_ORDER:
            for m in MODE_ORDER:
                rs = grouped.get((s, ng, m), [])
                if not rs:
                    continue
                ts = [tl if r["cost"] == "" else float(r["time_s"]) for r in rs]
                solved = sum(1 for r in rs if r["cost"] != "")
                lines.append(
                    f"| {s:<9} | {int(ng):>2} | {m:<16} | "
                    f"{shifted_geomean(ts, shift):>7.3f} | "
                    f"{arithmetic_mean(ts):>8.3f} | "
                    f"{solved:>3}/{len(rs):<3} |"
                )
    return "\n".join(lines)


def pathwyse_table(
    cmp_path: Path, bg_path: Path, shift: float = 1.0, tl: float = 120.0
) -> str:
    """bgspprc para_bidir vs patched Pathwyse on `.sppcc`/`.vrp`, in Plato
    style: one row per (set, ng, solver). Timeouts substitute as `tl`
    seconds (never dropped). Cost equality is implicit — both sides run
    pure-ng so optimal reduced cost matches modulo cost-scale rounding;
    inspect comparison_pathwyse.csv if exact agreement matters."""
    cmp_rows = list(csv.DictReader(open(cmp_path)))
    bg_set = {
        (r["instance"], r["ng"]): r["set"]
        for r in csv.DictReader(open(bg_path))
        if r["mode"] == "para_bidir"
    }

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in cmp_rows:
        s = bg_set.get((r["instance"], r["ng"]))
        if s in {"spprclib", "roberti"}:
            grouped[(s, r["ng"])].append(r)

    lines = [
        "| set      | ng | solver   | sgm (s) | mean (s) | solved |",
        "|----------|---:|----------|--------:|---------:|-------:|",
    ]
    for s in ["spprclib", "roberti"]:
        for ng in NG_ORDER:
            rs = grouped.get((s, ng), [])
            if not rs:
                continue
            for solver, col in (("bgspprc", "bgspprc_s"),
                                ("pathwyse", "pathwyse_s")):
                ts = [tl if r[col] == "" else float(r[col]) for r in rs]
                solved = sum(1 for r in rs if r[col] != "")
                lines.append(
                    f"| {s:<8} | {int(ng):>2} | {solver:<8} | "
                    f"{shifted_geomean(ts, shift):>7.3f} | "
                    f"{arithmetic_mean(ts):>8.3f} | "
                    f"{solved:>3}/{len(rs):<3} |"
                )
    return "\n".join(lines)


def rcspp_table(cmp_path: Path, shift: float = 1.0, tl: float = 120.0) -> str:
    """bgspprc para_bidir vs Petersen & Spoorendonk 2025 paper runtimes per
    ng, Plato-style. `TL` rows substitute as `tl` seconds (matches
    run_comparison.sh)."""
    rows = list(csv.DictReader(open(cmp_path)))

    def to_s(v: str) -> float:
        return tl if v == "TL" else float(v)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["ng"]].append(r)

    lines = [
        "| ng | solver   | sgm (s) | mean (s) | solved |",
        "|---:|----------|--------:|---------:|-------:|",
    ]
    for ng in sorted(grouped, key=int):
        rs = grouped[ng]
        for solver, col in (("bgspprc", "bgspprc_s"),
                            ("paper", "paper_all_s")):
            ts = [to_s(r[col]) for r in rs]
            solved = sum(1 for r in rs if r[col] != "TL")
            lines.append(
                f"| {int(ng):>2} | {solver:<8} | "
                f"{shifted_geomean(ts, shift):>7.3f} | "
                f"{arithmetic_mean(ts):>8.3f} | "
                f"{solved:>3}/{len(rs):<3} |"
            )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument(
        "table",
        nargs="?",
        default="all",
        choices=["pathwyse", "rcspp", "modes", "all"],
    )
    ap.add_argument("--bg", default=str(ROOT / "bgspprc.csv"))
    ap.add_argument("--cmp-pw", default=str(ROOT / "comparison_pathwyse.csv"))
    ap.add_argument("--cmp-rcspp", default=str(ROOT / "comparison_rcspp.csv"))
    args = ap.parse_args()

    if args.table in ("pathwyse", "all"):
        if args.table == "all":
            print("### Pathwyse comparison (sppcc + vrp)\n")
        print(pathwyse_table(Path(args.cmp_pw), Path(args.bg)))
        if args.table == "all":
            print()

    if args.table in ("rcspp", "all"):
        if args.table == "all":
            print("### Paper comparison (rcspp)\n")
        print(rcspp_table(Path(args.cmp_rcspp)))
        if args.table == "all":
            print()

    if args.table in ("modes", "all"):
        if args.table == "all":
            print("### bgspprc mode/SIMD ablation\n")
        print(modes_table(Path(args.bg)))

    return 0


if __name__ == "__main__":
    sys.exit(main())
