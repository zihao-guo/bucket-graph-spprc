#!/usr/bin/env python3
"""Compute the shifted-geometric-mean tables published in the README.

Three tables, each printed as github-flavored markdown so the output can be
pasted directly into `benchmarks/README.md`:

  runtime          bgspprc.csv per (set, ng, mode); shift=1s, TL→120s.
  pathwyse         comparison_pathwyse.csv per (set, ng); shift=1s, paired
                   rows only (TL on either side dropped); ratio = shifted
                   means; #bg_eq counts |bg_cost−pw_cost| ≤ 1e-3.
  rcspp            comparison_rcspp.csv per ng; shift=10s, TL→120s; ratio =
                   shifted means against the paper's `paper_all_s` column.

Usage:
    python3 benchmarks/compute_means.py [runtime|pathwyse|rcspp|all]

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


def runtime_table(csv_path: Path, shift: float = 1.0, tl: float = 120.0) -> str:
    """bgspprc-only wall-clock per (set, ng, mode). Empty cost = timeout
    substituted with `tl` seconds before the sgm."""
    rows = list(csv.DictReader(open(csv_path)))
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        s = SET_DISPLAY.get(r["set"], r["set"])
        grouped[(s, r["ng"], r["mode"])].append(r)

    lines = [
        "| set       | ng | mode             | sgm (s) | max (s) | #to |",
        "|-----------|---:|------------------|--------:|--------:|----:|",
    ]
    for s in SET_ORDER:
        for ng in NG_ORDER:
            for m in MODE_ORDER:
                rs = grouped.get((s, ng, m), [])
                if not rs:
                    continue
                ts = [tl if r["cost"] == "" else float(r["time_s"]) for r in rs]
                to = sum(1 for r in rs if r["cost"] == "")
                sgm = shifted_geomean(ts, shift)
                lines.append(
                    f"| {s:<9} | {int(ng):>2} | {m:<16} | "
                    f"{sgm:>7.3f} | {max(ts):>7.3f} | {to:>3} |"
                )
    return "\n".join(lines)


def pathwyse_table(
    cmp_path: Path, bg_path: Path, shift: float = 1.0, eq_tol: float = 1e-3
) -> str:
    """bgspprc para_bidir vs Pathwyse, per (set, ng). Paired rows only —
    TL on either side dropped from the sgm. Ratio computed on shifted
    means as (bg_sgm + shift) / (pw_sgm + shift)."""
    cmp_rows = list(csv.DictReader(open(cmp_path)))
    bg_set = {
        (r["instance"], r["ng"]): r["set"]
        for r in csv.DictReader(open(bg_path))
        if r["mode"] == "para_bidir"
    }

    def eq(a: str, b: str) -> bool:
        return a != "" and b != "" and abs(float(a) - float(b)) <= eq_tol

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in cmp_rows:
        s = bg_set.get((r["instance"], r["ng"]))
        if s in {"spprclib", "roberti"}:
            grouped[(s, r["ng"])].append(r)

    lines = [
        "| set      | ng | bgspprc sgm (s) | pathwyse sgm (s) |"
        " ratio | #bg_eq | n  | n_total |",
        "|----------|---:|----------------:|-----------------:|"
        "------:|-------:|---:|--------:|",
    ]
    for s in ["spprclib", "roberti"]:
        for ng in NG_ORDER:
            rs = grouped.get((s, ng), [])
            if not rs:
                continue
            pairs = [
                (float(r["bgspprc_s"]), float(r["pathwyse_s"]))
                for r in rs
                if r["bgspprc_s"] and r["pathwyse_s"]
            ]
            if not pairs:
                continue
            bgs = [p[0] for p in pairs]
            pps = [p[1] for p in pairs]
            beq = sum(1 for r in rs if eq(r["bgspprc_cost"], r["pathwyse_cost"]))
            bg_sgm = shifted_geomean(bgs, shift)
            pp_sgm = shifted_geomean(pps, shift)
            ratio = (bg_sgm + shift) / (pp_sgm + shift)
            lines.append(
                f"| {s:<8} | {int(ng):>2} | {bg_sgm:>15.3f} | "
                f"{pp_sgm:>16.3f} | {ratio:>5.3f} | {beq:>6} | "
                f"{len(pairs):>2} | {len(rs):>7} |"
            )
    return "\n".join(lines)


def rcspp_table(cmp_path: Path, shift: float = 10.0, tl: float = 120.0) -> str:
    """bgspprc vs Petersen & Spoorendonk 2025 paper runtimes per ng. Both
    sides share the 120 s budget — `TL` rows substituted with `tl` before
    the sgm."""
    rows = list(csv.DictReader(open(cmp_path)))

    def to_s(v: str) -> float:
        return tl if v == "TL" else float(v)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["ng"]].append(r)

    lines = [
        "| ng | bgspprc sgm (s) | paper sgm (s) | ratio | n  |",
        "|---:|----------------:|--------------:|------:|---:|",
    ]
    for ng in sorted(grouped, key=int):
        rs = grouped[ng]
        bgs = [to_s(r["bgspprc_s"]) for r in rs]
        pps = [to_s(r["paper_all_s"]) for r in rs]
        bg_sgm = shifted_geomean(bgs, shift)
        pp_sgm = shifted_geomean(pps, shift)
        ratio = (bg_sgm + shift) / (pp_sgm + shift)
        lines.append(
            f"| {int(ng):>2} | {bg_sgm:>15.3f} | {pp_sgm:>13.3f} | "
            f"{ratio:>5.3f} | {len(rs):>2} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument(
        "table",
        nargs="?",
        default="all",
        choices=["runtime", "pathwyse", "rcspp", "all"],
    )
    ap.add_argument("--bg", default=str(ROOT / "bgspprc.csv"))
    ap.add_argument("--cmp-pw", default=str(ROOT / "comparison_pathwyse.csv"))
    ap.add_argument("--cmp-rcspp", default=str(ROOT / "comparison_rcspp.csv"))
    args = ap.parse_args()

    if args.table in ("runtime", "all"):
        if args.table == "all":
            print("### Runtime\n")
        print(runtime_table(Path(args.bg)))
        if args.table == "all":
            print()

    if args.table in ("pathwyse", "all"):
        if args.table == "all":
            print("### Pathwyse comparison\n")
        print(pathwyse_table(Path(args.cmp_pw), Path(args.bg)))
        if args.table == "all":
            print()

    if args.table in ("rcspp", "all"):
        if args.table == "all":
            print("### Paper comparison (rcspp)\n")
        print(rcspp_table(Path(args.cmp_rcspp)))

    return 0


if __name__ == "__main__":
    sys.exit(main())
