#!/usr/bin/env python3
"""Verify that the bigram sweep produced complete evidence for all factors."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REQUIRED_FACTORS = {"5", "25", "100"}
REQUIRED_FIELDS = {
    "factor",
    "bigram_vocab_size",
    "rom_bigram",
    "rom_write",
    "val_loss",
    "train_time_ms",
    "step_avg_ms",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "run_id",
    "file",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()

    if not args.summary.exists():
        print(f"missing summary: {args.summary}", file=sys.stderr)
        return 1

    with args.summary.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("summary contains no rows", file=sys.stderr)
        return 1

    missing_fields = REQUIRED_FIELDS - set(rows[0])
    if missing_fields:
        print(f"summary missing columns: {sorted(missing_fields)}", file=sys.stderr)
        return 1

    by_factor = {row["factor"]: row for row in rows if row.get("factor") in REQUIRED_FACTORS}
    missing_factors = REQUIRED_FACTORS - set(by_factor)
    if missing_factors:
        print(f"summary missing factors: {sorted(missing_factors)}", file=sys.stderr)
        return 1

    for factor, row in sorted(by_factor.items(), key=lambda item: int(item[0])):
        empty = [field for field in REQUIRED_FIELDS if not row.get(field)]
        if empty:
            print(f"factor {factor} has empty fields: {empty}", file=sys.stderr)
            return 1
        try:
            loss = float(row["val_loss"])
            train_time_ms = float(row["train_time_ms"])
            step_avg_ms = float(row["step_avg_ms"])
            allocated = int(row["peak_allocated_mib"])
            reserved = int(row["peak_reserved_mib"])
        except ValueError as exc:
            print(f"factor {factor} has invalid numeric field: {exc}", file=sys.stderr)
            return 1
        if not (0 < loss < 20 and train_time_ms > 0 and step_avg_ms > 0 and allocated > 0 and reserved > 0):
            print(f"factor {factor} has implausible values: {row}", file=sys.stderr)
            return 1

    print("complete: factors 5, 25, and 100 have final loss/time/memory evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
