#!/usr/bin/env python3
"""Estimate bigram embedding memory for modded-nanogpt factors."""

from __future__ import annotations

import argparse


def gib(num_bytes: float) -> float:
    return num_bytes / 1024**3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=int, default=50304)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--factors", type=int, nargs="+", default=[5, 25, 100])
    parser.add_argument(
        "--param-bytes",
        type=int,
        default=2,
        help="BF16 parameter bytes",
    )
    parser.add_argument(
        "--state-multiplier",
        type=float,
        default=3.0,
        help="Approx param+grad+optimizer multiplier. Use 1 for parameters only.",
    )
    args = parser.parse_args()

    print("factor,rows,param_gib,param_plus_grad_opt_gib")
    for factor in args.factors:
        rows = factor * args.vocab
        param_bytes = rows * args.dim * args.param_bytes
        total_bytes = param_bytes * args.state_multiplier
        print(f"{factor},{rows},{gib(param_bytes):.3f},{gib(total_bytes):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
