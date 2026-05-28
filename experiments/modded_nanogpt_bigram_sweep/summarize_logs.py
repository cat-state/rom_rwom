#!/usr/bin/env python3
"""Summarize modded-nanogpt bigram sweep logs."""

from __future__ import annotations

import re
import sys
from pathlib import Path


CONFIG_RE = re.compile(
    r"Experiment config: .*?bigram_vocab_size=(?P<vocab>\d+) run_id=(?P<run_id>\S+)"
)
ROM_RE = re.compile(r"Experiment config: .*?rom_bigram=(?P<rom_bigram>[01])")
ROM_WRITE_RE = re.compile(r"Experiment config: .*?rom_write=(?P<rom_write>[01])")
ENGRAM_RE = re.compile(r"Experiment config: .*?engram_bigram=(?P<engram_bigram>[01])")
ENGRAM_DIM_RE = re.compile(r"Experiment config: .*?engram_dim=(?P<engram_dim>\d+)")
ENGRAM_HEADS_RE = re.compile(r"Experiment config: .*?engram_heads=(?P<engram_heads>\d+)")
ENGRAM_MAX_NGRAM_RE = re.compile(r"Experiment config: .*?engram_max_ngram=(?P<engram_max_ngram>\d+)")
ENGRAM_PER_HEAD_RE = re.compile(r"Experiment config: .*?engram_per_head=(?P<engram_per_head>[01])")
ENGRAM_CANONICALIZE_RE = re.compile(r"Experiment config: .*?engram_canonicalize=(?P<engram_canonicalize>[01])")
ROM_LAYER_RE = re.compile(r"Experiment config: .*?rom_layer_only=(?P<rom_layer_only>-?\d+)")
VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+) val_loss:(?P<loss>[0-9.]+) "
    r"train_time:(?P<time_ms>[0-9.]+)ms step_avg:(?P<step_avg>[0-9.]+)ms"
)
MEM_RE = re.compile(
    r"peak memory allocated: (?P<allocated>\d+) MiB reserved: (?P<reserved>\d+) MiB"
)


def parse_log(path: Path) -> dict[str, str]:
    text = path.read_text(errors="replace")
    config = CONFIG_RE.search(text)
    rom = ROM_RE.search(text)
    rom_write = ROM_WRITE_RE.search(text)
    engram = ENGRAM_RE.search(text)
    engram_dim = ENGRAM_DIM_RE.search(text)
    engram_heads = ENGRAM_HEADS_RE.search(text)
    engram_max_ngram = ENGRAM_MAX_NGRAM_RE.search(text)
    engram_per_head = ENGRAM_PER_HEAD_RE.search(text)
    engram_canonicalize = ENGRAM_CANONICALIZE_RE.search(text)
    rom_layer = ROM_LAYER_RE.search(text)
    vals = list(VAL_RE.finditer(text))
    mem = MEM_RE.search(text)
    final_val = vals[-1] if vals else None

    vocab = config.group("vocab") if config else ""
    factor = str(int(vocab) // 50304) if vocab else ""
    return {
        "file": str(path),
        "run_id": config.group("run_id") if config else path.stem,
        "factor": factor,
        "bigram_vocab_size": vocab,
        "rom_bigram": rom.group("rom_bigram") if rom else "0",
        "rom_write": rom_write.group("rom_write") if rom_write else "0",
        "engram_bigram": engram.group("engram_bigram") if engram else "0",
        "engram_dim": engram_dim.group("engram_dim") if engram_dim else "",
        "engram_heads": engram_heads.group("engram_heads") if engram_heads else "",
        "engram_max_ngram": engram_max_ngram.group("engram_max_ngram") if engram_max_ngram else "",
        "engram_per_head": engram_per_head.group("engram_per_head") if engram_per_head else "0",
        "engram_canonicalize": engram_canonicalize.group("engram_canonicalize") if engram_canonicalize else "0",
        "rom_layer_only": rom_layer.group("rom_layer_only") if rom_layer else "",
        "final_step": final_val.group("step") if final_val else "",
        "total_steps": final_val.group("total") if final_val else "",
        "val_loss": final_val.group("loss") if final_val else "",
        "train_time_ms": final_val.group("time_ms") if final_val else "",
        "step_avg_ms": final_val.group("step_avg") if final_val else "",
        "peak_allocated_mib": mem.group("allocated") if mem else "",
        "peak_reserved_mib": mem.group("reserved") if mem else "",
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: summarize_logs.py LOG [LOG ...]", file=sys.stderr)
        return 2

    rows = [parse_log(Path(arg)) for arg in sys.argv[1:]]
    cols = [
        "factor",
        "bigram_vocab_size",
        "rom_bigram",
        "rom_write",
        "engram_bigram",
        "engram_dim",
        "engram_heads",
        "engram_max_ngram",
        "engram_per_head",
        "engram_canonicalize",
        "rom_layer_only",
        "val_loss",
        "train_time_ms",
        "step_avg_ms",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "run_id",
        "file",
    ]
    print(",".join(cols))
    for row in rows:
        print(",".join(row[col] for col in cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
