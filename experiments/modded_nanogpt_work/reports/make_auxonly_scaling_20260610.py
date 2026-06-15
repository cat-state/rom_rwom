#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "tmp_remote_auxonly_20260610"
OUT_DIR = ROOT / "reports" / "bf_sota_scaling_20260609"


@dataclass
class AuxRun:
    label: str
    bf: int
    seed: int | None
    path: Path
    vals: list[tuple[int, float]]
    table_params: int
    table_rows: int
    store_dim: int
    completed: bool
    final_val: float


def parse_run(path: Path) -> AuxRun | None:
    name = path.name
    if ".console." in name:
        return None
    if "auxonlylayerrowsigns" not in name:
        return None

    bf_match = re.search(r"\bbf(\d+)_", name)
    if not bf_match:
        return None
    bf = int(bf_match.group(1))
    seed_match = re.search(r"_seed(\d+)_", name)
    seed = int(seed_match.group(1)) if seed_match else None

    label = "aux-only"
    if "aux025" in name:
        label = "aux-only aux0.25"
    if "layerheadmixdelta" in name:
        label = "aux-only + layer-head-mix-delta"
    if "attnreslayergain" in name:
        label = "aux-only + attnres layer gain"
    if "headmixfreeze" in name:
        label = "aux-only + frozen head mix"
    if "headmixinit00" in name:
        label = "aux-only headmix init 0,0"

    vals: list[tuple[int, float]] = []
    table_rows: int | None = None
    table_params: int | None = None
    store_dim: int | None = None

    for line in path.read_text(errors="replace").splitlines():
        m = re.search(r"step:(\d+)/(\d+) val_loss:([0-9.]+)", line)
        if m:
            vals.append((int(m.group(1)), float(m.group(3))))
            row_metric = re.search(r"\bengram_table_rows:([0-9.eE+-]+)", line)
            numel_metric = re.search(r"\bengram_table_numel:([0-9.eE+-]+)", line)
            if row_metric:
                table_rows = int(float(row_metric.group(1)))
            if numel_metric:
                table_params = int(float(numel_metric.group(1)))
        m = re.search(r"Engram memory shape: rows=(\d+) store_dim=(\d+) embedding_shape=\((\d+),\s*(\d+)\)", line)
        if m:
            table_rows = int(m.group(1))
            store_dim = int(m.group(2))
            table_params = int(m.group(3)) * int(m.group(4))

    if not vals or table_rows is None or table_params is None:
        return None
    if store_dim is None:
        store_dim = table_params // table_rows
    final_step, final_val = vals[-1]
    return AuxRun(
        label=label,
        bf=bf,
        seed=seed,
        path=path,
        vals=vals,
        table_params=table_params,
        table_rows=table_rows,
        store_dim=store_dim,
        completed=final_step == 1500,
        final_val=final_val,
    )


def collect() -> list[AuxRun]:
    runs = []
    for path in sorted(LOG_DIR.glob("*.txt")):
        run = parse_run(path)
        if run is not None:
            runs.append(run)
    runs.sort(key=lambda r: (r.label, r.bf, r.seed if r.seed is not None else -1, r.path.name))
    return runs


def write_csv(runs: list[AuxRun]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "summary_auxonly_20260610.csv"
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "bf",
                "seed",
                "completed",
                "last_step",
                "last_val",
                "final_val_if_complete",
                "table_rows",
                "store_dim",
                "table_params",
                "log",
            ]
        )
        for run in runs:
            last_step, last_val = run.vals[-1]
            writer.writerow(
                [
                    run.label,
                    run.bf,
                    "" if run.seed is None else run.seed,
                    int(run.completed),
                    last_step,
                    f"{last_val:.6f}",
                    f"{run.final_val:.6f}" if run.completed else "",
                    run.table_rows,
                    run.store_dim,
                    run.table_params,
                    run.path.name,
                ]
            )
    return out


def plot(runs: list[AuxRun]) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    complete = [r for r in runs if r.completed]
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    markers = {
        "aux-only": "o",
        "aux-only aux0.25": "s",
        "aux-only + layer-head-mix-delta": "D",
        "aux-only + attnres layer gain": "X",
        "aux-only + frozen head mix": "v",
        "aux-only headmix init 0,0": "P",
    }
    for label in sorted({r.label for r in complete}):
        group = [r for r in complete if r.label == label]
        if not group:
            continue
        ax.scatter(
            [r.table_params for r in group],
            [r.final_val for r in group],
            s=70,
            marker=markers.get(label, "o"),
            label=label,
        )
        for r in group:
            ax.annotate(
                f"BF{r.bf}s{r.seed}",
                (r.table_params, r.final_val),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
    ax.axhline(3.2411, color="#444", linewidth=1.2, linestyle=":", label="archived best 3.2411")
    ax.set_xscale("log")
    ax.set_xlabel("Engram table parameters")
    ax.set_ylabel("Final validation loss @ 1500")
    ax.set_title("Aux-only row-sign Engram variants vs table parameter count")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    scatter_path = OUT_DIR / "auxonly_variants_vs_params_20260610.png"
    fig.savefig(scatter_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    key_runs = [
        r
        for r in runs
        if r.label in {"aux-only", "aux-only + layer-head-mix-delta", "aux-only headmix init 0,0"}
        and (r.completed or r.vals[-1][0] >= 500)
    ]
    for r in sorted(key_runs, key=lambda x: (x.bf, x.seed or -1, x.label)):
        linestyle = "-" if r.completed else "--"
        ax.plot(
            [step for step, _ in r.vals],
            [val for _, val in r.vals],
            marker="o",
            linewidth=1.7,
            linestyle=linestyle,
            label=f"{r.label} BF{r.bf}s{r.seed}{'' if r.completed else ' partial'}",
        )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation loss")
    ax.set_title("Aux-only row-sign validation curves")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    curves_path = OUT_DIR / "auxonly_variants_curves_20260610.png"
    fig.savefig(curves_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return scatter_path, curves_path


def main() -> None:
    runs = collect()
    if not runs:
        raise SystemExit(f"no aux-only runs found in {LOG_DIR}")
    csv_path = write_csv(runs)
    scatter_path, curves_path = plot(runs)
    completed = sum(1 for r in runs if r.completed)
    print(f"wrote {csv_path} ({len(runs)} runs, {completed} complete)")
    print(f"wrote {scatter_path}")
    print(f"wrote {curves_path}")
    best = min((r for r in runs if r.completed), key=lambda r: r.final_val)
    print(f"best complete: {best.final_val:.4f} {best.path.name}")


if __name__ == "__main__":
    main()
