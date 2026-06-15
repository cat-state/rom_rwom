#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "tmp_remote_auxonly_20260610"
OUT_DIR = ROOT / "reports" / "bf_sota_scaling_20260609"


@dataclass
class Run:
    family: str
    bf: int
    seed: int | None
    path: Path
    vals: list[tuple[int, float]]
    table_rows: int
    store_dim: int
    table_params: int
    completed: bool
    last_step: int
    last_val: float


def family_for(name: str) -> str | None:
    if "slotreadout_slotmix_dimsigns_balanced" in name and "hashseed1" in name and "attnresextra5to8" in name and "scale1p25to0" in name:
        return "count-sketch slot-mix + hashseed1 + AttnRes extra 1.25 decay"
    if "slotreadout_slotmix_dimsigns_balanced" in name and "attnresextra5to8" in name and "scale0to1p25_s500_500" in name:
        return "count-sketch slot-mix + late AttnRes extra 1.25"
    if "slotreadout_slotmix_dimsigns_balanced" in name and "aux025" in name and "attnresextra5to8" in name and "scale1p25to0" in name:
        return "count-sketch slot-mix aux0.25 + AttnRes extra 1.25 decay"
    if "slotreadout_slotmix_dimsigns_balanced" in name and "attnresextra5to8" in name and "scale1p5to0" in name:
        return "count-sketch slot-mix + AttnRes extra 1.5 decay"
    if "slotreadout_slotmix_dimsigns_balanced" in name and "attnresextra5to8" in name and "scale1p25to0" in name:
        return "count-sketch slot-mix + AttnRes extra 1.25 decay"
    if "slotreadout_slotmix_dimsigns_balanced" in name and "attnresextra5to8" in name and "scale1p0to0" in name:
        return "count-sketch slot-mix + AttnRes extra 1.0 decay"
    if "sketchk2_combinemix_bounded01_dimsigns_balanced" in name and "hashseed1" in name:
        if "aux05to0" in name:
            return "count-sketch bounded combine-mix + hash seed 1 aux decay"
        if "basehist" in name:
            return "count-sketch bounded combine-mix + hash seed 1 base hit-hist"
        if "layerdelta_scale025" in name:
            return "count-sketch bounded combine-mix + hash seed 1 + layerdelta 0.25"
        if "layerdelta_0to1" in name:
            return "count-sketch bounded combine-mix + hash seed 1 + layerdelta ramp"
        if "layerdelta" in name:
            return "count-sketch bounded combine-mix + hash seed 1 + layerdelta"
        return "count-sketch bounded combine-mix + hash seed 1"
    if "sketchk3_combinemix_bounded01_dimsigns_balanced" in name:
        if "basehist" in name:
            return "count-sketch k3 bounded combine-mix base hit-hist"
        return "count-sketch k3 bounded combine-mix"
    if "sketchk2_combinemix_bounded01_dimsigns_balanced" in name:
        if "basehist" in name and "layerdelta_learnscale" in name:
            return "count-sketch bounded combine-mix base hit-hist + learned layerdelta scale"
        if "basehist" in name and "layerdelta_scale025" in name:
            return "count-sketch bounded combine-mix base hit-hist + layerdelta 0.25"
        if "basehist" in name and "layerdelta_0to025" in name:
            return "count-sketch bounded combine-mix base hit-hist + layerdelta ramp 0.25"
        if "basehist" in name and "layerdelta_0to05" in name:
            return "count-sketch bounded combine-mix base hit-hist + layerdelta ramp 0.5"
        if "basehist" in name and "layerdelta_0to1" in name:
            return "count-sketch bounded combine-mix base hit-hist + layerdelta ramp"
        if "layerdelta_scale025" in name:
            return "count-sketch bounded combine-mix + layerdelta 0.25"
        if "layerdelta_0to1" in name:
            return "count-sketch bounded combine-mix + layerdelta ramp"
        if "basehist" in name:
            return "count-sketch bounded combine-mix base hit-hist"
        if "layerdelta" in name:
            return "count-sketch bounded combine-mix + layerdelta"
        return "count-sketch bounded combine-mix"
    if "sketchk2_combinemix_bounded02_dimsigns_balanced" in name and "hashseed1" in name:
        return "count-sketch bounded0.2 combine-mix + hash seed 1"
    if "sketchk2_combinemix_softmax_dimsigns_balanced" in name and "hashseed1" in name:
        if "aux025" in name:
            return "count-sketch softmax combine-mix + hash seed 1 aux0.25"
        return "count-sketch softmax combine-mix + hash seed 1"
    if "attnresextra5to8_biasm2_scale0to1p25_s500_500" in name:
        return "late AttnRes extra 1.25"
    if "avalanchehash_hashseed1" in name:
        return "avalanche hash + hash seed 1"
    if "sota_k2_headmix_layerdelta_norowsigns" in name and "basehist" in name:
        return "no-row-sign control + base hit-hist"
    if "sota_k2_headmix_layerdelta_norowsigns_hashseed0" in name:
        return "no-row-sign control seed sweep"
    if "partgrp2_hashseed1" in name:
        return "two partition groups + hash seed 1"
    if "partgrp2" in name:
        return "two layer partition groups"
    if "layerhash0" in name:
        return "shared relative layer hash"
    hash_seed_match = re.search(r"hashseed(\d+)", name)
    if hash_seed_match:
        return f"hash seed {hash_seed_match.group(1)}"
    if "avalanchehash" in name:
        return "avalanche hash"
    if "ngramread115_085" in name:
        return "ngram read-scale 1.15/0.85"
    if "rowadagrad" in name:
        return "row-wise Adagrad"
    if "rowrmsfloor" in name:
        return "cold-row RMS floor"
    if "batchfreqnorm" in name:
        return "batch-frequency norm"
    if "ifal_sched" in name:
        return "scheduled inverse FAL"
    if "ifal" in name:
        return "inverse FAL scalar Adam"
    if "latentfsq_aux010to0_s0_250" in name:
        return "latent FSQ aux 0.1 to zero by 250"
    if "latentfsq_aux010to0" in name:
        return "latent FSQ aux 0.1 to zero"
    if "latentaux" in name and "s01to0" in name:
        return "latent aux 0.1 to zero"
    if "latentaux" in name and "s003" in name:
        return "latent aux 0.03"
    if "latentaux" in name and "s01" in name:
        return "latent aux 0.1"
    if "latentaux" in name and "s05" in name:
        return "latent aux 0.5"
    if "latentfsq_mixngram_nopart" in name:
        return "latent FSQ addressing"
    if "latentbsq24_mixngram_nopart" in name:
        return "latent BSQ24 addressing"
    if "k2_headmix_nopart_control" in name:
        return "no-partition control"
    if "hotsplitv9_full" in name:
        return "hot-split full train-only"
    if "hotsplitv8_full" in name:
        return "hot-split full detach decay"
    if "hotsplitv7_full" in name:
        return "hot-split full dedup aux decay"
    if "hotsplitv6_full" in name:
        return "hot-split full dedup aux"
    if "hotsplitv5_full" in name:
        return "hot-split full auxslot1"
    if "hotsplitv4_full" in name and "detachaux" in name:
        return "hot-split full detach-aux"
    if "hotsplitv3_full" in name:
        return "hot-split full manual coalesce"
    if "hotsplitv2_full" in name and "gradhookstart500" in name:
        return "hot-split full grad hook start500"
    if "hotsplitv2_full" in name and "gradhookstart501" in name:
        return "hot-split full grad hook start501"
    if "hotsplitv2_full" in name and "gradhook" in name:
        return "hot-split full grad hook"
    if "hotsplitv2_full" in name:
        return "hot-split full superpose"
    if "hotsplit_valueonly" in name and "layerdelta" in name:
        return "hot-split value-only + layer delta"
    if "hotsplit_valueonly" in name:
        return "hot-split value-only"
    if "slotreadout_slotmix_aux05to01" in name and "norowsigns" in name:
        return "slot-mix aux decay"
    if "slotreadout_slotmix_aux05to0" in name and "norowsigns" in name:
        return "slot-mix aux to zero"
    if (
        "sketchk2_slotreadout_slotmix" in name
        and "norowsigns" in name
        and "layerreadouts" not in name
        and "dimsigns" not in name
    ):
        return "slot-mix"
    if "slotreadout_slotmix_layerreadouts_base_aux05_norowsigns" in name:
        return "slot-mix + layer readouts"
    if "slotattention_layerreadouts_base_aux05_norowsigns" in name:
        return "slot-attn + layer readouts"
    if "slotattention_base_aux05_norowsigns" in name:
        return "slot-attn"
    if "readoutdelta_sketchk2_dimsigns_balanced" in name:
        return "count-sketch sum"
    if "slotreadout_slotmix_dimsigns_balanced" in name:
        return "count-sketch slot-mix"
    if "attnresextra5to8_biasm4" in name:
        return "AttnRes extra 5->8 bias -4"
    if "attnresextra5to8_biasm2_singlefix_scale1to0" in name:
        return "AttnRes extra 5->8 bias -2 single decay"
    if "attnresextra5to8_biasm2_singlefix_scale2to0" in name:
        return "AttnRes extra 5->8 bias -2 scale2 decay"
    if "attnresextra5to8_biasm2_singlefix_scale1p5to0" in name:
        return "AttnRes extra 5->8 bias -2 scale1.5 decay"
    if "attnresextra5to8_biasm2_singlefix_scale1p25to0" in name:
        return "AttnRes extra 5->8 bias -2 scale1.25 decay"
    if "attnresextra5to8_biasm2_singlefix" in name:
        return "AttnRes extra 5->8 bias -2 single"
    if "attnresextra5to8_biasm2" in name:
        return "AttnRes extra 5->8 bias -2"
    if "attnresextra5to8_biasm1p5" in name:
        return "AttnRes extra 5->8 bias -1.5"
    if "attnresextra5to8_biasm1" in name:
        return "AttnRes extra 5->8 bias -1"
    if "attnresextra5to8" in name:
        return "AttnRes extra 5->8"
    if "attnresextra2to8" in name:
        return "AttnRes extra 2->8"
    if "hitdropv1_p010_min32" in name:
        return "hot-drop 10% >=32"
    if "hitdrop10_min1024" in name:
        return "hot-drop 10% >=1024"
    if "hitdrop10_min256" in name:
        return "hot-drop 10% >=256"
    if "ngramrows_trigramheavy_0p5_1p5_readhit025" in name:
        return "SOTA ngramrows replay"
    if "layerheaddelta_norowsigns" in name:
        return "per-layer head-mix delta"
    if "layerheadmix_norowsigns" in name:
        return "per-layer head mix"
    if "control_currentcode" in name:
        return "current-code control"
    if "headmix_norowsigns_aux05_currentcode" in name:
        return "aux0.5 current-code control"
    if "headmix_norowsigns_outdrop005to0" in name:
        return "output dropout 0.05 to 0"
    if "headmix_norowsigns_outdrop010to0" in name:
        return "output dropout 0.10 to 0"
    if "noheadmix_norowsigns_aux05" in name:
        return "aux0.5 no head mix"
    if "headmix_norowsigns_aux035" in name:
        if "aux035to05" in name:
            return "aux0.35 to 0.5 schedule"
        return "aux0.35 control"
    if "headmix_norowsigns_aux025" in name:
        return "aux0.25 control"
    if "headmix_norowsigns_aux065" in name:
        return "aux0.65 control"
    if "noheadmix_norowsigns" in name:
        return "no head mix"
    if "superposek2_base_aux05to025" in name and "norowsigns" in name:
        return "K2 aux decay"
    if "norowsigns_control" in name:
        return "no-row-sign control"
    if "norowsigns_seed" in name and "superposek2" in name:
        return "no-row-sign control"
    return None


def parse_run(path: Path) -> Run | None:
    name = path.name
    if ".console." in name or ".nohup." in name or not name.endswith(".txt"):
        return None
    family = family_for(name)
    if family is None:
        return None
    bf_match = re.search(r"\bbf(\d+)_", name)
    if not bf_match:
        return None
    seed_match = re.search(r"_seed(\d+)_", name)
    bf = int(bf_match.group(1))
    seed = int(seed_match.group(1)) if seed_match else None

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
    last_step, last_val = vals[-1]
    return Run(
        family=family,
        bf=bf,
        seed=seed,
        path=path,
        vals=vals,
        table_rows=table_rows,
        store_dim=store_dim,
        table_params=table_params,
        completed=last_step == 1500,
        last_step=last_step,
        last_val=last_val,
    )


def collect() -> list[Run]:
    runs = []
    for path in sorted(LOG_DIR.glob("*.txt")):
        run = parse_run(path)
        if run is not None:
            runs.append(run)
    return sorted(runs, key=lambda r: (r.family, r.bf, r.seed or -1, r.path.name))


def write_csv(runs: list[Run]) -> Path:
    out = OUT_DIR / "summary_structural_20260610.csv"
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "family",
                "bf",
                "seed",
                "completed",
                "last_step",
                "last_val",
                "table_rows",
                "store_dim",
                "table_params",
                "log",
            ]
        )
        for r in runs:
            writer.writerow(
                [
                    r.family,
                    r.bf,
                    "" if r.seed is None else r.seed,
                    int(r.completed),
                    r.last_step,
                    f"{r.last_val:.6f}",
                    r.table_rows,
                    r.store_dim,
                    r.table_params,
                    r.path.name,
                ]
            )
    return out


def plot(runs: list[Run]) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    families = sorted({r.family for r in runs})
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]
    marker_for = {family: markers[i % len(markers)] for i, family in enumerate(families)}

    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    for family in families:
        group = [r for r in runs if r.family == family]
        complete = [r for r in group if r.completed]
        partial = [r for r in group if not r.completed]
        if complete:
            ax.scatter(
                [r.table_params for r in complete],
                [r.last_val for r in complete],
                s=72,
                marker=marker_for[family],
                label=family,
            )
        if partial:
            ax.scatter(
                [r.table_params for r in partial],
                [r.last_val for r in partial],
                s=72,
                marker=marker_for[family],
                facecolors="none",
                linewidths=1.7,
                label=f"{family} partial",
            )
        for r in group:
            suffix = "" if r.completed else f" @{r.last_step}"
            ax.annotate(
                f"BF{r.bf}s{r.seed}{suffix}",
                (r.table_params, r.last_val),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
            )
    ax.axhline(3.2411, color="#444", linewidth=1.1, linestyle=":", label="archived best 3.2411")
    ax.axhline(3.2440, color="#777", linewidth=1.0, linestyle="--", label="June best 3.2440")
    ax.set_xscale("log")
    ax.set_xlabel("Engram table parameters")
    ax.set_ylabel("Validation loss (final if complete; latest if partial)")
    ax.set_title("Structural Engram variants vs parameter count")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    scatter = OUT_DIR / "structural_variants_vs_params_20260610.png"
    fig.savefig(scatter, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    for r in sorted(runs, key=lambda x: (x.bf, x.family, x.seed or -1)):
        if r.last_step < 250:
            continue
        linestyle = "-" if r.completed else "--"
        ax.plot(
            [s for s, _ in r.vals],
            [v for _, v in r.vals],
            marker="o",
            linewidth=1.5,
            linestyle=linestyle,
            label=f"{r.family} BF{r.bf}s{r.seed}{'' if r.completed else ' partial'}",
        )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation loss")
    ax.set_title("Structural Engram validation curves")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    fig.tight_layout()
    curves = OUT_DIR / "structural_variants_curves_20260610.png"
    fig.savefig(curves, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return scatter, curves


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = collect()
    if not runs:
        raise SystemExit(f"no structural runs found in {LOG_DIR}")
    csv_path = write_csv(runs)
    scatter, curves = plot(runs)
    completed = sum(r.completed for r in runs)
    print(f"wrote {csv_path} ({len(runs)} runs, {completed} complete)")
    print(f"wrote {scatter}")
    print(f"wrote {curves}")
    best_complete = [r for r in runs if r.completed]
    if best_complete:
        best = min(best_complete, key=lambda r: r.last_val)
        print(f"best complete: {best.last_val:.4f} {best.path.name}")
    best_latest = min(runs, key=lambda r: r.last_val)
    print(f"best latest: {best_latest.last_val:.4f} step {best_latest.last_step} {best_latest.path.name}")


if __name__ == "__main__":
    main()
