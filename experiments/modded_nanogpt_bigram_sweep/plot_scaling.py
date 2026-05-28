#!/usr/bin/env python3
"""Plot observed bigram/ROM scaling from completed modded-nanogpt runs."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "summary.csv"
OUT_PNG = ROOT / "rom_scaling.png"
OUT_SVG = ROOT / "rom_scaling.svg"


def load_rows() -> list[dict[str, str]]:
    with SUMMARY.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> int:
    rows = load_rows()
    complete = [r for r in rows if r.get("val_loss") and r.get("peak_allocated_mib")]
    vector = [r for r in complete if r["rom_bigram"] == "0" and r.get("engram_bigram", "0") == "0"]
    rom_read = [r for r in complete if r["rom_bigram"] == "1" and r["rom_write"] == "0"]
    rom_write = [r for r in complete if r["rom_bigram"] == "1" and r["rom_write"] == "1"]

    # Two measured vector points define a capacity trendline. This is a memory
    # extrapolation only; factor 100 vector OOMed before producing a final loss.
    vec_x = np.array([as_float(r, "factor") for r in vector])
    vec_mem = np.array([as_float(r, "peak_allocated_mib") / 1024 for r in vector])
    mem_slope, mem_intercept = np.polyfit(vec_x, vec_mem, 1)
    x_fit = np.linspace(vec_x.min(), 100, 200)
    mem_fit = mem_slope * x_fit + mem_intercept
    predicted_vec_100 = mem_slope * 100 + mem_intercept

    vec_loss = np.array([as_float(r, "val_loss") for r in vector])
    loss_slope, loss_intercept = np.polyfit(np.log(vec_x), vec_loss, 1)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#202124",
            "xtick.color": "#3c4043",
            "ytick.color": "#3c4043",
        }
    )

    fig, (ax_loss, ax_mem) = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    fig.suptitle("Rememberance-of-Memories scaling check on SM120", fontsize=16, fontweight="bold")

    vector_color = "#1f77b4"
    read_color = "#2ca02c"
    write_color = "#d62728"
    fit_color = "#6c757d"
    cap_color = "#8c564b"

    def scatter(ax, subset, color, marker, label, y_key, y_scale=1.0):
        if not subset:
            return
        xs = [as_float(r, "factor") for r in subset]
        ys = [as_float(r, y_key) * y_scale for r in subset]
        ax.scatter(xs, ys, s=95, color=color, marker=marker, edgecolor="white", linewidth=1.2, label=label, zorder=3)

    # Validation-loss panel.
    x_loss_fit = np.linspace(vec_x.min(), vec_x.max(), 100)
    ax_loss.plot(
        x_loss_fit,
        loss_slope * np.log(x_loss_fit) + loss_intercept,
        color=fit_color,
        linewidth=1.5,
        linestyle="--",
        label=f"vector fit: L={loss_intercept:.3f}{loss_slope:+.3f} ln(f)",
    )
    scatter(ax_loss, vector, vector_color, "o", "vector bigram", "val_loss")
    scatter(ax_loss, rom_read, read_color, "s", "ROM read", "val_loss")
    scatter(ax_loss, rom_write, write_color, "D", "ROM write", "val_loss")
    ax_loss.set_xscale("log")
    ax_loss.set_xticks([5, 25, 100], labels=["5x", "25x", "100x"])
    ax_loss.set_xlabel("bigram table factor")
    ax_loss.set_ylabel("final validation loss")
    ax_loss.set_title("Loss is flat across tested scale")
    ax_loss.grid(True, axis="y", alpha=0.22)
    ax_loss.set_ylim(3.275, 3.305)
    ax_loss.annotate(
        "factor 100 vector table OOMed;\nROM reaches same loss scale",
        xy=(100, as_float(rom_write[0], "val_loss")),
        xytext=(39, 3.299),
        arrowprops=dict(arrowstyle="->", color="#5f6368", lw=1.2),
        fontsize=9,
        color="#3c4043",
    )
    ax_loss.legend(frameon=False, fontsize=9, loc="lower left")

    # Memory-scaling panel.
    ax_mem.plot(x_fit, mem_fit, color=fit_color, linewidth=1.8, linestyle="--", label="vector memory extrapolation")
    scatter(ax_mem, vector, vector_color, "o", "vector bigram", "peak_allocated_mib", 1 / 1024)
    scatter(ax_mem, rom_read, read_color, "s", "ROM read", "peak_allocated_mib", 1 / 1024)
    scatter(ax_mem, rom_write, write_color, "D", "ROM write", "peak_allocated_mib", 1 / 1024)
    ax_mem.axhline(94.97, color=cap_color, linewidth=1.2, linestyle=":", label="96 GB GPU capacity")
    ax_mem.set_xscale("log")
    ax_mem.set_xticks([5, 25, 100], labels=["5x", "25x", "100x"])
    ax_mem.set_xlabel("bigram table factor")
    ax_mem.set_ylabel("peak allocated memory (GiB)")
    ax_mem.set_title("ROM bends the capacity curve")
    ax_mem.grid(True, axis="y", alpha=0.22)
    ax_mem.set_ylim(34, max(122, predicted_vec_100 + 5))
    ax_mem.annotate(
        f"vector 100x extrapolates to\n~{predicted_vec_100:.1f} GiB allocated",
        xy=(100, predicted_vec_100),
        xytext=(30, predicted_vec_100 - 18),
        arrowprops=dict(arrowstyle="->", color="#5f6368", lw=1.2),
        fontsize=9,
        color="#3c4043",
    )
    ax_mem.annotate(
        f"ROM write: {as_float(rom_write[0], 'peak_allocated_mib') / 1024:.1f} GiB",
        xy=(100, as_float(rom_write[0], "peak_allocated_mib") / 1024),
        xytext=(39, 67),
        arrowprops=dict(arrowstyle="->", color="#5f6368", lw=1.2),
        fontsize=9,
        color="#3c4043",
    )
    ax_mem.legend(frameon=False, fontsize=9, loc="upper left")

    note = (
        "All runs use the SM120 finite-eval path. Loss fit uses only measured vector points; "
        "memory extrapolation is from factor 5/25 vector runs."
    )
    fig.text(0.5, -0.01, note, ha="center", fontsize=9, color="#5f6368")
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")

    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_SVG}")
    print(f"vector memory fit: {mem_intercept:.2f} GiB + {mem_slope:.3f} GiB/factor")
    print(f"predicted vector 100x peak allocated: {predicted_vec_100:.2f} GiB")
    print(f"vector loss fit: L = {loss_intercept:.4f} {loss_slope:+.4f} ln(factor)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
