#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load(path: Path) -> dict:
    data = json.loads(path.read_text())
    data["_path"] = str(path)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("analyses", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("engram_usage_scaling.png"))
    args = parser.parse_args()

    rows = sorted((load(path) for path in args.analyses), key=lambda d: d["engram_dim"])
    labels = [f"d{row['engram_dim']}" for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), dpi=180)
    colors = ["#3b6fb6", "#2e8b57", "#7a4eb3", "#c46a3a", "#555555"]

    axes[0, 0].plot(
        [row["engram_dim"] for row in rows],
        [row["loss"] for row in rows],
        color="#333333",
        linewidth=2.0,
    )
    axes[0, 0].scatter(
        [row["engram_dim"] for row in rows],
        [row["loss"] for row in rows],
        s=70,
        color=colors[: len(rows)],
        zorder=3,
    )
    for row in rows:
        axes[0, 0].annotate(f"{row['loss']:.4f}", (row["engram_dim"], row["loss"]), xytext=(5, 4), textcoords="offset points", fontsize=9)
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_xticks([row["engram_dim"] for row in rows], labels=labels)
    axes[0, 0].set_title("Analysis-Slice Loss")
    axes[0, 0].set_xlabel("Engram dim")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.7)

    gate_metrics = {
        "mean": [row["gate"]["mean"] for row in rows],
        ">0.75": [row["gate"]["frac_gt_0_75"] for row in rows],
        ">0.9": [row["gate"]["frac_gt_0_9"] for row in rows],
    }
    x = list(range(len(rows)))
    width = 0.24
    for idx, (name, values) in enumerate(gate_metrics.items()):
        axes[0, 1].bar([v + (idx - 1) * width for v in x], values, width=width, label=name, color=colors[idx])
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Gate Activity")
    axes[0, 1].set_ylabel("fraction / mean")
    axes[0, 1].legend(frameon=False, fontsize=9)
    axes[0, 1].grid(True, axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)

    bigram_cov = []
    trigram_cov = []
    for row in rows:
        heads = row["head_summaries"]
        bigram = [h["unique_fraction"] for h in heads if h["ngram"] == 2]
        trigram = [h["unique_fraction"] for h in heads if h["ngram"] == 3]
        bigram_cov.append(sum(bigram) / len(bigram))
        trigram_cov.append(sum(trigram) / len(trigram))
    axes[1, 0].plot(labels, bigram_cov, marker="o", linewidth=2, color="#3b6fb6", label="bigram")
    axes[1, 0].plot(labels, trigram_cov, marker="o", linewidth=2, color="#c46a3a", label="trigram")
    axes[1, 0].set_ylim(0, max(max(bigram_cov), max(trigram_cov)) * 1.15)
    axes[1, 0].set_title("Rows Touched Per Head")
    axes[1, 0].set_ylabel("mean unique fraction")
    axes[1, 0].legend(frameon=False, fontsize=9)
    axes[1, 0].grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.7)

    category_names = sorted({name for row in rows for name in row.get("token_category_stats", {})})
    selected = [name for name in ["newline", "pipe", "digit", "alpha", "punct", "mixed", "whitespace"] if name in category_names]
    for idx, row in enumerate(rows):
        stats = row.get("token_category_stats", {})
        values = [stats[name]["avg_output_norm"] for name in selected]
        axes[1, 1].plot(selected, values, marker="o", linewidth=2, color=colors[idx % len(colors)], label=labels[idx])
    axes[1, 1].set_title("Contribution Norm By Token Category")
    axes[1, 1].set_ylabel("avg output norm")
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].legend(frameon=False, fontsize=9)
    axes[1, 1].grid(True, axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)

    fig.suptitle("Engram Usage Scaling At 500 Steps")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    if args.out.suffix.lower() != ".svg":
        fig.savefig(args.out.with_suffix(".svg"))


if __name__ == "__main__":
    main()
