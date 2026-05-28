#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--out", type=Path, default=Path("engram_analysis_summary.png"))
    args = parser.parse_args()

    data = json.loads(args.analysis.read_text())
    heads = data["head_summaries"]
    top_slots = data["top_slots"][:12]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), dpi=180)

    gate_q = data["gate"]["quantiles"]
    qx = [float(k) for k in gate_q]
    qy = [gate_q[str(k)] if str(k) in gate_q else gate_q[k] for k in gate_q]
    q_pairs = sorted(zip(qx, qy))
    axes[0].plot([q for q, _ in q_pairs], [v for _, v in q_pairs], color="#6f4aa2", linewidth=2.2)
    axes[0].axhline(0.5, color="#777777", linewidth=1.0, linestyle="--")
    axes[0].set_title("Gate Distribution")
    axes[0].set_xlabel("quantile")
    axes[0].set_ylabel("sigmoid gate")
    axes[0].grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.7)
    axes[0].text(
        0.03,
        0.96,
        f"mean {data['gate']['mean']:.3f}\n>0.75 {100*data['gate']['frac_gt_0_75']:.1f}%\n>0.9 {100*data['gate']['frac_gt_0_9']:.1f}%",
        transform=axes[0].transAxes,
        va="top",
        fontsize=9,
    )

    x = list(range(len(heads)))
    unique = [h["unique_fraction"] for h in heads]
    colors = ["#3b6fb6" if h["ngram"] == 2 else "#c46a3a" for h in heads]
    axes[1].bar(x, unique, color=colors)
    axes[1].set_title("Slot Coverage By Hash Head")
    axes[1].set_xlabel("hash head index")
    axes[1].set_ylabel("fraction rows touched")
    axes[1].set_ylim(0, max(unique) * 1.15)
    axes[1].grid(True, axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)
    axes[1].legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#3b6fb6", label="bigram"),
            plt.Rectangle((0, 0), 1, 1, color="#c46a3a", label="trigram"),
        ],
        frameon=False,
        loc="upper left",
        fontsize=9,
    )

    labels = [f"h{s['head_idx']} n{s['ngram']}\n{s['examples'][0]['ngram_text'][:10] if s['examples'] else ''}" for s in top_slots]
    values = [s["avg_output_norm"] for s in top_slots]
    axes[2].barh(range(len(top_slots)), values, color="#557f5f")
    axes[2].set_yticks(range(len(top_slots)), labels)
    axes[2].invert_yaxis()
    axes[2].set_title("Top Slots By Output Norm")
    axes[2].set_xlabel("avg contribution norm")
    axes[2].grid(True, axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.7)

    fig.suptitle(f"Engram Analysis: {data['run_id']} ({data['analysis_tokens']:,} tokens)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    if args.out.suffix.lower() != ".svg":
        fig.savefig(args.out.with_suffix(".svg"))


if __name__ == "__main__":
    main()
