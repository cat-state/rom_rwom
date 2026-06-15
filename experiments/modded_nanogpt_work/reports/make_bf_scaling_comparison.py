#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
OUT = ROOT / "reports" / "engram_vs_builtin_bf_scaling.html"


ENGRAM_RUNS = {
    5: "bf5_h1_gain1p5_tail1_beststack_1500_20260518_223733.console.txt",
    10: "bf10_h1_gain1p5_tail1_beststack_1500_20260518_223733.console.txt",
    20: "bf20_h1_gain1p5_tail1_beststack_1500_20260518_223749.console.txt",
    40: "bf40_h1_gain1p5_tail1_beststack_1500_20260518_231124.console.txt",
    80: "bf80_h1_gain1p5_tail1_beststack_1500_20260518_231149.console.txt",
}

BUILTIN_RUNS = {
    5: "builtin_bigram_bf5_1500_20260518_234659.console.txt",
    10: "builtin_bigram_bf10_1500_20260518_234659.console.txt",
    20: "builtin_bigram_bf20_1500_20260519_001641.console.txt",
    40: "builtin_bigram_bf40_1500_20260519_001650.console.txt",
    80: "builtin_bigram_bf80_ga16_1500_20260519_000821.console.txt",
}


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    vals = []
    trains = []
    peak = None
    for line in text.splitlines():
        m = re.search(r"step:(\d+)/(\d+) val_loss:([0-9.]+)", line)
        if m:
            vals.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
        m = re.search(r"step:(\d+)/(\d+) train_loss:([0-9.]+).*?step_avg:([0-9.]+)ms", line)
        if m:
            trains.append((int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))))
        m = re.search(r"peak memory allocated: (\d+) MiB", line)
        if m:
            peak = int(m.group(1))
    if not vals:
        raise RuntimeError(f"no val losses in {path}")
    final_step, total, final_loss = vals[-1]
    step_avg = trains[-1][3] if trains else math.nan
    return {
        "path": path.name,
        "vals": vals,
        "final_step": final_step,
        "total": total,
        "final_loss": final_loss,
        "step_avg_ms": step_avg,
        "peak_mib": peak,
    }


def svg_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/svg+xml;base64,{data}"


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    engram = {bf: parse_log(LOG_DIR / name) for bf, name in ENGRAM_RUNS.items()}
    builtin = {bf: parse_log(LOG_DIR / name) for bf, name in BUILTIN_RUNS.items()}
    bfs = sorted(engram)

    eng_losses = [engram[bf]["final_loss"] for bf in bfs]
    base_losses = [builtin[bf]["final_loss"] for bf in bfs]
    deltas = [builtin[bf]["final_loss"] - engram[bf]["final_loss"] for bf in bfs]
    rel = [d / builtin[bf]["final_loss"] * 100 for d, bf in zip(deltas, bfs)]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    ax.plot(bfs, base_losses, marker="o", linewidth=2.2, label="Built-in bigram baseline")
    ax.plot(bfs, eng_losses, marker="s", linewidth=2.2, label="Engram best stack")
    ax.set_xscale("log", base=2)
    ax.set_xticks(bfs)
    ax.set_xticklabels([str(x) for x in bfs])
    ax.set_xlabel("BF / bigram table factor")
    ax.set_ylabel("Final validation loss @ 1500 steps")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.bar([str(bf) for bf in bfs], deltas, color="#2b6cb0")
    ax.set_xlabel("BF")
    ax.set_ylabel("Loss advantage vs built-in")
    ax.grid(True, axis="y", alpha=0.25)
    for i, (d, r) in enumerate(zip(deltas, rel)):
        ax.text(i, d + 0.00025, f"{d:.4f}\n{r:.2f}%", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Engram Best Stack vs Built-in Speedrun Bigram Scaling", y=1.03, fontsize=14)
    fig.tight_layout()
    plot_uri = svg_data_uri(fig)
    plt.close(fig)

    rows = []
    for bf in bfs:
        e = engram[bf]
        b = builtin[bf]
        rows.append(
            "<tr>"
            f"<td>{bf}</td>"
            f"<td>{e['final_loss']:.4f}</td>"
            f"<td>{b['final_loss']:.4f}</td>"
            f"<td>{b['final_loss'] - e['final_loss']:.4f}</td>"
            f"<td>{(b['final_loss'] - e['final_loss']) / b['final_loss'] * 100:.2f}%</td>"
            f"<td>{e['step_avg_ms']:.0f} ms</td>"
            f"<td>{b['step_avg_ms']:.0f} ms</td>"
            f"<td>{e['peak_mib'] or 0}</td>"
            f"<td>{b['peak_mib'] or 0}</td>"
            "</tr>"
        )

    curve_rows = []
    for bf in bfs:
        e_vals = {step: loss for step, _, loss in engram[bf]["vals"]}
        b_vals = {step: loss for step, _, loss in builtin[bf]["vals"]}
        for step in sorted(set(e_vals) | set(b_vals)):
            curve_rows.append(
                "<tr>"
                f"<td>{bf}</td><td>{step}</td>"
                f"<td>{e_vals.get(step, float('nan')):.4f}</td>"
                f"<td>{b_vals.get(step, float('nan')):.4f}</td>"
                "</tr>"
            )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Engram vs Built-in BF Scaling</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #172033; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin-top: 28px; font-size: 18px; }}
    p {{ max-width: 920px; line-height: 1.45; }}
    table {{ border-collapse: collapse; margin-top: 14px; font-size: 14px; }}
    th, td {{ border: 1px solid #d9dee8; padding: 7px 10px; text-align: right; }}
    th {{ background: #f3f5f9; }}
    td:first-child, th:first-child {{ text-align: left; }}
    .note {{ color: #536071; }}
    .plot {{ width: min(1180px, 100%); margin-top: 18px; }}
    code {{ background: #f3f5f9; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Engram vs Built-in BF Scaling</h1>
  <p class="note">Generated from local logs in <code>{LOG_DIR}</code>. Built-in baseline uses the speedrun bigram table path with all-layer injection; Engram uses the current best stack: BF-scaled table, h1, gain 1.5, low init, normalized readout, short conv, AttnRes, tail1 sparse Adam, layers 2/8.</p>
  <img class="plot" src="{plot_uri}" alt="BF scaling plots">
  <h2>Final Validation Loss</h2>
  <table>
    <thead><tr><th>BF</th><th>Engram</th><th>Built-in</th><th>Delta</th><th>Relative</th><th>Engram Step Avg</th><th>Built-in Step Avg</th><th>Engram Peak MiB</th><th>Built-in Peak MiB</th></tr></thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
  <h2>Validation Curve Points</h2>
  <table>
    <thead><tr><th>BF</th><th>Step</th><th>Engram</th><th>Built-in</th></tr></thead>
    <tbody>
      {''.join(curve_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    OUT.write_text(html)
    print(OUT)
    for bf, d in zip(bfs, deltas):
        print(f"BF{bf}: Engram {engram[bf]['final_loss']:.4f} vs built-in {builtin[bf]['final_loss']:.4f}; delta {d:.4f}")


if __name__ == "__main__":
    main()
