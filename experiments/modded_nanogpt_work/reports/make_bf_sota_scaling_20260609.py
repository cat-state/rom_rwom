#!/usr/bin/env python3
from __future__ import annotations

import base64
import csv
import io
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "tmp_remote_bf_scaling_20260609"
READOUTDELTA_LOG_DIR = ROOT / "tmp_remote_readoutdelta_20260609"
CURRENT_META_LOG_DIR = ROOT / "tmp_remote_current_meta_bfscale_20260609"
OUT_DIR = ROOT / "reports" / "bf_sota_scaling_20260609"
BF_VALUES = [40, 80, 120, 200, 300, 400]
READOUTDELTA_RUNS = {
    120: "bf120_sota_readoutdelta_seed5_1500_20260609.txt",
    200: "bf200_sota_readoutdelta_seed5_1500_20260609.txt",
    300: "bf300_sota_readoutdelta_seed5_1500_20260609.txt",
}
MEASURED_COLD_TAIL_PRUNING = {
    # Post-hoc checkpoint evals from report.md. Rows with hit < 32 are zeroed.
    # Parameter count assumes compacting the table down to retained hit>=32 rows.
    300: {"base_eval_loss": 3.24435, "pruned_row_fraction": 0.2875, "eval_loss": 3.24494},
    400: {"base_eval_loss": 3.24525, "pruned_row_fraction": 0.5229, "eval_loss": 3.24794},
}


@dataclass
class Run:
    bf: int
    path: Path
    final_val: float
    final_step: int
    vals: list[tuple[int, float]]
    table_rows: int
    store_dim: int
    table_params: int
    peak_mib: int | None
    step_avg_ms: float
    hit_frac_ever: float | None
    hit_frac_gt1: float | None
    touched_fraction: float | None


def parse_last_float(line: str, key: str) -> float | None:
    m = re.search(rf"\b{re.escape(key)}:([0-9.eE+-]+)", line)
    return float(m.group(1)) if m else None


def parse_log(path: Path, bf: int) -> Run:
    text = path.read_text(errors="replace")
    vals: list[tuple[int, float]] = []
    trains: list[tuple[int, float, float]] = []
    peak_mib: int | None = None
    table_rows: int | None = None
    store_dim: int | None = None
    table_params: int | None = None
    hit_frac_ever: float | None = None
    hit_frac_gt1: float | None = None
    touched_fraction: float | None = None

    for line in text.splitlines():
        m = re.search(r"step:(\d+)/(\d+) val_loss:([0-9.]+)", line)
        if m:
            vals.append((int(m.group(1)), float(m.group(3))))
            hit_frac_ever = parse_last_float(line, "engram_hit_frac_ever") or hit_frac_ever
            hit_frac_gt1 = parse_last_float(line, "engram_hit_frac_gt1") or hit_frac_gt1
            touched_fraction = parse_last_float(line, "engram_touched_fraction") or touched_fraction
            row_metric = parse_last_float(line, "engram_table_rows")
            numel_metric = parse_last_float(line, "engram_table_numel")
            if row_metric is not None:
                table_rows = int(row_metric)
            if numel_metric is not None:
                table_params = int(numel_metric)
        m = re.search(r"step:(\d+)/(\d+) train_loss:([0-9.]+).*?step_avg:([0-9.]+)ms", line)
        if m:
            trains.append((int(m.group(1)), float(m.group(3)), float(m.group(4))))
        m = re.search(r"Engram memory shape: rows=(\d+) store_dim=(\d+) embedding_shape=\((\d+),\s*(\d+)\)", line)
        if m:
            table_rows = int(m.group(1))
            store_dim = int(m.group(2))
            table_params = int(m.group(3)) * int(m.group(4))
        m = re.search(r"peak memory allocated: (\d+) MiB", line)
        if m:
            peak_mib = int(m.group(1))

    if not vals:
        raise RuntimeError(f"no validation losses found in {path}")
    if table_rows is None or table_params is None:
        raise RuntimeError(f"no Engram table shape found in {path}")
    if store_dim is None:
        store_dim = table_params // table_rows
    final_step, final_val = vals[-1]
    if final_step != 1500:
        raise RuntimeError(f"{path} is incomplete; last validation step is {final_step}")
    return Run(
        bf=bf,
        path=path,
        final_val=final_val,
        final_step=final_step,
        vals=vals,
        table_rows=table_rows,
        store_dim=store_dim,
        table_params=table_params,
        peak_mib=peak_mib,
        step_avg_ms=trains[-1][2] if trains else math.nan,
        hit_frac_ever=hit_frac_ever,
        hit_frac_gt1=hit_frac_gt1,
        touched_fraction=touched_fraction,
    )


def svg_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    return "data:image/svg+xml;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def fmt_billions(x: int) -> str:
    return f"{x / 1e9:.2f}B"


def collect_runs(log_dir: Path, filename_template: str) -> tuple[list[Run], list[int]]:
    runs: list[Run] = []
    missing: list[int] = []
    for bf in BF_VALUES:
        path = log_dir / filename_template.format(bf=bf)
        if not path.exists():
            missing.append(bf)
            continue
        try:
            runs.append(parse_log(path, bf))
        except RuntimeError:
            missing.append(bf)
    runs.sort(key=lambda r: r.bf)
    return runs, missing


def write_summary(path: Path, runs: list[Run]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "bf",
                "final_val",
                "table_rows",
                "store_dim",
                "table_params",
                "peak_mib",
                "step_avg_ms",
                "hit_frac_ever",
                "hit_frac_gt1",
                "touched_fraction",
                "log",
            ]
        )
        for r in runs:
            writer.writerow(
                [
                    r.bf,
                    f"{r.final_val:.6f}",
                    r.table_rows,
                    r.store_dim,
                    r.table_params,
                    r.peak_mib or "",
                    f"{r.step_avg_ms:.3f}",
                    "" if r.hit_frac_ever is None else f"{r.hit_frac_ever:.6g}",
                    "" if r.hit_frac_gt1 is None else f"{r.hit_frac_gt1:.6g}",
                    "" if r.touched_fraction is None else f"{r.touched_fraction:.6g}",
                    r.path.name,
                ]
            )


def make_pruned_runs(runs: list[Run], pruning: dict[int, dict[str, float]]) -> list[Run]:
    pruned_runs: list[Run] = []
    for r in runs:
        spec = pruning.get(r.bf)
        if spec is None:
            continue
        retained_fraction = 1.0 - float(spec["pruned_row_fraction"])
        retained_rows = max(1, int(round(r.table_rows * retained_fraction)))
        pruned_runs.append(
            replace(
                r,
                final_val=float(spec["eval_loss"]),
                table_rows=retained_rows,
                table_params=retained_rows * r.store_dim,
                peak_mib=None,
                touched_fraction=retained_fraction,
            )
        )
    return pruned_runs


def write_pruning_summary(path: Path, baseline_runs: list[Run], pruned_runs: list[Run]) -> None:
    baseline_by_bf = {r.bf: r for r in baseline_runs}
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "bf",
                "checkpoint_base_eval_loss",
                "pruned_eval_loss",
                "delta",
                "baseline_table_params",
                "pruned_table_params",
                "retained_fraction",
                "pruning_rule",
            ]
        )
        for r in pruned_runs:
            base = baseline_by_bf[r.bf]
            base_eval_loss = float(MEASURED_COLD_TAIL_PRUNING[r.bf]["base_eval_loss"])
            writer.writerow(
                [
                    r.bf,
                    f"{base_eval_loss:.6f}",
                    f"{r.final_val:.6f}",
                    f"{r.final_val - base_eval_loss:+.6f}",
                    base.table_params,
                    r.table_params,
                    f"{r.table_params / base.table_params:.6g}",
                    "zero-mask rows with hit < 32; params assume compaction",
                ]
            )


def plot_scaling(
    runs: list[Run],
    overlays: list[tuple[str, list[Run]]] | None,
    title: str,
    png_path: Path,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))
    xs = [r.table_params for r in runs]
    ys = [r.final_val for r in runs]

    ax = axes[0]
    ax.plot(xs, ys, marker="o", linewidth=2.2, label="main")
    for label, overlay_runs in overlays or []:
        if not overlay_runs:
            continue
        ax.plot(
            [r.table_params for r in overlay_runs],
            [r.final_val for r in overlay_runs],
            marker="D",
            linewidth=2.0,
            linestyle="--",
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Engram table parameters")
    ax.set_ylabel("Final validation loss @ 1500")
    ax.grid(True, alpha=0.25)
    for r in runs:
        ax.annotate(f"BF{r.bf}", (r.table_params, r.final_val), textcoords="offset points", xytext=(4, 5), fontsize=9)
    for label, overlay_runs in overlays or []:
        for r in overlay_runs:
            ax.annotate(
                f"BF{r.bf} {label}",
                (r.table_params, r.final_val),
                textcoords="offset points",
                xytext=(4, -12),
                fontsize=9,
            )
    if overlays:
        ax.legend(frameon=False)

    ax = axes[1]
    ax.plot([r.bf for r in runs], ys, marker="s", linewidth=2.2, color="#2b6cb0", label="main")
    for label, overlay_runs in overlays or []:
        if not overlay_runs:
            continue
        if "pruned" in label:
            continue
        ax.plot(
            [r.bf for r in overlay_runs],
            [r.final_val for r in overlay_runs],
            marker="D",
            linewidth=2.0,
            linestyle="--",
            label=label,
        )
    ax.set_xlabel("BIGRAM_FACTOR")
    ax.set_ylabel("Final validation loss @ 1500")
    ax.grid(True, alpha=0.25)
    if len(runs) > 1:
        ax.set_xticks([r.bf for r in runs])
    if overlays:
        ax.legend(frameon=False)
    fig.suptitle(title, y=1.03, fontsize=14)
    fig.tight_layout()
    plot_uri = svg_data_uri(fig)
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    return fig, plot_uri


def plot_curves(runs: list[Run], overlays: list[tuple[str, list[Run]]] | None, png_path: Path):
    import matplotlib.pyplot as plt

    curve_fig, curve_ax = plt.subplots(figsize=(7.2, 4.6))
    for r in runs:
        curve_ax.plot([s for s, _ in r.vals], [v for _, v in r.vals], marker="o", label=f"BF{r.bf}")
    for label, overlay_runs in overlays or []:
        for r in overlay_runs:
            curve_ax.plot(
                [s for s, _ in r.vals],
                [v for _, v in r.vals],
                marker="D",
                linestyle="--",
                label=f"BF{r.bf} {label}",
            )
    curve_ax.set_xlabel("Step")
    curve_ax.set_ylabel("Validation loss")
    curve_ax.grid(True, alpha=0.25)
    curve_ax.legend(frameon=False, ncol=2)
    curve_fig.tight_layout()
    curve_uri = svg_data_uri(curve_fig)
    curve_fig.savefig(png_path, dpi=180, bbox_inches="tight")
    return curve_fig, curve_uri


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs, missing = collect_runs(LOG_DIR, "bf{bf}_sota_meta_scaling_seed5_1500_20260609.txt")

    if not runs:
        raise SystemExit(f"no complete logs found under {LOG_DIR}")

    readoutdelta_runs: list[Run] = []
    for bf, name in READOUTDELTA_RUNS.items():
        path = READOUTDELTA_LOG_DIR / name
        if not path.exists():
            continue
        try:
            readoutdelta_runs.append(parse_log(path, bf))
        except RuntimeError:
            continue
    readoutdelta_runs.sort(key=lambda r: r.bf)
    current_meta_runs, current_meta_missing = collect_runs(
        CURRENT_META_LOG_DIR, "bf{bf}_sota_meta_bfscale_seed5_1500_20260609.txt"
    )
    pruned_current_meta_runs = make_pruned_runs(current_meta_runs, MEASURED_COLD_TAIL_PRUNING)

    write_summary(OUT_DIR / "summary.csv", runs)

    if readoutdelta_runs:
        write_summary(OUT_DIR / "summary_readoutdelta.csv", readoutdelta_runs)
    if current_meta_runs:
        write_summary(OUT_DIR / "summary_current_meta_bfscale.csv", current_meta_runs)
    if pruned_current_meta_runs:
        write_pruning_summary(
            OUT_DIR / "summary_current_meta_pruned_hit32.csv",
            current_meta_runs,
            pruned_current_meta_runs,
        )

    fig, plot_uri = plot_scaling(
        runs,
        [("readoutdelta", readoutdelta_runs)],
        "Original SOTA Meta: BF Scaling vs Engram Parameter Count",
        OUT_DIR / "bf_sota_scaling_vs_params.png",
    )
    plt.close(fig)

    curve_fig, curve_uri = plot_curves(
        runs,
        [("readoutdelta", readoutdelta_runs)],
        OUT_DIR / "bf_sota_scaling_curves.png",
    )
    plt.close(curve_fig)

    current_meta_html = ""
    if current_meta_runs:
        current_fig, current_plot_uri = plot_scaling(
            current_meta_runs,
            [("pruned hit>=32", pruned_current_meta_runs)] if pruned_current_meta_runs else None,
            "Current Readoutdelta Meta: BF Scaling vs Engram Parameter Count",
            OUT_DIR / "bf_current_meta_bfscale_vs_params.png",
        )
        plt.close(current_fig)
        current_curve_fig, current_curve_uri = plot_curves(
            current_meta_runs,
            None,
            OUT_DIR / "bf_current_meta_bfscale_curves.png",
        )
        plt.close(current_curve_fig)
        current_rows = []
        for r in current_meta_runs:
            original = next((base for base in runs if base.bf == r.bf), None)
            delta = "" if original is None else f"{r.final_val - original.final_val:+.4f}"
            current_rows.append(
                "<tr>"
                f"<td>{r.bf}</td>"
                f"<td>{r.final_val:.4f}</td>"
                f"<td>{delta}</td>"
                f"<td>{fmt_billions(r.table_params)}</td>"
                f"<td>{r.table_rows:,}</td>"
                f"<td>{'' if r.peak_mib is None else r.peak_mib}</td>"
                f"<td>{'' if r.touched_fraction is None else f'{r.touched_fraction:.4f}'}</td>"
                "</tr>"
            )
        pruned_rows = []
        baseline_by_bf = {r.bf: r for r in current_meta_runs}
        for r in pruned_current_meta_runs:
            base = baseline_by_bf[r.bf]
            base_eval_loss = float(MEASURED_COLD_TAIL_PRUNING[r.bf]["base_eval_loss"])
            pruned_rows.append(
                "<tr>"
                f"<td>{r.bf}</td>"
                f"<td>{base_eval_loss:.5f}</td>"
                f"<td>{r.final_val:.5f}</td>"
                f"<td>{r.final_val - base_eval_loss:+.5f}</td>"
                f"<td>{fmt_billions(r.table_params)}</td>"
                f"<td>{r.table_params / base.table_params:.3f}</td>"
                f"<td>zero-mask hit &lt; 32; compact retained rows</td>"
                "</tr>"
            )
        current_note = (
            "All six current-meta BF points are present."
            if not current_meta_missing
            else f"Partial current-meta report: still missing complete logs for BF {', '.join(map(str, current_meta_missing))}."
        )
        current_meta_html = f"""
  <h2>Current Readoutdelta Meta BF Sweep</h2>
  <p class="note">{current_note} Logs were pulled from <code>{CURRENT_META_LOG_DIR.relative_to(ROOT)}</code>.</p>
  <img class="plot" src="{current_plot_uri}" alt="Current-meta BF scaling vs parameter count">
  <img class="plot" src="{current_curve_uri}" alt="Current-meta BF validation curves">
  <table>
    <thead><tr><th>BF</th><th>Final val</th><th>Delta vs original same-BF</th><th>Table params</th><th>Rows</th><th>Peak MiB</th><th>Touched fraction</th></tr></thead>
    <tbody>{''.join(current_rows)}</tbody>
  </table>
  {f'''<h3>Measured Cold-Tail Pruning Overlay</h3>
  <p class="note">These are post-hoc checkpoint evals, not retrained runs. Deltas are against the checkpoint-loaded base evals, not the inline BF-sweep final losses. The plotted parameter count assumes physical compaction after dropping rows with hit &lt; 32.</p>
  <table>
    <thead><tr><th>BF</th><th>Checkpoint base</th><th>Pruned eval loss</th><th>Delta vs ckpt base</th><th>Pruned params</th><th>Retained fraction</th><th>Rule</th></tr></thead>
    <tbody>{''.join(pruned_rows)}</tbody>
  </table>''' if pruned_rows else ''}
"""

    rows = []
    for r in runs:
        rows.append(
            "<tr>"
            f"<td>{r.bf}</td>"
            f"<td>{r.final_val:.4f}</td>"
            f"<td>{fmt_billions(r.table_params)}</td>"
            f"<td>{r.table_rows:,}</td>"
            f"<td>{r.store_dim}</td>"
            f"<td>{'' if r.peak_mib is None else r.peak_mib}</td>"
            f"<td>{r.step_avg_ms:.0f} ms</td>"
            f"<td>{'' if r.hit_frac_ever is None else f'{r.hit_frac_ever:.3f}'}</td>"
            f"<td>{'' if r.hit_frac_gt1 is None else f'{r.hit_frac_gt1:.3f}'}</td>"
            "</tr>"
        )

    readoutdelta_rows = []
    for r in readoutdelta_runs:
        baseline = next((base for base in runs if base.bf == r.bf), None)
        delta = "" if baseline is None else f"{r.final_val - baseline.final_val:+.4f}"
        readoutdelta_rows.append(
            "<tr>"
            f"<td>BF{r.bf} readoutdelta</td>"
            f"<td>{r.final_val:.4f}</td>"
            f"<td>{delta}</td>"
            f"<td>{fmt_billions(r.table_params)}</td>"
            f"<td>{'' if r.peak_mib is None else r.peak_mib}</td>"
            f"<td>{'' if r.touched_fraction is None else f'{r.touched_fraction:.4f}'}</td>"
            "</tr>"
        )

    best = min(runs, key=lambda r: r.final_val)
    complete_note = (
        "All six planned BF points are present."
        if not missing
        else f"Partial report: still missing complete logs for BF {', '.join(map(str, missing))}."
    )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>BF SOTA Scaling 2026-06-09</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #172033; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin-top: 28px; font-size: 18px; }}
    p {{ max-width: 980px; line-height: 1.45; }}
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
  <h1>BF Scaling Of The Current SOTA Meta</h1>
  <p class="note">{complete_note} Logs were pulled from the remote instance into <code>{LOG_DIR.relative_to(ROOT)}</code>. X-axis parameter count is the learned Engram table parameter count; all non-memory model parameters are held fixed across BF, so the table is the scaling variable.</p>
  <p>Best observed point in this sweep so far: <b>BF{best.bf}</b> at <b>{best.final_val:.4f}</b> final validation loss with <b>{fmt_billions(best.table_params)}</b> Engram table parameters.</p>
  <img class="plot" src="{plot_uri}" alt="BF scaling vs parameter count">
  <h2>Validation Curves</h2>
  <img class="plot" src="{curve_uri}" alt="BF validation curves">
  <h2>Summary</h2>
  <table>
    <thead><tr><th>BF</th><th>Final val</th><th>Table params</th><th>Rows</th><th>Store dim</th><th>Peak MiB</th><th>Step avg</th><th>Hit ever</th><th>Hit &gt;1</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  {f'''<h2>Readoutdelta Overlay</h2>
  <table>
    <thead><tr><th>Run</th><th>Final val</th><th>Delta vs same-BF baseline</th><th>Table params</th><th>Peak MiB</th><th>Touched fraction</th></tr></thead>
    <tbody>{''.join(readoutdelta_rows)}</tbody>
  </table>''' if readoutdelta_rows else ''}
  {current_meta_html}
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(html)
    print(OUT_DIR / "index.html")
    print(OUT_DIR / "summary.csv")
    for r in runs:
        print(f"BF{r.bf}: val={r.final_val:.4f} params={fmt_billions(r.table_params)}")
    for r in readoutdelta_runs:
        print(f"BF{r.bf} readoutdelta: val={r.final_val:.4f} params={fmt_billions(r.table_params)}")
    for r in current_meta_runs:
        print(f"BF{r.bf} current-meta: val={r.final_val:.4f} params={fmt_billions(r.table_params)}")
    if missing:
        print("missing:", ",".join(map(str, missing)))
    if current_meta_missing and current_meta_runs:
        print("current-meta missing:", ",".join(map(str, current_meta_missing)))


if __name__ == "__main__":
    main()
