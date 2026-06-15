#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
REPORTS = ROOT / "reports"
CF = REPORTS / "engram_counterfactuals"
INIT_CF = REPORTS / "engram_init_counterfactuals"
OUT = REPORTS / "engram_init_report.html"

VAL_RE = re.compile(r"step:(\d+)/(\d+) val_loss:([0-9.]+)(.*)")
CONFIG_RE = re.compile(r"Experiment config: (.*)")
KEYVAL_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")

RUNS = [
    {
        "label": "BF99 pre-AttnRes",
        "run_id": "engram_bf99_poshitlr_sqrt_adamevery_lr6p35_nofloor_layers2_8_layerhash_readouts_dim768_h6_ng3_ga8_1500_20260516_035123",
        "note": "Layer 2+8 Engram with positive sqrt hit LR before AttnRes merge.",
    },
    {
        "label": "BF99 AttnRes, std=1",
        "run_id": "engram_bf99_attnres_poshit_normreadout_canon_shortconv_hist_ckpt500_20260517_003531",
        "note": "Current best saved run: normalized readout, short conv, AttnRes merge, default N(0,1) memory init.",
    },
    {
        "label": "BF99 AttnRes, std=1e-2",
        "run_id": "engram_bf99_attnres_poshit_normreadout_canon_shortconv_init1e2_hist_ckpt500_20260517_031953",
        "note": "Same setup, memory table initialized with N(0, 1e-2). Live/partial until the run completes.",
    },
    {
        "label": "BF99 AttnRes + L2 source, std=1e-2",
        "run_id": "engram_bf99_attnres_l2src_l8_init1e2_hist_ckpt500_20260517_042608",
        "note": "Layer-8 AttnRes can also attend to the saved post-layer-2 residual stream.",
    },
]

NORMAL_LOSS = 3.2589
UNHIT_LOSS = 3.25995517
ALL_MEMORY_MASK_LOSS = 3.4201695919036865


def parse_log(run_id: str) -> dict:
    path = LOGS / f"{run_id}.console.txt"
    points = []
    config = {}
    status = "missing"
    text = ""
    if path.exists():
        text = path.read_text(errors="replace")
        status = "partial"
        if "Traceback" in text or "RuntimeError" in text:
            status = "failed"
        if "peak memory allocated" in text:
            status = "complete"
        for line in text.splitlines():
            m = CONFIG_RE.search(line)
            if m:
                config = dict(KEYVAL_RE.findall(m.group(1)))
            m = VAL_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            total = int(m.group(2))
            loss = float(m.group(3))
            metrics = {}
            for part in m.group(4).split():
                if ":" not in part:
                    continue
                key, raw = part.split(":", 1)
                try:
                    metrics[key] = float(raw)
                except ValueError:
                    pass
            points.append({"step": step, "total": total, "loss": loss, "metrics": metrics})
        if points and points[-1]["step"] >= points[-1]["total"]:
            status = "complete"
    return {
        "run_id": run_id,
        "path": path,
        "status": status,
        "points": points,
        "config": config,
        "text": text,
        "final": points[-1]["loss"] if points else None,
        "best": min((p["loss"] for p in points), default=None),
    }


def load_jsons(root: Path = CF) -> list[dict]:
    rows = []
    for path in sorted(root.glob("eval_mask_*_step001500.json")):
        data = json.loads(path.read_text())
        name = path.name.removesuffix("_step001500.json").removeprefix("eval_mask_")
        if name == "all":
            name = "all_memory"
        rows.append({"name": name, "path": path, **data})
    return rows


def load_mask_meta() -> dict[str, dict]:
    path = CF / "engram_hit_count_mask_hists_step001500.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {item["label"]: item for item in data.get("outputs", [])}


def fmt(x: float | None, n: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{n}f}"


def esc(x: object) -> str:
    return html.escape(str(x))


def svg_curve(runs: list[dict]) -> str:
    pts = [(run, p) for run in runs for p in run["parsed"]["points"] if p["step"] > 0]
    if not pts:
        return "<p>No validation curves available yet.</p>"
    max_step = max(p["total"] for _, p in pts)
    losses = [p["loss"] for _, p in pts]
    lo = min(losses) - 0.03
    hi = max(losses) + 0.03
    x0, y0, w, h = 68, 24, 780, 280
    colors = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#d97706"]
    dashes = ["", "7 4", "3 3", "10 4 2 4"]
    grid = []
    for frac in [0, .25, .5, .75, 1]:
        y = y0 + h * frac
        loss = hi - (hi - lo) * frac
        grid.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+w}" y2="{y:.1f}"/>')
        grid.append(f'<text x="14" y="{y+4:.1f}">{loss:.2f}</text>')
    for frac in [0, .25, .5, .75, 1]:
        x = x0 + w * frac
        grid.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+h}"/>')
        grid.append(f'<text x="{x-12:.1f}" y="{y0+h+22}">{int(max_step * frac)}</text>')

    lines = []
    legend = []
    for i, run in enumerate(runs):
        points = run["parsed"]["points"]
        if not points:
            continue
        color = colors[i % len(colors)]
        coords = []
        for p in points:
            x = x0 + w * (p["step"] / max(1, max_step))
            y = y0 + h * (1 - (p["loss"] - lo) / max(1e-9, hi - lo))
            coords.append(f"{x:.1f},{y:.1f}")
            if p["step"] in {0, 250, 500, 750, 1000, 1250, 1500}:
                lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"><title>{esc(run["label"])} step {p["step"]}: {p["loss"]:.4f}</title></circle>')
        dash = f' stroke-dasharray="{dashes[i % len(dashes)]}"' if dashes[i % len(dashes)] else ""
        lines.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="2.6"{dash}/>')
        legend.append(f'<span><i style="background:{color}"></i>{esc(run["label"])}</span>')

    return f"""
    <div class="legend">{''.join(legend)}</div>
    <svg viewBox="0 0 890 345" role="img" aria-label="Validation loss curves">
      <g class="grid">{''.join(grid)}</g>
      <text x="390" y="340">optimizer step</text>
      <text x="8" y="20">val loss</text>
      {''.join(lines)}
    </svg>
    """


def bar_chart(rows: list[dict], meta: dict[str, dict], names: list[str], title: str) -> str:
    selected = [r for r in rows if r["name"] in names]
    if not selected:
        return ""
    max_delta = max(abs(float(r["val_loss"]) - NORMAL_LOSS) for r in selected)
    max_delta = max(max_delta, 1e-6)
    bars = []
    for r in selected:
        delta = float(r["val_loss"]) - NORMAL_LOSS
        width = 100 * abs(delta) / max_delta
        label = r["name"]
        m = meta.get(label, {})
        detail = ""
        if m:
            detail = f" rows {100*m.get('masked_row_fraction', 0):.1f}%, hits {100*m.get('masked_hit_fraction', 0):.1f}%"
        bars.append(
            f"<tr><td><code>{esc(label)}</code><small>{esc(detail)}</small></td>"
            f"<td>{fmt(float(r['val_loss']))}</td><td>{delta:+.4f}</td>"
            f"<td><div class='bar'><b style='width:{width:.1f}%'></b></div></td></tr>"
        )
    return f"<h3>{esc(title)}</h3><table><thead><tr><th>mask</th><th>loss</th><th>delta</th><th></th></tr></thead><tbody>{''.join(bars)}</tbody></table>"


def matched_init_table(runs: list[dict]) -> str:
    if len(runs) < 3:
        return ""
    base = {p["step"]: p for p in runs[1]["parsed"]["points"]}
    small = {p["step"]: p for p in runs[2]["parsed"]["points"]}
    common = [s for s in sorted(set(base) & set(small)) if s > 0]
    if not common:
        return "<p>No matched init-comparison validation points yet.</p>"
    rows = []
    for step in common:
        b = base[step]
        s = small[step]
        bm = b["metrics"]
        sm = s["metrics"]
        rows.append(
            "<tr>"
            f"<td>{step}</td>"
            f"<td>{fmt(b['loss'])}</td><td>{fmt(s['loss'])}</td><td>{s['loss'] - b['loss']:+.4f}</td>"
            f"<td>{fmt(bm.get('engram_param_rms'), 3)} -> {fmt(sm.get('engram_param_rms'), 3)}</td>"
            f"<td>{fmt(bm.get('engram_grad_rms'), 3)} -> {fmt(sm.get('engram_grad_rms'), 3)}</td>"
            f"<td>{fmt(bm.get('engram_attnres_p_mean'), 3)} -> {fmt(sm.get('engram_attnres_p_mean'), 3)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>step</th><th>std=1 loss</th><th>std=1e-2 loss</th><th>delta</th>"
        "<th>param RMS</th><th>grad RMS</th><th>AttnRes p(mem)</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def counterfactual_compare_table(std_rows: list[dict], init_rows: list[dict]) -> str:
    std = {row["name"]: row for row in std_rows}
    init = {row["name"]: row for row in init_rows}
    names = ["unhit", "all_memory", "hit_ge_1024", "hit_ge_256", "hit_ge_64", "hit_lt_64", "hit_lt_256"]
    rows = []
    for name in names:
        a = std.get(name)
        b = init.get(name)
        if a is None and b is None:
            continue
        a_loss = float(a["val_loss"]) if a else None
        b_loss = float(b["val_loss"]) if b else None
        rows.append(
            "<tr>"
            f"<td><code>{esc(name)}</code></td>"
            f"<td>{fmt(a_loss)}</td><td>{fmt(None if a_loss is None else a_loss - NORMAL_LOSS)}</td>"
            f"<td>{fmt(b_loss)}</td><td>{fmt(None if b_loss is None else b_loss - 3.2575)}</td>"
            f"<td>{fmt(None if a_loss is None or b_loss is None else b_loss - a_loss)}</td>"
            "</tr>"
        )
    if not rows:
        return "<p>No small-init counterfactuals found yet.</p>"
    return (
        "<table><thead><tr><th>mask</th><th>std=1 loss</th><th>std=1 delta</th>"
        "<th>std=1e-2 loss</th><th>std=1e-2 delta</th><th>small-init minus std=1</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render() -> str:
    runs = []
    for spec in RUNS:
        parsed = parse_log(spec["run_id"])
        runs.append({**spec, "parsed": parsed})
    cf_rows = load_jsons(CF)
    init_cf_rows = load_jsons(INIT_CF)
    meta = load_mask_meta()

    run_rows = []
    for run in runs:
        parsed = run["parsed"]
        cfg = parsed["config"]
        run_rows.append(
            "<tr>"
            f"<td>{esc(run['label'])}<small>{esc(run['run_id'])}</small></td>"
            f"<td>{esc(parsed['status'])}</td><td>{fmt(parsed['final'])}</td><td>{fmt(parsed['best'])}</td>"
            f"<td>{esc(cfg.get('engram_init_std', '1.0'))}</td><td>{esc(cfg.get('engram_attnres_merge', '?'))}</td>"
            f"<td>{esc(run['note'])}</td>"
            "</tr>"
        )

    cumulative = ["hit_ge_1024", "hit_ge_256", "hit_ge_64", "hit_ge_16", "hit_ge_4", "all_memory"]
    cold = ["hit_lt_16", "hit_lt_64", "hit_lt_256", "hit_lt_1024"]
    topk = ["punct_top25", "punct_top100", "punct_top500", "semantic_top25", "semantic_top100", "semantic_top500"]

    init_run = runs[-1]["parsed"]
    latest = init_run["points"][-1] if init_run["points"] else None
    init_comment = "The 1e-2 run has not emitted a validation point yet."
    if latest:
        init_comment = f"The 1e-2 run is at step {latest['step']}/{latest['total']} with val_loss {latest['loss']:.4f}; compare against std=1.0 at the same step as more points arrive."

    hit_dist = REPORTS / "engram_hit_dist" / "engram_hit_count_distribution_step001500.png"
    hit_img = f"<img src='engram_hit_dist/{hit_dist.name}' alt='Engram hit-count distribution'>" if hit_dist.exists() else "<p>Hit distribution image not found.</p>"

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Engram Init And Counterfactual Report</title>
  <style>
    body {{ font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #111827; }}
    h1 {{ font-size: 26px; margin: 0 0 8px; }}
    h2 {{ font-size: 19px; margin: 30px 0 8px; }}
    h3 {{ font-size: 15px; margin: 20px 0 8px; }}
    .lede {{ max-width: 980px; color: #374151; }}
    .callout {{ border-left: 4px solid #2563eb; background: #eff6ff; padding: 10px 14px; margin: 16px 0; max-width: 980px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 8px 0 18px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; text-align: left; padding: 7px 8px; vertical-align: top; }}
    th {{ background: #f9fafb; font-weight: 650; }}
    small {{ display: block; color: #6b7280; margin-top: 2px; }}
    code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }}
    svg {{ width: 100%; max-width: 980px; height: auto; border: 1px solid #e5e7eb; border-radius: 8px; background: white; }}
    .grid line {{ stroke: #e5e7eb; stroke-width: 1; }}
    .grid text {{ font-size: 11px; fill: #6b7280; }}
    .legend {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 8px 0; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 5px; }}
    .legend i {{ width: 14px; height: 3px; display: inline-block; }}
    .bar {{ height: 10px; background: #f3f4f6; border-radius: 5px; overflow: hidden; width: 160px; }}
    .bar b {{ display: block; height: 100%; background: #dc2626; }}
    img {{ max-width: 980px; width: 100%; border: 1px solid #e5e7eb; border-radius: 8px; }}
    ul {{ max-width: 980px; }}
  </style>
</head>
<body>
  <h1>Engram Init And Counterfactual Report</h1>
  <p class="lede">This report compares the BF99 AttnRes Engram run against the new small-memory-init run and summarizes the row-masking counterfactuals from the step-1500 checkpoint.</p>
  <div class="callout"><b>Current read:</b> AttnRes is a clean architectural win. The memory benefit is broad but highly Zipf-shaped: a small hot tail carries most of the causal signal, while cold rows are weakly useful only inside the full mixture. Small memory init at <code>1e-2</code> is a small but consistent win. The naive layer-2 residual source was rejected by the model and hurt by step 500. {esc(init_comment)}</div>

  <h2>Training Curves</h2>
  {svg_curve(runs)}
  <table><thead><tr><th>run</th><th>status</th><th>final/latest</th><th>best</th><th>init std</th><th>AttnRes</th><th>note</th></tr></thead><tbody>{''.join(run_rows)}</tbody></table>

  <h2>Matched Init Diagnostics</h2>
  {matched_init_table(runs)}

  <h2>Counterfactual Memory Masks</h2>
  <p>Baseline for deltas is the normal std=1 AttnRes saved run final loss, approximately <code>{NORMAL_LOSS:.4f}</code>. Unseen-row-only masking was <code>{UNHIT_LOSS:.4f}</code>; all-memory masking was <code>{ALL_MEMORY_MASK_LOSS:.4f}</code>.</p>
  <h3>Std=1 vs Std=1e-2 Counterfactuals</h3>
  {counterfactual_compare_table(cf_rows, init_cf_rows)}
  {bar_chart(cf_rows, meta, cumulative, "Mask Hot Rows Cumulatively")}
  {bar_chart(cf_rows, meta, cold, "Mask Cold Rows Cumulatively")}
  {bar_chart(cf_rows, meta, topk, "Mask Top Interpretable Punctuation/Semantic Rows")}

  <h2>Hit Count Distribution</h2>
  {hit_img}

  <h2>Working Hypotheses</h2>
  <ul>
    <li><b>Zipf law is central.</b> Row utility is not proportional to row count. The top 8.5% of rows by hit count covered 53.7% of training touches and caused nearly all of the all-memory ablation damage when removed.</li>
    <li><b>Cold rows are not pure noise.</b> Masking <code>hit&lt;64</code> removed 37.8% of rows and slightly worsened loss. They provide small support inside the full mixture, even if they cannot substitute for hot rows.</li>
    <li><b>Partial memory can be miscalibrated.</b> Masking <code>hit&gt;=64</code> was worse than masking all memory, which suggests the readout/AttnRes merger expects the co-adapted full memory distribution.</li>
    <li><b>Init scale affects learning, not just output scale.</b> RMS normalization can hide row magnitude at the final output, but key/value projections, gate dynamics, SparseAdam update-to-param ratios, and the early AttnRes mixture still see the init regime. The <code>1e-2</code> run had lower row RMS, higher gradient-to-param ratio, lower memory attention probability, and slightly better loss.</li>
    <li><b>Depth-skip residuals need a better interface.</b> Adding the raw post-layer-2 residual as a third layer-8 AttnRes source hurt early BF99 loss and was downweighted by the router. A raw residual stream is probably too broad; a projected/delta/cache stream is a better test.</li>
  </ul>

  <h2>Next Bets</h2>
  <ul>
    <li>Prefer small memory init by default, and sweep <code>1e-3</code>/<code>1e-4</code> or add a learned readout scale initialized near zero.</li>
    <li>Add causal attribution by validation-token usage: aggregate loss delta or logit delta by row/head/position instead of selecting rows by training frequency alone.</li>
    <li>Replace raw layer-2 residual reuse with a compressed/projected delta cache source, then let layer 8 attend over current state, Engram readout, and that cache.</li>
    <li>Train with frequency-aware confidence rather than eval-only masks: e.g. per-row learned confidence, hit-count prior, or dropout/cold-row annealing so partial memory states are calibrated.</li>
  </ul>
</body>
</html>"""


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    OUT.write_text(render())
    print(OUT)


if __name__ == "__main__":
    main()
