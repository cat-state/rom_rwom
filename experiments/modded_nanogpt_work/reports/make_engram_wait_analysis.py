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
OUT = REPORTS / "engram_wait_analysis.html"

VAL_RE = re.compile(r"step:(\d+)/(\d+) val_loss:([0-9.]+)(.*)")
CONFIG_RE = re.compile(r"Experiment config: (.*)")
KEYVAL_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")
BF_RE = re.compile(r"_bf(\d+)", re.I)


RUN_LABELS = {
    "engram_bf99_attnres_poshit_normreadout_canon_layers2_8_dim768_h6_ng3_ga8_compilebody_1500_20260516_235031": "BF99 AttnRes/norm readout best",
    "engram_bf99_attnres_poshit_normreadout_canon_shortconv_init1e2_hist_ckpt500_20260517_031953": "BF99 AttnRes low-init hist",
    "engram_bf99_attnres_poshit_normreadout_canon_shortconv_hist_ckpt500_20260517_003531": "BF99 AttnRes std=1 hist",
    "engram_bf99_poshitlr_sqrt_cap100_adamevery_lr6p35_nofloor_layers2_8_layerhash_readouts_dim768_h6_ng3_ga8_1500_20260516_033153": "BF99 positive-hit pre-AttnRes",
    "engram_bf99_mhc_inversehit_adamevery_lr6p35_nofloor_layers2_8_layerhash_readouts_dim768_h6_ng3_ga8_1500_20260516_042315": "BF99 mHC inverse-hit",
}

CURRENT_SUITE = "engram_hotbf_beststack_suite_20260517_163326"
OLD_SUITE_PREFIX = "engram_scale_bf"
OLD_SUITE_SUFFIX = "_sqrthit_max1e9_canon_layers2_8_layerhash_readouts_dim768_h6_ng3_ga8_1500_20260516"


def esc(x: object) -> str:
    return html.escape(str(x))


def fmt(x: float | None, n: int = 4) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{n}f}"


def parse_metrics(rest: str) -> dict[str, float]:
    metrics = {}
    for part in rest.split():
        if ":" not in part:
            continue
        key, raw = part.split(":", 1)
        try:
            metrics[key] = float(raw)
        except ValueError:
            pass
    return metrics


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    points = []
    config = {}
    for line in text.splitlines():
        config_match = CONFIG_RE.search(line)
        if config_match:
            config = dict(KEYVAL_RE.findall(config_match.group(1)))
        val_match = VAL_RE.search(line)
        if val_match:
            points.append(
                {
                    "step": int(val_match.group(1)),
                    "total": int(val_match.group(2)),
                    "loss": float(val_match.group(3)),
                    "metrics": parse_metrics(val_match.group(4)),
                }
            )
    status = "missing"
    if points:
        status = "complete" if ("peak memory allocated" in text or points[-1]["step"] >= points[-1]["total"]) else "partial"
    if "Traceback" in text or "RuntimeError" in text:
        status = "failed"
    return {
        "run_id": path.name.removesuffix(".console.txt"),
        "path": path,
        "points": points,
        "config": config,
        "status": status,
        "last": points[-1] if points else None,
        "final": points[-1]["loss"] if points else None,
        "best": min((p["loss"] for p in points), default=None),
    }


def all_runs() -> dict[str, dict]:
    parsed = {}
    for path in LOGS.glob("*.console.txt"):
        run = parse_log(path)
        if run["points"]:
            parsed[run["run_id"]] = run
    return parsed


def label(run_id: str) -> str:
    if run_id in RUN_LABELS:
        return RUN_LABELS[run_id]
    if "hitlrexpm0p5" in run_id:
        return "BF80 hit LR exponent -0.5"
    if "hitlrexpm025" in run_id:
        return "BF80 hit LR exponent -0.25"
    if "hitlrexp0p25" in run_id:
        return "BF80 hit LR exponent +0.25"
    if "hitlrexp0_" in run_id:
        return "BF80 hit LR exponent 0.0"
    if "smoke_finalsmear_bf80_beststack" in run_id:
        return "BF80 +Final-Smear smoke"
    if "smoke_snoo_bf80_beststack" in run_id:
        return "BF80 +Snoo smoke"
    if "smoke_update_smoothing_bf80_beststack" in run_id:
        return "BF80 +Update-Smoothing smoke"
    if CURRENT_SUITE in run_id:
        match = BF_RE.search(run_id)
        return f"Current best-stack BF{match.group(1) if match else '?'}"
    if run_id.startswith(OLD_SUITE_PREFIX):
        match = BF_RE.search(run_id)
        return f"Old positive-hit BF{match.group(1) if match else '?'}"
    return run_id


def bf(run_id: str) -> int | None:
    match = BF_RE.search(run_id)
    return int(match.group(1)) if match else None


def interesting_runs(runs: dict[str, dict]) -> list[dict]:
    selected = []
    for run_id, run in runs.items():
        if (
            run_id in RUN_LABELS
            or CURRENT_SUITE in run_id
            or "hitlrexp" in run_id
            or "smoke_finalsmear_bf80_beststack" in run_id
            or "smoke_snoo_bf80_beststack" in run_id
            or "smoke_update_smoothing_bf80_beststack" in run_id
        ):
            selected.append(run)
    selected.sort(key=lambda r: (999 if r["final"] is None else r["final"], r["run_id"]))
    return selected


def hit_lr_exponent(run_id: str) -> float | None:
    if "hitlrexpm0p5" in run_id:
        return -0.5
    if "hitlrexpm025" in run_id:
        return -0.25
    if "hitlrexp0p25" in run_id:
        return 0.25
    if "hitlrexp0_" in run_id:
        return 0.0
    return None


def hit_lr_runs(runs: dict[str, dict]) -> list[dict]:
    rows = [run for run_id, run in runs.items() if hit_lr_exponent(run_id) is not None]
    rows.sort(key=lambda run: (hit_lr_exponent(run["run_id"]) or 0.0, run["run_id"]))
    return rows


def derisk_runs(runs: dict[str, dict]) -> list[dict]:
    needles = ("smoke_finalsmear_bf80_beststack", "smoke_snoo_bf80_beststack", "smoke_update_smoothing_bf80_beststack")
    rows = [run for run_id, run in runs.items() if any(n in run_id for n in needles)]
    rows.sort(key=lambda run: run["run_id"])
    return rows


def old_suite_runs(runs: dict[str, dict]) -> dict[int, dict]:
    out = {}
    for run_id, run in runs.items():
        if run_id.startswith(OLD_SUITE_PREFIX) and OLD_SUITE_SUFFIX in run_id:
            b = bf(run_id)
            if b in {5, 10, 20, 40, 80} and run["status"] == "complete":
                out[b] = run
    return out


def current_suite_runs(runs: dict[str, dict]) -> dict[int, dict]:
    out = {}
    for run_id, run in runs.items():
        if CURRENT_SUITE in run_id:
            b = bf(run_id)
            if b is not None:
                out[b] = run
    return out


def latest_metrics(run: dict) -> dict[str, float]:
    for point in reversed(run["points"]):
        if point["metrics"]:
            return point["metrics"]
    return {}


def render_leaderboard(runs: list[dict]) -> str:
    rows = []
    for run in runs:
        last = run["last"] or {}
        metrics = latest_metrics(run)
        rows.append(
            "<tr>"
            f"<td>{esc(label(run['run_id']))}<small><code>{esc(run['run_id'])}</code></small></td>"
            f"<td>{esc(run['status'])}</td>"
            f"<td>{last.get('step', 'n/a')}/{last.get('total', 'n/a')}</td>"
            f"<td>{fmt(run['final'])}</td>"
            f"<td>{fmt(metrics.get('engram_attnres_p_mean'), 3)}</td>"
            f"<td>{fmt(metrics.get('engram_param_rms'), 3)}</td>"
            f"<td>{fmt(metrics.get('engram_hit_lr_scale_mean'), 2)}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>run</th><th>status</th><th>step</th><th>latest loss</th><th>AttnRes p</th><th>param rms</th><th>hit LR mean</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def run_table(runs: list[dict]) -> str:
    rows = []
    for run in runs:
        last = run["last"] or {}
        cfg = run.get("config", {})
        metrics = latest_metrics(run)
        rows.append(
            "<tr>"
            f"<td>{esc(label(run['run_id']))}<small><code>{esc(run['run_id'])}</code></small></td>"
            f"<td>{esc(run['status'])}</td>"
            f"<td>{last.get('step', 'n/a')}/{last.get('total', 'n/a')}</td>"
            f"<td>{fmt(run['final'])}</td>"
            f"<td>{esc(cfg.get('engram_sparse_hit_lr_exponent', ''))}</td>"
            f"<td>{esc(cfg.get('normuon_update_smoothing', ''))}</td>"
            f"<td>{fmt(metrics.get('engram_update_grad'), 3)}</td>"
            f"<td>{fmt(metrics.get('engram_param_rms'), 3)}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>run</th><th>status</th><th>step</th><th>loss</th><th>hit exp</th><th>smooth</th><th>update/grad</th><th>param rms</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def scale_chart(old: dict[int, dict], current: dict[int, dict]) -> str:
    points = []
    for suite, rows in [("old", old), ("current", current)]:
        for b, run in rows.items():
            if run["final"] is not None:
                points.append((suite, b, run["final"], run["status"]))
    if not points:
        return "<p>No scaling points available.</p>"
    min_b = min(b for _, b, _, _ in points)
    max_b = max(b for _, b, _, _ in points)
    min_y = min(y for _, _, y, _ in points) - 0.006
    max_y = max(y for _, _, y, _ in points) + 0.006
    x0, y0, w, h = 72, 28, 760, 285
    def xy(b: int, y: float) -> tuple[float, float]:
        lx = (math.log(b) - math.log(min_b)) / max(1e-9, math.log(max_b) - math.log(min_b))
        yy = 1 - (y - min_y) / max(1e-9, max_y - min_y)
        return x0 + w * lx, y0 + h * yy
    grid = []
    for frac in [0, .25, .5, .75, 1]:
        y = y0 + h * frac
        val = max_y - (max_y - min_y) * frac
        grid.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+w}" y2="{y:.1f}"></line><text x="16" y="{y+4:.1f}">{val:.3f}</text>')
    for b in sorted({b for _, b, _, _ in points}):
        x, _ = xy(b, min_y)
        grid.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+h}"></line><text x="{x-10:.1f}" y="{y0+h+24}">{b}</text>')
    series = []
    legend = []
    for suite, color, dash in [("old", "#64748b", "6 4"), ("current", "#2563eb", "")]:
        rows = sorted((b, y, status) for s, b, y, status in points if s == suite)
        coords = " ".join(f"{xy(b, y)[0]:.1f},{xy(b, y)[1]:.1f}" for b, y, _ in rows)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        series.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.7"{dash_attr} points="{coords}"></polyline>')
        for b, y, status in rows:
            px, py = xy(b, y)
            opacity = "0.55" if status != "complete" else "1"
            series.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{color}" opacity="{opacity}"><title>{suite} BF{b}: {y:.4f} ({status})</title></circle>')
        legend.append(f'<span><i style="border-top:3px {"dashed" if dash else "solid"} {color}"></i>{suite}</span>')
    return f"""
    <div class="legend">{''.join(legend)}</div>
    <svg viewBox="0 0 880 360" role="img" aria-label="BF scaling curve">
      <g class="grid">{''.join(grid)}</g>
      <text x="360" y="354">bigram factor, log scale</text>
      <text x="12" y="18">val loss</text>
      {''.join(series)}
    </svg>
    """


def scaling_table(old: dict[int, dict], current: dict[int, dict]) -> str:
    rows = []
    for b in sorted(set(old) | set(current)):
        old_loss = old.get(b, {}).get("final")
        cur_loss = current.get(b, {}).get("final")
        delta = cur_loss - old_loss if old_loss is not None and cur_loss is not None else None
        status = current.get(b, {}).get("status", "")
        rows.append(
            "<tr>"
            f"<td>BF{b}</td><td>{fmt(old_loss)}</td><td>{fmt(cur_loss)}</td><td>{fmt(delta)}</td><td>{esc(status)}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>BF</th><th>old suite</th><th>current best-stack</th><th>current-old</th><th>current status</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def load_counterfactuals(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("eval_mask*_step001500.json")):
        data = json.loads(path.read_text())
        name = path.name.removesuffix("_step001500.json").removeprefix("eval_mask_")
        rows.append({"name": name, "path": path, "loss": float(data["val_loss"])})
    return rows


def load_mask_meta(root: Path) -> dict[str, dict]:
    path = root / "engram_hit_count_mask_hists_step001500.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {item["label"]: item for item in data.get("outputs", [])}


def counterfactual_table(title: str, root: Path, base_loss: float) -> str:
    rows = load_counterfactuals(root)
    meta = load_mask_meta(root)
    order = ["unhit", "hit_lt_64", "hit_lt_256", "hit_ge_1024", "hit_ge_256", "hit_ge_64", "all", "all_memory"]
    ordered = [r for name in order for r in rows if r["name"] == name]
    ordered += [r for r in rows if r not in ordered and ("punct" in r["name"] or "semantic" in r["name"])][:10]
    trs = []
    for r in ordered:
        m = meta.get(r["name"], {})
        detail = ""
        if m:
            detail = f"{100*m.get('masked_row_fraction', 0):.1f}% rows, {100*m.get('masked_hit_fraction', 0):.1f}% hits"
        trs.append(
            "<tr>"
            f"<td><code>{esc(r['name'])}</code><small>{esc(detail)}</small></td>"
            f"<td>{fmt(r['loss'], 6)}</td>"
            f"<td>{r['loss'] - base_loss:+.6f}</td>"
            "</tr>"
        )
    return f"<h3>{esc(title)}</h3><table><thead><tr><th>mask</th><th>loss</th><th>delta vs run</th></tr></thead><tbody>{''.join(trs)}</tbody></table>"


def hit_distribution_summary() -> str:
    path = REPORTS / "engram_hit_dist" / "engram_hit_count_distribution_step001500.json"
    if not path.exists():
        return "<p>No hit distribution artifact found.</p>"
    data = json.loads(path.read_text())
    qs = dict(zip(data.get("quantile_probs", []), data.get("all_row_quantiles", [])))
    return (
        "<ul>"
        f"<li>Rows: <b>{data['rows']:,}</b>; ever hit: <b>{100*data['frac_ever_hit']:.3f}%</b>; hit more than once: <b>{100*data['frac_hit_gt1']:.3f}%</b>.</li>"
        f"<li>Total training hits: <b>{data['total_hits']:,}</b>; mean per row: <b>{data['mean_hits_per_row']:.1f}</b>; max row hit count: <b>{data['max_hit']:,}</b>.</li>"
        f"<li>Hit count quantiles: p50 <b>{qs.get(0.5, 'n/a')}</b>, p90 <b>{qs.get(0.9, 'n/a')}</b>, p99 <b>{qs.get(0.99, 'n/a')}</b>, p99.9 <b>{qs.get(0.999, 'n/a')}</b>.</li>"
        "</ul>"
    )


def hot_rows_summary() -> str:
    path = REPORTS / "engram_hot_rows" / "engram_analysis_step001500_top1000_top_slots.csv"
    if not path.exists():
        return "<p>No hot-row analysis artifact found.</p>"
    rows = []
    for line in path.read_text(errors="replace").splitlines()[1:16]:
        parts = line.split(",", 8)
        if len(parts) != 9:
            continue
        head_idx, ngram, hash_head, row, absolute_row, count, gate, norm, text = parts
        rows.append(
            "<tr>"
            f"<td>{esc(absolute_row)}</td><td>{esc(ngram)}g/h{esc(hash_head)}</td><td>{esc(count)}</td><td>{float(gate):.3f}</td><td>{float(norm):.0f}</td><td><code>{esc(text)}</code></td>"
            "</tr>"
        )
    return "<table><thead><tr><th>row</th><th>bucket</th><th>analysis hits</th><th>gate</th><th>gated norm</th><th>example ngram</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def curve_chart(runs: list[dict]) -> str:
    points = [p for run in runs for p in run["points"] if p["step"] > 0]
    if not points:
        return ""
    max_step = max(p["total"] for p in points)
    losses = [p["loss"] for p in points]
    min_y = min(losses) - 0.03
    max_y = max(losses) + 0.03
    x0, y0, w, h = 72, 28, 760, 285
    colors = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#d97706", "#0891b2", "#be185d"]
    grid = []
    for frac in [0, .25, .5, .75, 1]:
        y = y0 + h * frac
        val = max_y - (max_y - min_y) * frac
        grid.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+w}" y2="{y:.1f}"></line><text x="16" y="{y+4:.1f}">{val:.2f}</text>')
    for frac in [0, .25, .5, .75, 1]:
        x = x0 + w * frac
        grid.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+h}"></line><text x="{x-12:.1f}" y="{y0+h+24}">{int(max_step*frac)}</text>')
    series = []
    legend = []
    for i, run in enumerate(runs):
        pts = [p for p in run["points"] if p["step"] > 0]
        if not pts:
            continue
        color = colors[i % len(colors)]
        coords = []
        for p in pts:
            x = x0 + w * p["step"] / max_step
            y = y0 + h * (1 - (p["loss"] - min_y) / max(1e-9, max_y - min_y))
            coords.append(f"{x:.1f},{y:.1f}")
            series.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.7" fill="{color}"><title>{esc(label(run["run_id"]))} step {p["step"]}: {p["loss"]:.4f}</title></circle>')
        series.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(coords)}"></polyline>')
        legend.append(f'<span><i style="border-top:3px solid {color}"></i>{esc(label(run["run_id"]))}</span>')
    return f"<div class='legend'>{''.join(legend)}</div><svg viewBox='0 0 880 360'><g class='grid'>{''.join(grid)}</g>{''.join(series)}<text x='360' y='354'>optimizer step</text><text x='12' y='18'>val loss</text></svg>"


def main() -> None:
    runs = all_runs()
    old = old_suite_runs(runs)
    current = current_suite_runs(runs)
    selected = interesting_runs(runs)[:16]
    hitlr = hit_lr_runs(runs)
    derisk = derisk_runs(runs)
    base_std = runs.get("engram_bf99_attnres_poshit_normreadout_canon_shortconv_hist_ckpt500_20260517_003531", {}).get("final", 3.2589)
    base_low = runs.get("engram_bf99_attnres_poshit_normreadout_canon_shortconv_init1e2_hist_ckpt500_20260517_031953", {}).get("final", 3.2563)
    REPORTS.mkdir(exist_ok=True)
    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Engram Wait-Time Analysis</title>
  <style>
    body {{ font: 14px/1.45 -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 28px; color: #172033; background: #f8fafc; }}
    h1, h2, h3 {{ color: #0f172a; }}
    section {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px 20px; margin: 18px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f1f5f9; color: #334155; }}
    td small {{ display: block; color: #64748b; margin-top: 2px; }}
    code {{ background: #f1f5f9; padding: 1px 4px; border-radius: 4px; }}
    .grid line {{ stroke: #e2e8f0; }}
    .grid text, svg text {{ fill: #64748b; font-size: 12px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 8px 18px; margin: 8px 0 4px; }}
    .legend span {{ display: inline-flex; gap: 6px; align-items: center; color: #334155; }}
    .legend i {{ display: inline-block; width: 26px; height: 0; }}
    .note {{ color: #475569; }}
    .good {{ color: #166534; font-weight: 600; }}
    .warn {{ color: #9a3412; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>Engram Wait-Time Analysis</h1>
  <p class="note">Generated from local logs and existing analysis artifacts. Partial current-suite points are intentionally included but marked as partial.</p>

  <section>
    <h2>Decision Summary</h2>
    <ul>
      <li>The current best-stack sweep is already clearly above the old positive-hit suite at BF5/10/20: roughly 0.016 to 0.018 lower loss at the same BF.</li>
      <li>The best full run remains <b>{fmt(runs.get('engram_bf99_attnres_poshit_normreadout_canon_layers2_8_dim768_h6_ng3_ga8_compilebody_1500_20260516_235031', {}).get('final'))}</b>, with low-init hist close at <b>{fmt(base_low)}</b>.</li>
      <li>Hit distribution is extremely saturated at BF99: almost every row is touched, so unhit masking is mostly a guardrail, not a large effect.</li>
      <li>Counterfactual masks say the common/high-hit rows matter disproportionately; masking low-hit rows is mostly harmless until very aggressive thresholds.</li>
    </ul>
  </section>

  <section>
    <h2>Recent Leaderboard</h2>
    {render_leaderboard(selected)}
  </section>

  <section>
    <h2>Hit LR Exponent Probes</h2>
    <p class="note">BF80, current best-stack config, 500-step probes. Later probes include train-loss logging; hit-bucket moment stats are available only for runs launched after the bucket-stat patch.</p>
    {run_table(hitlr)}
    {curve_chart(hitlr[-8:])}
  </section>

  <section>
    <h2>Feature Derisk Smokes</h2>
    <p class="note">BF80, 80-step smokes for incremental features before spending a full 1500-step run.</p>
    {run_table(derisk)}
    {curve_chart(derisk)}
  </section>

  <section>
    <h2>Scaling: Old Suite vs Current Best Stack</h2>
    {scale_chart(old, current)}
    {scaling_table(old, current)}
  </section>

  <section>
    <h2>Validation Curves</h2>
    {curve_chart(selected[:8])}
  </section>

  <section>
    <h2>Hit Distribution</h2>
    {hit_distribution_summary()}
  </section>

  <section>
    <h2>Counterfactual Masks</h2>
    {counterfactual_table('Std=1 AttnRes hist run', REPORTS / 'engram_counterfactuals', float(base_std))}
    {counterfactual_table('Low-init AttnRes hist run', REPORTS / 'engram_init_counterfactuals', float(base_low))}
  </section>

  <section>
    <h2>Top Gated Memory Rows</h2>
    <p class="note">These are dominated by formatting/document-boundary grams, not factual long-tail semantics. The semantic-ish filtered list has examples, but its mask effect was tiny relative to broad hit-count masks.</p>
    {hot_rows_summary()}
  </section>
</body>
</html>
"""
    OUT.write_text(html_doc)
    print(OUT)


if __name__ == "__main__":
    main()
