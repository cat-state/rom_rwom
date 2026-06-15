#!/usr/bin/env python3
import html
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIRS = [ROOT / "logs", ROOT / "tmp_remote_logs"]
OUT = ROOT / "reports" / "engram_mhc_report.html"
PROMPT_REPORT_DIR = ROOT / "reports" / "engram_prompt_gating"


RUNS = [
    {
        "name": "Old BF99 inverse-hit best",
        "run_id": "engram_bf99_hitlr_sqrt_adamevery_lr6p35_nofloor_layers2_8_layerhash_readouts_dim768_h6_ng3_ga8_1500_20260515_024954",
        "kind": "reference",
        "params_b": 3.82,
        "note": "Best known full Engram baseline before corrected positive hit-count scaling.",
    },
    {
        "name": "BF99 sqrt hit LR, uncapped",
        "run_id": "engram_bf99_poshitlr_sqrt_adamevery_lr6p35_nofloor_layers2_8_layerhash_readouts_dim768_h6_ng3_ga8_1500_20260516_035123",
        "kind": "hit-lr",
        "params_b": 3.82,
        "note": "Correct positive sqrt(hit_count) scaling, no max scale cap.",
    },
    {
        "name": "BF99 sqrt hit LR, cap 100",
        "run_id": "engram_bf99_poshitlr_sqrt_cap100_adamevery_lr6p35_nofloor_layers2_8_layerhash_readouts_dim768_h6_ng3_ga8_1500_20260516_033153",
        "kind": "hit-lr",
        "params_b": 3.82,
        "note": "Same as uncapped positive sqrt, clamped to max LR scale 100.",
    },
]


BF99_PARAMS_B = 3.824764928


def infer_params_b(run_id: str) -> float:
    match = re.search(r"bf(\d+)", run_id.lower())
    if not match:
        return 0.0
    return BF99_PARAMS_B * int(match.group(1)) / 99


def discover_runs():
    runs = list(RUNS)
    known = {run["run_id"] for run in runs}
    patterns = [
        "*mhc*.console.txt",
        "*mhc*.txt",
        "*cache*.console.txt",
        "*cache*.txt",
        "*cfg*.console.txt",
        "*cfg*.txt",
        "*scale*.console.txt",
        "*scale*.txt",
        "*hitlr*.console.txt",
        "*hitlr*.txt",
        "*finalsmear*.console.txt",
        "*finalsmear*.txt",
        "*rowrmscap*.console.txt",
        "*rowrmscap*.txt",
        "*attnres*.console.txt",
        "*attnres*.txt",
        "*layerpart*.console.txt",
        "*layerpart*.txt",
        "*normmem*.console.txt",
        "*normmem*.txt",
        "*novelty*.console.txt",
        "*novelty*.txt",
        "*rom*.console.txt",
        "*rom*.txt",
        "*diagnostic*.console.txt",
        "*diagnostic*.txt",
        "*full_h*.console.txt",
        "*full_h*.txt",
    ]
    for log_dir in LOG_DIRS:
        if not log_dir.exists():
            continue
        discovered = sorted({path for pattern in patterns for path in log_dir.glob(pattern)})
        for path in discovered:
            run_id = path.name.removesuffix(".console.txt")
            if run_id in known:
                continue
            known.add(run_id)
            lower = run_id.lower()
            if "scale" in lower:
                kind = "scale"
                note = "Auto-discovered scaling run."
            elif "rowrmscap" in lower:
                kind = "row-rms-cap"
                note = "Auto-discovered row RMS cap run."
            elif "finalsmear" in lower:
                kind = "final-smear"
                note = "Auto-discovered final smear run."
            elif "attnres" in lower:
                kind = "attnres"
                note = "Auto-discovered attention-residual run."
            elif "rom" in lower:
                kind = "rom"
                note = "Auto-discovered ROM memory run."
            elif "novelty" in lower:
                kind = "attnres"
                note = "Auto-discovered novelty-metric run."
            elif "hitlr" in lower:
                kind = "hit-lr"
                note = "Auto-discovered hit LR run."
            else:
                kind = "mhc/cache"
                note = "Auto-discovered mHC/cache run."
            runs.append({
                "name": run_id,
                "run_id": run_id,
                "kind": kind,
                "params_b": infer_params_b(run_id),
                "note": note,
            })
    return runs


VAL_RE = re.compile(r"step:(\d+)/(\d+) val_loss:([0-9.]+)(.*)")
CONFIG_RE = re.compile(r"Experiment config: (.*)")
KEYVAL_RE = re.compile(r"([a-zA-Z0-9_]+)=([^ ]+)")


def find_log(run_id: str) -> Path | None:
    candidates = []
    for log_dir in LOG_DIRS:
        candidates.extend([log_dir / f"{run_id}.console.txt", log_dir / f"{run_id}.txt"])
    for path in candidates:
        if path.exists():
            return path
    return None


def parse_run(run: dict) -> dict:
    path = find_log(run["run_id"])
    points = []
    config = {}
    status = "missing"
    if path is not None:
        status = "running"
        text = path.read_text(errors="replace")
        if "Traceback" in text or "ChildFailedError" in text or "\nFAILED\n" in text:
            status = "failed"
        if "peak memory allocated" in text:
            status = "complete"
        for line in text.splitlines():
            config_match = CONFIG_RE.search(line)
            if config_match:
                config = dict(KEYVAL_RE.findall(config_match.group(1)))
            val_match = VAL_RE.search(line)
            if val_match:
                step = int(val_match.group(1))
                total = int(val_match.group(2))
                loss = float(val_match.group(3))
                rest = val_match.group(4)
                metrics = {}
                for item in rest.split():
                    if ":" not in item:
                        continue
                    key, value = item.split(":", 1)
                    try:
                        metrics[key] = float(value)
                    except ValueError:
                        pass
                points.append({"step": step, "total": total, "loss": loss, "metrics": metrics})
        if points and points[-1]["step"] >= points[-1]["total"]:
            status = "complete"
        elif points and status == "running":
            status = "partial"
    params_b = run["params_b"]
    for point in reversed(points):
        table_numel = point["metrics"].get("engram_table_numel")
        if table_numel:
            params_b = table_numel / 1e9
            break
    best = min((p["loss"] for p in points), default=None)
    final = points[-1]["loss"] if points else None
    return {**run, "path": path, "status": status, "points": points, "best": best, "final": final, "config": config, "params_b": params_b}


def fmt(x, digits=4):
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def color_for(i: int) -> str:
    colors = [
        "#2563eb", "#d97706", "#16a34a", "#7c3aed", "#dc2626", "#0891b2",
        "#4d7c0f", "#be185d", "#0f766e", "#9333ea", "#ea580c", "#0284c7",
        "#65a30d", "#c026d3", "#b91c1c", "#0d9488", "#4338ca", "#a16207",
        "#047857", "#e11d48",
    ]
    return colors[i % len(colors)]


def dash_for(i: int) -> str:
    patterns = ["", "8 4", "3 3", "10 3 2 3", "2 5", "12 4", "6 2 2 2", "1 4"]
    return patterns[i % len(patterns)]


def marker_shape_for(i: int) -> str:
    return ["circle", "square", "diamond", "triangle", "cross"][i % 5]


def marker_svg(shape: str, x: float, y: float, color: str, size: float = 4.8, title: str | None = None) -> str:
    title_svg = f"<title>{html.escape(title)}</title>" if title else ""
    if shape == "square":
        return f'<rect x="{x-size:.1f}" y="{y-size:.1f}" width="{2*size:.1f}" height="{2*size:.1f}" fill="{color}">{title_svg}</rect>'
    if shape == "diamond":
        points = f"{x:.1f},{y-size:.1f} {x+size:.1f},{y:.1f} {x:.1f},{y+size:.1f} {x-size:.1f},{y:.1f}"
        return f'<polygon points="{points}" fill="{color}">{title_svg}</polygon>'
    if shape == "triangle":
        points = f"{x:.1f},{y-size:.1f} {x+size:.1f},{y+size:.1f} {x-size:.1f},{y+size:.1f}"
        return f'<polygon points="{points}" fill="{color}">{title_svg}</polygon>'
    if shape == "cross":
        return (
            f'<g stroke="{color}" stroke-width="2.2" stroke-linecap="round">{title_svg}'
            f'<line x1="{x-size:.1f}" y1="{y-size:.1f}" x2="{x+size:.1f}" y2="{y+size:.1f}"></line>'
            f'<line x1="{x-size:.1f}" y1="{y+size:.1f}" x2="{x+size:.1f}" y2="{y-size:.1f}"></line>'
            '</g>'
        )
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size:.1f}" fill="{color}">{title_svg}</circle>'


def make_polyline(points, x0, y0, w, h, max_step, min_loss, max_loss):
    coords = []
    for point in points:
        x = x0 + w * (point["step"] / max(max_step, 1))
        if math.isclose(max_loss, min_loss):
            y = y0 + h / 2
        else:
            y = y0 + h * (1 - (point["loss"] - min_loss) / (max_loss - min_loss))
        coords.append(f"{x:.1f},{y:.1f}")
    return " ".join(coords)


def point_xy(point, x0, y0, w, h, max_step, min_loss, max_loss):
    x = x0 + w * (point["step"] / max(max_step, 1))
    if math.isclose(max_loss, min_loss):
        y = y0 + h / 2
    else:
        y = y0 + h * (1 - (point["loss"] - min_loss) / (max_loss - min_loss))
    return x, y


def render_curve(runs):
    all_points = [p for run in runs for p in run["points"] if p["step"] > 0]
    if not all_points:
        return "<p>No validation points beyond step 0 yet.</p>"
    max_step = max(p["total"] for run in runs for p in run["points"])
    losses = [p["loss"] for p in all_points]
    min_loss = min(losses) - 0.03
    max_loss = max(losses) + 0.03
    x0, y0, w, h = 70, 32, 800, 280
    grid = []
    for frac in [0, .25, .5, .75, 1]:
        y = y0 + h * frac
        loss = max_loss - (max_loss - min_loss) * frac
        grid.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+w}" y2="{y:.1f}"></line>')
        grid.append(f'<text x="16" y="{y+4:.1f}">{loss:.2f}</text>')
    for frac in [0, .25, .5, .75, 1]:
        x = x0 + w * frac
        step = int(max_step * frac)
        grid.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+h}"></line>')
        grid.append(f'<text x="{x-12:.1f}" y="{y0+h+24}">{step}</text>')
    lines = []
    legend = []
    for i, run in enumerate(runs):
        pts = [p for p in run["points"] if p["step"] > 0]
        if not pts:
            continue
        color = color_for(i)
        dash = dash_for(i)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        shape = marker_shape_for(i)
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.6"{dash_attr} points="{make_polyline(pts, x0, y0, w, h, max_step, min_loss, max_loss)}"></polyline>')
        for point in pts:
            px, py = point_xy(point, x0, y0, w, h, max_step, min_loss, max_loss)
            lines.append(marker_svg(shape, px, py, color, size=3.8, title=f"{run['name']} step {point['step']}: {point['loss']:.4f}"))
        dash_css = f"border-top:3px {'dashed' if dash else 'solid'} {color};"
        legend.append(f'<span><i style="{dash_css}"></i><b class="marker marker-{shape}" style="color:{color}"></b>{html.escape(run["name"])}</span>')
    return f"""
    <svg viewBox="0 0 900 370" role="img" aria-label="Engram mHC validation loss curves">
      <rect width="900" height="370" fill="#fff"></rect>
      <g class="gridlines">{''.join(grid)}</g>
      {''.join(lines)}
      <text x="390" y="356">optimizer steps</text>
      <text x="12" y="22">val loss</text>
    </svg>
    <div class="legend">{''.join(legend)}</div>
    """


def render_size_curve(runs):
    points = []
    for run in runs:
        if run["status"] != "complete" or run["final"] is None or run["params_b"] <= 0:
            continue
        points.append((run["params_b"], run["final"], run["name"]))
    if len(points) < 2:
        return "<p>Need at least two completed table-size points before this plot is meaningful.</p>"
    x_min = min(p[0] for p in points)
    x_max = max(p[0] for p in points)
    y_min = min(p[1] for p in points) - 0.01
    y_max = max(p[1] for p in points) + 0.01
    x0, y0, w, h = 70, 28, 800, 260

    def x_pos(value):
        if math.isclose(x_min, x_max):
            return x0 + w / 2
        return x0 + w * ((math.log10(value) - math.log10(x_min)) / (math.log10(x_max) - math.log10(x_min)))

    def y_pos(value):
        if math.isclose(y_min, y_max):
            return y0 + h / 2
        return y0 + h * (1 - (value - y_min) / (y_max - y_min))

    grid = []
    for frac in [0, .25, .5, .75, 1]:
        y = y0 + h * frac
        loss = y_max - (y_max - y_min) * frac
        grid.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+w}" y2="{y:.1f}"></line>')
        grid.append(f'<text x="16" y="{y+4:.1f}">{loss:.2f}</text>')
    for value in sorted({p[0] for p in points}):
        x = x_pos(value)
        grid.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+h}"></line>')
        grid.append(f'<text x="{x-18:.1f}" y="{y0+h+24}">{value:.2g}B</text>')

    dots = []
    for i, (params_b, loss, name) in enumerate(points):
        color = color_for(i)
        dots.append(marker_svg(
            marker_shape_for(i),
            x_pos(params_b),
            y_pos(loss),
            color,
            size=5.4,
            title=f"{name}: {params_b:.2f}B, {loss:.4f}",
        ))
    return f"""
    <svg viewBox="0 0 900 340" role="img" aria-label="Final validation loss versus Engram table size">
      <rect width="900" height="340" fill="#fff"></rect>
      <g class="gridlines">{''.join(grid)}</g>
      {''.join(dots)}
      <text x="360" y="326">Engram table parameters, log scale</text>
      <text x="12" y="20">final val loss</text>
    </svg>
    """


def find_final(runs, needle):
    return next((r["final"] for r in runs if needle in r["run_id"]), None)


def render_takeaways(runs):
    scale_runs = sorted(
        [r for r in runs if "scale_bf" in r["run_id"] and r["status"] == "complete" and r["final"] is not None],
        key=lambda r: r["params_b"],
    )
    items = []
    if scale_runs:
        best_scale = min(scale_runs, key=lambda r: r["final"])
        bf40 = next((r for r in scale_runs if "scale_bf40_" in r["run_id"]), None)
        bf80 = next((r for r in scale_runs if "scale_bf80_" in r["run_id"]), None)
        items.append(
            f"Scaling suite best: {html.escape(best_scale['name'])} at "
            f"{best_scale['final']:.4f} with {best_scale['params_b']:.2f}B table params."
        )
        if bf40 and bf80:
            diff = bf40["final"] - bf80["final"]
            direction = "better" if diff > 0 else "worse"
            items.append(
                f"BF80 is {abs(diff):.4f} {direction} than BF40 while using roughly 2x table params; "
                "this is a very shallow gain, not a decisive scale break."
            )
    mhc_final = find_final(runs, "mhc_inversehit")
    if mhc_final is not None:
        items.append(f"The tested mHC mixer underperformed the additive Engram baseline, ending at {mhc_final:.4f}.")
    cache_final = find_final(runs, "cache_cosine_detached_w0p01_learncfg0")
    if cache_final is not None:
        items.append(f"The learned-CFG cache pilot ended at {cache_final:.4f}; the CE path learned to downweight/subtract it.")
    if not items:
        return "<p>No completed takeaways yet.</p>"
    return "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"


def render_prompt_reports():
    if not PROMPT_REPORT_DIR.exists():
        return "<p>No prompt gating reports found yet.</p>"
    reports = sorted(PROMPT_REPORT_DIR.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)[:8]
    if not reports:
        return "<p>No prompt gating reports found yet.</p>"
    rows = []
    for path in reports:
        rel = path.relative_to(ROOT)
        rows.append(
            "<tr>"
            f"<td><a href=\"{html.escape(str(rel))}\">{html.escape(path.stem)}</a></td>"
            f"<td>{path.stat().st_size / 1024:.1f} KiB</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Prompt Report</th><th>Size</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render():
    runs = [parse_run(run) for run in discover_runs()]
    completed = [r for r in runs if r["final"] is not None]
    best = min(completed, key=lambda r: r["final"]) if completed else None
    rows = []
    for run in runs:
        last = run["points"][-1] if run["points"] else None
        hit_max = last["metrics"].get("engram_hit_lr_scale_max") if last else None
        mhc_gate = last["metrics"].get("mhc_gate_mean") if last else None
        cache_recon = last["metrics"].get("cache_recon_loss") if last else None
        cache_cfg = last["metrics"].get("cache_cfg_l2") if last else None
        if cache_cfg is None and last is not None:
            cache_cfg = last["metrics"].get("cache_cfg_mean")
        path = str(run["path"].relative_to(ROOT)) if run["path"] else "missing"
        rows.append(
            "<tr>"
            f"<td>{html.escape(run['name'])}</td>"
            f"<td>{html.escape(run['status'])}</td>"
            f"<td>{last['step'] if last else 'n/a'}</td>"
            f"<td>{fmt(run['final'])}</td>"
            f"<td>{fmt(run['best'])}</td>"
            f"<td>{run['params_b']:.2f}B</td>"
            f"<td>{fmt(hit_max, 1)}</td>"
            f"<td>{fmt(mhc_gate, 3)}</td>"
            f"<td>{fmt(cache_recon, 4)}</td>"
            f"<td>{fmt(cache_cfg, 3)}</td>"
            f"<td><code>{html.escape(path)}</code></td>"
            f"<td class=\"note\">{html.escape(run['note'])}</td>"
            "</tr>"
        )
    best_text = "n/a" if best is None else f"{best['name']} at {best['final']:.4f}"
    mhc_full = find_final(runs, "mhc_inversehit")
    detached_cache = find_final(runs, "cache_cosine_detached_w0p01_cfg0")
    learned_cfg = find_final(runs, "cache_cosine_detached_w0p01_learncfg0")
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Engram + mHC Report</title>
  <style>
    :root {{ --bg:#f7f8f7; --ink:#171717; --muted:#5f6660; --line:#d9ded8; --panel:#fff; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ padding:28px 36px 18px; background:#fff; border-bottom:1px solid var(--line); }}
    main {{ max-width:1180px; margin:0 auto; padding:24px 24px 48px; }}
    h1,h2 {{ margin:0; line-height:1.15; letter-spacing:0; }}
    h1 {{ font-size:30px; }}
    h2 {{ font-size:19px; margin:28px 0 12px; }}
    p {{ color:var(--muted); }}
    .cards {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; }}
    .label {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
    .metric {{ font-size:25px; font-weight:750; margin-top:5px; }}
    table {{ width:100%; border-collapse:collapse; background:#fff; border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
    th,td {{ text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); white-space:nowrap; vertical-align:top; }}
    th {{ background:#eef2ee; color:#394039; }}
    tr:last-child td {{ border-bottom:0; }}
    .note {{ white-space:normal; min-width:260px; }}
    code {{ background:#edf0ed; padding:1px 4px; border-radius:4px; }}
    a {{ color:#1d4ed8; text-decoration:none; }}
    svg {{ width:100%; height:auto; background:#fff; border:1px solid var(--line); border-radius:8px; }}
    .gridlines line {{ stroke:#e5e9e4; }}
    .gridlines text, svg text {{ fill:#626962; font-size:12px; }}
    .legend {{ display:flex; gap:16px; flex-wrap:wrap; margin-top:10px; color:var(--muted); }}
    .legend i {{ display:inline-block; width:16px; height:0; border-radius:2px; margin-right:5px; vertical-align:middle; }}
    .legend .marker {{ display:inline-block; width:10px; height:10px; margin-right:6px; vertical-align:-1px; position:relative; }}
    .marker-circle {{ border-radius:50%; background:currentColor; }}
    .marker-square {{ background:currentColor; }}
    .marker-diamond {{ background:currentColor; transform:rotate(45deg) scale(.82); }}
    .marker-triangle {{ width:0!important; height:0!important; border-left:5px solid transparent; border-right:5px solid transparent; border-bottom:10px solid currentColor; }}
    .marker-cross::before,.marker-cross::after {{ content:""; position:absolute; left:4px; top:-1px; width:2px; height:12px; background:currentColor; border-radius:1px; }}
    .marker-cross::before {{ transform:rotate(45deg); }}
    .marker-cross::after {{ transform:rotate(-45deg); }}
    @media (max-width:860px) {{ header {{ padding:24px 18px; }} main {{ padding:18px 12px 36px; }} .cards {{ grid-template-columns:1fr; }} th,td {{ white-space:normal; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Engram + mHC Report</h1>
    <p>Updated from local logs. Current best in this report: {html.escape(best_text)}.</p>
  </header>
  <main>
    <section class="cards">
      <div class="card"><div class="label">Best Result</div><div class="metric">{fmt(best['final'] if best else None)}</div><p>{html.escape(best['name'] if best else 'No completed runs yet.')}</p></div>
      <div class="card"><div class="label">mHC Full Run</div><div class="metric">{fmt(mhc_full)}</div><p>Two-lane constrained mixer on the BF99 inverse-hit baseline.</p></div>
      <div class="card"><div class="label">Detached Cache Recon</div><div class="metric">{fmt(detached_cache)}</div><p>Cache target is learnable when detached, but not useful for CE yet.</p></div>
      <div class="card"><div class="label">Learned CFG Pilot</div><div class="metric">{fmt(learned_cfg)}</div><p>Cache CFG scalar starts at zero and is optimized as a replicated Adam parameter.</p></div>
    </section>
    <h2>Takeaways</h2>
    {render_takeaways(runs)}
    <h2>Validation Curves</h2>
    {render_curve(runs)}
    <h2>Table Size Scaling</h2>
    {render_size_curve(runs)}
    <h2>Prompt Gating</h2>
    {render_prompt_reports()}
    <h2>Run Table</h2>
    <table>
      <thead><tr><th>Run</th><th>Status</th><th>Last step</th><th>Final/last loss</th><th>Best loss</th><th>Table params</th><th>Hit LR max</th><th>mHC gate</th><th>Cache loss</th><th>CFG</th><th>Log</th><th>Note</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    <h2>Implementation Notes</h2>
    <p>The current reproducible default runner is <code>scripts/run_engram_best_bf80.sh</code>, which launches the best completed additive Engram configuration from this report.</p>
    <p>The first mHC path keeps Engram disabled by default. With <code>ENGRAM_MHC=1</code>, a second residual lane is mixed with the main lane using a symmetric doubly-stochastic 2x2 mixer <code>[[1-a,a],[a,1-a]]</code>. The BF99 full run and the BF80 direct baseline test both underperformed additive Engram after early checkpoints, so the current cache line separates future-direction reconstruction from the CE path and uses learned CFG to test whether the model wants that stream.</p>
  </main>
</body>
</html>
"""
    html_doc = "\n".join(line.rstrip() for line in html_doc.splitlines()) + "\n"
    OUT.write_text(html_doc)
    return OUT


if __name__ == "__main__":
    print(render())
