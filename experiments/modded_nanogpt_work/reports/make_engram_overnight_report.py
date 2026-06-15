#!/usr/bin/env python3
from __future__ import annotations

import base64
import html
import io
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
REMOTE_LOGS = ROOT / "tmp_remote_logs"
MIRRORED_LOGS = ROOT / "remote_logs"
OUT = ROOT / "reports" / "engram_overnight_report.html"


ENGRAM_SCALING = {
    5: "bf5_h1_gain1p5_tail1_beststack_1500_20260518_223733.console.txt",
    10: "bf10_h1_gain1p5_tail1_beststack_1500_20260518_223733.console.txt",
    20: "bf20_h1_gain1p5_tail1_beststack_1500_20260518_223749.console.txt",
    40: "bf40_h1_gain1p5_tail1_beststack_1500_20260518_231124.console.txt",
    80: "bf80_h1_gain1p5_tail1_beststack_1500_20260518_231149.console.txt",
}

BUILTIN_SCALING = {
    5: "builtin_bigram_bf5_1500_20260518_234659.console.txt",
    10: "builtin_bigram_bf10_1500_20260518_234659.console.txt",
    20: "builtin_bigram_bf20_1500_20260519_001641.console.txt",
    40: "builtin_bigram_bf40_1500_20260519_001650.console.txt",
    80: "builtin_bigram_bf80_ga16_1500_20260519_000821.console.txt",
}

ACTIVE_RUNS = [
    {
        "name": "BF40 beststack + shadow grad",
        "path": MIRRORED_LOGS / "bf40_beststack_shadow_wrms002_scale005_1500_20260520_0114.txt",
        "hypothesis": "Full-length BF40 best stack with shadow-gradient row writes; tests whether gradient-derived writes add value beyond tail1 sparse Adam.",
    },
    {
        "name": "BF40 shadow grad + extra7 memory Adam",
        "path": MIRRORED_LOGS / "bf40_beststack_shadow_extra7_lrhalf_wrms002_scale005_500_20260520_013514.txt",
        "hypothesis": "Smoke test for whether shadow-gradient writes and multiple Engram-only Adam updates are complementary.",
    },
    {
        "name": "BF40 shadow grad + extra7 memory Adam full",
        "path": MIRRORED_LOGS / "bf40_beststack_shadow_extra7_lrhalf_wrms002_scale005_1500_20260520_015216.txt",
        "hypothesis": "Full-length confirmation of the shadow-gradient plus extra memory Adam combination after the 500-step smoke improved slightly.",
    },
    {
        "name": "Single-token ROM sparse Adam LR2",
        "path": MIRRORED_LOGS / "rom_single_tok_sparseadam_diagfrob001_h1_k8_v32_scale1_lr2_bf40_l2_8_500_20260520_0240.txt",
        "hypothesis": "500-step continuation of the best short single-token ROM setup: sparse Adam, small diagonal/frob init, short conv, AttnRes.",
    },
    {
        "name": "Single-token ROM sparse Adam LR2 full",
        "path": MIRRORED_LOGS / "rom_single_tok_sparseadam_diagfrob001_h1_k8_v32_scale1_lr2_bf40_l2_8_1500_20260520_014300.txt",
        "hypothesis": "Full-length version of the best current single-token ROM setup to test whether its 500-step trajectory cools down competitively.",
    },
    {
        "name": "Single-token ROM sparse Adam LR1",
        "path": MIRRORED_LOGS / "rom_single_tok_sparseadam_diagfrob001_h1_k8_v32_scale1_lr1_uuid_fb1411af_20260520.txt",
        "hypothesis": "Lower-LR bracket after LR2 beat LR5/LR10 in short ROM probes.",
    },
    {
        "name": "Single-token ROM direct normwrite",
        "path": MIRRORED_LOGS / "rom_single_tok_normwrite_direct_diagfrob001_h1_k8_v32_scale1_wrms001_bf40_l2_8_160_20260520_014123.txt",
        "hypothesis": "Direct sparse state-gradient normalized write, contrasting with the recovered memory-vector normalized writes.",
    },
    {
        "name": "Single-token ROM row-scalar Adam",
        "path": MIRRORED_LOGS / "rom_single_tok_rowscalaradam_diagfrob001_h1_k8_v32_scale1_lr2_bf40_l2_8_160_20260520_014846.txt",
        "hypothesis": "Engram-style row-scalar second moment for the ROM state table, testing whether moment geometry matters for single-token memory.",
    },
    {
        "name": "BF80 h2 bank AttnRes",
        "path": REMOTE_LOGS / "bf80_h2_bankattnres_beststack_1500_20260519_033624.console.txt",
        "hypothesis": "Route over two memory banks instead of merging all heads first.",
    },
    {
        "name": "BF80 h4 bank AttnRes",
        "path": REMOTE_LOGS / "bf80_h4_bankattnres_beststack_1500_20260519_0350.console.txt",
        "hypothesis": "More independent memory banks for the AttnRes router.",
    },
    {
        "name": "BF80 h1 layer partitions",
        "path": REMOTE_LOGS / "bf80_h1_layerpart_beststack_1500_20260519_0350.console.txt",
        "hypothesis": "Reduce layer-2/layer-8 row coupling and collisions by partitioning rows by injection layer.",
    },
    {
        "name": "BF80 h4 bank AttnRes + layer partitions",
        "path": REMOTE_LOGS / "bf80_h4_bankattnres_layerpart_beststack_1500_20260519_0458.console.txt",
        "hypothesis": "Combine multi-bank routing with layer-partitioned rows to reduce collisions while keeping bank diversity.",
    },
    {
        "name": "BF120 h1 layer partitions hitLR",
        "path": REMOTE_LOGS / "bf120_h1_layerpart_beststack_ga16_1500_20260519_0558.console.txt",
        "hypothesis": "Scale layer partitions with the hit-LR optimizer path; kept as a contrast to the tail1 winner.",
    },
    {
        "name": "BF120 h1 layer partitions tail1",
        "path": REMOTE_LOGS / "bf120_h1_layerpart_tail1_beststack_ga16_1500_20260519_0710.console.txt",
        "hypothesis": "Scale the actual BF80 layer-partition tail1 winner to a larger table under GA16 memory pressure.",
    },
    {
        "name": "BF40 h2 bank AttnRes store384",
        "path": REMOTE_LOGS / "bf40_h2_bankattnres_store384_beststack_1500_20260519_0610.console.txt",
        "hypothesis": "Trade row count for wider per-bank memory vectors to test whether bank routing was bottlenecked by narrow store dim.",
    },
    {
        "name": "BF80 h1 layer partitions tail1 checkpointed",
        "path": REMOTE_LOGS / "bf80_h1_layerpart_tail1_beststack_ckpt_1500_20260519_0635.console.txt",
        "hypothesis": "Stopped after weak step-1000; attempted checkpointed reproduction for analysis but seed looked worse.",
    },
    {
        "name": "BF80 h1 layer partitions tail1 repro2",
        "path": REMOTE_LOGS / "bf80_h1_layerpart_tail1_repro2_1500_20260519_0730.console.txt",
        "hypothesis": "Repeat the current best layer-partition tail1 setup without checkpointing to estimate run-to-run stability.",
    },
    {
        "name": "BF80 h1 layer partitions tail1 orig schedule",
        "path": REMOTE_LOGS / "bf80_h1_layerpart_tail1_origsched_1500_20260519_0745.console.txt",
        "hypothesis": "Reproduce the current best layer-partition tail1 setup with the original 1460+40 schedule after identifying an LR schedule mismatch.",
    },
    {
        "name": "BF120 h1 layer partitions tail1 orig schedule",
        "path": REMOTE_LOGS / "bf120_h1_layerpart_tail1_origsched_ga16_1500_20260519_0745.console.txt",
        "hypothesis": "Scale the layer-partition tail1 setup under the original 1460+40 schedule.",
    },
    {
        "name": "BF160 h1 layer partitions tail1 orig schedule",
        "path": REMOTE_LOGS / "bf160_h1_layerpart_tail1_origsched_ga16_1500_20260519_0830.console.txt",
        "hypothesis": "Push the corrected layer-partition tail1 scaling one more step to test where the GPU-resident table stops helping or fitting.",
    },
    {
        "name": "BF120 h1 layer partitions tail1 orig schedule checkpointed",
        "path": REMOTE_LOGS / "bf120_h1_layerpart_tail1_origsched_ckpt_1500_20260519_0840.console.txt",
        "hypothesis": "Re-run the current best BF120 setup with checkpoint saving enabled for semantic and counterfactual analysis.",
    },
    {
        "name": "BF140 h1 layer partitions tail1 orig schedule",
        "path": REMOTE_LOGS / "bf140_h1_layerpart_tail1_origsched_ga16_1500_20260519_0640.console.txt",
        "hypothesis": "Probe the GPU-resident scaling boundary between the fitting BF120 run and the OOMing BF160 run.",
    },
    {
        "name": "BF140 h1 layer partitions tail1 ngram=2",
        "path": REMOTE_LOGS / "bf140_h1_layerpart_tail1_origsched_ng2_ga16_1500_20260519_0755.console.txt",
        "hypothesis": "Reallocate the same memory parameter budget to 2-grams only after BF120 gated-row analysis showed top readouts were dominated by 2-grams.",
    },
    {
        "name": "Built-in bigram BF120 GA16",
        "path": REMOTE_LOGS / "builtin_bigram_bf120_ga16_1500_20260519_0735.console.txt",
        "hypothesis": "Check whether simply scaling the speedrun built-in bigram table to BF120 fits and catches the Engram stack.",
    },
    {
        "name": "Built-in bigram BF120 GA32",
        "path": REMOTE_LOGS / "builtin_bigram_bf120_ga32_1500_20260519_0745.console.txt",
        "hypothesis": "Retry the built-in BF120 comparison with higher grad accumulation to reduce activation pressure.",
    },
    {
        "name": "BF120 AttnRes extra L2 source to L8",
        "path": REMOTE_LOGS / "bf120_h1_layerpart_tail1_origsched_attnres_extra2to8_ga16_1500_20260519_0910.console.txt",
        "hypothesis": "At layer 8, route over current residual, Engram readout, and saved layer-2 residual as an attention-residual memory routing test.",
    },
]

SEMANTIC_MD = ROOT / "reports" / "engram_hot_rows" / "bf99_h1_top1000_semanticish_20260519_0357.md"
BF120_HOT_MD = ROOT / "reports" / "engram_hot_rows" / "bf120_h1_top1000_20260519_0720.md"
PROMPT_HTML = ROOT / "reports" / "engram_prompt_gating" / "bf99_h1_promptgate_20260519_0402.html"
BF120_PROMPT_HTML = ROOT / "reports" / "engram_prompt_gating" / "bf120_h1_promptgate_20260519_0745.html"
BF120_COUNTERFACTUAL_DIR = ROOT / "reports" / "engram_bf120_counterfactuals"
BF120_COUNTERFACTUAL_FULL_DIR = ROOT / "reports" / "engram_bf120_counterfactuals_full"
SCALING_HTML = ROOT / "reports" / "engram_vs_builtin_bf_scaling.html"


VAL_RE = re.compile(r"step:(\d+)/(\d+) val_loss:([0-9.]+)(.*)")
TRAIN_RE = re.compile(r"step:(\d+)/(\d+) train_loss:([0-9.]+).*?step_avg:([0-9.]+)ms")
CONFIG_RE = re.compile(r"Experiment config: (.*)")
KEYVAL_RE = re.compile(r"([a-zA-Z0-9_]+)=([^ ]+)")
METRIC_RE = re.compile(r"([a-zA-Z0-9_]+):([^ ]+)")


def parse_log(path: Path) -> dict:
    out = {
        "path": path,
        "exists": path.exists(),
        "status": "missing",
        "vals": [],
        "trains": [],
        "config": {},
        "peak_mib": None,
        "error": "",
    }
    if not path.exists():
        return out
    text = path.read_text(errors="replace")
    if "Traceback" in text or "ChildFailedError" in text or "\nFAILED\n" in text:
        out["status"] = "failed"
    else:
        out["status"] = "running"
    for line in text.splitlines():
        m = CONFIG_RE.search(line)
        if m:
            out["config"] = dict(KEYVAL_RE.findall(m.group(1)))
        m = TRAIN_RE.search(line)
        if m:
            out["trains"].append({
                "step": int(m.group(1)),
                "total": int(m.group(2)),
                "loss": float(m.group(3)),
                "step_avg_ms": float(m.group(4)),
            })
        m = VAL_RE.search(line)
        if m:
            metrics = {}
            for key, value in METRIC_RE.findall(m.group(4)):
                try:
                    metrics[key] = float(value)
                except ValueError:
                    pass
            out["vals"].append({
                "step": int(m.group(1)),
                "total": int(m.group(2)),
                "loss": float(m.group(3)),
                "metrics": metrics,
            })
        if "peak memory allocated:" in line:
            m = re.search(r"peak memory allocated: (\d+) MiB", line)
            if m:
                out["peak_mib"] = int(m.group(1))
                out["status"] = "complete"
    if out["vals"]:
        last = out["vals"][-1]
        if last["step"] >= last["total"]:
            out["status"] = "complete"
        elif out["status"] == "running":
            out["status"] = "partial"
    return out


def svg_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    return "data:image/svg+xml;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def last_train(run: dict) -> dict | None:
    return run["trains"][-1] if run["trains"] else None


def last_val(run: dict) -> dict | None:
    vals = [point for point in run["vals"] if point["step"] > 0]
    return vals[-1] if vals else None


def make_plots(engram: dict[int, dict], builtin: dict[int, dict], active: list[dict]) -> tuple[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bfs = sorted(engram)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), dpi=140)
    ax = axes[0]
    ax.plot(bfs, [builtin[bf]["vals"][-1]["loss"] for bf in bfs], marker="o", linewidth=2.2, label="Built-in")
    ax.plot(bfs, [engram[bf]["vals"][-1]["loss"] for bf in bfs], marker="s", linewidth=2.2, label="Engram best stack")
    ax.set_xscale("log", base=2)
    ax.set_xticks(bfs)
    ax.set_xticklabels([str(bf) for bf in bfs])
    ax.set_xlabel("BF")
    ax.set_ylabel("final val loss")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax = axes[1]
    deltas = [builtin[bf]["vals"][-1]["loss"] - engram[bf]["vals"][-1]["loss"] for bf in bfs]
    ax.bar([str(bf) for bf in bfs], deltas, color="#2563eb")
    ax.set_xlabel("BF")
    ax.set_ylabel("Engram advantage")
    ax.grid(True, axis="y", alpha=0.25)
    for i, d in enumerate(deltas):
        ax.text(i, d + 0.00025, f"{d:.4f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Matched BF Scaling", fontsize=13)
    fig.tight_layout()
    scaling_uri = svg_uri(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=140)
    ref = engram[80]
    if ref["vals"]:
        ax.plot([p["step"] for p in ref["vals"]], [p["loss"] for p in ref["vals"]], marker="s", linewidth=2.1, label="clean BF80 best")
    colors = [
        "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
    ]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    linestyles = ["-", "--", "-.", ":"]
    for i, run in enumerate(active):
        vals = [point for point in run["parsed"]["vals"] if point["step"] > 0]
        if vals:
            ax.plot(
                [p["step"] for p in vals],
                [p["loss"] for p in vals],
                marker=markers[i % len(markers)],
                linestyle=linestyles[(i // len(markers)) % len(linestyles)],
                color=colors[i % len(colors)],
                linewidth=2.0,
                label=run["name"],
            )
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("validation loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Active Routing / ROM / Shadow Runs", fontsize=13)
    fig.tight_layout()
    active_uri = svg_uri(fig)
    plt.close(fig)
    return scaling_uri, active_uri


def semantic_summary() -> tuple[str, str]:
    path = BF120_HOT_MD if BF120_HOT_MD.exists() else SEMANTIC_MD
    if not path.exists():
        return "missing", "<p>Semantic summary is missing.</p>"
    lines = path.read_text(errors="replace").splitlines()
    intro = []
    bullets = []
    for line in lines:
        if line.startswith("Total top slots") or line.startswith("Semantic-ish") or line.startswith("Validation tokens scanned"):
            intro.append(html.escape(line))
        if line.startswith("- #") and len(bullets) < 12:
            bullets.append(html.escape(line))
        if line.startswith("- row `") and len(bullets) < 12:
            bullets.append(html.escape(line))
    body = "<p>" + "<br>".join(intro) + "</p>" if intro else ""
    body += "<ul>" + "".join(f"<li><code>{b}</code></li>" for b in bullets) + "</ul>"
    return "present", body


def counterfactual_summary(counterfactual_dir: Path, title: str) -> str:
    if not counterfactual_dir.exists():
        return f"<h3>{html.escape(title)}</h3><p>Missing.</p>"
    rows = []
    base = None
    payloads = []
    for path in sorted(counterfactual_dir.glob("*.json")):
        import json

        data = json.loads(path.read_text())
        if path.stem == "base_mask_unhit":
            base = float(data["val_loss"])
        payloads.append((path.stem, path, data))
    for label, path, data in payloads:
        loss = float(data["val_loss"])
        delta = loss - base if base is not None else None
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{loss:.6f}</td>"
            f"<td>{fmt(delta, 6)}</td>"
            f"<td>{int(data.get('val_tokens', 0)):,}</td>"
            f"<td><code>{html.escape(rel(path))}</code></td>"
            "</tr>"
        )
    return (
        f"<h3>{html.escape(title)}</h3>"
        "<table>"
        "<thead><tr><th>Mask</th><th>Val Loss</th><th>Delta vs Base</th><th>Tokens</th><th>JSON</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def main() -> None:
    engram = {bf: parse_log(LOGS / name) for bf, name in ENGRAM_SCALING.items()}
    builtin = {bf: parse_log(LOGS / name) for bf, name in BUILTIN_SCALING.items()}
    for bf, run in {**engram, **builtin}.items():
        if not run["vals"]:
            raise RuntimeError(f"missing validation points for BF{bf}: {run['path']}")
    active = []
    for item in ACTIVE_RUNS:
        parsed = parse_log(item["path"])
        active.append({**item, "parsed": parsed})

    scaling_uri, active_uri = make_plots(engram, builtin, active)
    semantic_status, semantic_body = semantic_summary()
    counterfactual_body = counterfactual_summary(BF120_COUNTERFACTUAL_DIR, "Quick 1M-token counterfactuals")
    counterfactual_body += counterfactual_summary(BF120_COUNTERFACTUAL_FULL_DIR, "Full 10M-token counterfactuals")

    scaling_rows = []
    for bf in sorted(engram):
        e = engram[bf]["vals"][-1]["loss"]
        b = builtin[bf]["vals"][-1]["loss"]
        scaling_rows.append(
            "<tr>"
            f"<td>{bf}</td><td>{e:.4f}</td><td>{b:.4f}</td><td>{b - e:.4f}</td>"
            f"<td>{rel(engram[bf]['path'])}</td><td>{rel(builtin[bf]['path'])}</td>"
            "</tr>"
        )

    active_rows = []
    for run in active:
        parsed = run["parsed"]
        train = last_train(parsed)
        val = last_val(parsed)
        step = val["step"] if val else (train["step"] if train else "n/a")
        loss = val["loss"] if val else None
        step_avg = train["step_avg_ms"] if train else None
        metrics = val["metrics"] if val else {}
        active_rows.append(
            "<tr>"
            f"<td>{html.escape(run['name'])}</td>"
            f"<td>{html.escape(parsed['status'])}</td>"
            f"<td>{step}</td>"
            f"<td>{fmt(loss)}</td>"
            f"<td>{fmt(step_avg, 1)} ms</td>"
            f"<td>{fmt(metrics.get('engram_attnres_p_mean'), 3)}</td>"
            f"<td>{fmt(metrics.get('engram_attnres_extra_p_mean'), 3)}</td>"
            f"<td>{html.escape(run['hypothesis'])}</td>"
            f"<td><code>{html.escape(rel(parsed['path']))}</code></td>"
            "</tr>"
        )

    artifact_rows = []
    for label, path, status in [
        ("BF scaling report", SCALING_HTML, "present" if SCALING_HTML.exists() else "missing"),
        ("Semantic-ish top rows", SEMANTIC_MD, semantic_status),
        ("BF120 hot/gated rows", BF120_HOT_MD, "present" if BF120_HOT_MD.exists() else "missing"),
        ("Prompt gating HTML", PROMPT_HTML, "present" if PROMPT_HTML.exists() else "missing"),
        ("BF120 prompt gating HTML", BF120_PROMPT_HTML, "present" if BF120_PROMPT_HTML.exists() else "missing"),
        ("BF120 mask counterfactuals", BF120_COUNTERFACTUAL_DIR, "present" if BF120_COUNTERFACTUAL_DIR.exists() else "missing"),
        ("BF120 full mask counterfactuals", BF120_COUNTERFACTUAL_FULL_DIR, "present" if BF120_COUNTERFACTUAL_FULL_DIR.exists() else "missing"),
    ]:
        link = f'<a href="{html.escape(rel(path))}">{html.escape(rel(path))}</a>' if path.exists() else html.escape(rel(path))
        artifact_rows.append(f"<tr><td>{html.escape(label)}</td><td>{html.escape(status)}</td><td>{link}</td></tr>")

    best_bf80 = engram[80]["vals"][-1]["loss"]
    builtin_bf80 = builtin[80]["vals"][-1]["loss"]
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Engram Overnight Report</title>
  <style>
    body {{ margin: 0; color: #172033; font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f7f8fb; }}
    header {{ background: #fff; border-bottom: 1px solid #dce2eb; padding: 26px 34px 18px; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 28px 0 10px; font-size: 18px; }}
    p {{ max-width: 950px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; margin: 12px 0 20px; }}
    th, td {{ border: 1px solid #dce2eb; padding: 7px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    td:nth-child(2), td:nth-child(3), td:nth-child(4), td:nth-child(5), td:nth-child(6), td:nth-child(7) {{ text-align: right; }}
    .card {{ background: #fff; border: 1px solid #dce2eb; padding: 14px 16px; margin: 14px 0; }}
    .plot {{ width: min(100%, 1080px); background: #fff; border: 1px solid #dce2eb; padding: 8px; }}
    code {{ background: #eef2f7; padding: 1px 4px; border-radius: 4px; }}
    a {{ color: #1d4ed8; }}
  </style>
</head>
<body>
<header>
  <h1>Engram Overnight Report</h1>
  <p>Focused snapshot for routing, collision reduction, and semantic probing. Clean BF80 best-stack is <b>{best_bf80:.4f}</b>; matched built-in BF80 baseline is <b>{builtin_bf80:.4f}</b>.</p>
</header>
<main>
  <h2>Matched Scaling</h2>
  <img class="plot" src="{scaling_uri}" alt="Matched BF scaling">
  <table>
    <thead><tr><th>BF</th><th>Engram</th><th>Built-in</th><th>Delta</th><th>Engram Log</th><th>Built-in Log</th></tr></thead>
    <tbody>{''.join(scaling_rows)}</tbody>
  </table>

  <h2>Active Routing / Collision Experiments</h2>
  <img class="plot" src="{active_uri}" alt="Active run validation curves">
  <table>
    <thead><tr><th>Run</th><th>Status</th><th>Last Step</th><th>Last Val</th><th>Train Step Avg</th><th>AttnRes p mean</th><th>Max-bank p mean</th><th>Hypothesis</th><th>Log</th></tr></thead>
    <tbody>{''.join(active_rows)}</tbody>
  </table>

  <h2>Semantic Probe</h2>
  <div class="card">
    {semantic_body}
  </div>

  <h2>BF120 Mask Counterfactuals</h2>
  <div class="card">
    {counterfactual_body}
  </div>

  <h2>Artifacts</h2>
  <table>
    <thead><tr><th>Artifact</th><th>Status</th><th>Path</th></tr></thead>
    <tbody>{''.join(artifact_rows)}</tbody>
  </table>
</main>
</body>
</html>
"""
    OUT.write_text(html_doc)
    print(OUT)


if __name__ == "__main__":
    main()
