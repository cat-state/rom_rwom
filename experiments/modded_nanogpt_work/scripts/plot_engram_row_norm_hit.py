#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import torch


class ParamConfig:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot all-row Engram row RMS vs log hit count.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--row-rms", type=Path)
    parser.add_argument("--hit-hist", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--chunk-rows", default=262144, type=int)
    args = parser.parse_args()
    if (args.checkpoint is None) == (args.row_rms is None):
        parser.error("provide exactly one of --checkpoint or --row-rms")
    return args


def get_state_dict(ckpt: object) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict"):
            if key in ckpt:
                return ckpt[key]
        if all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)!r}")


def pick_memory_weight(sd: dict[str, torch.Tensor]) -> tuple[str, torch.Tensor]:
    for key in ("bigram_embed.embedding.weight", "module.bigram_embed.embedding.weight"):
        if key in sd and isinstance(sd[key], torch.Tensor):
            return key, sd[key]
    candidates = [
        (k, v) for k, v in sd.items()
        if isinstance(v, torch.Tensor) and "bigram_embed" in k and v.ndim == 2
    ]
    if not candidates:
        raise KeyError("Could not find Engram memory table in checkpoint")
    return max(candidates, key=lambda kv: kv[1].numel())


def load_hit_hist(path: Path) -> torch.Tensor:
    hist = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(hist, dict):
        for key in ("hit_hist", "hit_count", "hits", "hist"):
            if key in hist:
                hist = hist[key]
                break
    if not isinstance(hist, torch.Tensor):
        raise TypeError(f"Unsupported hit histogram type: {type(hist)!r}")
    return hist.detach().cpu().long().flatten()


def load_row_rms(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ("row_rms", "rms"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, torch.Tensor):
        raise TypeError(f"Unsupported row RMS artifact type: {type(payload)!r}")
    return payload.detach().cpu().float().flatten()


def compute_row_rms(weight: torch.Tensor, n_rows: int, chunk_rows: int) -> np.ndarray:
    rms = np.empty(n_rows, dtype=np.float32)
    for start in range(0, n_rows, chunk_rows):
        end = min(start + chunk_rows, n_rows)
        rows = weight[start:end].float()
        rms[start:end] = rows.square().mean(dim=1).sqrt().cpu().numpy()
        print(f"computed row rms {end:,}/{n_rows:,}", flush=True)
    return rms


def save_plot(out_dir: Path, log_hit: np.ndarray, row_rms: np.ndarray, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7), dpi=180)
    hb = ax.hexbin(
        log_hit,
        row_rms,
        gridsize=120,
        bins="log",
        mincnt=1,
        cmap="magma",
    )
    ax.set_xlabel("log10(hit_count + 1)")
    ax.set_ylabel("row RMS")
    ax.set_title(title)
    ax.grid(alpha=0.16)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("log10(rows per hex)")

    # Median trend over hit-count bins.
    edges = np.linspace(float(log_hit.min()), float(log_hit.max()), 64)
    centers = []
    medians = []
    p10 = []
    p90 = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (log_hit >= lo) & (log_hit < hi)
        if mask.sum() < 50:
            continue
        vals = row_rms[mask]
        centers.append((lo + hi) * 0.5)
        medians.append(float(np.median(vals)))
        p10.append(float(np.quantile(vals, 0.10)))
        p90.append(float(np.quantile(vals, 0.90)))
    if centers:
        centers = np.asarray(centers)
        medians = np.asarray(medians)
        p10 = np.asarray(p10)
        p90 = np.asarray(p90)
        ax.plot(centers, medians, color="#38bdf8", linewidth=2.0, label="median")
        ax.fill_between(centers, p10, p90, color="#38bdf8", alpha=0.18, label="10-90%")
        ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_dir / "row_norm_vs_log_hit_all.png")
    fig.savefig(out_dir / "row_norm_vs_log_hit_all.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hit = load_hit_hist(args.hit_hist)
    weight_key = ""
    weight_shape = []
    if args.row_rms is not None:
        print(f"loading row RMS {args.row_rms}", flush=True)
        row_rms_tensor = load_row_rms(args.row_rms)
        n_rows = min(int(row_rms_tensor.numel()), int(hit.numel()))
        row_rms = row_rms_tensor[:n_rows].numpy()
    else:
        print(f"loading checkpoint {args.checkpoint}", flush=True)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        sd = get_state_dict(ckpt)
        weight_key, weight = pick_memory_weight(sd)
        weight_shape = list(weight.shape)
        n_rows = min(int(weight.shape[0]), int(hit.numel()))
        row_rms = compute_row_rms(weight, n_rows, args.chunk_rows)
    hit_np = hit[:n_rows].numpy()
    log_hit = np.log10(hit_np.astype(np.float64) + 1.0).astype(np.float32)
    corr = float(np.corrcoef(log_hit, row_rms)[0, 1])
    source_name = args.checkpoint.parent.name if args.checkpoint is not None else args.row_rms.parent.name
    title = f"Engram row RMS vs log hit count ({args.run_name or source_name})"
    save_plot(args.out_dir, log_hit, row_rms, title)

    quantile_levels = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    meta = {
        "run_name": args.run_name or source_name,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else "",
        "row_rms": str(args.row_rms) if args.row_rms is not None else "",
        "hit_hist": str(args.hit_hist),
        "weight_key": weight_key,
        "weight_shape": weight_shape,
        "rows": n_rows,
        "hit_fraction": float((hit_np > 0).mean()),
        "corr_log_hit_row_rms": corr,
        "hit_quantiles": {str(q): float(np.quantile(hit_np, q)) for q in quantile_levels},
        "log_hit_quantiles": {str(q): float(np.quantile(log_hit, q)) for q in quantile_levels},
        "row_rms_quantiles": {str(q): float(np.quantile(row_rms, q)) for q in quantile_levels},
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    np.savez_compressed(args.out_dir / "row_norm_hit_all.npz", log_hit=log_hit, row_rms=row_rms, hit_count=hit_np)

    page = f"""<!doctype html>
<meta charset="utf-8">
<title>Engram Row Norm vs Hit Count</title>
<style>
body {{ font-family: system-ui, -apple-system, sans-serif; margin: 32px; max-width: 1100px; }}
img {{ max-width: 100%; border: 1px solid #ddd; }}
code, pre {{ background: #f6f6f6; padding: 2px 4px; }}
pre {{ padding: 16px; overflow: auto; }}
</style>
<h1>Engram Row Norm vs Hit Count</h1>
<p><b>Run:</b> {html.escape(meta["run_name"])}</p>
<p><b>Rows:</b> {n_rows:,}; <b>corr(log10(hit+1), row RMS):</b> {corr:.4f}</p>
<img src="row_norm_vs_log_hit_all.png" alt="row norm vs log hit count">
<h2>Metadata</h2>
<pre>{html.escape(json.dumps(meta, indent=2))}</pre>
"""
    (args.out_dir / "index.html").write_text(page)
    print(args.out_dir / "index.html")


if __name__ == "__main__":
    main()
