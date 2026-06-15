#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


class ParamConfig:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize sampled Engram memory rows with PCA/UMAP.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--hit-hist", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("reports/engram_table_viz"), type=Path)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--sample-rows", default=30000, type=int)
    parser.add_argument("--umap-rows", default=12000, type=int)
    parser.add_argument("--top-rows", default=5000, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    return parser.parse_args()


def get_state_dict(ckpt: object) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict"):
            if key in ckpt:
                return ckpt[key]
        if all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)!r}")


def pick_memory_weight(sd: dict[str, torch.Tensor]) -> tuple[str, torch.Tensor]:
    candidates = [
        (k, v) for k, v in sd.items()
        if isinstance(v, torch.Tensor) and k.endswith("bigram_embed.embedding.weight")
    ]
    if not candidates:
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


def sample_indices(hit: torch.Tensor, n_rows: int, sample_rows: int, top_rows: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    hit = hit[:n_rows]
    sample_parts: list[torch.Tensor] = []

    positive = torch.nonzero(hit > 0, as_tuple=False).flatten()
    zero = torch.nonzero(hit == 0, as_tuple=False).flatten()
    if positive.numel() > 0 and top_rows > 0:
        k = min(top_rows, positive.numel())
        sample_parts.append(torch.topk(hit, k=k).indices)

    # Log-hit bins keep both frequent and tail rows visible.
    if positive.numel() > 0:
        log_hit = torch.log10(hit[positive].float() + 1.0)
        edges = torch.linspace(float(log_hit.min()), float(log_hit.max()) + 1e-6, steps=9)
        per_bin = max(256, sample_rows // 18)
        for lo, hi in zip(edges[:-1], edges[1:]):
            bin_rows = positive[(log_hit >= lo) & (log_hit < hi)]
            if bin_rows.numel() == 0:
                continue
            take = min(per_bin, bin_rows.numel())
            sample_parts.append(bin_rows[torch.randperm(bin_rows.numel(), generator=gen)[:take]])

    if zero.numel() > 0:
        take = min(max(1024, sample_rows // 8), zero.numel())
        sample_parts.append(zero[torch.randperm(zero.numel(), generator=gen)[:take]])

    remaining = max(0, sample_rows - sum(int(x.numel()) for x in sample_parts))
    if remaining > 0:
        all_rows = torch.randperm(n_rows, generator=gen)[:remaining]
        sample_parts.append(all_rows)

    idx = torch.unique(torch.cat(sample_parts), sorted=True)
    if idx.numel() > sample_rows:
        perm = torch.randperm(idx.numel(), generator=gen)
        # Preserve top rows, randomly trim the rest.
        top = sample_parts[0] if sample_parts else idx[:0]
        keep_top = torch.isin(idx, top)
        rest = idx[~keep_top]
        need_rest = max(0, sample_rows - int(keep_top.sum()))
        rest = rest[torch.randperm(rest.numel(), generator=gen)[:need_rest]]
        idx = torch.unique(torch.cat([idx[keep_top], rest]), sorted=True)
    return idx


def row_rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(x), axis=1))


def standardize_for_pca(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True) + 1e-6
    return x / scale


def fit_pca(x: np.ndarray, n_components: int = 10):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    z = pca.fit_transform(standardize_for_pca(x))
    return z, pca.explained_variance_ratio_


def fit_umap(x: np.ndarray, n_rows: int, seed: int) -> np.ndarray:
    import umap

    if x.shape[0] > n_rows:
        rng = np.random.default_rng(seed)
        sel = rng.choice(x.shape[0], size=n_rows, replace=False)
        x = x[sel]
        return sel, umap.UMAP(
            n_neighbors=30,
            min_dist=0.08,
            metric="cosine",
            random_state=seed,
            low_memory=True,
        ).fit_transform(x.astype(np.float32))
    sel = np.arange(x.shape[0])
    return sel, umap.UMAP(
        n_neighbors=30,
        min_dist=0.08,
        metric="cosine",
        random_state=seed,
        low_memory=True,
    ).fit_transform(x.astype(np.float32))


def save_scatter(path: Path, x: np.ndarray, y: np.ndarray, color: np.ndarray, title: str, color_label: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7), dpi=160)
    sc = ax.scatter(x, y, c=color, s=3, alpha=0.55, cmap="viridis", linewidths=0, rasterized=True)
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.grid(alpha=0.18)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(color_label)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_norm_hit(path: Path, df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    ax.hexbin(
        np.log10(df["hit_count"].to_numpy() + 1.0),
        df["row_rms"].to_numpy(),
        gridsize=80,
        bins="log",
        mincnt=1,
        cmap="magma",
    )
    ax.set_xlabel("log10(hit_count + 1)")
    ax.set_ylabel("row RMS")
    ax.set_title("Engram row norm vs hit count")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = get_state_dict(ckpt)
    weight_key, weight = pick_memory_weight(sd)
    weight = weight.detach().cpu()
    hit = load_hit_hist(args.hit_hist)
    n_rows = min(weight.shape[0], hit.numel())
    idx = sample_indices(hit, n_rows, args.sample_rows, args.top_rows, args.seed)
    print(f"weight={weight_key} shape={tuple(weight.shape)} sample={idx.numel()} n_rows={n_rows}")

    rows = weight.index_select(0, idx).float().numpy()
    hit_s = hit.index_select(0, idx).numpy()
    rms = row_rms(rows)
    rows_dir = rows / (rms[:, None] + 1e-6)
    log_hit = np.log10(hit_s.astype(np.float64) + 1.0)

    pca_raw, evr_raw = fit_pca(rows)
    pca_dir, evr_dir = fit_pca(rows_dir)

    df = pd.DataFrame({
        "row": idx.numpy(),
        "hit_count": hit_s,
        "log10_hit": log_hit,
        "row_rms": rms,
        "pca_raw_1": pca_raw[:, 0],
        "pca_raw_2": pca_raw[:, 1],
        "pca_dir_1": pca_dir[:, 0],
        "pca_dir_2": pca_dir[:, 1],
    })

    save_scatter(args.out_dir / "pca_raw_by_hit.png", pca_raw[:, 0], pca_raw[:, 1], log_hit, "PCA of raw Engram rows", "log10(hit + 1)")
    save_scatter(args.out_dir / "pca_dir_by_hit.png", pca_dir[:, 0], pca_dir[:, 1], log_hit, "PCA of direction-normalized Engram rows", "log10(hit + 1)")
    save_scatter(args.out_dir / "pca_dir_by_norm.png", pca_dir[:, 0], pca_dir[:, 1], rms, "PCA of row directions, colored by row RMS", "row RMS")
    save_norm_hit(args.out_dir / "row_norm_vs_hit.png", df)

    umap_error = None
    try:
        sel, u = fit_umap(rows_dir, args.umap_rows, args.seed)
        df_umap = df.iloc[sel].copy()
        df_umap["umap_1"] = u[:, 0]
        df_umap["umap_2"] = u[:, 1]
        df_umap.to_csv(args.out_dir / "umap_sample.csv", index=False)
        save_scatter(args.out_dir / "umap_dir_by_hit.png", u[:, 0], u[:, 1], df_umap["log10_hit"].to_numpy(), "UMAP of direction-normalized Engram rows", "log10(hit + 1)")
    except Exception as exc:
        umap_error = repr(exc)

    df.to_csv(args.out_dir / "pca_sample.csv", index=False)
    meta = {
        "run_name": args.run_name,
        "checkpoint": str(args.checkpoint),
        "hit_hist": str(args.hit_hist),
        "weight_key": weight_key,
        "weight_shape": list(weight.shape),
        "rows_total": int(n_rows),
        "sample_rows": int(idx.numel()),
        "hit_fraction_total": float((hit[:n_rows] > 0).float().mean().item()),
        "sample_hit_min": int(hit_s.min()),
        "sample_hit_max": int(hit_s.max()),
        "sample_hit_median": float(np.median(hit_s)),
        "row_rms_min": float(rms.min()),
        "row_rms_max": float(rms.max()),
        "row_rms_median": float(np.median(rms)),
        "pca_raw_explained_variance_ratio": evr_raw.tolist(),
        "pca_direction_explained_variance_ratio": evr_dir.tolist(),
        "umap_error": umap_error,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    images = [
        "pca_raw_by_hit.png",
        "pca_dir_by_hit.png",
        "pca_dir_by_norm.png",
        "row_norm_vs_hit.png",
    ]
    if (args.out_dir / "umap_dir_by_hit.png").exists():
        images.append("umap_dir_by_hit.png")
    html_parts = [
        "<!doctype html><meta charset='utf-8'><title>Engram Table Visualization</title>",
        "<style>body{font-family:system-ui,-apple-system,sans-serif;margin:32px;max-width:1200px} img{max-width:100%;border:1px solid #ddd} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px} code{background:#f5f5f5;padding:2px 4px}</style>",
        "<h1>Engram Table Visualization</h1>",
        f"<p><b>Run:</b> {html.escape(args.run_name or args.checkpoint.parent.name)}</p>",
        f"<p><b>Table:</b> <code>{html.escape(weight_key)}</code>, shape {tuple(weight.shape)}, sampled {idx.numel():,} rows.</p>",
        "<h2>Summary</h2><pre>" + html.escape(json.dumps(meta, indent=2)[:4000]) + "</pre>",
        "<h2>Plots</h2><div class='grid'>",
    ]
    for image in images:
        html_parts.append(f"<figure><img src='{image}'><figcaption>{html.escape(image)}</figcaption></figure>")
    html_parts.append("</div>")
    (args.out_dir / "index.html").write_text("\n".join(html_parts))
    print(args.out_dir / "index.html")


if __name__ == "__main__":
    main()
