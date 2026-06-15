from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def label_for_threshold(threshold: int) -> str:
    return f"hit_ge_{threshold}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Engram eval hit histograms that mask rows by final training hit count.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--hist", default="engram_hit_hist_step001500.pt")
    parser.add_argument("--thresholds", default="1024,256,64,16,4,2,1")
    parser.add_argument("--below-thresholds", default="")
    parser.add_argument("--buckets", default="1-1,2-3,4-7,8-15,16-31,32-63,64-255,256-1023,1024-")
    parser.add_argument("--step", default="001500")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    payload = torch.load(run_dir / args.hist, map_location="cpu", weights_only=False)
    base_hist = payload["hit_hist"].to(torch.int32)
    total_rows = int(base_hist.numel())
    total_hits = int(base_hist.sum(dtype=torch.int64).item())
    outputs: list[dict[str, object]] = []

    for raw in (part.strip() for part in args.thresholds.split(",")):
        if not raw:
            continue
        threshold = int(raw)
        rows = base_hist >= threshold
        hist = base_hist.clone()
        hist[rows] = 0
        masked = dict(payload)
        masked["hit_hist"] = hist
        masked["mask_source"] = f"mask_rows_with_training_hit_count_ge_{threshold}"
        masked["masked_threshold"] = threshold
        masked["masked_rows_count"] = int(rows.sum().item())
        masked["masked_training_hits"] = int(base_hist[rows].sum(dtype=torch.int64).item())
        label = label_for_threshold(threshold)
        out = run_dir / f"engram_hit_hist_step{args.step}_mask_{label}.pt"
        torch.save(masked, out)
        outputs.append(
            {
                "kind": "threshold",
                "label": label,
                "path": str(out),
                "threshold": threshold,
                "masked_rows": masked["masked_rows_count"],
                "masked_row_fraction": masked["masked_rows_count"] / max(1, total_rows),
                "masked_hits": masked["masked_training_hits"],
                "masked_hit_fraction": masked["masked_training_hits"] / max(1, total_hits),
            }
        )

    for raw in (part.strip() for part in args.below_thresholds.split(",")):
        if not raw:
            continue
        threshold = int(raw)
        rows = (base_hist > 0) & (base_hist < threshold)
        hist = base_hist.clone()
        hist[rows] = 0
        masked = dict(payload)
        masked["hit_hist"] = hist
        masked["mask_source"] = f"mask_rows_with_training_hit_count_lt_{threshold}"
        masked["masked_below_threshold"] = threshold
        masked["masked_rows_count"] = int(rows.sum().item())
        masked["masked_training_hits"] = int(base_hist[rows].sum(dtype=torch.int64).item())
        label = f"hit_lt_{threshold}"
        out = run_dir / f"engram_hit_hist_step{args.step}_mask_{label}.pt"
        torch.save(masked, out)
        outputs.append(
            {
                "kind": "below_threshold",
                "label": label,
                "path": str(out),
                "threshold": threshold,
                "masked_rows": masked["masked_rows_count"],
                "masked_row_fraction": masked["masked_rows_count"] / max(1, total_rows),
                "masked_hits": masked["masked_training_hits"],
                "masked_hit_fraction": masked["masked_training_hits"] / max(1, total_hits),
            }
        )

    for raw in (part.strip() for part in args.buckets.split(",")):
        if not raw:
            continue
        lo_raw, hi_raw = raw.split("-", 1)
        lo = int(lo_raw)
        hi = int(hi_raw) if hi_raw else None
        rows = base_hist >= lo
        if hi is not None:
            rows = rows & (base_hist <= hi)
            label = f"hit_{lo}_to_{hi}"
            source = f"mask_rows_with_training_hit_count_{lo}_to_{hi}"
        else:
            label = f"hit_ge_{lo}_bucket"
            source = f"mask_rows_with_training_hit_count_ge_{lo}_bucket"
        hist = base_hist.clone()
        hist[rows] = 0
        masked = dict(payload)
        masked["hit_hist"] = hist
        masked["mask_source"] = source
        masked["masked_bucket"] = {"lo": lo, "hi": hi}
        masked["masked_rows_count"] = int(rows.sum().item())
        masked["masked_training_hits"] = int(base_hist[rows].sum(dtype=torch.int64).item())
        out = run_dir / f"engram_hit_hist_step{args.step}_mask_{label}.pt"
        torch.save(masked, out)
        outputs.append(
            {
                "kind": "bucket",
                "label": label,
                "path": str(out),
                "lo": lo,
                "hi": hi,
                "masked_rows": masked["masked_rows_count"],
                "masked_row_fraction": masked["masked_rows_count"] / max(1, total_rows),
                "masked_hits": masked["masked_training_hits"],
                "masked_hit_fraction": masked["masked_training_hits"] / max(1, total_hits),
            }
        )

    out_json = run_dir / f"engram_hit_count_mask_hists_step{args.step}.json"
    out_json.write_text(json.dumps({"total_rows": total_rows, "total_hits": total_hits, "outputs": outputs}, indent=2))
    print(json.dumps({"total_rows": total_rows, "total_hits": total_hits, "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
