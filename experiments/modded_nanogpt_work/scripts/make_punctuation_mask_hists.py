from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def is_punct_control(text: str) -> bool:
    stripped = text.strip()
    if not text:
        return True
    if "<|endoftext|>" in text:
        return True
    if "\n" in text:
        return True
    if "�" in text:
        return True
    letters = sum(ch.isalpha() for ch in text)
    alnum = sum(ch.isalnum() for ch in text)
    punct = sum((not ch.isalnum()) and (not ch.isspace()) for ch in text)
    if alnum == 0:
        return True
    if punct >= alnum and letters <= 2:
        return True
    if stripped in {"’s", "'s", "’re", "'re", "n't", "n’t"}:
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--analysis", default="engram_analysis_step001500_top1000.json")
    parser.add_argument("--hist", default="engram_hit_hist_step001500.pt")
    parser.add_argument("--limits", default="25,50,100,200,500")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    analysis = json.loads((run_dir / args.analysis).read_text())
    payload = torch.load(run_dir / args.hist, map_location="cpu", weights_only=False)
    base_hist = payload["hit_hist"]

    punct = []
    for rank, slot in enumerate(analysis["top_slots"], 1):
        examples = slot.get("examples") or []
        text = examples[0].get("ngram_text", "") if examples else ""
        if is_punct_control(text):
            row = int(slot["absolute_row"])
            punct.append(
                {
                    "rank": rank,
                    "row": row,
                    "text": text.replace("\n", "\\n"),
                    "gate": float(slot["avg_gate"]),
                    "norm": float(slot["avg_output_norm"]),
                    "count": int(slot["count"]),
                    "train_hits": int(base_hist[row].item()),
                }
            )

    print(f"punct_control_rows_in_top_slots={len(punct)}")
    for limit_raw in args.limits.split(","):
        limit = int(limit_raw)
        rows = []
        seen = set()
        for item in punct[:limit]:
            if item["row"] not in seen:
                rows.append(item["row"])
                seen.add(item["row"])
        masked = dict(payload)
        hist = base_hist.clone()
        hist[torch.tensor(rows, dtype=torch.long)] = 0
        masked["hit_hist"] = hist
        masked["masked_rows"] = rows
        masked["mask_source"] = f"top{limit}_punct_control_by_gated_norm_from_top1000"
        out = run_dir / f"engram_hit_hist_step001500_mask_punct_top{limit}.pt"
        torch.save(masked, out)
        print(f"top{limit}: unique_rows={len(rows)} path={out}")

    (run_dir / "punct_control_rows_top50.json").write_text(json.dumps(punct[:50], indent=2))
    print(json.dumps(punct[:15], indent=2))


if __name__ == "__main__":
    main()
