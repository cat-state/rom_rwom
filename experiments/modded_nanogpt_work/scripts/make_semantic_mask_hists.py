from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


STOP = {
    "and", "or", "of", "to", "in", "on", "for", "with", "the", "a", "an", "is", "are", "was", "were",
    "as", "by", "from", "at", "it", "this", "that", "be", "has", "have", "had", "not", "but", "his",
    "her", "their", "its", "you", "your", "we", "they", "he", "she", "will", "can", "would", "could",
    "one", "all", "more", "new", "also", "about", "into", "than", "then", "when", "where", "which",
    "who", "what", "been", "there", "other", "some", "may", "such",
}


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


def is_semanticish(text: str) -> bool:
    if is_punct_control(text):
        return False
    stripped = text.strip()
    if not stripped or "�" in stripped:
        return False
    letters = sum(ch.isalpha() for ch in stripped)
    if letters < 2:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z0-9']+", stripped)
    if words and all(word.lower().strip("'") in STOP for word in words):
        return False
    if any(ch in stripped for ch in "\n\t") and letters < 5:
        return False
    return True


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

    semantic = []
    for rank, slot in enumerate(analysis["top_slots"], 1):
        examples = slot.get("examples") or []
        text = examples[0].get("ngram_text", "") if examples else ""
        if is_semanticish(text):
            row = int(slot["absolute_row"])
            semantic.append(
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

    print(f"semanticish_rows_in_top_slots={len(semantic)}")
    for limit_raw in args.limits.split(","):
        limit = int(limit_raw)
        rows = []
        seen = set()
        for item in semantic[:limit]:
            if item["row"] not in seen:
                rows.append(item["row"])
                seen.add(item["row"])
        masked = dict(payload)
        hist = base_hist.clone()
        hist[torch.tensor(rows, dtype=torch.long)] = 0
        masked["hit_hist"] = hist
        masked["masked_rows"] = rows
        masked["mask_source"] = f"top{limit}_semanticish_by_gated_norm_from_top1000"
        out = run_dir / f"engram_hit_hist_step001500_mask_semantic_top{limit}.pt"
        torch.save(masked, out)
        print(f"top{limit}: unique_rows={len(rows)} path={out}")

    (run_dir / "semanticish_rows_top50.json").write_text(json.dumps(semantic[:50], indent=2))
    print(json.dumps(semantic[:15], indent=2))


if __name__ == "__main__":
    main()
