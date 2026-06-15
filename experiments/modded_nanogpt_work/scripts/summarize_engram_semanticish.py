from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--limit", type=int, default=150)
    args = parser.parse_args()

    data = json.loads(Path(args.analysis_json).read_text())
    slots = data["top_slots"]

    stop = {
        "and", "or", "of", "to", "in", "on", "for", "with", "the", "a", "an", "is", "are", "was", "were",
        "as", "by", "from", "at", "it", "this", "that", "be", "has", "have", "had", "not", "but", "his",
        "her", "their", "its", "you", "your", "we", "they", "he", "she", "will", "can", "would", "could",
        "one", "all", "more", "new", "also", "about", "into", "than", "then", "when", "where", "which",
        "who", "what", "been", "were", "there", "other", "some", "may", "such",
    }

    def slot_text(slot: dict) -> str:
        if slot.get("examples"):
            return slot["examples"][0].get("ngram_text", "").replace("\n", "\\n")
        return ""

    def slot_window(slot: dict) -> str:
        if slot.get("examples"):
            return slot["examples"][0].get("window_text", "").replace("\n", "\\n")
        return ""

    def is_semanticish(text: str) -> bool:
        stripped = text.strip()
        if not stripped or "�" in stripped:
            return False
        letters = sum(ch.isalpha() for ch in stripped)
        if letters < 2:
            return False
        words = re.findall(r"[A-Za-z][A-Za-z0-9']+", stripped)
        if words and all(word.lower().strip("'") in stop for word in words):
            return False
        if any(ch in stripped for ch in "\n\t") and letters < 5:
            return False
        return True

    semantic = []
    for rank, slot in enumerate(slots, start=1):
        text = slot_text(slot)
        if is_semanticish(text):
            semantic.append((rank, slot, text, slot_window(slot)))

    lines = [
        "# Semantic-ish Engram Rows in Top 1000",
        "",
        "Ranked by average gated output norm in the Engram analysis sample.",
        "",
        f"Total top slots: {len(slots)}",
        f"Semantic-ish after filter: {len(semantic)}",
        "",
    ]
    for rank, slot, text, window in semantic[: args.limit]:
        lines.append(
            f"- #{rank} row `{slot['absolute_row']}` {slot['ngram']}-gram/head{slot['hash_head']}, "
            f"count `{slot['count']}`, gate `{slot['avg_gate']:.3f}`, norm `{slot['avg_output_norm']:.1f}`: `{text}`"
        )
        if window:
            lines.append(f"  - window: `{window[:220]}`")

    Path(args.out_md).write_text("\n".join(lines))
    print(f"slots={len(slots)} semanticish={len(semantic)}")
    for rank, slot, text, window in semantic[:80]:
        print(
            f"#{rank:04d} row={slot['absolute_row']} {slot['ngram']}-g/h{slot['hash_head']} "
            f"count={slot['count']} gate={slot['avg_gate']:.3f} norm={slot['avg_output_norm']:.1f} "
            f"text={text!r} window={window[:160]!r}"
        )
    print(args.out_md)


if __name__ == "__main__":
    main()
