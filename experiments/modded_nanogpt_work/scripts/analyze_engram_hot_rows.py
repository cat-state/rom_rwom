from __future__ import annotations

import argparse
import glob
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import torch


def load_data_shard(file: Path) -> torch.Tensor:
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert int(header[0]) == 20240520
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens
    return tokens


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def next_prime_after(start: int, seen: set[int]) -> int:
    n = max(2, start + 1)
    while n in seen or not is_prime(n):
        n += 1
    seen.add(n)
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--hist", default="engram_hit_hist_step001500.pt")
    parser.add_argument("--analysis", default="engram_analysis_step001500_top100.json")
    parser.add_argument("--data-glob", default="data/fineweb10B/fineweb_val_*.bin")
    parser.add_argument("--token-limit", type=int, default=2_097_152)
    parser.add_argument("--top-rows", type=int, default=200)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    hist_path = run_dir / args.hist
    analysis_path = run_dir / args.analysis
    out_json = run_dir / "engram_hot_rows_and_gated_ngrams_step001500.json"
    out_md = run_dir / "engram_hot_rows_and_gated_ngrams_step001500.md"
    out_csv = run_dir / "engram_hot_rows_scan_step001500.csv"

    vocab_size = 50257
    pad_id = 50256
    max_ngram = 3
    layers = (2, 8)
    seed = 0

    payload = torch.load(hist_path, map_location="cpu", weights_only=False)
    hit_hist = payload["hit_hist"].to(torch.int64)
    head_mods = payload["head_mods"].to(torch.int64)
    offsets = payload["offsets"].to(torch.int64)
    total_hash_heads = int(head_mods.numel())
    num_heads = max(1, total_hash_heads // (max_ngram - 1))
    first_layer_rows = int(head_mods.sum().item())
    layer_count = max(1, round(int(hit_hist.numel()) / max(1, first_layer_rows)))
    if layer_count > len(layers):
        layers = tuple(range(layer_count))
    if layer_count > 1:
        seen_primes: set[int] = set()
        prime_start = int(head_mods.min().item()) - 1
        layer_head_mods = []
        flat_offsets = [0]
        flat_sizes = []
        for _layer in range(layer_count):
            mods = []
            for _head_idx in range(total_hash_heads):
                mod = next_prime_after(prime_start, seen_primes)
                mods.append(mod)
                flat_sizes.append(mod)
            layer_head_mods.append(torch.tensor(mods, dtype=torch.int64))
        for size in flat_sizes[:-1]:
            flat_offsets.append(flat_offsets[-1] + size)
        layer_offsets = []
        cursor = 0
        for _layer in range(layer_count):
            layer_offsets.append(torch.tensor(flat_offsets[cursor: cursor + total_hash_heads], dtype=torch.int64))
            cursor += total_hash_heads
    else:
        layer_head_mods = [head_mods]
        layer_offsets = [offsets]
    top_counts, top_rows = torch.topk(hit_hist, k=args.top_rows)
    top_row_rank = {int(r): i + 1 for i, r in enumerate(top_rows.tolist())}
    top_row_count = {int(r): int(c) for r, c in zip(top_rows.tolist(), top_counts.tolist())}

    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    def decode(ids: list[int]) -> str:
        ids = [int(x) for x in ids]
        if enc is None:
            return " ".join(map(str, ids))
        try:
            return enc.decode(ids)
        except Exception:
            return " ".join(map(str, ids))

    def canonical_token_key(token_id: int) -> str:
        if enc is None:
            return str(token_id)
        token_bytes = enc.decode_single_token_bytes(token_id)
        try:
            text = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return f"bytes:{token_bytes!r}"
        if "\ufffd" in text:
            return f"bytes:{token_bytes!r}"
        text = unicodedata.normalize("NFKC", text)
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return text.casefold()

    key_to_id: dict[str, int] = {}
    canonical = torch.empty(vocab_size, dtype=torch.int64)
    for token_id in range(vocab_size):
        key = canonical_token_key(token_id)
        canonical[token_id] = key_to_id.setdefault(key, len(key_to_id))
    canonical_vocab_size = len(key_to_id)
    hash_pad_id = int(canonical[pad_id].item())
    max_multiplier = max(1, (2**31 - 1) // max(1, canonical_vocab_size))
    layer_multipliers = {}
    for layer in layers:
        generator = torch.Generator()
        generator.manual_seed(seed + 10007 * layer)
        layer_multipliers[layer] = torch.randint(1, max_multiplier, (max_ngram,), generator=generator, dtype=torch.int64) * 2 + 1

    files = sorted(Path(path) for path in glob.glob(args.data_glob))
    if not files:
        raise FileNotFoundError(args.data_glob)
    tokens = load_data_shard(files[0]).to(torch.int64)[: args.token_limit]
    canonical_tokens = canonical[tokens.clamp_min(0).clamp_max(vocab_size - 1)]

    row_hits = defaultdict(lambda: {"scan_count": 0, "examples": Counter(), "layers": Counter()})
    chunk_size = 262_144
    for start in range(0, canonical_tokens.numel(), chunk_size):
        end = min(canonical_tokens.numel(), start + chunk_size)
        context_start = max(0, start - (max_ngram - 1))
        x = canonical_tokens[context_start:end]
        orig = tokens[context_start:end]
        offset_pos = start - context_start
        shifted = [x]
        for k in range(1, max_ngram):
            pad = torch.full((k,), hash_pad_id, dtype=torch.int64)
            shifted.append(torch.cat([pad, x[:-k]], dim=0))
        valid = torch.arange(offset_pos, x.numel())

        for layer_idx, layer in enumerate(layers):
            multipliers = layer_multipliers[layer]
            mods_for_layer = layer_head_mods[min(layer_idx, len(layer_head_mods) - 1)]
            offsets_for_layer = layer_offsets[min(layer_idx, len(layer_offsets) - 1)]
            head_idx = 0
            for ngram in range(2, max_ngram + 1):
                mix = shifted[0] * multipliers[0]
                for k in range(1, ngram):
                    mix = torch.bitwise_xor(mix, shifted[k] * multipliers[k])
                for hash_head in range(num_heads):
                    addresses = (mix % mods_for_layer[head_idx]) + offsets_for_layer[head_idx]
                    addr_valid = addresses[valid]
                    mask = torch.isin(addr_valid, top_rows)
                    if bool(mask.any()):
                        positions = valid[mask]
                        rows = addr_valid[mask]
                        for pos_t, row_t in zip(positions.tolist(), rows.tolist()):
                            ngram_start = max(0, pos_t - ngram + 1)
                            ids = orig[ngram_start : pos_t + 1].tolist()
                            text = decode(ids).replace("\n", "\\n")
                            entry = row_hits[int(row_t)]
                            entry["scan_count"] += 1
                            entry["examples"][(layer, head_idx, ngram, text)] += 1
                            entry["layers"][layer] += 1
                    head_idx += 1

    hot_rows = []
    for row in top_rows.tolist():
        row = int(row)
        entry = row_hits.get(row)
        examples = []
        if entry:
            for (layer, head_idx, ngram, text), count in entry["examples"].most_common(8):
                examples.append({"layer": layer, "head_idx": head_idx, "ngram": ngram, "text": text, "count_in_scan": count})
            scan_count = entry["scan_count"]
            layers_seen = dict(entry["layers"])
        else:
            scan_count = 0
            layers_seen = {}
        bucket_head = None
        bucket_layer = None
        for layer_idx, (mods_for_layer, offsets_for_layer) in enumerate(zip(layer_head_mods, layer_offsets)):
            for head_idx, (offset, mod) in enumerate(zip(offsets_for_layer.tolist(), mods_for_layer.tolist())):
                if offset <= row < offset + mod:
                    bucket_head = head_idx
                    bucket_layer = layers[layer_idx] if layer_idx < len(layers) else layer_idx
                    break
            if bucket_head is not None:
                break
        hot_rows.append(
            {
                "rank_by_training_hits": top_row_rank[row],
                "absolute_row": row,
                "training_hit_count": top_row_count[row],
                "bucket_layer": bucket_layer,
                "bucket_head_idx": bucket_head,
                "bucket_ngram": (2 + bucket_head // num_heads) if bucket_head is not None else None,
                "bucket_hash_head": (bucket_head % num_heads) if bucket_head is not None else None,
                "scan_count_in_val_sample": scan_count,
                "scan_layer_counts": layers_seen,
                "top_ngram_examples": examples,
            }
        )

    analysis_data = json.loads(analysis_path.read_text()) if analysis_path.exists() else {}
    top_gated = []
    for slot in analysis_data.get("top_slots", [])[:50]:
        example = slot.get("examples", [{}])[0] if slot.get("examples") else {}
        absolute_row = slot.get("absolute_row")
        top_gated.append(
            {
                "head_idx": slot.get("head_idx"),
                "ngram": slot.get("ngram"),
                "hash_head": slot.get("hash_head"),
                "absolute_row": absolute_row,
                "count_in_analysis": slot.get("count"),
                "avg_gate": slot.get("avg_gate"),
                "avg_gated_output_norm": slot.get("avg_output_norm"),
                "example_ngram_text": example.get("ngram_text", "").replace("\n", "\\n"),
                "example_window_text": example.get("window_text", "").replace("\n", "\\n"),
                "training_hit_count": int(hit_hist[int(absolute_row)].item()) if absolute_row is not None else None,
            }
        )

    result = {
        "val_tokens_scanned": int(tokens.numel()),
        "top_rows_considered": args.top_rows,
        "canonical_vocab_size": canonical_vocab_size,
        "hot_rows_by_training_hits": hot_rows,
        "top_memory_rows_after_internal_gating": top_gated,
        "analysis_json": str(analysis_path),
    }
    out_json.write_text(json.dumps(result, indent=2))
    with out_csv.open("w") as f:
        f.write("rank,absolute_row,training_hit_count,bucket_ngram,bucket_hash_head,scan_count,top_example\n")
        for row in hot_rows:
            top_example = row["top_ngram_examples"][0]["text"] if row["top_ngram_examples"] else ""
            f.write(
                f"{row['rank_by_training_hits']},{row['absolute_row']},{row['training_hit_count']},"
                f"{row['bucket_ngram']},{row['bucket_hash_head']},{row['scan_count_in_val_sample']},"
                f"\"{top_example.replace(chr(34), chr(34) + chr(34))}\"\n"
            )

    lines = ["# Engram Hot Rows and Gated Ngrams\n", f"Validation tokens scanned for hot-row examples: {tokens.numel():,}\n"]
    lines.append("\n## Hottest Training-Hit Rows: Example Ngrams\n")
    for row in hot_rows[:25]:
        lines.append(
            f"- rank {row['rank_by_training_hits']}: row `{row['absolute_row']}`, train hits `{row['training_hit_count']:,}`, "
            f"bucket {row['bucket_ngram']}-gram/head{row['bucket_hash_head']}, scan hits `{row['scan_count_in_val_sample']}`"
        )
        for example in row["top_ngram_examples"][:3]:
            lines.append(f"  - L{example['layer']} H{example['head_idx']} {example['ngram']}-gram x{example['count_in_scan']}: `{example['text']}`")
    lines.append("\n## Top Memory Rows After Internal Gating\n")
    for slot in top_gated[:25]:
        lines.append(
            f"- row `{slot['absolute_row']}`, {slot['ngram']}-gram/head{slot['hash_head']}, "
            f"analysis count `{slot['count_in_analysis']}`, avg gate `{slot['avg_gate']:.3f}`, "
            f"avg gated norm `{slot['avg_gated_output_norm']:.3f}`, train hits `{slot['training_hit_count']:,}`: "
            f"`{slot['example_ngram_text']}`"
        )
    out_md.write_text("\n".join(lines))

    print(out_json)
    print(out_md)
    print(out_csv)
    print(json.dumps({"top_hot_rows_with_scan_hits": sum(1 for row in hot_rows if row["scan_count_in_val_sample"] > 0), "top_gated_rows": len(top_gated)}, indent=2))


if __name__ == "__main__":
    main()
