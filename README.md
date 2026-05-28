# rom-rwom

`rememberance_of_memories` is an experimental memory architecture inspired by
DeepSeek Engram and Gated DeltaNet.

The first target keeps Engram's deterministic 2-gram hash addressing, but swaps
the static embedding payload for a recurrent-state payload compatible with the
GDN update:

```text
2-gram -> hash row -> recurrent state cell -> gated read/write -> residual add
```

Why keep 2-grams first: they are cheap, deterministic, and prefetchable. Latent
or quantized memory keys are the more flexible direction, but they remove the
systems advantage until the state-payload baseline works.

## Repo Layout

- `rom_rwom/ngram_hash.py`: Engram-style deterministic suffix n-gram hashing.
- `rom_rwom/torch_state_memory.py`: naive Torch GDN-state memory payload and
  zero-init residual branch prototype.
- `docs/rom_rwom_architecture.md`: current architecture notes and staged plan.
- `references/Engram`: official DeepSeek Engram demo/source checkout.
- `references/engram_arxiv_src`: arXiv source for `2601.07372`.
- `references/flash-linear-attention`: full FLA checkout, including GDN ops.

## Quick Check

```bash
uv run --with pytest pytest -q
uv run --with pytest --with torch pytest -q
```

## Current Direction

1. Validate Engram-compatible 2-gram addressing.
2. Validate a small GDN recurrent-state payload with explicit read/write. Done
   in the naive Torch prototype.
3. Add a Torch vector-memory branch as a baseline against the state payload.
4. Attach beside an existing GDN layer in a Qwen3-Next/Qwen3.5-style hybrid.
