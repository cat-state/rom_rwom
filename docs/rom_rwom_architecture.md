# Rememberance of Memories (ROM/RWOM)

## What Engram Gives Us

The official Engram demo in `references/Engram/engram_demo_v1.py` is a compact
reference for the data path:

1. Compress tokenizer IDs so superficial token variants collide intentionally.
2. Build suffix `N`-grams ending at each current token.
3. Hash each suffix with per-layer multiplicative-XOR multipliers.
4. Use several prime-sized tables per `N`-gram order to reduce correlated
   collisions.
5. Concatenate retrieved vectors, gate them against the current hidden state,
   pass the gated value through a short causal depthwise conv, and add it before
   the block's attention/MoE.

The paper configuration uses Engram at early/mid layers, especially layer 2 and
15, with 2/3-grams, eight hash heads, zero-initialized conv, higher LR for
memory embeddings, and no weight decay on memory tables.

## FLA/GDN Interface We Can Reuse

FLA's `GatedDeltaNet` in
`references/flash-linear-attention/fla/layers/gated_deltanet.py` projects
hidden states into:

- `q, k`: `[B, T, H, K]`
- `v`: `[B, T, H, V]`
- `beta`: `[B, T, H]`
- `g`: `[B, T, H]`
- recurrent state: `[B, H, K, V]`

The reference recurrence in
`references/flash-linear-attention/fla/ops/gated_delta_rule/naive.py` is:

```text
h_t = exp(g_t) * h_{t-1}
prediction = h_t @ k_t
error = v_t - prediction
h_t = h_t + outer(k_t, beta_t * error)
o_t = q_t @ h_t
```

That state matrix is the natural candidate for a ROM memory cell.

## First Target: 2-Gram Address, Recurrent-State Payload

Start with Engram's deterministic 2-gram address and replace each static vector
row with a small recurrent-state payload:

```text
addr_t = hash(token_{t-1}, token_t, layer, head)
cell_t = memory_table[addr_t]
read_t = q_t @ cell_t
cell'_t = gdn_update(cell_t, k_t, v_t, beta_t, g_t)
memory_table[addr_t] <- write_gate_t * cell'_t + (1 - write_gate_t) * cell_t
hidden_t <- hidden_t + out_proj(read_t)
```

This keeps Engram's strongest systems property: addresses are known from tokens
before the target layer executes, so prefetch/offload remains possible.

The full `[H,K,V]` cell is expensive. With `H=8, K=64, V=128`, one row is
65,536 values. At 1M rows this is not a parameter table, it is a storage system.
The first practical version should therefore use one of:

- low-rank cell factors, e.g. `A: [r,K]`, `B: [r,V]`;
- smaller ROM heads than the backbone GDN heads;
- a vector payload that is reshaped/projected into a tiny state matrix;
- top-k hot rows on GPU and cold rows on CPU/NVMe, only for inference.

## Can ROM Remove 2-Grams?

Eventually, yes. For the first target, no.

The 2-gram is not the memory content; it is the address. If we remove it
immediately, we need another addressing scheme that is cheap, stable, and
prefetchable. Learned latent addressing is possible, but it changes the problem:

- deterministic token address: cheap, prefetchable, collision behavior is easy
  to measure;
- quantized latent address: more flexible, but depends on hidden states, is
  harder to prefetch, may collapse without commitment/load-balancing losses;
- content-addressed lookup: most flexible, but no longer Engram-like `O(1)`
  sparse lookup unless heavily approximated.

Recommended sequence:

1. `ROM-v0`: Engram-compatible 2-gram hash, recurrent-state payload.
2. `ROM-v1`: add 3-grams and compare vector payload vs state payload.
3. `RWOM-v2`: write-gated updates during finetuning, with detach/EMA variants
   to test stability.
4. `Latent-ROM-v3`: replace token n-gram addresses with quantized latent codes
   after the state-payload baseline is understood.

## Qwen/GDN Finetuning Path

Qwen3-Next and later Qwen3.5/Qwen3.6 models are a good target because their
hybrid stacks already contain GDN layers. FLA's README notes Qwen3-Next uses GDN,
and current Qwen/NVIDIA docs describe Qwen3.5/3.6 as GDN + standard attention
hybrids. The most conservative integration is to attach ROM beside an existing
GDN layer rather than replacing it:

```text
prenorm hidden
  -> existing Qwen GDN
  -> ROM read/write branch keyed by token 2-gram
  -> residual add
  -> MLP/MoE
```

This lets us freeze most base weights, train only ROM tables and small
projections first, and later unfreeze selected GDN projections if the branch
shows signal.

## Immediate Implementation Plan

1. Build and test Engram-compatible hash indexing. Done in
   `rom_rwom/ngram_hash.py`.
2. Add a tiny recurrent-state payload implementation using the naive GDN update,
   small dimensions, and explicit scatter/write semantics. Done in
   `RomStateMemory`.
3. Add a Torch module that can sit beside a transformer/GDN block with
   `hidden -> q/k/v/beta/decay`, Engram-style addresses, write gates, and a
   zero-initialized residual projection. Done in `RomGatedDeltaMemory`.
4. Port the recurrence to FLA kernels only after the math and training behavior
   are validated in the tiny prototype.
