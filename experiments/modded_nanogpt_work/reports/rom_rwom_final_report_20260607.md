# rom_rwom Final Progress Report, 2026-06-07

This report summarizes the cumulative rom_rwom / Engram work using the compacted
chat history, local saved logs, local W&B cache, and prior written reports.

Primary durable sources:

- `reports/engram_scalar_frontier_20260522.md`
- `reports/engram_sota_progress_20260527.md`
- `reports/engram_mhc_report.html`
- `reports/engram_overnight_report.html`
- `reports/engram_scaling_report.html`
- `reports/engram_vs_builtin_bf_scaling.html`
- `logs/*.console.txt`, `tmp_remote_logs/*.console.txt`, `remote_logs/*.txt`
- local W&B cache under `logs/wandb/wandb`

The local W&B cache only contains a small May 17 subset. Later non-smoke runs
were W&B-enabled on the remote instance, but the local workspace mostly has them
through saved console logs and the two SOTA reports rather than full W&B history
files.

## Best Performance Found

Best observed 1500-step validation loss:

| Run | Final val | Notes |
| --- | ---: | --- |
| `bf400_ngramrows_trigramheavy_0p5_1p5_readhit025_s5_1500_20260527` | `3.2411` | Best single measured run |
| `bf400_ngramrows_trigramheavy_0p5_1p5_s5_ckpt_1500_20260527` | `3.2413` | Checkpointed repeat of same line |
| `bf400_ngramrows_0p5_1p5_sharepart_readhit025_s5_1500_20260527` | `3.2414` | Removes explicit row partitions |
| `bf400_ngramrows_0p5_1p5_layerheadmix_readhit025_s5_1500_20260528` | `3.2415` | Learned per-layer bigram/trigram mix |
| Hot-row dropout `p=0.05`, inverted scale | `3.2418` | Close, not SOTA |

Current SOTA recipe:

- `BIGRAM_FACTOR=400`
- `ENGRAM_DIM=768`
- `ENGRAM_HEADS=1`
- `ROM_LAYERS=2,8`
- `ENGRAM_MAX_NGRAM=3`
- `ENGRAM_NGRAM_ROW_FACTORS=0.5,1.5`
- `ENGRAM_READ_HIT_SCALE_EXPONENT=0.25`
- `ENGRAM_READ_HIT_SCALE_OFFSET=1.0`
- `ENGRAM_READ_HIT_SCALE_MIN=0.25`
- `ENGRAM_READ_HIT_SCALE_MAX=4.0`
- `ENGRAM_READ_HIT_SCALE_NORM_MEAN=1`
- `ENGRAM_NORMALIZE_READOUT=1`
- `ENGRAM_NORMALIZE_MEMORY_HEADS=1`
- `ENGRAM_LAYER_HASHES=1`
- `ENGRAM_LAYER_READOUTS=1`
- `ENGRAM_LAYER_PARTITION_GROUPS=1`
- `ENGRAM_PER_HEAD=1`
- `ENGRAM_CANONICALIZE=1`
- `ENGRAM_ATTNRES_MERGE=1`
- `ENGRAM_ATTNRES_MERGE_GAIN=1.5`
- `ENGRAM_UNTIED_PROJ=1`
- `ENGRAM_SPARSE_SCALAR_ADAM=1`
- `GRAD_ACCUM_STEPS=16`
- `MODEL_SEED=5`
- `TRAIN_DATA_SEED=0`

Important caveat: seed variance is material. The reports estimate meaningful
noise around `0.002-0.005`, sometimes larger for unlucky runs. Single-run
improvements below that band are diagnostic, not settled wins.

## Timeline Of Discoveries

### 1. Early ROM / Hashed Memory Feasibility

The first runs established that extra sparse memory paths could train without
breaking the base modded-nanogpt loop, but the early ROM token variants were not
competitive.

Examples from saved logs/reports:

- `rom_token_l2_ga4_blackwell_sm120_20260512_080330` completed around `3.8590`
  final-ish validation.
- `rom_token_engram_gate_shortconv_l2_ga8_20260512_152410` completed around
  `3.3149`, a large improvement over the first ROM attempts but still far from
  later Engram.
- Hashed ROM smoke runs were useful implementation tests, not final quality
  candidates.

Key lesson: the project needed the memory path to behave less like a generic
residual state cache and more like a sparse, content-addressed learned table.

### 2. Engram Became The Main Productive Path

The Engram table shifted the work from token-state ROM toward ngram-addressed
learned memory. The productive ingredients were:

- sparse learned rows,
- canonicalized ngrams,
- layer-specific hashes/readouts,
- sparse optimizer support,
- AttnRes-style learned residual merge,
- readout normalization.

By mid-project, BF80/BF99 Engram runs were already in the `3.25-3.29` region,
with analysis logs showing that masking or removing the Engram memory could
substantially hurt loss. That confirmed Engram was carrying real predictive
signal.

### 3. AttnRes And Readout Normalization Were Real Structural Wins

The best additive Engram line used an attention-residual merge rather than a
plain fixed injection. The merge learns how much memory to add at L2/L8.

Two normalization knobs mattered:

- `ENGRAM_NORMALIZE_READOUT=1`: normalize the final Engram output before adding
  it to the model.
- `ENGRAM_NORMALIZE_MEMORY_HEADS=1`: normalize each retrieved memory head before
  key/value readout.

Interpretation: row norm is a dangerous side channel. Normalizing the read path
lets row direction carry content while preventing raw row scale from becoming an
unstable shortcut.

### 4. Scalar-Row Adam Was The Memory Unlock

The largest practical advance was `ENGRAM_SPARSE_SCALAR_ADAM=1`.

It did not clearly beat full/vector Adam at smaller BF, but it saved enough
optimizer memory to fit BF400 on the 96 GB RTX PRO 6000B instance. BF400 scalar
Adam at LR mul 5.0 reached a replicated seed3/4/5 mean around `3.2457`.

This optimizer is not ordinary Adam with compressed tensors. It stores
row-scalar moments:

- first scalar moment tracks row gradient RMS,
- second scalar moment tracks squared row gradient RMS,
- update direction is the current row gradient normalized by row RMS.

Practical interpretation: Engram mostly needed row-wise adaptive step sizing,
not directional momentum, so scalar moments were good enough and much cheaper.

### 5. BF400 Was The Practical Full-Width Row Ceiling

BF400 full-width `dim768/h1/scalar Adam/GA16` sat near the memory ceiling:

- about `95.4 GB` resident GPU memory,
- PyTorch peak around `89.8-90.0 GiB` allocated/reserved in the scalar-frontier
  report.

BF410/BF420 failed with OOM. Reducing microbatch pressure or disabling compile
did not rescue full-width BF420.

Store-dim compression let larger tables fit, but quality got worse:

- BF400/store256 seed4: `3.2513`
- BF400/store192 seed4: `3.2515`
- BF600/store256 seed4: `3.2492`
- BF800/store192 seed4: `3.2496`

Lesson: "smaller wider table" was not a free capacity win. `ENGRAM_STORE_DIM`
adds a projection bottleneck, and larger row counts become update-starved over a
fixed 1500-step budget.

### 6. Shared Row Address Space Beat Per-Layer Row Isolation

`ENGRAM_LAYER_PARTITION_GROUPS=1` was the clearest replicated same-memory
structural win.

It lets L2 and L8 share the full BF400 row address space while keeping
layer-specific hashes and readouts. Across seeds 3/4/5/6, the mean moved from
about `3.2463` to about `3.2447`.

Key interpretation: row isolation was self-limiting. The table wants shared
address capacity, but not necessarily identical addressing or identical
readouts.

Later simplifications showed:

- no explicit row partitions with layer hashes/readouts was close: `3.2414`,
- shared readout was close but worse: `3.2421`,
- layer-specific hashes mattered more than literal row partitioning.

### 7. More Memory Layers Were Not Monotonic

Adding L10 sometimes helped, but was seed-mixed:

- `2,8,10` beat matched `2,8` on seeds 3/4/5,
- lost on seeds 6/7,
- essentially tied on seed8.

Four-layer probes either finished worse or destabilized. Partition-group splits
for four-layer traffic did not rescue them.

Lesson: more table traffic is not automatically better. Extra consumers can
over-couple or over-train the shared hot rows.

### 8. Trigram-Heavy Rows + Hit-Scaled Reads Produced The Best Line

The best final run came from changing the ngram row allocation and read scaling:

- row factors `0.5,1.5` for bigram/trigram,
- read hit scale exponent `0.25`,
- mean-normalized scaling with min/max bounds.

This produced the `3.2411` SOTA run.

Related probes constrained the region:

- `0.45,1.55`: `3.2450`
- `0.4,1.6`: `3.2466`
- seed4 at `0.5,1.5`: `3.2433`
- seed6 at `0.5,1.5`: `3.2458`

Interpretation: trigram capacity matters, but the useful row split is narrow.
The table is not uniformly capacity-limited.

### 9. Hot Rows Dominate; Cold Rows Are Nearly Disposable

Pruning and masking were highly informative.

From the current-best checkpoint evals:

| Intervention | Val loss |
| --- | ---: |
| Base eval | `3.2418` |
| Random replace cold rows with hit `<2` | `3.2418` |
| Random replace cold rows with hit `<4` | `3.2419` |
| Random replace cold rows with hit `<8` | `3.2430` |
| Random replace cold rows with hit `<16` | `3.2495` |
| Random replace hot rows with hit `>=256` | `3.4121` |
| Random replace hot rows with hit `>=1024` | `3.3316` |
| Random replace hot rows with hit `>=4096` | `3.2899` |

Rows hit fewer than about four times are almost free at eval. The hot tail is
loss-critical.

This moved our understanding more than it moved SOTA: it suggests memory
compression/pruning opportunities, but naive hot-row dropout or hot splitting
did not improve final loss.

### 10. Count-Sketch / Multi-Row Codes Helped Early, Then Faded

The distributed-code direction was conceptually strong but did not beat final
SOTA in tested forms.

Best sketch-slot result:

| Run | 250 | 500 | 750 | 1000 | 1250 | 1500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sketchslotk2_base_aux05_sanitize_cap4` | `8.5079` | `7.2229` | `5.3425` | `3.4963` | `3.3430` | `3.2442` |

It beat SOTA at step 500 (`7.2229` vs `7.2329`) but finished worse
(`3.2442` vs `3.2411`). Unsanitized versions could NaN around step 1000.

Interpretation: auxiliary sketch rows currently act like an early optimization
scaffold or perturbation, not a better final representation.

### 11. Recsys-Inspired Optimizers Did Not Move SOTA

Tested interventions included:

- row-wise AdaGrad,
- frequency-aware learning-rate scaling (FAL),
- within-batch frequency normalization,
- hot-row dropout/suppression.

Observed results:

- FAL was weak at 250 (`8.5168` vs SOTA `8.4993`).
- row-wise AdaGrad was weaker (`8.5299` at 250).
- batch frequency normalization was behind by step 500 (`7.2351` vs `7.2329`).
- hot dropout had some early edges but faded by 750/1000.

Lesson: hot-row dominance is real, but simple recommender-style update
averaging/suppression did not improve this training setup.

### 12. Frozen Memory Rows Were A Useful Control

`ENGRAM_FREEZE_MEMORY=1` tested whether Engram was merely a frozen random
feature table plus learned readout.

Result:

- frozen memory `freezemem_readhit025_s5`: `8.5677` at 250 and `7.3222` at 500,
  much worse than the learned-table SOTA trace (`8.4993`, `7.2329`).

Conclusion: the rows themselves are learning useful content.

### 13. Learned Bigram/Trigram Head Mixing Was Diagnostic, Not A Win

`ENGRAM_LAYER_HEAD_MIX=1` learned per-layer weights over bigram/trigram heads.
It finished close:

- `layerheadmix_readhit025_s5`: `3.2415`

The learned weights leaned bigram in both L2 and L8, especially L8, despite the
SOTA row allocation being trigram-heavy. Global head mixing and static read
scales were worse.

Interpretation: the model wants layer-dependent readout balance. More trigram
row capacity does not mean every layer should read trigrams more heavily.

### 14. Hash Changes Were Not The Bottleneck

Changing hash seed, using stronger avalanche mixing, and adding layer row signs
did not produce a final win.

Notable signal:

- coarse `ENGRAM_LAYER_SIGNS=1` helped early (`8.4975` at 250, `7.2298` at 500)
  but faded by 750/1000.
- combining layer signs with layer-head-mix had an even stronger early 250
  result (`8.4949`) but also faded.

Hard handoff and smooth schedules failed because they changed row semantics
mid-training too abruptly.

Lesson: there may be early interference benefits from signed layer views, but
the representation learned under the signed view does not transfer cleanly to
the unsigned view.

## Best Working Model Of What Engram Is Doing

1. Engram is a learned sparse ngram memory, not just random features.
2. Its useful signal is concentrated in a hot-row tail.
3. Cold rows are mostly disposable after training.
4. Shared row address capacity across L2/L8 is better than isolating layers.
5. Layer-specific hashes/readouts remain useful even when row capacity is
   shared.
6. Row norm is an unstable content channel; read-path normalization helps.
7. More rows help only when they preserve row rank and update density.
8. More consumers/layers increase table traffic and can over-couple the hot
   rows.
9. Scalar-row Adam is good enough because row-adaptive step size matters more
   than vector directional momentum for this table.
10. Early perturbations often help optimization but fade unless the final
    representation remains compatible.

## Negative Results Worth Preserving

- BF > 400 full-width did not fit on 96 GB.
- BF600/BF800 through store-dim compression fit but lost quality.
- Extra memory layers were seed-mixed or worse.
- Four-layer variants and partition-group splits did not help.
- Delta AttnRes was worse.
- Extra-source AttnRes was worse after completion.
- Same-address/shared-readout simplification was close but worse.
- Vector/scalar optimizer variants did not clearly beat the BF400 scalar line
  once memory was accounted for.
- Row-wise AdaGrad, FAL, and batch-frequency normalization did not help.
- Frozen memory rows were much worse.
- Count-sketch / slot attention helped early or was slower, then faded.
- Hot-row dropout/splitting did not produce a durable final gain.
- Avalanche hash and hash seed changes were weak.
- Signed layer views helped early but did not survive late training.

## Code / Infrastructure Fixes That Mattered

- Sparse optimizer diagnostics and row-RMS computation were hardened with stable
  row RMS (`norm / sqrt(width)`) to avoid fp32 square overflow in abnormal rows.
- Distributed hit histograms were fixed to use global reduction for eval/save,
  avoiding rank-local undercount and incorrect unhit masking.
- Launcher run IDs were fixed to include suffix/seed/data-seed, preventing
  same-second jobs from colliding in logs and W&B.
- W&B was turned on for non-smoke runs after the request, though the local cache
  here only preserves the small May 17 subset.
- Queue scripts and remote logs became the reliable source of run provenance.

## Final Takeaway

The project improved the Engram line from early ROM/hashed-memory feasibility
into a strong learned sparse-memory system. The biggest real unlock was not one
small hyperparameter; it was the combination of scalar-row Adam, BF400 row
capacity, shared L2/L8 address space, read-path normalization, trigram-heavy row
allocation, and hit-scaled reads.

The best measured 1500-step validation loss is `3.2411`. The most important
scientific discovery is that Engram capacity is highly nonuniform: a small hot
tail carries most of the loss-critical signal, while many cold rows can be
randomized with almost no eval damage. Future progress is likely to come from
better hot-row/collision learning dynamics or compression-aware training, not
simply larger tables or more memory layers.
