# Engram SOTA Progress Report, 2026-05-27

This report rolls up the Engram work after the older scalar-frontier note
(`engram_scalar_frontier_20260522.md`). It is meant to preserve the SOTA trace,
the negative results, and the working model of what the table is doing.

Remote evidence source for the latest runs:

- Host: `ubuntu@154.54.100.57`
- Repo: `/home/ubuntu/modded-nanogpt`
- Main logs: `/home/ubuntu/modded-nanogpt/logs/*20260527*.console.txt`
- Local prior report:
  `experiments/modded_nanogpt_work/reports/engram_scalar_frontier_20260522.md`

## Current SOTA Anchor

Best 1500-step run so far:

| Run | Seed | 250 | 500 | 750 | 1000 | 1250 | 1500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bf400_ngramrows_trigramheavy_0p5_1p5_readhit025_s5_1500_20260527` | 5 | `8.4993` | `7.2329` | `5.3425` | `3.4964` | `3.3406` | `3.2411` |
| `bf400_ngramrows_trigramheavy_0p5_1p5_readhit025_s5_ckpt_1500_20260527` | 5 | `8.4993` | `7.2329` | `5.3425` | `3.4964` | `3.3403` | `3.2413` |

Common current-SOTA config:

- `BIGRAM_FACTOR=400`, `ENGRAM_DIM=768`, `ENGRAM_HEADS=1`
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
- `ENGRAM_LAYER_PARTITIONS=1`
- `ENGRAM_LAYER_PARTITION_GROUPS=1`
- `ENGRAM_PER_HEAD=1`, `ENGRAM_CANONICALIZE=1`
- `ENGRAM_ATTNRES_MERGE=1`, `ENGRAM_ATTNRES_MERGE_GAIN=1.5`
- `ENGRAM_UNTIED_PROJ=1`
- `ENGRAM_SPARSE_SCALAR_ADAM=1`
- `GRAD_ACCUM_STEPS=16`
- `MODEL_SEED=5`, `TRAIN_DATA_SEED=0`

The practical memory ceiling on the 96 GB RTX PRO 6000 Blackwell instance is
still BF400 at this width. BF400 runs sit around `95.4 GB` resident GPU memory.

## SOTA Trajectory

The trajectory from the older scalar-frontier report to the current best is:

| Change | Main evidence | Interpretation |
| --- | --- | --- |
| Scalar-row Adam enabled BF400 | BF400 scalar-Adam mean around `3.2457` on seeds 3/4/5 | The memory saving was the key unlock; scalar moments were good enough to scale row count. |
| `ENGRAM_LAYER_PARTITION_GROUPS=1` | Four-seed mean moved from about `3.2463` to `3.2447` in the 2026-05-22 report | Best robust structural win: L2/L8 keep layer-specific hashes/readouts but share the full row address space. |
| Extra memory layers | `2,8,10` won seeds 3/4/5 but lost/tied seeds 6/7/8; four-layer probes failed or NaN'd | More table traffic is not monotonic. Extra consumers can over-couple or over-train the shared table. |
| Memory-head normalization | Seeds 3/4/5/6 mean improved from about `3.2447` to about `3.2442` | Normalizing each retrieved memory head helps more often than not and fits the scalar-Adam/normed-readout setup. |
| Trigram-heavy row split + hit-scaled read | Best seed5 final `3.2411` | Biggest recent jump. A larger trigram row share plus hit-frequency read scaling is the current SOTA recipe. |

## Structure Findings

### Layer Partitions

The old per-layer partitioning was probably self-limiting. Letting both memory
layers use the full BF400 row address space while preserving layer hashes and
layer readouts was the cleanest robust structural improvement.

Later simplification checks:

| Run | Meaning | Result |
| --- | --- | ---: |
| `bf400_ngramrows_0p5_1p5_sharepart_readhit025_s5_1500_20260527` | No row partitions, keep layer hashes + layer readouts | `3.2414` |
| `bf400_ngramrows_0p5_1p5_layerhash_sharedreadout_readhit025_s5_retry_1500_20260527` | Layer hashes, shared readout, no partitions | `3.2421` |

Interpretation:

- Row partitions are not needed for the current best behavior.
- Layer-specific readouts still have a small but measurable value.
- Layer-specific hashes matter more than literal row partitioning.

### Extra Layers

Adding layers increases table traffic at fixed row count. That sometimes helps
mid-run but did not become a robust final-loss improvement.

Important negatives:

- `2,8,10` was seed-mixed: promising on seeds 3/4/5, weak on 6/7, tie on 8.
- `2,5,8` did not replicate as a better structure.
- Four-layer variants (`2,5,8,10`, `2,4,8,10`, `2,8,9,10`) either finished
  worse or destabilized.
- Splitting four-layer traffic into two partition groups did not rescue it.

Working interpretation: the table wants enough shared row address space, not
arbitrarily more consumers. Extra consumers can increase hit pressure on hot
rows and degrade late generalization.

## Optimizer And Scaling Findings

### Scalar Adam

The scalar Engram optimizer is not ordinary vector Adam with compressed state.
It stores row-scalar moments and uses the current row gradient direction,
normalized by row RMS. This preserves row-wise adaptive step sizing while
discarding directional momentum.

Consequences:

- The memory saving is real and central: it enables BF400.
- Quality loss versus vector moments is small enough to tolerate.
- Adam beta intuition transfers only partially: `beta1` smooths magnitude, not
  vector direction.

### Store-Dim And Wider Tables

Lowering `ENGRAM_STORE_DIM` to fit wider tables was not a pure row-count win.
It inserts a projection bottleneck before key/value readout.

Prior results:

- BF400/store256 and BF400/store192 were already worse at the same row count.
- BF600/store256 and BF800/store192 fit but did not beat BF400/full-width.

Interpretation: simply trading width for rows is not the right capacity axis
for this setup. It reduces per-row representational rank and also update-starves
the larger table over a fixed 1500-step budget.

## Normalization Findings

Two normalizations matter:

- `ENGRAM_NORMALIZE_READOUT=1`: normalize the final Engram readout before it is
  injected into the model.
- `ENGRAM_NORMALIZE_MEMORY_HEADS=1`: normalize each looked-up memory head before
  key/value readout.

`normmem` is not identical to normalizing rows after every optimizer step. It
normalizes each read path, so optimizer state and row magnitudes can still exist,
but row norm is prevented from acting as an uncontrolled direct signal during
the forward read.

This seems especially compatible with scalar Adam: scalar Adam already treats
row scale as part of adaptive step sizing, while normmem prevents row norm from
becoming an overfit content channel.

## N-Gram And Hit-Scaling Findings

The current best uses:

- `ENGRAM_MAX_NGRAM=3`
- `ENGRAM_NGRAM_ROW_FACTORS=0.5,1.5`
- hit-frequency read scaling with exponent `0.25`, min `0.25`, max `4.0`, and
  mean normalization.

This is the best evidence that the table is not uniformly capacity-limited.
Giving the trigram channel more rows and then compensating reads by hit
frequency improved final loss more than simply adding more layers or more total
rows through store-dim compression.

Other row-factor/read-scale probes around this line mostly faded or hurt:

| Probe | Final or status |
| --- | --- |
| `0.45,1.55` row factors | `3.2450` |
| `0.4,1.6` row factors | `3.2466` |
| seed4 at `0.5,1.5` | `3.2433` |
| seed6 at `0.5,1.5` | `3.2458` |

The seed5 `3.2411` is the current best single run, but seed variance is still
material at the `0.002-0.005` scale.

## Pruning And Masking Findings

Checkpoint masking/pruning probes show a strongly skewed importance profile.

From the current best checkpoint evals:

| Eval | Val loss |
| --- | ---: |
| Base eval | `3.2418` |
| Random replace cold rows with hit `<2` | `3.2418` |
| Random replace cold rows with hit `<4` | `3.2419` |
| Random replace cold rows with hit `<8` | `3.2430` |
| Random replace cold rows with hit `<16` | `3.2495` |
| Random replace cold rows with hit `<32` | `3.2653` |
| Random replace hot rows with hit `>=256` | `3.4121` |
| Random replace hot rows with hit `>=1024` | `3.3316` |
| Random replace hot rows with hit `>=4096` | `3.2899` |
| Random replace hot rows with hit `>=16384` | `3.2646` |

Interpretation:

- Rows hit fewer than about 4 times are nearly free at eval.
- The hot tail is extremely loss-critical.
- This is useful for memory/compression/pruning, but naive hot-row dropout or
  hot splitting has not moved SOTA.

Important code review result: the hit histogram is now globally reduced for
distributed eval/save via `global_hit_hist()`, and validation uses
`begin_global_hit_hist_eval()`. Earlier concern about rank-local hit histograms
silently undercounting rows is addressed in the current worktree.

## Sketching And Distributed-Code Findings

### Basic Count Sketch / Superposition

The simple signed multi-row representation (`ENGRAM_SKETCH_K=2` or superpose
variants) gave occasional early bumps but generally faded. It did not beat the
current baseline as a permanent representation change.

Representative negatives:

- Superpose base `k=2` aux schedules often improved step 250 slightly, then
  lost by 500/750.
- Basic sketch `aux=0.5` could improve step 500, but not final.
- Learned slot/combine mix was worse early in tested forms.

### Sketch Slot Readout

The most interesting new branch is slot-level readout with a base row plus one
auxiliary sketch row:

| Run | 250 | 500 | 750 | 1000 | 1250 | 1500 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `sketchslotk2_base_aux05` | `8.5079` | `7.2229` | `5.3425` | `3.4963` | - | - | NaN after about step 1000 |
| `sketchslotk2_base_aux05_sanitize_cap4` | `8.5079` | `7.2229` | `5.3425` | `3.4963` | `3.3430` | `3.2442` | complete |
| `sketchslotk2_base_aux05to025_s500_start750` | `8.5079` | `7.2229` | `5.3425` | `3.4974` | - | - | NaN right after 1000 |
| `sketchslotk2_base_aux05to025_s500_start750_sanitize_cap4` | `8.5079` | `7.2229` | `5.3425` | `3.4974` | `3.3441` | pending | active but weak |
| `sketchslotk2_base_aux05to0_s500_start750_sanitize_cap4` | `8.5079` | `7.2229` | `5.3425` | `3.4972` | `3.3468` | killed | weak |
| `sketchslotk2_base_aux05to0_s250_start500_sanitize_cap4` | `8.5079` | `7.2229` | `5.3493` | killed | - | - | weak |
| `sketchslotk2_base_aux05_basehist_sanitize_cap4` | `8.5065` | `7.2450` | killed | - | - | - | weak |
| `sketchslotattnk2_base` | `8.5077` | `7.2358` | `5.3487` | killed | - | - | weak and slower |
| `sketchslotattnk2_base_aux05` | `8.5096` | `7.2395` | `5.3493` | killed | - | - | weak and slower |

Interpretation so far:

- The auxiliary sketch row clearly helps early optimization at step 500
  (`7.2229` versus SOTA trace `7.2329`).
- Without sanitizer, this branch can NaN around step 1000.
- Sanitizer fixes the NaN but not the final-loss fade: `3.2442`, worse than
  `3.2411`.
- Annealing aux contribution to `0.25` keeps the run finite only when combined
  with sanitizer, but its step-1000 and step-1250 losses are worse than SOTA.
- The live `aux05 -> 0` run is the sharper test of "aux row as early
  curriculum only"; the `start750/s500` version is also weak at step 1000, so
  an earlier shutoff (`start500/s250`) is now running.
- Recording only the base slot in the hit histogram did not preserve the early
  sketch-slot gain. It reached `7.2450` at step 500 versus `7.2229` for the
  original sanitized sketch-slot run, so aux-address hit statistics are not the
  simple cause of the late fade.

Working interpretation: extra sketch rows are currently useful as an early
training perturbation/scaffold, not as a better final representation. If the
branch is worth continuing, the next useful knobs are earlier/faster shutoff or
making the aux path stop affecting hit statistics once annealed away.

A separate `ENGRAM_SKETCH_SLOT_ATTENTION=1` branch was added on 2026-05-28.
This keeps `ENGRAM_SKETCH_SLOT_READOUT=1`, but replaces the independent
sigmoid-gated sum over slots with a query-dependent softmax over the `k`
candidate hashed rows. This is the first direct test of "k hash functions pull
k rows, then attention reads out the K/Vs". The initial run is
`bf400_ngramrows_0p5_1p5_sketchslotattnk2_base_readhit025_s5_1500_20260528`.
Both slot-attention probes were killed after step 750 because they trailed the
SOTA trace by about `0.006` to `0.007` and ran slower than the additive sketch
slot-readout path.

### Frozen Memory Rows

`ENGRAM_FREEZE_MEMORY=1` was added on 2026-05-28 as a clean non-learned-row
control. It freezes the actual `bigram_embed.embedding.weight`, removes the
table from dense/sparse optimizers, and leaves the readout/projection/AttnRes
and dense model trainable.

Live control:

| Run | 250 | 500 | 750 | 1000 | 1250 | 1500 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `freezemem_readhit025_s5` | `8.5677` | `7.3222` | killed | - | - | - | weak |

Result: the fixed random table is much worse than the learned-table SOTA trace
at both 250 (`8.5677` versus `8.4993`) and 500 (`7.3222` versus `7.2329`).
The useful conclusion is that Engram is not just a frozen random-feature table
with learned readout; learning the rows is doing real work.

### Hot-Row Training Masking

Prior real hot-row dropout with inverted scaling, `p=0.05` and
`min_hits=16384`, completed at `3.2418`, close to but not better than the
`3.2411` SOTA. A no-inverted-scale mode was added on 2026-05-28 via
`ENGRAM_HIT_DROPOUT_INVERT_SCALE=0` to test actual hot-row suppression instead
of expectation-preserving dropout.

| Run | 250 | 500 | 750 | Status |
| --- | ---: | ---: | ---: | --- |
| `hotdrop005_min16384_noscale` | `8.5217` | killed | - | too harsh |
| `hotdrop002_min16384_noscale` | `8.5060` | `7.2315` | `5.3499` | killed; early help faded |
| `hotdrop002to0_s500_250_min16384_noscale` | `8.5060` | `7.2315` | `5.3452` | killed at 1000 |
| `hotdrop001_min16384_noscale` | `8.5007` | `7.2364` | killed | weak at 500 |

The `p=0.02` no-scale run slightly beat the SOTA trace at step 500
(`7.2315` versus `7.2329`) but faded badly by step 750 (`5.3499` versus
`5.3425`). The current follow-up tests whether this is useful only as early
regularization: keep `p=0.02` through step 500 then anneal to 0 by 750. It
preserved the same step-500 early edge and reduced the step-750 fade versus
continuous no-scale dropout (`5.3452` versus `5.3499`), but still trails the
SOTA trace at 750 (`5.3425`). Because the gap is only about `0.0027`, it is
was carried to step 1000. It reached `3.4984` at step 1000 versus SOTA
`3.4964`, so it was killed. A continuous `p=0.01` no-scale bracket was less
disruptive at step 250 but weak at step 500, so it was killed.

### Frequency-Aware Optimizer Probes

`bf400_ngramrows_0p5_1p5_fal_readhit025_s5_1500_20260528` was launched on
2026-05-28. It keeps the current SOTA structure but enables
`ENGRAM_SPARSE_FAL=1` on top of scalar Adam. This is a recommender-inspired
frequency-aware update test: rows with fewer accumulated sparse hits get
smaller update scale relative to the hottest rows.

| Run | 250 | 500 | 750 | Status |
| --- | ---: | ---: | ---: | --- |
| `fal_readhit025_s5` | `8.5168` | killed | - | weak at 250 |
| `rowadagrad_readhit025_s5` | `8.5299` | killed | - | weak at 250 |
| `batchfreqnorm_readhit025_s5` | `8.5091` | `7.2351` | killed | weak at 500 |

FAL was too far behind at the first eval (`8.5168` versus SOTA `8.4993`), so
it was killed and GPU 0 was recycled into the cleaner row-wise AdaGrad probe
using `ENGRAM_SPARSE_ROW_ADAGRAD=1`. Row-wise AdaGrad was even weaker at 250
(`8.5299`), so it was also killed. GPU 0 is now testing
`ENGRAM_SPARSE_BATCH_FREQ_NORM=1` with scalar Adam: this divides a row's
coalesced sparse gradient by its within-batch occurrence count before the
optimizer step, a direct hot-row dominance intervention that does not change
the forward table structure. Its 250 result is behind SOTA (`8.5091` versus
`8.4993`) but not catastrophic, so it was carried to 500. It remained behind
there (`7.2351` versus SOTA `7.2329`) and was killed. The useful conclusion is
that simply averaging repeated within-batch row updates is not enough to fix
hot-row dominance.

### Scheduled Hot Splitting

`ENGRAM_HOT_SPLIT_AUX_SCALE_FINAL`,
`ENGRAM_HOT_SPLIT_AUX_SCALE_SCHEDULE_START`, and
`ENGRAM_HOT_SPLIT_AUX_SCALE_SCHEDULE_STEPS` were added on 2026-05-28. The
purpose is to test hot-split aux rows as an early scaffold rather than a
permanent representation change.

Motivation: the prior value-only, two-aux-slot hot split at
`min_hits=16384, aux_scale=0.025` reached `8.4992` / `7.2314` at steps
250/500, but faded to `5.3482` by step 750. The new live run keeps the same
setup through step 500 and anneals aux scale to zero by step 750:
`bf400_ngramrows_0p5_1p5_hotsplit_valueonly16384_auxslots2_aux0025to0_s500_250_readhit025_s5_1500_20260528`.
The first launch missed `ENGRAM_MANUAL_SPARSE_COALESCE=1` and hit the known
two-aux-slot sparse coalesce CUDA illegal-address failure during the first
optimizer step. It was relaunched with manual sparse coalescing as
`bf400_ngramrows_0p5_1p5_hotsplit_valueonly16384_auxslots2_aux0025to0_s500_250_manualcoal_readhit025_s5_1500_20260528`.

Current trace:

| Run | 250 | 500 | 750 | 1000 | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `hotsplit_valueonly16384_auxslots2_aux0025to0_s500_250_manualcoal` | `8.5128` | `7.2278` | `5.3443` | `3.4995` | killed |

This recovered much of the old hot-split fade: `5.3443` at 750 is still behind
SOTA `5.3425`, but much better than the old unscheduled two-slot fade
(`5.3482`). The recovery did not last to 1000 (`3.4995` versus SOTA `3.4964`),
so the run was killed. Conclusion: hot-split aux rows are another early
optimization perturbation, not a durable SOTA improvement in this form.

### Learned Head Mixing

`bf400_ngramrows_0p5_1p5_layerheadmix_readhit025_s5_1500_20260528` was
launched on 2026-05-28. It keeps the SOTA table structure but enables
`ENGRAM_LAYER_HEAD_MIX=1`, adding learned per-layer weights over the two
Engram heads, currently bigram and trigram. The initialization is functionally
equivalent to the baseline uniform `sum/sqrt(2)` merge, so this tests whether
L2 and L8 want different learned bigram/trigram readout balances rather than
changing capacity allocation.

Early trace:

| Run | 250 | 500 | 750 | 1000 | 1250 | 1500 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `layerheadmix_readhit025_s5` | `8.4999` | `7.2336` | `5.3420` | `3.4968` | `3.3416` | `3.2415` | complete; close but not SOTA |

At step 250, the learned head weights already moved both layers toward head 0:
L2 `0.557/0.443`, L8 `0.560/0.440`. By step 500 they moved farther:
L2 `0.568/0.432`, L8 `0.587/0.413`. Since head 0 is the bigram head in this
two-head setup, this suggests the SOTA trigram-heavy row allocation may not mean
the readout should be trigram-heavy at every layer/time. The 500 loss is just
behind SOTA (`7.2336` versus `7.2329`), but the 750 loss is slightly ahead
(`5.3420` versus `5.3425`). It gives the tiny edge back at 1000 (`3.4968`
versus `3.4964`) and finishes at `3.2415`, narrowly behind the `3.2411` anchor.
The final learned weights remain bigram-leaning in both layers: L2
`0.566/0.434`, L8 `0.582/0.418`. Conclusion: learned layer-specific
bigram/trigram readout balance is tolerated and diagnostic, but the extra
freedom alone is not a quality win.

### Static N-Gram Read Scaling

`bf400_ngramrows_0p5_1p5_readscale112_088_readhit025_s5_1500_20260528` was
launched after the layer-head-mix 500 eval. It sets
`ENGRAM_NGRAM_READ_SCALES=1.12,0.88`, which approximately matches the effective
bigram/trigram multiplier implied by the learned 250 head-mix weights while
keeping the rest of the SOTA structure unchanged.

| Run | 250 | 500 | 750 | Status |
| --- | ---: | ---: | ---: | --- |
| `readscale112_088_readhit025_s5` | `8.5033` | `7.2367` | killed | weak at 500 |

The fixed read-scale test trails layer-head mix at 250 (`8.5033` versus
`8.4999`) and gets worse at 500 (`7.2367` versus layer-head-mix `7.2336` and
SOTA `7.2329`). This suggests the learned/per-layer part matters more than a
fixed global bigram lean.

`bf400_ngramrows_0p5_1p5_headmix_readhit025_s5_1500_20260528` was launched
after this miss. It enables `ENGRAM_HEAD_MIX=1`, a single learned global
bigram/trigram mixture shared by L2 and L8. This separates "learned mixing is
useful" from "per-layer learned mixing is specifically needed."

| Run | 250 | 500 | 750 | Status |
| --- | ---: | ---: | ---: | --- |
| `headmix_readhit025_s5` | `8.5011` | `7.2353` | killed | weak at 500 |

The global mix quickly moved toward bigram (`0.563/0.437` at 250 and
`0.582/0.418` at 500), but it was worse than both SOTA and layer-head mix. This
keeps the useful signal narrow: the model appears to want layer-dependent
bigram/trigram balance, not just one global shift toward bigram.

`bf400_ngramrows_0p75_1p25_layerheadmix_readhit025_s5_1500_20260528` was
launched next. It keeps the learned per-layer head mix but relaxes row
allocation from `0.5,1.5` to `0.75,1.25`, testing whether the learned bigram
readout preference means the SOTA row split is too trigram-heavy once per-layer
mixing is available.

This direction failed immediately: 250 val was `8.5624`, much worse than
`8.4999` for `0.5,1.5` layer-head-mix. It was killed. The important
interpretation is that the learned bigram-leaning readout does not imply that
bigram row capacity is under-allocated; if anything, reducing trigram capacity
damages the run sharply.

`bf400_ngramrows_0p25_1p75_layerheadmix_readhit025_s5_1500_20260528` was
launched in parallel after the original layer-head-mix run completed. It pushes
the opposite direction, testing whether the bigram-leaning readout is actually
compensating for an allocation that should be even more trigram-heavy.

This also failed early: 250 val was `8.5127`, clearly worse than `8.4999` for
the original `0.5,1.5` layer-head-mix run. It was killed. Combined with the
`0.75,1.25` miss, the row-allocation lesson is sharper than expected: the
existing `0.5,1.5` split appears close to a narrow useful region. Learned
readout weights can lean bigram, but both giving bigrams more rows and giving
trigrams more rows hurt early optimization.

`bf400_ngramrows_0p25_1p75_noheadmix_readhit025_s5_1500_20260528` was launched
after the `0.75,1.25` miss. It uses the same more-trigram-heavy row split
without learned head mixing, isolating whether extra trigram rows help
independently or only together with adaptive layer readout weights.

This no-head-mix isolation run was also weak: 250 val was `8.5060`, so it was
killed. The conclusion now looks robust: the `0.5,1.5` row allocation is not
obviously leaving quality on the table in either direction.

`bf400_ngramrows_0p5_1p5_avalanche_readhit025_s5_1500_20260528` was launched
after the `0.25,1.75` layer-head-mix miss. It returns to the SOTA row split but
sets `ENGRAM_AVALANCHE_HASH=1`, testing whether stronger hash mixing changes
collision/hot-row structure enough to help.

Avalanche was weak at 250: `8.5048` versus anchor `8.4993`, and was killed.
This does not support the idea that a stronger avalanche mixer improves the
current hash/collision structure.

`bf400_ngramrows_0p5_1p5_hashseed1_readhit025_s5_1500_20260528` was launched
after the `0.25,1.75` no-head-mix miss. It keeps the baseline hash form but
changes `ENGRAM_HASH_SEED=1`, testing whether the current seed-0 hash draw is
materially lucky or unlucky.

Hash seed 1 was weak at 250: `8.5043` versus anchor `8.4993`, and was killed.
Together with the avalanche miss, this suggests the current seed-0/simple mixer
is not obviously the bottleneck.

`bf400_ngramrows_0p5_1p5_layerrowsigns_readhit025_s5_1500_20260528` was
launched after the avalanche miss. It keeps the shared table and layer-specific
hash/readout setup, but enables `ENGRAM_LAYER_ROW_SIGNS=1`. This tests whether
L2/L8 sharing would benefit from signed per-layer row views that reduce direct
interference while preserving shared capacity.

Layer-row signs were weak at 250: `8.5039` versus anchor `8.4993`, and were
killed. Randomizing each layer's row/dim view does not seem to help the shared
table.

`bf400_ngramrows_0p5_1p5_layersigns_readhit025_s5_1500_20260528` was launched
after the hashseed1 miss. This is the coarser companion to layer-row signs:
`ENGRAM_LAYER_SIGNS=1` applies a fixed per-layer/per-head sign view rather than
a per-row/per-dimension sign view.

Initial signal is positive: 250 val was `8.4975`, ahead of the `8.4993` anchor
and much better than layer-row signs (`8.5039`). It held at 500 with `7.2298`
versus anchor `7.2329`, but faded at 750 with `5.3463` versus anchor `5.3425`,
then stayed behind at 1000 with `3.4981` versus anchor `3.4964`. This suggests
coarse signed layer/head views help early optimization or interference, but
the always-on version does not preserve the late SOTA trajectory. It was killed
at 1000.

`bf400_ngramrows_0p5_1p5_detachkey_readhit025_s5_1500_20260528` was launched
after the layer-row-sign miss. It sets `ENGRAM_DETACH_KEY_MEMORY=1`, so memory
rows learn through the value path but not through key/gate gradients. This tests
whether key-side gradients are adding noise or hot-row pressure.

Detach-key was weak at 250: `8.5090`, and was killed. The key/gate gradient
path appears useful rather than merely noisy.

`bf400_ngramrows_0p5_1p5_layersigns_layerheadmix_readhit025_s5_1500_20260528`
was launched after detach-key missed. It combines the promising coarse
`ENGRAM_LAYER_SIGNS=1` signal with `ENGRAM_LAYER_HEAD_MIX=1`, testing whether
signed layer/head views and learned bigram/trigram readout balance compose.

Early signal is strong: 250 val was `8.4949`, better than both layer signs alone
(`8.4975`) and the anchor (`8.4993`). The learned head mix is already
bigram-leaning, especially in L8 (`0.565/0.435`). At 500 it remained ahead of
the anchor (`7.2307` versus `7.2329`), though the always-on layer-signs-only
run was slightly better at that point (`7.2298`). It then faded harder at 750
(`5.3477` versus anchor `5.3425`) and was killed. The combination improves
early optimization but does not fix the late sign trajectory.

Because always-on signs faded late, a scheduled variant was added in code:
`ENGRAM_LAYER_SIGN_SCALE`, `ENGRAM_LAYER_SIGN_SCALE_FINAL`,
`ENGRAM_LAYER_SIGN_SCALE_SCHEDULE_START`, and
`ENGRAM_LAYER_SIGN_SCALE_SCHEDULE_STEPS`. These blend the coarse layer sign view
toward a weaker signed view over training. The first scheduled run is
`bf400_ngramrows_0p5_1p5_layersigns1to025_s500_250_layerheadmix_readhit025_s5_1500_20260528`,
which starts at full layer signs and anneals to scale `0.25` from step 500 to
750, while keeping layer-head-mix enabled. As expected, before the schedule
changes the effective model, it matches the always-on combo at 250/500:
`8.4949`, `7.2307`. The smooth anneal was bad at 750 (`5.4014`), likely
because the linear sign-strength blend crosses near-zero factors for negatively
signed heads. It was killed.

After the always-on combo faded at 750, a second scheduled run was launched:
`bf400_ngramrows_0p5_1p5_layersigns1to0_s500_250_layerheadmix_readhit025_s5_1500_20260528`.
It uses the same schedule window but anneals the layer-sign scale all the way to
`0.0`, testing whether the early sign benefit can be treated as a curriculum
that hands off to the original unsigned representation.

That smooth `1 -> 0` schedule was also killed before reaching the anneal because
the `1 -> 0.25` result showed the smooth interpolation itself is suspect. Two
hard-handoff probes replaced it:
`bf400_ngramrows_0p5_1p5_layersignshardoff500_layerheadmix_readhit025_s5_1500_20260528`
and
`bf400_ngramrows_0p5_1p5_layersignshardoff650_layerheadmix_readhit025_s5_1500_20260528`.
Both use full signs early and switch to unsigned in one step, at 500 and 650
respectively.

The hard handoff failed. The 500-step switch reached `5.3810` at step 750, and
the 650-step switch reached `5.3970` at step 750, versus the `5.3425` SOTA
anchor. This is worse than the always-on signed-layer fade, not a rescue. The
likely reason is representational discontinuity: rows/readouts were trained
under the signed view, and flipping back to unsigned changes the semantics of
half the layer/head channels at once.

A cleaner follow-up branch was added: `ENGRAM_LAYER_SIGN_AUX_SCALE`. This keeps
the unsigned base path active throughout training, then adds a signed layer view
as an auxiliary value/key branch:

`base + scale * signed_aux`, RMS-normalized by `sqrt(1 + scale^2)`.

This tests whether the early benefit of coarse layer signs is usable as an
auxiliary gradient/readout perturbation without ever forcing the model to hand
off between incompatible row views. Two probes are live:

- `bf400_ngramrows_0p5_1p5_layersignaux05to0_s500_500_layerheadmix_readhit025_s5_1500_20260528`:
  starts with aux scale `0.5`, decays to `0.0` from step 500 to 1000.
- `bf400_ngramrows_0p5_1p5_layersignaux025_layerheadmix_readhit025_s5_1500_20260528`:
  keeps a mild permanent aux scale `0.25`.

Both signed-aux probes were weak immediately: each reached `8.5249` at step
250, versus the `8.4993` SOTA anchor and the `8.4949` early value from
multiplicative signs plus layer-head mix. They were killed. This rules out the
simple "signed view as an extra readout branch" version; the early layer-sign
benefit appears tied to changing/replacing the effective memory view, not just
adding signed gradients alongside the unsigned path.

After the signed branch miss, two non-sign structural probes were launched:

- `bf400_ngramrows_0p5_1p5_readoutdelta_readhit025_s5_1500_20260528`:
  disables full layer readouts and enables `ENGRAM_LAYER_READOUT_DELTA=1`, so
  the model has a shared Engram readout plus per-layer residual key/value
  projections. This tests a middle point between the close-but-worse shared
  readout run (`3.2421`) and fully separate layer readouts.
- `bf400_ngramrows_0p5_1p5_latentmixfsq_retry_readhit025_s5_1500_20260528`:
  enables `ENGRAM_LATENT=1` and `ENGRAM_LATENT_MIX_NGRAM=1`, with layer
  partitions disabled because latent addressing does not support them. This
  mixes learned hidden-state FSQ codes with ngram hashes, testing an adaptive
  latent/stable-hash direction rather than pure lexical ngram hashing.

## Negative Results Worth Remembering

These are important because they constrain future work:

- BF > 400 at full width does not fit on the current 96 GB GPUs.
- BF600/BF800 via store-dim compression fits but loses quality.
- More memory layers are not monotonically better.
- Four-layer variants and partition-group splits did not rescue extra-layer
  traffic.
- Delta AttnRes was worse.
- Extra-source/direct AttnRes variants were close mid-run but faded or hurt.
- Same-address layer readout was worse; layer-specific hashing matters.
- Removing layer readouts is close but worse (`3.2421` vs `3.2411`).
- Row partitions can be removed/simplified, but this is mostly a simplification
  (`3.2414`) rather than a clear new best.
- Hot-row dropout/splitting has not become a quality win.
- Learned sketch slot mix was bad in the tested form.

## Working Model

The Engram table is not behaving like a generic "more rows always better"
embedding table.

Current model:

1. The table is hot-row dominated. A small tail of highly-hit rows carries much
   of the loss-critical signal.
2. Low-hit rows are mostly disposable, at least after training.
3. Sharing the row address space across L2/L8 is better than isolating them, but
   adding too many consumers over-trains or entangles the shared rows.
4. Row norm is a dangerous side channel; normalizing memory heads/readouts helps
   make the table behave more like directional content.
5. Trigram capacity is more valuable than uniform capacity in the current
   data/model regime.
6. Count-sketch style distributed codes can improve early optimization, but the
   extra rows currently become harmful or unstable late.

## Current Live Runs

As of the latest poll on 2026-05-28:

| GPU | Run | Last known status |
| ---: | --- | --- |
| 0 | `bf400_ngramrows_0p5_1p5_latentmixfsq_retry_readhit025_s5_1500_20260528` | active; latent FSQ code mixed with ngram hash, no layer partitions |
| 1 | `bf400_ngramrows_0p5_1p5_readoutdelta_readhit025_s5_1500_20260528` | active; shared readout plus per-layer residual readout deltas |

Decision rule:

- If layer-head mix is close or better early, keep it through 1000 because its
  learned weights may need time to move away from the uniform baseline.
- Compare global head mix to layer-head mix. If global is weak while per-layer
  stays close, the L2/L8-specific weighting is likely carrying the signal.

## Next High-Signal Directions

1. Finish the layer-head-mix and global-head-mix probes.
2. If early auxiliary paths keep fading, move away from permanent sketch/hot-row
   capacity and toward controlled early curriculum/readout perturbations.
3. Try a cleaner pruning/memory-saving path based on the cold-row result:
   rows hit `<4` appear nearly free at eval.
4. Consider making hit statistics slot-aware for sketch-slot runs. The current
   slot path records all aux addresses even if the aux contribution is annealed
   down, which may contaminate hit-scaled reads.
5. Keep `2,8`, shared row address space, layer-specific hashes/readouts,
   normmem, trigram-heavy rows, and read-hit scaling as the default baseline
   unless a new run beats the `3.2411` anchor.
