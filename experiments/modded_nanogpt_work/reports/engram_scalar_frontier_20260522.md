# Engram Scalar-Adam Frontier, 2026-05-22

This note records the BF400 scalar-moment Engram runs and the memory boundary probes from the 2x RTX PRO 6000B 96GB instance (`ubuntu@154.54.100.57`, repo `/home/ubuntu/modded-nanogpt`).

## Configuration

Common setup unless noted:

- `BIGRAM_FACTOR=400`
- `ENGRAM_DIM=768`
- `ENGRAM_HEADS=1`
- `ROM_LAYERS=2,8`
- `ENGRAM_ATTNRES_MERGE=1`
- `ENGRAM_LAYER_HASHES=1`
- `ENGRAM_LAYER_READOUTS=1`
- `ENGRAM_LAYER_PARTITIONS=1`
- `ENGRAM_SPARSE_SCALAR_ADAM=1`
- `ENGRAM_LR_MUL=5.0` for the frontier line
- `GRAD_ACCUM_STEPS=16`
- `MODEL_SEED` varied, `TRAIN_DATA_SEED=0`
- W&B enabled for non-smoke runs.

The scalar optimizer stores row-scalar Adam moments for the Engram table. This did not improve quality by itself at BF120, but it reduced memory enough to scale the number of rows.

## Main Results

Validation losses are shown as step 500 / step 1000 / step 1500.

| Run | Seed | LR mul | Val losses |
| --- | ---: | ---: | --- |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_121712` | 3 | 5.0 | `7.2345 / 3.5027 / 3.2457` |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_125723_seed4_data0` | 4 | 5.0 | `7.2383 / 3.5007 / 3.2462` |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_125727_seed5_data0` | 5 | 5.0 | `7.2324 / 3.5012 / 3.2452` |
| `bf400_h1_dim768_lrmul4p5_layerpart_scalaradam_1500_20260522_140958_seed3_data0` | 3 | 4.5 | `7.2350 / 3.5019 / 3.2465` |
| `bf400_h1_dim768_lrmul4p5_layerpart_scalaradam_1500_20260522_133544_seed4_data0` | 4 | 4.5 | `7.2393 / 3.5009 / 3.2456` |
| `bf400_h1_dim768_lrmul4p5_layerpart_scalaradam_1500_20260522_140958_seed5_data0` | 5 | 4.5 | `7.2320 / 3.5009 / 3.2457` |

Summary:

- BF400 LR5 mean final over seeds 3/4/5: about `3.2457`.
- BF400 LR4.5 mean final over seeds 3/4/5: about `3.2459`.
- LR4.5 is effectively neutral/slightly worse. Keep LR5 as the default frontier point.
- BF400 LR5 is a replicated improvement over the BF320 scalar line (`3.2460`, `3.2464`) and over the earlier BF120/BF200 scalar/full-Adam baselines.

## Capacity Boundary

BF400 fits but is near the 96GB ceiling:

- `nvidia-smi` during BF400 training: about `95.4GB` used.
- PyTorch peak for BF400 runs: about `89794 MiB` allocated, `90020 MiB` reserved.

The next row-count probes failed:

| Probe | Result | Notes |
| --- | --- | --- |
| BF410 LR5, GA16 | OOM | Failed in compiled dense MLP/Triton path after table allocation. |
| BF420 LR5, GA16 | OOM | Same failure class. |
| BF420 LR5, GA16, `COMPILE_DENSE_LAYER_BODY=0` | OOM | Failed in hand-written Triton MLP kernel. |
| BF420 LR5, GA32 | OOM | Reducing microbatch pressure did not rescue the larger table. |

Conclusion: for `dim768/h1/scalar Adam/GA16` on 96GB RTX PRO 6000B GPUs, BF400 is the practical row-count ceiling. The next likely improvements need a different memory layout, smaller stored dimension, or a non-row-count idea rather than BF > 400.

## Store-Dimension Tradeoffs

After BF410/BF420 hit the memory ceiling at the default stored width, we tested the existing `ENGRAM_STORE_DIM` path as a memory/compute tradeoff rather than introducing new quality knobs.

| Run | Seed | Rows / store dim | Val losses |
| --- | ---: | --- | --- |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_160935_store256_isolate_seed4_data0` | 4 | `40243256 / 256` | `7.2487 / 3.5084 / 3.2513` |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_160935_store192_isolate_seed4_data0` | 4 | `40243256 / 192` | `7.2558 / 3.5095 / 3.2515` |
| `bf800_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_145437_store192_seed4_data0` | 4 | `80486626 / 192` | `7.2489 / 3.5080 / 3.2496` |
| `bf600_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_152841_store256_seed4_data0` | 4 | `60364854 / 256` | `7.2488 / 3.5069 / 3.2492` |

Interpretation so far:

- BF800/store192 fits and trains at about the same GPU memory level as BF400/store384, but final quality is worse than the BF400 LR5 replicated mean.
- `ENGRAM_STORE_DIM < head_dim` is not a pure row-count knob. It inserts `bigram_embed.memory_proj`, so each memory row is stored in a lower-dimensional subspace and projected back to the head dimension before key/value readout.
- More rows also lower repeat density per row over the fixed 1500-step budget. BF800/store192 had fewer hits per touched row than the BF400-style runs, so it may be update-starved even though it has more address capacity.
- BF600/store256 is the most natural midpoint: 1.5x the BF400 row count with less rank loss than store192. It finished slightly better than BF800/store192 but still behind the BF400 LR5 replicated mean.
- The same-row-count isolate runs show that store-dim itself is already costly: BF400/store256 and BF400/store192 land around `3.251`, about `+0.005` worse than the BF400/store384 seed4 control (`3.2462`). That makes store projection a weak path for quality unless the projection/readout training is changed.

## Extra AttnRes Source Probe

Older logs contained one promising but incomplete structural result: BF120 with an extra L2 source merged at L8 reached step 500 at `7.2174` but did not complete. The completed BF80 version of that idea ended at `3.2583`, so it was not a small-table win. Because this changes the routing/mixing structure rather than only table size or LR, the next targeted rerun is BF400 scalar Adam with `ENGRAM_ATTNRES_EXTRA_SOURCE_LAYER=2` and `ENGRAM_ATTNRES_EXTRA_TARGET_LAYER=8`.

Code detail: the non-delta extra-source path stacks `(main, memory, extra)` for routing, but returns `main + memory_weight * memory + extra_weight * extra`. With zero-initialized query, L8 initially receives about `main + 0.5 * memory + 0.5 * extra` after the merge gain. This is a strong learned skip/source injection, not a tiny routing perturbation.

## Partition-Group Hypothesis

Current frontier runs use `ENGRAM_LAYER_PARTITIONS=1` with the default group assignment, so the L2 and L8 memories get disjoint row ranges. This isolates layers but halves the row budget seen by each layer.

The existing `ENGRAM_LAYER_PARTITION_GROUPS` code can change this without adding a new hyperparameter family: setting `ENGRAM_LAYER_PARTITION_GROUPS=1` keeps layer-specific hashes/readouts while letting all memory layers use the full BF400 row range. This is a cleaner structural probe than larger BF/store-dim, because it tests whether the row ceiling is partly self-inflicted by splitting the table across layers.

Current BF400/partition-group-1 runs:

| Run | Seed | Step 500 val | Control 500 | Step 1000 val | Control 1000 | Step 1500 val | Control 1500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_182255_partgrp1_bf400_seed3_data0` | 3 | `7.2216` | `7.2345` | `3.4987` | `3.5027` | `3.2436` | `3.2457` |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_174707_partgrp1_bf400_seed4_data0` | 4 | `7.2260` | `7.2383` | `3.5026` | `3.5007` | `3.2452` | `3.2462` |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_174707_partgrp1_bf400_seed5_data0` | 5 | `7.2455` | `7.2324` | `3.4989` | `3.5012` | `3.2448` | `3.2452` |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_182255_partgrp1_bf400_seed6_data0` | 6 | `7.2274` | `7.2434` | `3.5001` | `3.5055` | `3.2452` | `3.2481` |

The final result is a consistent improvement across the four matching seeds. Seed3 improves by `0.0021`, seed4 by `0.0010`, seed5 by `0.0004`, and seed6 by `0.0029`. On seeds 3/4/5, the mean moves from `3.2457` to about `3.2445`; including seed6, the mean moves from about `3.2463` to `3.2447`. This is the best structural result in this batch so far: it improves quality at the same BF400 memory footprint by sharing the row address space across L2/L8 while keeping layer-specific hashes/readouts.

## Delta AttnRes Probe

We isolated `ENGRAM_ATTNRES_DELTA=1` on the current BF400 scalar frontier setup.

| Run | Seed | Delta AttnRes | Val losses |
| --- | ---: | --- | --- |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_145833_deltaattnres_seed6_data0` | 6 | on | `7.2507 / 3.5074 / 3.2516` |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_153347_control_seed6_data0` | 6 | off | `7.2434 / 3.5055 / 3.2481` |

Delta AttnRes was worse than the replicated BF400 LR5 line. The same-seed control is better at step 500 (`7.2434` vs `7.2507`), step 1000 (`3.5055` vs `3.5074`), and final (`3.2481` vs `3.2516`), so delta is a real regression here rather than just seed noise.

## More Memory Layers

Using BF400 plus `ENGRAM_LAYER_PARTITION_GROUPS=1` as the new same-memory structural baseline, two seed4 probes add a third memory layer while keeping the shared row address space.

| Run | Layers | Step 500 val | Control 500 | Step 1000 val | Control 1000 | Step 1500 val | Control 1500 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_185909_partgrp1_layers258_bf400_seed4_data0` | `2,5,8` | `7.2364` | `7.2260` | `3.4990` | `3.5026` | `3.2448` | `3.2452` |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_185909_partgrp1_layers2810_bf400_seed4_data0` | `2,8,10` | `7.2292` | `7.2260` | `3.4960` | `3.5026` | `3.2443` | `3.2452` |

The third-layer probes were not obvious at step 500, but both beat the seed4 two-layer partition-group control by final validation. `2,8,10` is the stronger candidate, improving seed4 by `0.0009` over the `2,8` partition-group control. Because that is small relative to run noise, the guarded next step is replication on seeds 3 and 5 before treating it as a new baseline.

## Code Review Notes

- The current "scalar Adam" path is closer to row-wise adaptive normalized SGD than ordinary Adam. It stores scalar moments per row, but the first moment tracks row grad RMS, not a signed vector direction. The direction is the current row gradient normalized by row RMS.
- This explains why scalar Adam can save memory without an obvious catastrophic quality drop: the model may mostly need row-adaptive step sizing, not directional momentum, for the Engram table.
- It also means Adam beta intuition transfers imperfectly. `beta1` smooths scalar gradient magnitude, not direction.
- Store-dim runs have an extra dense projection trained with replicated Adam at `lr_mul=0.5`; this is another reason they are not apples-to-apples row-count scaling.
- The launcher now appends `RUN_SUFFIX`, `MODEL_SEED`, and `TRAIN_DATA_SEED` to run IDs, preventing same-second parallel jobs from sharing console/W&B names.

## Structural Audit

- `ENGRAM_LAYER_PARTITION_GROUPS=1` is a same-memory change, not a hidden capacity increase. The constructor maps all layer partitions to one partition group, so the table row count stays at the BF400 size while L2/L8 keep separate layer hashes and readouts.
- Adding a third memory layer under `ENGRAM_LAYER_PARTITION_GROUPS=1` also does not add rows. It increases how often the shared table is queried and trained, while adding one more layer-specific readout/hash consumer of the same row address space.
- This makes the current `2,5,8` and `2,8,10` probes good tests of whether Engram is under-used structurally rather than simply row-capacity limited.
- The failed extra-source AttnRes probe is a stronger topology edit than it first appears. In non-delta mode, the merge computes weights over `(main, memory, extra)` but returns `main + gain * p_memory * memory + gain * p_extra * extra`, so the extra residual source is injected as an additive branch rather than just a routing reference.
- The current AttnRes query is per layer and zero initialized. At initialization with two sources, the memory branch coefficient is roughly `0.5 * gain = 0.75`; with three sources it is roughly `1/3 * gain = 0.5` per non-main source.

Next decision rule:

- `2,8,10` beat the seed4 `2,8` partition-group control at final validation, so seeds 3 and 5 were launched as replications at `20260522_193628`.
- If the 3-seed mean for `2,8,10` beats the matched `2,8` partition-group mean, promote it to the structural baseline. If not, keep `2,8` with `ENGRAM_LAYER_PARTITION_GROUPS=1`.

Replication status:

| Run | Seed | Layers | Step 500 val | Matched `2,8` 500 | Step 1000 val | Matched `2,8` 1000 | Step 1500 val | Matched `2,8` 1500 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_193628_partgrp1_layers2810_rep_bf400_seed3_data0` | 3 | `2,8,10` | `7.2244` | `7.2216` | `3.4963` | `3.4987` | `3.2426` | `3.2436` |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_185909_partgrp1_layers2810_bf400_seed4_data0` | 4 | `2,8,10` | `7.2292` | `7.2260` | `3.4960` | `3.5026` | `3.2443` | `3.2452` |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_193628_partgrp1_layers2810_rep_bf400_seed5_data0` | 5 | `2,8,10` | `7.2241` | `7.2455` | `3.4993` | `3.4989` | `3.2446` | `3.2448` |

The initial seed3/4/5 result cleared the first replication gate: `2,8,10` improved every matched seed at final validation by `0.0010`, `0.0009`, and `0.0002`. The matched seed3/4/5 mean moved from about `3.2445` for `2,8` partgrp1 to about `3.2438` for `2,8,10`. Later seed6/7 checks weakened this into a seed-mixed candidate rather than a settled baseline.

## Four-Layer Same-Memory Probes

After promoting `2,8,10`, two seed4 same-memory follow-ups were launched without changing BF, LR, optimizer, or partition grouping:

| Run | Layers | Step 500 val | Current `2,8,10` seed4 step 500 | Step 1000 val | Current `2,8,10` seed4 step 1000 |
| --- | --- | ---: | ---: | ---: | ---: |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_201425_partgrp1_layers25810_bf400_seed4_data0` | `2,5,8,10` | `7.2321` | `7.2292` | `3.4965` | `3.4960` |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_201425_partgrp1_layers28910_bf400_seed4_data0` | `2,8,9,10` | `7.2303` | `7.2292` | `3.4993` | `3.4960` |

At step 1000, `2,5,8,10` remained close enough to finish. `2,8,9,10` fell about `+0.0033` behind the current seed4 baseline and was killed to free GPU1.

`2,5,8,10` ultimately finished at `3.2497`, far behind the current `2,8,10` seed4 baseline final of `3.2443`. This means the step-1000 closeness was misleading; adding L5 appears to over-couple or over-train the shared table by the end.

A follow-up same-memory placement probe was launched on GPU1:

| Run | Layers | Status |
| --- | --- | --- |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_203800_partgrp1_layers24810_bf400_seed4_data0` | `2,4,8,10` | active |

Rationale: if four layers help, the useful fourth layer may be an earlier middle layer (`4` or `5`) rather than another tail layer near `8/10`. `2,8,9,10` was the tail-cluster test and looked worse at step 1000; `2,5,8,10` is still plausible; `2,4,8,10` tests whether moving that extra mid layer earlier is better.

Step-500 update for `2,4,8,10`: `7.2309`, better than `2,5,8,10` at `7.2321` and close to current `2,8,10` seed4 at `7.2292`. Step 1000 was also superficially close at `3.4967`, but the run immediately produced NaN train losses at steps 1026/1027 and was killed. Treat this as invalid, not as a close loss.

Because `2,5,8,10` failed final validation, GPU0 was reassigned to a seed6 replication of the promoted `2,8,10` baseline:

| Run | Seed | Layers | Status |
| --- | ---: | --- | --- |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_205332_partgrp1_layers2810_rep_bf400_seed6_data0` | 6 | `2,8,10` | complete: `7.2391 / 3.5015 / 3.2463` |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_210528_partgrp1_layers2810_rep_bf400_seed7_data0` | 7 | `2,8,10` | complete: `7.2365 / 3.4992 / 3.2464` |

The seed6 result weakens the promotion. It finished worse than its matched `2,8` partgrp1 control (`3.2463` vs `3.2452`). The previous `2,8,10` promotion still stands on matched seeds 3/4/5, but it is no longer robust across all checked seeds. Treat `2,8,10` as a promising candidate, not a settled replacement.

To make seed7 interpretable, a matched seed7 `2,8` partgrp1 control was launched:

| Run | Seed | Layers | Status |
| --- | ---: | --- | --- |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_213106_partgrp1_bf400_seed7_control_seed7_data0` | 7 | `2,8` | complete: `7.2519 / 3.4977 / 3.2452` |

Seed7 joins seed6 as a mixed result. Despite a much worse step-500 control (`7.2519` vs `7.2365`) and slightly better step-1000 control (`3.4977` vs `3.4992`), the final `2,8` control beat `2,8,10` (`3.2452` vs `3.2464`). The layer-10 addition is therefore not a robust replacement for `2,8` partgrp1: it wins seeds 3/4/5, loses seeds 6/7, and needs matched seed8 before deciding whether to keep exploring this branch.

A matched seed8 pair is running to reduce uncertainty:

| Run | Seed | Layers | Status |
| --- | ---: | --- | --- |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_220140_partgrp1_layers2810_rep_bf400_seed8_data0` | 8 | `2,8,10` | complete: `7.2352 / 3.4992 / 3.2458` |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_220544_partgrp1_bf400_seed8_control_seed8_data0` | 8 | `2,8` | complete: `7.2397 / 3.5003 / 3.2459` |

Seed8 ended as an effective tie: `2,8,10` beat the matched `2,8` control by only `0.0001`, after larger intermediate advantages at step 500 and step 1000. Across seeds 3/4/5/6/7/8, `2,8,10` is not a clear improvement over `2,8` partgrp1: it wins seeds 3/4/5/8, loses seeds 6/7, and the six-seed mean is essentially identical. Keep `2,8` partgrp1 as the stable structural line; treat L10 as an interesting traffic-increase probe rather than a new baseline.

Because the only other positive three-layer placement was seed4 `2,5,8`, matched seed3/seed5 replications are running. These reuse existing matched `2,8` controls, so no new control jobs are needed:

| Run | Seed | Layers | Status |
| --- | ---: | --- | --- |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_224020_partgrp1_layers258_rep_bf400_seed3_data0` | 3 | `2,5,8` | complete: `7.2318 / 3.5000 / 3.2441`; matched `2,8` final `3.2436` |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_224020_partgrp1_layers258_rep_bf400_seed5_data0` | 5 | `2,5,8` | complete: `7.2374 / 3.5002 / 3.2448`; matched `2,8` final `3.2448` |

The seed3/seed5 replications do not validate `2,5,8` as a better structure. Seed3 loses to matched `2,8` by about `0.0005`; seed5 ties; seed4 had only a `0.0004` win. The seed3/4/5 mean is effectively unchanged from `2,8` partgrp1. Together with the `2,8,10` seed-mixed result and failed four-layer probes, this argues against simply adding more memory layers at BF400/partgrp1.

Hit-density check: at step 500, `2,8` partgrp1 has `262,144,000` Engram hits, `2,8,10` has `393,216,000`, and four-layer probes have `524,288,000`. This confirms that adding memory layers under `ENGRAM_LAYER_PARTITION_GROUPS=1` increases actual shared-table traffic at fixed row count.

## Count-Sketch Distributed Read Probe

The next representation probe is fixed signed multi-row superposition. `ENGRAM_SKETCH_K=2` changes each original hash head from one table row to two signed rows, combined as `sum(sign_i * row_i) / sqrt(K)`. There is no learned router and no new routing optimizer state; the goal is to test a count-sketch / distributed-code version of the existing hash table while leaving `K=1` as the exact baseline behavior.

Two runs are active under the current `2,8` partgrp1 + normmem setup:

| Run | Seed | BF | Status |
| --- | ---: | ---: | --- |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260523_013441_partgrp1_normmem_sketchk2_bf400_seed4_data0` | 4 | 400 | active; reached step 31 at first poll; GPU memory about `95.4/97.9 GB` |
| `bf80_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260523_013441_partgrp1_normmem_sketchk2_bf80_seed4_data0` | 4 | 80 | active; reached step 237 at first poll; GPU memory about `47.5/97.9 GB` |

BF400 `K=2` fits, but it is very close to the memory ceiling because each hash head now touches two sparse rows. BF80 is the lower-memory sanity check for the same mechanism. Early throughput is roughly unchanged at `634-638 ms/step`; the decisive readout will be step-500 and final validation.

## Current Interpretation

- `ENGRAM_LAYER_PARTITION_GROUPS=1` is the clearest win because it removes an artificial per-layer row split while preserving layer-specific hashes and readouts. That means L2/L8 do not read the same exact rows for the same token context, but they do compete inside one full-size row address space instead of isolated half-size spaces.
- Adding L10 appears to exploit the same table a bit more on some seeds, but the seed6/seed7 misses and the four-layer failures suggest shared-table traffic has a narrow useful range. More memory layers are not monotonically better; after a point they likely over-couple rows or over-train high-frequency addresses.
- Step-1000 validation is not reliable enough for layer-count decisions. `2,5,8,10` looked close at step 1000 and then finished badly, while `2,4,8,10` looked close before NaNs. Final validation is the decision point for structural variants.
- `SparseScalarAdam` is not ordinary Adam with smaller moments. It stores scalar row moments: `exp_avg_row` tracks grad RMS, `exp_avg_sq_row` tracks grad RMS squared, and the update direction is the current row gradient normalized by current row RMS. This explains why it saves memory with only modest quality loss: it keeps row-wise adaptive step sizing but discards directional momentum.
- Because scalar Adam's `beta1` smooths magnitude rather than direction, changing Adam betas is a different intervention than in dense/vector Adam. The current evidence does not justify a beta sweep; LR and structural placement have been more informative.
- Store-dim widening is also not pure capacity scaling. Lower `ENGRAM_STORE_DIM` adds a projection bottleneck before readout, so the BF600/BF800 runs traded rank for rows and lost quality despite fitting.

Next structural probe: split four-layer traffic into two partition groups rather than forcing all layers through one group. This keeps the same BF400 row budget but maps `2,5,8,10` or `2,4,8,10` onto two row groups, so early/mid layers and late layers stop fully colliding. Seed4 runs are active:

| Run | Seed | Layers | Groups | Status |
| --- | ---: | --- | ---: | --- |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_232001_partgrp2_layers25810_bf400_seed4_data0` | 4 | `2,5,8,10` | 2 | complete: `7.2348 / 3.4992 / 3.2515` |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260522_232001_partgrp2_layers24810_bf400_seed4_data0` | 4 | `2,4,8,10` | 2 | killed after NaNs from step 1136; step 500 `7.2422`, step 1000 `3.5014` |

Step 500 was not encouraging. `2,5,8,10` partgrp2 was behind the seed4 `2,8,10` candidate (`7.2292`) and also behind the partgrp1 `2,5,8,10` step-500 value (`7.2321`). `2,4,8,10` partgrp2 was clearly weak at `7.2422`, reached only `3.5014` at step 1000, then produced NaNs from step 1136 onward and was killed. The remaining `2,5,8,10` partgrp2 run finished badly at `3.2515`, worse than the already-bad partgrp1 `2,5,8,10` final (`3.2497`). Splitting four-layer traffic into two partition groups did not rescue the "add more layers" direction.

## Memory-Head Normalization Probe

After the layer-count branch flattened out, the next structural probe changes the memory representation instead of adding more traffic. `ENGRAM_NORMALIZE_MEMORY_HEADS=1` normalizes each looked-up memory head before key/value readout, testing whether learned row norm is an uncontrolled channel that hurts generalization or stability. This keeps `ROM_LAYERS=2,8`, `ENGRAM_LAYER_PARTITION_GROUPS=1`, BF400, scalar Adam, LR5, and the existing matched controls.

| Run | Seed | Status | Matched control |
| --- | ---: | --- | --- |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260523_000219_partgrp1_normmem_bf400_seed4_data0` | 4 | complete: `7.2337 / 3.4976 / 3.2436` | partgrp1 seed4 `7.2260 / 3.5026 / 3.2452` |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260523_000219_partgrp1_normmem_bf400_seed6_data0` | 6 | complete: `7.2401 / 3.5033 / 3.2457` | partgrp1 seed6 `7.2274 / 3.5001 / 3.2452` |

Step 500 was worse on both matched seeds, step 1000 was mixed, and final remained mixed. Seed4 improved final validation by `0.0016`; seed6 lost by `0.0005`. The two-seed mean is `3.24465` versus `3.2452` for matched controls, so memory-head normalization is not a clean win yet but is alive enough to replicate on seeds 3 and 5. This is a representation change on the current `2,8` partgrp1 baseline, not an added hyperparameter sweep.

Replication status:

| Run | Seed | Status | Matched control |
| --- | ---: | --- | --- |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260523_003923_partgrp1_normmem_rep_bf400_seed3_data0` | 3 | complete: `7.2319 / 3.4982 / 3.2433` | partgrp1 seed3 `7.2216 / 3.4987 / 3.2436` |
| `bf400_h1_dim768_lrmul5p0_layerpart_scalaradam_1500_20260523_003923_partgrp1_normmem_rep_bf400_seed5_data0` | 5 | complete: `7.2281 / 3.4992 / 3.2442` | partgrp1 seed5 `7.2455 / 3.4989 / 3.2448` |

Replication step 500 was mixed, but finals were positive on both seeds: seed3 improves by `0.0003` and seed5 improves by `0.0006`. Across seeds 3/4/5/6, memory-head normalization improves 3 of 4 matched seeds and the mean moves from about `3.2447` to about `3.2442`. This is smaller than the original `partgrp1` gain, but it is the best current representation-side addition to the stable `2,8` shared-address baseline.

## Current Interpretation

The robust improvement is not simply "more Engram capacity." The strongest replicated line is `ROM_LAYERS=2,8` with `ENGRAM_LAYER_PARTITION_GROUPS=1`: it keeps the same BF400 table size but lets both memory layers use the full row address space while preserving layer-specific hashes/readouts. That consistently beats the normal layer-partition baseline across seeds 3/4/5/6.

The negative results point in the same direction:

- Bigger row counts through smaller stored dimension did not help. BF600/store256 and BF800/store192 fit, but the rank loss and lower update density outweighed the extra addresses.
- Extra memory layers did not help robustly. `2,8,10` had promising seed3/4/5 finals, then lost or tied on later matched seeds; four-layer variants either finished badly or destabilized.
- Splitting four-layer traffic into two partition groups did not rescue the extra-layer idea.
- Delta AttnRes and extra-source AttnRes were worse than the current BF400 scalar frontier.

Working hypothesis: the table is useful when it gives the existing L2/L8 memory path more shared address room, but quality is limited by training dynamics and row usage, not only by raw row count. Adding consumers increases shared-row traffic and can over-train or over-couple the table; widening by reducing store dimension creates a projection bottleneck and update-starves rows. The remaining plausible direction is to improve the representation/optimizer behavior of the same `2,8` shared-address setup, with minimal LR checks only when a structural change clearly shifts update scale.

## Implementation Patch

During review, several sparse optimizer paths still computed row RMS as `x.square().mean(dim=1).sqrt()`. That can overflow in fp32 before the mean if a row gets very large, even when the conceptual RMS would be representable. `train_gpt.py` now uses a stable `x.norm(dim=1) / sqrt(width)` helper for sparse row moments and sampled row-RMS W&B metrics. This should not change normal finite runs materially, but it makes the overflow diagnostics and future abnormal runs safer.

## Launcher Fix

`scripts/run_vec_scalar_delta_goal.sh` was updated so `run_train` appends `RUN_SUFFIX`, `MODEL_SEED`, and `TRAIN_DATA_SEED` to generated run IDs when present. This prevents same-second parallel launches of the same config from writing to the same console file or W&B run.

Relevant remote script path:

- `/home/ubuntu/modded-nanogpt/scripts/run_vec_scalar_delta_goal.sh`

Local mirrored path:

- `experiments/modded_nanogpt_work/scripts/run_vec_scalar_delta_goal.sh`
