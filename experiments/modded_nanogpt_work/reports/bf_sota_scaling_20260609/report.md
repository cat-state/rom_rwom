# BF Scaling Of Current SOTA Meta, 2026-06-09

This report started from the six-run BF sweep launched on
`root@31.22.104.101`. Everything except `BIGRAM_FACTOR` was held at the SOTA
meta available at launch time:

- `MODEL_SEED=5`, `TRAIN_DATA_SEED=0`
- `ROM_LAYERS=2,8`
- `ENGRAM_DIM=768`, `ENGRAM_HEADS=1`
- `ENGRAM_MAX_NGRAM=3`, `ENGRAM_NGRAM_ROW_FACTORS=0.5,1.5`
- read-hit scaling exponent `0.25`
- readout normalization + memory-head normalization
- AttnRes merge gain `1.5`
- scalar sparse Adam
- `GRAD_ACCUM_STEPS=16`

The parameter count plotted is the learned Engram table parameter count
(`engram_table_numel`). The base transformer and small projection/readout
parameters are fixed across this sweep, so the Engram table is the scaling axis.

Historical-SOTA caveat: the absolute best archived Engram run remains
`bf400_ngramrows_trigramheavy_0p5_1p5_readhit025_s5_1500_20260527` at `3.2411`.
That exact log is not present on the new 2026-06-09 instance, so this report uses
the June 9 rerun grid as the apples-to-apples scaling slice. Its BF400 endpoint
is `3.2458`, which should be interpreted as the reproduced BF400 anchor for this
sweep, not as a replacement for the historical `3.2411` record.

Artifacts:

- HTML report: `index.html`
- Plot: `bf_sota_scaling_vs_params.png`
- Validation curves: `bf_sota_scaling_curves.png`
- Data: `summary.csv`
- Readoutdelta overlay data: `summary_readoutdelta.csv`

Clarification after later readoutdelta results: the original table below is the
pre-readoutdelta SOTA-meta BF sweep. The corrected current-meta sweep is now
queued as `bf{40,80,120,200,300,400}_sota_meta_bfscale_seed5_1500_20260609`,
using the readoutdelta settings that became the best replicated
parameter-efficiency branch. GPU1 started the BF40 run on 2026-06-09 while GPU0
was still finishing the read-hit follow-up.

First corrected-sweep checkpoint: BF40 current-meta/readoutdelta reaches
`8.5289` at step 250. That is worse than the original BF40 SOTA-meta trajectory
and matches the broader pattern that readoutdelta was negative at BF120 but
useful around BF200-BF300. The low-BF end is still being carried to final for
the scaling curve, not because it looks locally promising.

## Results

| BF | Final val | Engram table params | Peak MiB | Touched fraction |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 3.2602 | 1.55B | 29,732 | 0.2223 |
| 80 | 3.2560 | 3.09B | 35,673 | 0.1186 |
| 120 | 3.2525 | 4.64B | 41,614 | 0.0808 |
| 200 | 3.2505 | 7.73B | 53,496 | 0.0494 |
| 300 | 3.2480 | 11.59B | 68,349 | 0.0332 |
| 400 | 3.2458 | 15.45B | 89,794 | 0.0250 |

## Interpretation

Scaling BF still helps, but the curve is shallow and clearly sublinear.

The best point is BF400 at `3.2458`, but BF200 is already `3.2505`. Doubling
from BF200 to BF400 adds about `7.7B` Engram table parameters and roughly
`36.3 GiB` peak allocation for a `0.0047` loss improvement. That improvement is
real in this single-seed sweep, but it is on the same order as the seed-variance
band established in prior reports.

The touched-row fraction falls monotonically as BF grows:

- BF40 touches about `22.2%` of rows by step 1500.
- BF400 touches about `2.5%` of rows by step 1500.

This reinforces the prior hot-row/cold-row story. Larger tables reduce
collision pressure and still improve loss, but a growing fraction of parameters
are cold or barely-trained under the fixed 1500-step budget. The scaling result
therefore argues against treating raw row count as the main future path.

## Consequence For Next Runs

The next GPU experiments should target parameter efficiency and collision/hot-row
structure rather than simply increasing BF:

1. BF200 + sketch-slot auxiliary capacity:
   `bf200_sota_sketchslotk2_base_aux05_sanitize_cap4_seed5_1500_20260609`
2. BF200 + learned layer-dependent ngram/head mix:
   `bf200_sota_layerheadmix_seed5_1500_20260609`

These completed on the same instance. The hypothesis was that if BF200 is
already close to BF400, a better allocation/read structure may recover some of
the BF400 gain without doubling table parameters.

## Follow-Up Run Status

Early follow-up results support continuing the parameter-efficiency branch:

| Run | 250 | 500 | 750 | 1000 | 1250 | 1500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF200 baseline SOTA meta | 8.5093 | 7.3076 | 5.3587 | 3.5038 | 3.3507 | 3.2505 |
| BF200 + sketch-slot k2 aux0.5 sanitize/cap4 | 8.5110 | 7.2435 | 5.3507 | 3.5053 | 3.3502 | 3.2505 |
| BF200 + layer-head-mix | 8.5147 | 7.2495 | 5.3569 | 3.5061 | 3.3518 | 3.2516 |
| BF200 + sketch-slot k2 aux0.25 sanitize/cap4 | 8.5180 | 7.2526 | 5.3543 | 3.5043 | 3.3509 | 3.2502 |
| BF200 + sketch-slot k2 aux0.5->0.1 after 500 sanitize/cap4 | 8.5110 | 7.2435 | 5.3555 | 3.5114 | 3.3526 | 3.2517 |

The first two BF200 variants completed. The step-500 result is exactly the type of
signal suggested by earlier reports: auxiliary/distributed-code and learned
mixing can improve optimization before the final representation bottleneck
becomes clear. By step 750 the gain had narrowed substantially, and by step
1000 both variants were slightly behind the baseline. Final results confirm the
pattern:

- Sketch-slot aux0.5 exactly matched the BF200 final (`3.2505`) after a large
  step-500 advantage. It improves optimization speed but not final loss in this
  form.
- Layer-head-mix finished slightly worse (`3.2516`) despite a strong step-500
  advantage. Its learned weights moved toward bigram-heavy reads, consistent
  with previous diagnostics, but that did not improve the final.

This is strong evidence that the auxiliary branch acts as an optimization
scaffold but becomes late-training interference unless its strength is reduced
or scheduled away.

Completed next:

| Run | Purpose |
| --- | --- |
| `bf200_sota_sketchslotk2_base_aux025_sanitize_cap4_seed5_1500_20260609` | Test whether lower constant auxiliary strength preserves the 500-step gain with less late interference. |
| `bf200_sota_sketchslotk2_base_aux05to01_s500_500_sanitize_cap4_seed5_1500_20260609` | Test whether starting with the useful `0.5` scaffold and annealing it down after step 500 keeps the early gain while reducing the 1000-step fade. At step 250 it matches the aux0.5 trace (`8.5110`), as expected before the schedule begins. |

Current follow-up read: constant aux0.25 is weaker at step 250 (`8.5180`) and
step 500 (`7.2526`) than constant aux0.5 (`8.5110`, `7.2435`), so the early
scaffold appears strength-dependent. By step 750 it recovers somewhat
(`5.3543`), beating BF200 baseline (`5.3587`) but still trailing constant
aux0.5 (`5.3507`). The annealed aux run matches constant aux0.5 through step
500, which is expected because the schedule begins at step 500. Its step-750
loss (`5.3555`) is slightly better than BF200 baseline but worse than constant
aux0.5, so this schedule is not preserving the full mid-curve gain. By step
1000, aux0.25 is essentially tied but slightly behind BF200 baseline (`3.5043`
versus `3.5038`), while the annealed run is clearly behind (`3.5114`). This
weakens the case for further auxiliary-strength schedule tuning. At step 1250,
aux0.25 is essentially tied with BF200 baseline (`3.3509` versus `3.3507`), and
the annealed run remains behind (`3.3526`).

Final result: aux0.25 finishes at `3.2502`, only `0.0003` better than BF200
baseline and far below the known seed-variance band. The annealed run finishes
worse at `3.2517`. The useful conclusion is not a new SOTA but a clearer
mechanistic one: extra sketch rows can speed mid-optimization, and lower aux
strength reduces late damage, but the representation converges back to the same
final basin. This points away from more aux-scale tuning and toward changing
which keys get extra capacity, or measuring hot-row collision composition
directly.

## Structural Follow-Ups

Because the BF200 sketch follow-ups converged back to the same final basin, the
next two GPU probes move away from auxiliary-strength schedules and test
addressing/readout structure:

| Run | 250 | 500 | 750 | 1000 | 1250 | 1500 | Purpose | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `bf200_sota_latentmixfsq_seed5_1500_20260609` | 8.5634 | - | - | - | - | - | Mix learned FSQ latent codes with ngram hashes, testing an adaptive/stable-hash addressing path instead of pure lexical ngram addressing. Layer partitions are disabled because latent addressing does not support them. | Killed after weak 250. |
| `bf200_sota_readoutdelta_seed5_1500_20260609` | 8.5172 | 7.2403 | 5.3588 | 3.5037 | 3.3495 | 3.2489 | Replace fully separate layer readouts with a shared readout plus per-layer residual readout deltas, testing a middle point between shared and fully layer-specific readers. | Finished; small BF200 gain. |
| `bf200_sota_sharepart_seed5_1500_20260609` | 8.5069 | 7.2644 | 5.3684 | - | - | - | Remove literal layer row partitions while keeping layer-specific hashes/readouts, testing whether BF200 benefits from sharing the full row address space across Engram layers. | Killed after weak 750. |
| `bf400_sota_readoutdelta_seed5_1500_20260609` | 8.5108 | 7.2474 | - | - | - | - | Scale the readoutdelta idea to BF400, testing whether the mid-curve BF200 gain can matter at the current largest fitted table size. | Killed after weak 500. |
| `bf200_sota_meta_scaling_seed6_1500_20260609` | 8.5143 | 7.2481 | 5.3637 | 3.5067 | 3.3518 | 3.2519 | BF200 baseline seed6 paired against readoutdelta seed6 to estimate whether the seed5 readoutdelta gain is real. | Complete. |
| `bf200_sota_readoutdelta_seed6_1500_20260609` | 8.5172 | 7.2386 | 5.3557 | 3.5056 | 3.3510 | 3.2497 | BF200 readoutdelta seed6 paired against baseline seed6. | Complete. |
| `bf120_sota_readoutdelta_seed5_1500_20260609` | 8.5132 | 7.2523 | 5.3637 | 3.5094 | 3.3533 | 3.2543 | Readoutdelta at BF120, to map whether the BF200 gain exists below BF200. | Complete; worse than BF120 baseline. |
| `bf300_sota_readoutdelta_seed5_1500_20260609` | 8.4990 | 7.2334 | 5.3539 | 3.5015 | 3.3464 | 3.2466 | Readoutdelta at BF300, to map whether the BF200 gain extends toward the BF300/BF400 frontier. | Complete; parameter-efficiency win. |

These are BF200, seed5, 1500-step probes using the same SOTA meta otherwise.
The goal is not another local hyperparameter sweep; it is to ask whether BF200
can spend its already-large table more intelligently through addressing or
readout structure.

First checkpoint: latentmix FSQ is far behind BF200 baseline (`8.5634` versus
`8.5093`) and was killed. Readoutdelta started behind (`8.5172`) but reached
`7.2403` at step 500, beating BF200 baseline (`7.3076`) and slightly beating
the BF200 sketch-slot aux0.5 checkpoint (`7.2435`). By step 750, though, it
faded to `5.3588`, essentially BF200 baseline (`5.3587`), so this may be another
mid-curve-only gain. At step 1000 it is `3.5037`, effectively tied with BF200
baseline (`3.5038`). Sharepart is stronger at step 250 (`8.5069` versus BF200
baseline `8.5093`) and remained ahead at step 500 (`7.2644` versus `7.3076`),
but it faded badly by step 750 (`5.3684` versus `5.3587`) and was killed. The
freed GPU is now running BF400 readoutdelta. That is the sharper test of whether
the readoutdelta mid-curve gain is only a BF200 optimization effect or can
actually improve the current largest fitted table setting.

BF200 readoutdelta remains interesting at step 1250 (`3.3495` versus BF200
baseline `3.3507`), although the margin is still well inside seed variance.
BF400 readoutdelta starts behind the BF400 baseline at 250 (`8.5108` versus
`8.5045`), but BF200 readoutdelta also started behind before its 500-step gain,
so the 500 checkpoint is the first fair decision point.

Final BF200 readoutdelta is `3.2489`, beating BF200 baseline `3.2505` by
`0.0016`. This is a useful BF200 improvement, but it remains below BF300
baseline (`3.2480`) and far below BF400 baseline (`3.2458`). BF400 readoutdelta
did not reproduce the BF200 mid-curve gain: at step 500 it was `7.2474` versus
BF400 baseline `7.2303`, so it was killed. The next rigorous check is a paired
BF200 seed6 baseline/readoutdelta run to separate a real structural effect from
seed noise.

Seed6 starts similarly to seed5: readoutdelta is behind at step 250 (`8.5172`
versus seed6 baseline `8.5143`). In seed5, readoutdelta also started behind and
then gained at step 500, so the paired 500 checkpoint is the first important
replication point.

The step-500 effect replicated on seed6: baseline is `7.2481`, readoutdelta is
`7.2386`, a `0.0095` mid-curve gain. This strengthens the conclusion that
readoutdelta changes optimization dynamics. The remaining question is whether it
survives the familiar late fade.

At step 750, seed6 readoutdelta still leads (`5.3557` versus baseline `5.3637`).
That is stronger than seed5, where readoutdelta had faded to the BF200 baseline
by 750. The paired seed6 run is now worth carrying to final.

By step 1000 the seed6 gap narrows but remains positive (`3.5056` versus
`3.5067`). This reinforces the same pattern: readoutdelta is a real optimizer/
trajectory change, but the final-loss question is still open.

At step 1250 the gap is very small but still positive (`3.3510` versus
baseline `3.3518`). The final checkpoint will decide whether this is a durable
BF200 improvement or another nearly-complete late fade.

Final seed6 confirms a small durable BF200 readoutdelta gain: `3.2497` versus
paired baseline `3.2519`, a `0.0022` improvement. Together with seed5
readoutdelta (`3.2489` versus seed5 BF200 baseline `3.2505`), the effect looks
real but small. It improves BF200 parameter efficiency, but it does not by
itself beat the BF300/BF400 row-count frontier, and the BF400 readoutdelta scale
probe was weak by step 500.

The next scale check is BF120 and BF300 readoutdelta at seed5. The purpose is to
see whether the readoutdelta gain is localized around BF200, appears at smaller
tables, or has a useful BF300 region despite the failed BF400 probe.

First scale-check checkpoints: BF120 readoutdelta is only marginally ahead at
250 (`8.5132` versus BF120 baseline `8.5144`), while BF300 readoutdelta is
meaningfully ahead at 250 (`8.4990` versus BF300 baseline `8.5071`). Both are
worth carrying to step 500, because the replicated BF200 readoutdelta signal was
strongest around that checkpoint.

At step 500, BF120 readoutdelta has faded slightly behind (`7.2523` versus
BF120 baseline `7.2490`), so it is not showing the BF200-style effect. BF300
readoutdelta remains ahead (`7.2334` versus BF300 baseline `7.2410`), a
`0.0076` gain. This suggests readoutdelta may have a useful mid/high-BF region
around BF200-BF300 even though BF400 readoutdelta was weak.

At step 750 the scale-check picture weakens. BF120 is worse (`5.3637` versus
BF120 baseline `5.3605`), and BF300 has mostly faded from its 500-step lead
(`5.3539` versus BF300 baseline `5.3526`). The current best interpretation is
that readoutdelta changes the trajectory and can give strong early/mid gains,
but it is not obviously improving the final representation at BF120/BF300.

At step 1000, BF120 remains worse (`3.5094` versus BF120 baseline `3.5070`),
but BF300 recovers to a small lead (`3.5015` versus BF300 baseline `3.5033`).
The BF300 run is therefore worth carrying to final. If the lead survives, the
readoutdelta sweet spot may extend from BF200 to BF300 while still failing at
BF120 and BF400.

At step 1250, BF120 is still a negative control (`3.3533` versus BF120 baseline
`3.3528`), but BF300 strengthens (`3.3464` versus BF300 baseline `3.3492`).
This also essentially ties the BF400 baseline at the same checkpoint (`3.3465`)
while using BF300's smaller table. The final checkpoint now matters: BF300
readoutdelta could be a parameter-efficiency win even if the BF400 readoutdelta
probe failed early.

Final scale-check result: BF120 readoutdelta finishes worse than BF120 baseline
(`3.2543` versus `3.2525`), but BF300 readoutdelta finishes at `3.2466`,
beating BF300 baseline (`3.2480`) by `0.0014`. It does not beat BF400 baseline
(`3.2458`), but it gets within `0.0008` while using `11.59B` table params and
`68,385 MiB` peak memory instead of BF400's `15.45B` table params and
`89,794 MiB` peak memory. That is the clearest structural parameter-efficiency
result from this branch.

The pattern is now:

- BF120 readoutdelta: negative.
- BF200 readoutdelta: replicated small positive final gain (`0.0016` seed5,
  `0.0022` seed6).
- BF300 readoutdelta: small positive final gain (`0.0014`) and close to BF400
  baseline with much less memory.
- BF400 readoutdelta: failed early by step 500 in the killed probe.

So readoutdelta is not a universal improvement. It appears to help in the
middle of the collision/capacity regime, where rows are numerous enough for
separating shared and layer-specific readout structure to matter, but not so
overprovisioned that the added readout structure becomes irrelevant or harmful.

## Next Iteration

The next two GPU runs launched from this result focus on BF300, because BF300
readoutdelta is the best current parameter-efficiency point:

| Run | Purpose |
| --- | --- |
| `bf300_sota_readoutdelta_seed6_1500_20260609` | Replicate the BF300 readoutdelta gain on seed6. If this also beats BF300 baseline, readoutdelta becomes more credible as a mid/high-BF structural improvement rather than a seed5 accident. |
| `bf300_sota_readoutdelta_attnresdelta_seed5_1500_20260609` | Combine readoutdelta with `ENGRAM_ATTNRES_DELTA=1`, testing whether making both the readout and AttnRes merge residual/delta-style can push BF300 past the BF400 baseline without BF400 memory. |

Launch script: `run_bf300_readoutdelta_next_20260609.sh`.

Early checkpoints:

| Run | 250 | 500 | Read |
| --- | ---: | ---: | --- |
| `bf300_sota_readoutdelta_seed6_1500_20260609` | 8.5161 | 7.2337 | Seed6 starts worse at 250 but matches the seed5 BF300 readoutdelta 500 point (`7.2334`) closely. At 750 it is `5.3533`, close to seed5 readoutdelta (`5.3539`) and baseline BF300 (`5.3526`). At 1000 it is `3.5033`, exactly BF300 baseline seed5 and behind BF300 readoutdelta seed5 (`3.5015`), but still worth finalizing for seed variance. |
| `bf300_sota_readoutdelta_attnresdelta_seed5_1500_20260609` | 8.4997 | 7.2340 | The AttnRes-delta combination is essentially tied with plain BF300 readoutdelta at 500, but weaker at 750 (`5.3578`) and worse at 1000 (`3.5063`). Killed after 1000; the zero/main delta AttnRes merge appears to hurt this setup. |

After killing the AttnRes-delta combination, GPU1 was recycled into
`bf300_sota_readoutdelta_hotdrop010_min1024_s250_500_seed5_1500_20260609`.
This keeps the BF300 readoutdelta recipe but ramps hot-row dropout from `0` to
`0.10` over steps 250-750 for rows with hit count at least `1024`. The point is
to test whether the known hot-row dominance can be regularized without changing
table size or readout structure.

Current follow-up read:

- BF300 readoutdelta seed6 reaches `3.3470` at step 1250, close to seed5
  readoutdelta (`3.3464`) and better than BF300 baseline seed5 (`3.3492`) at
  the same checkpoint. It finishes at `3.2466`, exactly matching BF300
  readoutdelta seed5 (`3.2466`). This strongly supports a real but small BF300
  readoutdelta effect.
- Hotdrop is `8.4990` at step 250, exactly matching plain BF300 readoutdelta
  seed5 at that checkpoint. That is expected because the dropout schedule starts
  at step 250. At step 500 it is worse (`7.2364` versus plain BF300
  readoutdelta `7.2334`), but the dropout ramp has not fully completed yet, so
  the fair kill point is step 750. At 750, hotdrop recovers to `5.3521`,
  beating both BF300 baseline (`5.3526`) and plain BF300 readoutdelta seed5
  (`5.3539`) at the same checkpoint. This is the first positive sign that
  training-time hot-row pressure might improve the BF300 readoutdelta basin. At
  step 1000 it remains slightly ahead (`3.5011` versus plain BF300 readoutdelta
  seed5 `3.5015` and BF300 baseline `3.5033`). By step 1250, though, it fades
  to `3.3478`, still better than BF300 baseline (`3.3492`) but worse than plain
  BF300 readoutdelta seed5 (`3.3464`). The final checkpoint will decide whether
  hotdrop is a durable improvement or another temporary trajectory effect. It
  finishes at `3.2470`: better than BF300 baseline (`3.2480`), but worse than
  plain BF300 readoutdelta seed5/seed6 (`3.2466`). The useful lesson is that
  hot-row dropout can improve the mid-training trajectory, but `0.10` appears a
  little too strong late.

Because hotdrop turned positive at 750, a seed6 replication was launched:
`bf300_sota_readoutdelta_hotdrop010_min1024_s250_500_seed6_1500_20260609`.
The seed6 hotdrop replication matches seed6 readoutdelta at 250 (`8.5161`) but
is much worse at 500 (`7.2444` versus seed6 readoutdelta `7.2337`). The seed5
hotdrop run recovered at 750 after a weaker 500, so seed6 gets the same 750
decision point; if it does not recover, the hotdrop effect is likely seed- or
trajectory-sensitive. It does recover at 750 (`5.3532`), essentially tied with
plain seed6 readoutdelta (`5.3533`). At step 1000 it improves over plain seed6
readoutdelta (`3.5018` versus `3.5033`), and at step 1250 it remains only
barely ahead (`3.3469` versus `3.3470`). This makes the effect look real as a
trajectory nudge but not yet convincing as a final-loss improvement. The final
confirms that read: seed6 hotdrop finishes at `3.2468`, better than BF300
baseline (`3.2480`) but worse than plain BF300 readoutdelta seed5/seed6
(`3.2466`). Across two seeds, constant `0.10` hot-row dropout is a mid-training
regularizer, not a new final-loss improvement.

Based on the seed5 shape, a weaker follow-up was launched on GPU1:
`bf300_sota_readoutdelta_hotdrop005_min1024_s250_500_seed5_1500_20260609`.
This tests whether `0.05` hot-row dropout preserves the 750/1000 benefit while
reducing the late drag seen with `0.10`. It starts normally at 250 (`8.4990`)
and is only mildly behind plain readoutdelta at 500 (`7.2393` versus `7.2334`),
but by 750 it is worse (`5.3564` versus `5.3539`). It was killed at 750. Lower
constant dropout does not preserve the useful 750 signal.

The current probes are now more structural:

| Run | Purpose |
| --- | --- |
| `bf300_sota_readoutdelta_hotdrop010_decay0_min1024_s250_500_d1000_500_seed5_1500_20260609` | Keep the useful `0.10` mid-training pressure, then decay it from `0.10` to `0` over steps 1000-1500. This directly tests whether the observed late drag is caused by holding dropout too long. |
| `bf300_sota_readoutdelta_hotdrop010_decay0_min1024_s250_500_d750_500_seed5_1500_20260609` | Start removing the same `0.10` hotdrop immediately after the useful 750 checkpoint, decaying to zero by 1250. This tests whether the 1000-1500 decay is too late. |
| `bf300_sota_readoutdelta_hotdrop010_decay0_min1024_s250_500_d750_250_seed5_1500_20260609` | Remove the same `0.10` hotdrop faster, decaying to zero by 1000. This tests whether the regularizer should only shape the 750 basin and be absent for the late representation fit. |
| `bf300_sota_readoutdelta_hotdrop010_min4096_s250_500_seed5_1500_20260609` | Apply `0.10` dropout only to ultra-hot rows with at least `4096` hits, testing whether the hot-row issue is concentrated in the very hottest rows rather than the broader `>=1024` bucket. Killed at 750: it reached `5.3538`, effectively plain readoutdelta (`5.3539`) and worse than broad `min1024` hotdrop (`5.3521`). The mid-run benefit is not explained by only the ultra-hottest rows. |

The decay-start-1000 run preserves the known broad-hotdrop trajectory through
1000: `8.4990`, `7.2364`, `5.3521`, `3.5011`. That is expected because decay
starts at 1000. The key comparison is 1250/1500, when it should diverge from
the constant `0.10` run. At 1250 it reaches `3.3462`, beating both constant
hotdrop (`3.3478`) and plain BF300 readoutdelta seed5 (`3.3464`) at the same
checkpoint. The margin over plain readoutdelta is tiny, but the direction is
right: decaying dropout late appears to preserve the mid-run regularization
without the same late drag at 1250. The final, however, is `3.2468`, not better
than plain BF300 readoutdelta (`3.2466`). Decay from 1000 to 1500 removes some
late drag but not enough to produce a final improvement.

The decay-start-750 run is normal through 500 (`8.4990`, `7.2364`). It should
match the late-decay run through 750 and then diverge earlier, so the first
meaningful comparison is 1000. At 1000 it is still `3.5011`, matching the
other broad hotdrop paths and preserving the useful trajectory. The key
checkpoint is 1250, when this schedule has fully decayed to zero. At 1250 it
reaches `3.3459`, the best late checkpoint from the hotdrop branch so far and
better than plain BF300 readoutdelta seed5 (`3.3464`). The final checkpoint
shows it does not survive: final is `3.2469`, still worse than plain BF300
readoutdelta (`3.2466`). This is now a repeated pattern: hotdrop improves the
mid/late trajectory but does not improve the final basin at 1500.

The faster decay-to-zero-by-1000 run is normal through 750 (`8.4990`, `7.2364`,
`5.3521`) but gives back some of the 1000-step benefit (`3.5014` versus
`3.5011` for the slower decay runs). At 1250 it is `3.3468`, worse than plain
BF300 readoutdelta seed5 (`3.3464`) and much worse than the 750-to-1250 decay
run (`3.3459`). It was killed at 1250. Turning the regularizer fully off by
1000 is too early.

## Read-Hit Scale Follow-Up

The hotdrop branch shows that perturbing hot-row reliance improves the trajectory
but does not change the final basin. The next probe asks whether the deterministic
read-hit scaling itself is too aggressive. Both runs keep the BF300 readoutdelta
recipe and remove hit dropout:

| Run | Purpose |
| --- | --- |
| `bf300_sota_readoutdelta_readhitexp0125_seed5_1500_20260609` | Reduce `ENGRAM_READ_HIT_SCALE_EXPONENT` from `0.25` to `0.125`, weakening all hit-count amplification while preserving normalization and max cap. |
| `bf300_sota_readoutdelta_readhitmax2_seed5_1500_20260609` | Keep exponent `0.25` but cap `ENGRAM_READ_HIT_SCALE_MAX` at `2.0` instead of `4.0`, targeting only the strongest hot-row boosts. |

The decision points are 500 and 750. If either variant loses the known BF300
readoutdelta trajectory by then, the evidence will point away from weakening
read-hit scaling globally.

At 250 both variants are weaker than plain BF300 readoutdelta seed5 (`8.4990`):
exponent `0.125` reaches `8.5028`, and max cap `2.0` reaches `8.5088`. The cap
run also lowers the observed mean read-hit scale to `0.847` after clamping,
so that change is not just limiting the hottest rows; it reduces overall Engram
read strength. Both are carried to 500, but the early signal is negative.

At 500 the signal remains negative. Exponent `0.125` reaches `7.2361`, and max
cap `2.0` reaches `7.2366`, both behind plain BF300 readoutdelta seed5
(`7.2334`). They are carried to 750 because hotdrop also looked worse at 500
before recovering, but the read-hit weakening branch is currently disfavored.

At 750, exponent `0.125` recovers to `5.3523`, slightly ahead of plain BF300
readoutdelta seed5 (`5.3539`) and BF300 baseline (`5.3526`). The max-cap `2.0`
run reaches `5.3531`, which is weaker than exponent `0.125` and only marginally
better than plain readoutdelta while also reducing the mean read scale to about
`0.844`. The max-cap run was stopped to free GPU1 for the corrected BF scaling
sweep.

At 1000, exponent `0.125` reaches `3.5007` versus plain BF300 readoutdelta
seed5 at `3.5015`. This is directionally positive but only a `0.0008` gap, far
below the seed-variance scale. The run is being carried to 1250 before deciding
whether deterministic read-hit weakening is anything more than another
trajectory-shaping nudge.

At 1250, exponent `0.125` reaches `3.3469`. This is worse than BF300
readoutdelta seed5 (`3.3464`) and essentially tied with BF300 readoutdelta
seed6 (`3.3470`). That resolves the branch: weakening deterministic read-hit
scaling is another small trajectory perturbation, not a convincing final-loss
improvement. The run was stopped after 1250 to free GPU0 for the corrected
BF40-BF400 current-meta scaling queue.

The corrected BF40 current-meta run reaches `7.2686` at step 500 after `8.5289`
at step 250. This low-BF point continues to look weak, consistent with the
earlier BF120 readoutdelta negative control. It is still worth finishing because
the requested artifact is the full BF scaling curve, not just the promising
BF200-BF300 region.

At step 750, corrected BF40 reaches `5.3720`. This confirms the low-BF
readoutdelta/current-meta point is not promising on its own; it is being kept
only to complete the BF-vs-parameter-count curve. GPU0 is now running the BF200
current-meta point in parallel.

BF200 current-meta reaches `8.5088` at step 250. This is effectively the same
early trajectory as the original BF200 SOTA-meta baseline (`8.5093`) and slightly
ahead of the earlier BF200 readoutdelta standalone run (`8.5172`), but the
250-step region has not been predictive enough to draw a final-loss conclusion.

At step 1000, corrected BF40 is `3.5188`. This is much worse than the original
BF40 SOTA-meta trajectory and confirms that the current readoutdelta meta does
not transfer to the smallest table in this sweep.

At step 1250, corrected BF40 is `3.3606`, again well behind the original BF40
SOTA-meta curve. The final BF40 point is expected to be a low-BF negative
control for readoutdelta/current meta, not a candidate improvement.

BF40 final is `3.2596` with `1.55B` Engram table params and `29,768 MiB` peak
memory. This is slightly better than the original BF40 SOTA-meta final
(`3.2602`) but only by `0.0006`, far below the seed-variance scale. The useful
conclusion is not a low-BF win; it is that readoutdelta/current meta does not
materially improve small-table scaling, even though it also does not catastrophically
hurt the final after its very weak mid-curve.

BF200 reaches `7.2482` at step 500. That is far better than the original BF200
SOTA-meta baseline at step 500 (`7.3076`) and close to the earlier BF200
readoutdelta run (`7.2403`) and seed6 readoutdelta (`7.2386`). The result
supports the existing interpretation: readoutdelta is a real mid-curve
optimization improvement around BF200, but the final-loss question remains open
because previous readoutdelta gains faded late to a small but durable margin.

At step 750, BF200 current-meta is `5.3696`. This is weaker than both earlier
BF200 readoutdelta runs (`5.3588` seed5, `5.3557` seed6) and also weaker than
the original BF200 SOTA-meta baseline at 750 (`5.3587`). This queued BF200 point
therefore does not currently reproduce the earlier BF200 readoutdelta trajectory
despite matching the mid-500 behavior. It is still being finished for the full
scaling curve, but it should not be treated as a new structural win unless it
recovers late.

At step 1000, BF200 current-meta is `3.5033`. This is slightly better than the
original BF200 SOTA-meta baseline (`3.5038`) and earlier BF200 readoutdelta
seed5 (`3.5037`), but only by `0.0004-0.0005`. Given the established variance
scale, the honest read is that BF200 has mostly converged back into the same
late basin after the weak 750 checkpoint.

At step 1250, BF200 current-meta reaches `3.3489`. This is a small recovery:
it is ahead of the original BF200 baseline (`3.3507`) and the earlier BF200
readoutdelta seed5 run (`3.3495`), and also ahead of the paired seed6
readoutdelta/baseline checkpoints (`3.3510`/`3.3518`). The gap is still within
the noise band, but the trajectory is no longer purely negative after the weak
750 point.

BF200 current-meta final is `3.2487` with `53,532 MiB` peak memory. This beats
the original BF200 SOTA-meta baseline (`3.2505`) by `0.0018` and slightly beats
the earlier BF200 readoutdelta seed5 run (`3.2489`), but the margin is still
small. It does not beat BF300 baseline (`3.2480`) or the replicated BF300
readoutdelta result (`3.2466`). The conclusion is that current-meta/readoutdelta
is a real BF200 parameter-efficiency improvement, but not yet a new frontier
against the BF300-BF400 row-count curve.

BF80 current-meta starts weak: step 250 is `8.5255`, behind the original BF80
SOTA-meta checkpoint (`8.5176`). This matches the BF40/BF120 evidence that the
readoutdelta/current-meta structure is not a low-BF improvement; the interesting
region remains BF200-BF300 unless the later BF80 trajectory unexpectedly
recovers.

At step 500, BF80 current-meta is `7.2504`, still behind the original BF80
baseline (`7.2420`). This makes the low-BF negative-control pattern stronger:
the same structure that improves BF200 does not seem to help when the table is
too narrow in rows.

At step 750, BF80 current-meta is `5.3733`, behind the original BF80 baseline
(`5.3601`) by `0.0132`. This is no longer a tiny comparison; the low-BF branch
is consistently worse through the middle of training.

Next queued research branch: `run_bf300_superpose_next_20260609.sh` waits behind
the current BF sweep and tests `ENGRAM_SUPERPOSE_K=2` on the BF300
readoutdelta recipe. This is the cleaner version of the distributed-code idea:
keep the original hash row, add one salted extra row, and combine with norm
correction. Two variants are queued: constant auxiliary scale `0.5`, and the
same scale decayed to zero from steps 750-1250. The purpose is to test whether
extra independent row support can improve BF300's final basin without the
incompatibilities of sketch-slot readout.

BF300 current-meta starts at `8.5045` at step 250. This is ahead of the old
BF300 baseline checkpoint (`8.5071`) but behind the earlier BF300 readoutdelta
seed5 run (`8.4990`). So the full corrected sweep is not yet reproducing the
strongest BF300 readoutdelta trace, though the first checkpoint is still in the
right broad region.

At step 1000, BF80 current-meta is `3.5118`, only slightly behind the original
BF80 baseline (`3.5104`). The low-BF curve has narrowed the gap late, but after
being weaker at 250/500/750 it still does not look like a useful low-BF
improvement unless the final unexpectedly flips.

At step 1250, BF80 current-meta reaches `3.3545`, slightly ahead of the original
BF80 baseline (`3.3557`). This is a late recovery after a consistently worse
early/mid curve. The result may still finish as a tiny BF80 gain, but given the
noise scale and weak earlier trajectory, the final checkpoint is needed before
calling the low-BF conclusion either way.

BF80 current-meta final is `3.2546` with `35,709 MiB` peak memory. This beats
the original BF80 SOTA-meta baseline (`3.2560`) by `0.0014`, but it is still
worse than the original BF120 baseline (`3.2525`) and BF120 readoutdelta
negative-control final (`3.2543`). The conclusion is narrower than the late
recovery suggested: current-meta/readoutdelta can slightly improve BF80 final
loss, but it does not change the low-BF scaling frontier.

BF120 current-meta starts at `8.5184` at step 250. This is behind both the old
BF120 baseline (`8.5144`) and earlier BF120 readoutdelta (`8.5132`). Given that
BF120 readoutdelta previously finished negative, this first checkpoint keeps
BF120 in the low-BF negative-control bucket unless it shows a BF80-like late
recovery.

At step 500, BF120 current-meta is `7.2516`. This is still behind the old BF120
baseline (`7.2490`) but slightly ahead of the earlier BF120 readoutdelta run
(`7.2523`). It remains a low-BF watch point rather than a likely improvement.

At step 750, BF120 current-meta recovers to `5.3571`, beating both the old
BF120 baseline (`5.3605`) and the earlier BF120 readoutdelta run (`5.3637`).
This mirrors the BF80 pattern: weak early, then better mid/late. The important
question is whether this late recovery persists to final, because BF120
readoutdelta previously faded to a negative final despite mid-curve movement.

At step 1000, BF120 current-meta is `3.5110`, behind both the old BF120
baseline (`3.5070`) and earlier BF120 readoutdelta (`3.5094`). The 750 recovery
did not hold. BF120 is again looking like a negative control unless it performs
an unusual late reversal.

At step 1250, BF120 current-meta is `3.3532`. This is still behind the old
BF120 baseline (`3.3528`) and essentially tied with the earlier BF120
readoutdelta run (`3.3533`). The corrected current-meta package has not moved
the low-BF frontier; BF120 final is now mostly a confirmation point.

BF120 current-meta final is `3.2531` with `41,650 MiB` peak memory. It is worse
than the old BF120 baseline (`3.2525`) but better than the earlier BF120
readoutdelta run (`3.2543`). This confirms the BF80/BF120 story: the corrected
current-meta package can recover some readoutdelta weakness, but it does not
make small tables competitive with the original SOTA-meta scaling curve.

BF400 current-meta starts weak at step 250: `8.5109`, behind the old BF400
baseline (`8.5045`) and essentially no better than the earlier failed BF400
readoutdelta probe (`8.5108`). This is consistent with the prior BF400
readoutdelta result: the structure that helps BF200-BF300 does not obviously
transfer to the largest table.

At step 500, BF400 current-meta is `7.2590`, far behind the old BF400 baseline
(`7.2303`) and worse than the earlier BF400 readoutdelta probe (`7.2474`). This
makes BF400 current-meta a clear negative at the largest table so far; it is
being carried only to complete the requested BF scaling curve.

At step 750, BF400 current-meta recovers to `5.3497`, but it remains behind the
old BF400 baseline (`5.3460`). The weak 500 checkpoint was not just an eval
glitch, and the largest-table corrected-meta run still looks worse than simply
using the old BF400 SOTA-meta recipe.

At step 1000, BF400 current-meta reaches `3.5024`, barely ahead of the old BF400
baseline (`3.5027`). This is a late recovery from the weak early curve, but the
gap is only `0.0003`, well inside the seed-noise scale we have been using for
decisions. Treat this as "caught up" rather than evidence of a new BF400
frontier.

At step 1250, BF400 current-meta is `3.3466`, essentially tied with the old
BF400 baseline (`3.3465`) and slightly behind the earlier BF300 readoutdelta
checkpoint (`3.3464`). The BF400 run has fully recovered from its weak 500, but
it has not created a larger-table advantage.

BF400 current-meta final is `3.2462` with `89,799 MiB` peak memory. This is
worse than the old BF400 SOTA-meta baseline (`3.2458`), though it is better than
BF300 current-meta (`3.2472`) and BF300 readoutdelta (`3.2466`). The full
corrected BF sweep therefore does not produce a new SOTA; its useful result is
mostly a cleaner mapping of where readoutdelta/current-meta helps (BF200-BF300)
and where it fails to justify extra rows (BF400).

At step 500, BF300 current-meta is `7.2377`. This beats the old BF300 baseline
checkpoint (`7.2410`) but trails the earlier BF300 readoutdelta seed5 run
(`7.2334`). This keeps the BF300 corrected sweep alive as a possible small
positive, but it is not currently stronger than the best BF300 readoutdelta
evidence already in hand.

At step 750, BF300 current-meta improves to `5.3510`. This is now better than
both the old BF300 baseline (`5.3526`) and the earlier BF300 readoutdelta seed5
run (`5.3539`). This is the first checkpoint in the corrected BF300 sweep that
beats the prior best BF300 trace, so the run is worth watching closely through
1000-1500 rather than treating it as a weaker reproduction.

At step 1000, BF300 current-meta reaches `3.5010`. This beats the old BF300
baseline (`3.5033`), the earlier BF300 readoutdelta seed5 run (`3.5015`), and
even the old BF400 baseline at the same checkpoint (`3.5027`). This is the
strongest parameter-efficiency signal in the corrected sweep so far; the key
question is whether it persists through the late 1250/1500 region where several
previous mid-curve gains faded.

At step 1250, BF300 current-meta is `3.3469`. This is still better than the old
BF300 baseline (`3.3492`), but it has slipped behind both earlier BF300
readoutdelta seed5 (`3.3464`) and the old BF400 baseline (`3.3465`) by about
`0.0004-0.0005`. The corrected BF300 run still looks useful, but the strongest
1000-step parameter-efficiency signal has partially faded. Final decides whether
it lands as a new BF300 best or another mid-curve-only improvement.

BF300 current-meta final is `3.2472` with `68,385 MiB` peak memory. It beats
the old BF300 baseline (`3.2480`) by `0.0008`, but it does not beat the earlier
BF300 readoutdelta result (`3.2466`) or the old BF400 baseline (`3.2458`). This
confirms that the excellent 750/1000 signal partially faded late: the corrected
BF300 point is good, but it is not a new SOTA.

Operational note: after BF120 completed, the BF300 superpose follow-up was
relaunched. GPU1 is running the aux-scale-to-zero variant while the GPU0 branch
waits behind BF400. This avoids the earlier transient-memory race while still
using the idle GPU after BF120 freed it.

First superpose checkpoint: the aux-scale-to-zero BF300 variant is weak at step
250 (`8.5156`). That trails old BF300 baseline/current-meta (`8.5045`) and the
earlier BF300 readoutdelta run (`8.4990`). This does not fully rule it out
because the branch changes the read geometry and may need the 500 checkpoint to
settle, but it starts with a clear early optimization tax.

At step 500, the same superpose variant recovers to `7.2339`. This beats old
BF300 baseline (`7.2410`) and current-meta (`7.2377`), and is essentially tied
with prior BF300 readoutdelta (`7.2334`). The distributed-code readout is
therefore not an early kill; it has a rough 250-step warmup but reaches the
existing BF300 readoutdelta frontier by 500.

At step 750, aux-to-zero superpose reaches `5.3516`. This beats old BF300
baseline (`5.3526`) and earlier BF300 readoutdelta (`5.3539`), but trails BF300
current-meta (`5.3510`). It remains worth running to 1000, but at this point it
looks like another mid-curve competitor rather than a clearly stronger idea.

At step 1000, aux-to-zero superpose reaches `3.5018`. This is better than old
BF300 baseline (`3.5033`), but worse than BF300 current-meta (`3.5010`) and
earlier BF300 readoutdelta (`3.5015`). The branch is competitive enough to
finish, but so far the distributed-code readout is not improving on the
readoutdelta/current-meta family.

The constant-aux superpose variant matches aux-to-zero exactly through the
pre-schedule checkpoints: `8.5156` at step 250 and `7.2339` at step 500. This is
expected because the aux-to-zero schedule only starts at step 750; the meaningful
comparison begins at 750/1000.

Constant-aux also matches at the 750 boundary (`5.3516`), before the schedule has
had time to separate the traces. The first meaningful schedule-vs-constant
comparison should therefore be constant-aux 1000 against aux-to-zero 1000.

At step 1250, aux-to-zero superpose falls to `3.3484`. This is better than old
BF300 baseline (`3.3492`), but worse than BF300 current-meta (`3.3469`) and
earlier BF300 readoutdelta (`3.3464`). The scheduled decay of the auxiliary
superposed readout did not rescue the late-curve fade.

Aux-to-zero superpose final is `3.2479` with `69,477 MiB` peak memory. That is
roughly old BF300 baseline quality (`3.2480`), worse than BF300 current-meta
(`3.2472`), worse than earlier BF300 readoutdelta (`3.2466`), and worse than old
BF400 baseline (`3.2458`). Decaying the auxiliary superposed branch to zero is
therefore a negative variant.

Constant-aux superpose at step 1000 is the first genuinely interesting result
from this branch: `3.4998`. That beats BF300 current-meta (`3.5010`), earlier
BF300 readoutdelta (`3.5015`), old BF300 baseline (`3.5033`), and the BF400
baseline/current-meta 1000 checkpoints (`3.5027`/`3.5024`). The open question is
whether it preserves this advantage at 1250/final, where many prior BF300 gains
have faded.

Because constant-aux is the first superpose variant with a real 1000-step lead,
a seed6 replication was launched on the freed GPU1:
`bf300_sota_readoutdelta_superposek2_base_aux05_seed6_1500_20260609`.

At step 1250, constant-aux seed5 fades to `3.3480`. This is better than the
aux-to-zero variant (`3.3484`) and old BF300 baseline (`3.3492`), but worse than
BF300 current-meta (`3.3469`) and earlier BF300 readoutdelta (`3.3464`). The
1000-step lead was real enough to justify the seed6 replication, but it did not
survive the late-window test on seed5.

Seed6 constant-aux starts much better than seed5 at step 250: `8.5051` versus
seed5's `8.5156`. This is close to the BF300/BF400 SOTA-meta starts and means
the rough 250-step warmup seen on seed5 is not intrinsic to constant-aux
superpose.

Constant-aux seed5 final reverses the 1250 fade and lands at `3.2454` with
`69,477 MiB` peak memory. This is the best single 1500-step value found so far:
it beats BF300 current-meta (`3.2472`), earlier BF300 readoutdelta (`3.2466`),
current-meta BF400 (`3.2462`), and old BF400 SOTA-meta (`3.2458`) while using
the BF300 table size. The margin over old BF400 is only `0.0004`, so the seed6
replication and an added seed7 replication are needed before treating it as a
robust new SOTA, but it is the first distributed-code idea that actually moves
the observed frontier.

Seed7 constant-aux was launched on the freed GPU0:
`bf300_sota_readoutdelta_superposek2_base_aux05_seed7_1500_20260609`.

Seed6 constant-aux reaches `5.3508` at step 750. That is slightly better than
seed5 constant-aux (`5.3516`) and BF300 current-meta (`5.3510`), so the
replication remains live. The important checks are still 1000/1250/final, since
seed5 only became a final winner after a misleading 1250 fade.

Seed6 constant-aux then weakens at step 1000 to `3.5028`, worse than BF300
current-meta (`3.5010`), earlier BF300 readoutdelta (`3.5015`), and seed5
constant-aux (`3.4998`). This is a negative replication signal for the strong
seed5 1000-step bump, though seed6 is still being carried because seed5's final
improved after a weak 1250.

Seed7 constant-aux starts strong at step 250 (`8.5046`), similar to seed6
(`8.5051`) and much better than seed5 (`8.5156`). The early roughness is
therefore seed-dependent rather than a fixed cost of superpose.

Seed7 constant-aux reaches `7.2368` at step 500. This is slightly better than
BF300 current-meta (`7.2377`) and seed6 (`7.2373`), but still worse than seed5
constant-aux (`7.2339`) and prior BF300 readoutdelta (`7.2334`). Replication is
mixed: seeds 6/7 have good starts, but neither has yet reproduced seed5's
stronger 500/1000 trajectory.

Seed7 reaches `5.3505` at step 750, now the best 750-step checkpoint among the
constant-aux replications. Together with seed6's `5.3508`, the 750-step
superpose signal is replicated and slightly ahead of BF300 current-meta
(`5.3510`). The unresolved question is whether either seed can avoid seed6's
weak 1000 or seed5's 1250 fade.

Seed6 then rebounds at step 1250 to `3.3461`. This beats seed5 constant-aux
(`3.3480`), BF300 current-meta (`3.3469`), earlier BF300 readoutdelta (`3.3464`),
and old BF400 SOTA-meta (`3.3465`) at the same checkpoint. The constant-aux
superpose improvement is therefore not only a seed5 final artifact; seed6 shows
a real late-window advantage even though its 1000 checkpoint was weak.

Seed7 reaches `3.4998` at step 1000, matching seed5's strong 1000 checkpoint and
beating BF300 current-meta (`3.5010`) and earlier BF300 readoutdelta (`3.5015`).
The 1000-step superpose lead is now present in two seeds (5 and 7), while seed6
instead shows the stronger 1250 checkpoint. This makes the branch meaningfully
more credible than a single lucky trace.

Seed6 final is `3.2460` with `69,477 MiB` peak memory. It does not beat seed5
constant-aux (`3.2454`) or old BF400 SOTA-meta (`3.2458`), but it does beat
BF300 current-meta (`3.2472`), earlier BF300 readoutdelta (`3.2466`), and
current-meta BF400 (`3.2462`). This supports the branch as a real BF300
parameter-efficiency improvement, while showing the seed5 SOTA margin is not yet
fully replicated.

With GPU1 free after seed6, a constant-superpose aux-scale `1.0` variant was
launched to test whether the auxiliary readout should be stronger than `0.5`:
`bf300_sota_readoutdelta_superposek2_base_aux10_seed5_1500_20260609`.

Seed7 reaches `3.3452` at step 1250, the best 1250 checkpoint observed so far.
It beats seed6 constant-aux (`3.3461`), seed5 constant-aux (`3.3480`), BF300
current-meta (`3.3469`), earlier BF300 readoutdelta (`3.3464`), and old BF400
SOTA-meta (`3.3465`). This gives the strongest evidence yet that constant-aux
superpose is a real structural improvement at BF300, not just a noisy final from
seed5.

Aux-scale `1.0` starts at `8.5055` at step 250, similar to the strong seed6/7
aux-scale `0.5` starts (`8.5051`/`8.5046`) and much better than seed5 aux-scale
`0.5` (`8.5156`). A stronger auxiliary branch does not hurt the first checkpoint;
the useful comparison begins at 500/750.

Aux-scale `1.0` reaches `7.2345` at step 500. That is close to seed5 aux-scale
`0.5` (`7.2339`), better than BF300 current-meta (`7.2377`) and seed6/7
aux-scale `0.5` (`7.2373`/`7.2368`), but still just behind earlier BF300
readoutdelta (`7.2334`). So far, increasing the auxiliary scale to `1.0` is not
obviously harmful; the 750/1000 checkpoints will decide whether it improves the
late curve or only matches the early readout scaffold.

At step 750, aux-scale `1.0` is `5.3511`. That is essentially tied with BF300
current-meta (`5.3510`) and slightly behind the best K2 aux-scale `0.5`
replication, seed7 (`5.3505`). The stronger branch is therefore neutral to
slightly worse by 750, not a clear improvement over aux-scale `0.5`.

At step 1000, aux-scale `1.0` is `3.5026`. That is worse than K2 aux-scale
`0.5` seed5/seed7 (`3.4998`/`3.4998`) and only comparable to the weak seed6
curve (`3.5028`). Because this is a same-seed comparison against the strongest
K2 aux-scale `0.5` run, aux-scale `1.0` is not worth carrying unless extra GPU
time is abundant; GPU1 was freed for a higher-value BF400 superpose test.

With GPU0 free after seed7, a K3 superpose follow-up was launched:
`bf300_sota_readoutdelta_superposek3_base_aux05_seed5_1500_20260609`. It keeps
the same BF300/readoutdelta/current-meta base and aux-scale `0.5`, but increases
`ENGRAM_SUPERPOSE_K` from `2` to `3` to test whether a wider distributed code
improves the readout or just adds optimization noise. It is still before its
first validation checkpoint.

K3 starts well: step 250 is `8.5031`, better than the K2 aux-scale `0.5`
replications (`8.5156`/`8.5051`/`8.5046`) and slightly ahead of the reproduced
BF300/BF400 SOTA-meta anchors at that checkpoint (`8.5045`/`8.5045`). This is
not enough to claim K3 is better, but it is a strong enough early signal to keep
the run alive through at least 500.

K3 at step 500 is `7.2373`. That gives back the 250-step lead: it matches K2
aux-scale `0.5` seed6 (`7.2373`), is slightly behind seed7 (`7.2368`), and is
well behind seed5 (`7.2339`). K3 is therefore not an obvious upgrade by 500, but
it remains close enough to keep through 750 before making a kill/replacement
decision.

K3 at step 750 is `5.3555`, clearly behind all K2 aux-scale `0.5` replications
(`5.3516`/`5.3508`/`5.3505`) and BF300 current-meta (`5.3510`). The early 250
lead did not survive. K3 was killed after 750; adding a second auxiliary salted
row appears to add interference rather than useful distributed capacity in this
configuration.

Next launched branch after killing aux-scale `1.0` at 1000: BF400 K2 superpose
aux-scale `0.5`, `bf400_sota_readoutdelta_superposek2_base_aux05_seed5_1500_20260609`.
The motivation is direct: BF300 K2 already reaches reproduced-BF400 quality with
BF300 table size, so the most useful scale test is whether the same distributed
readout can recover the missing BF400 margin without the late weakness seen in
plain BF400 readoutdelta/current-meta.

BF400 K2 superpose fits but is tight during warmup: GPU1 showed about
`95,351 MiB` allocated before the first train/eval checkpoint. It is alive for
now, but this is close enough to the device limit that any future BF400 variant
with extra activations or wider superposition should be treated as risky.

BF400 K2 superpose starts weak: step 250 is `8.5190`, far behind the reproduced
BF400 baseline/current-meta starts (`8.5045`/`8.5109`) and even worse than the
weak BF300 K2 seed5 start (`8.5156`). Because BF300 K2 seed5 later recovered and
became the best K2 final, the BF400 run is being kept to 500 before deciding,
but the first signal is negative.

BF400 K2 reaches `7.2374` at step 500. That is still behind the reproduced BF400
baseline (`7.2303`) and BF300 K2 seed5 (`7.2339`) while using almost the full
95GB device. The run was killed at 500. Scaling the K2 superpose branch from
BF300 to BF400 does not look like the missing SOTA margin; it appears to import
the same early weakness as BF300 seed5 without enough recovery to justify the
memory.

After killing K3, GPU0 was assigned to a pure K2/no-base superpose test:
`bf300_sota_readoutdelta_superposek2_nobase_seed5_1500_20260609`. This disables
`ENGRAM_SUPERPOSE_INCLUDE_BASE`, so the readout averages two salted rows by
`sqrt(2)` instead of reading the original lexical row plus a scaled auxiliary
row. This tests whether the winning K2 behavior is general multi-row
superposition or specifically a residual augmentation of the base hash row.

No-base K2 starts at `8.5120` at step 250. That is worse than base+aux K2
seed6/seed7 (`8.5051`/`8.5046`) but better than the unusually weak base+aux
seed5 start (`8.5156`). The first checkpoint is therefore not enough to answer
the structural question; the run is being carried to 500.

No-base K2 reaches `7.2343` at step 500. This is essentially competitive with
the winning base+aux branch: just behind seed5 (`7.2339`) and ahead of seed6/7
(`7.2373`/`7.2368`). That means the K2 result may not require preserving the
base lexical row; a pure two-row distributed code can at least match the early
readout scaffold by 500. The run is being carried to 750.

No-base K2 falls back at step 750: `5.3542`, worse than all base+aux K2
replications (`5.3516`/`5.3508`/`5.3505`) and BF300 current-meta (`5.3510`).
This suggests pure two-row averaging can provide an early scaffold, but it does
not preserve the mid-curve advantage. The winning K2 form is more likely a
residual augmentation of the base lexical hash row than a fully symmetric
two-row code. No-base was killed after 750.

After killing no-base, GPU0 was assigned to constant aux-scale `0.25`:
`bf300_sota_readoutdelta_superposek2_base_aux025_seed5_1500_20260609`. This
brackets residual strength against constant `0.5` (best so far), constant `1.0`
(weak at 1000), full `0.5 -> 0` decay (weak final), and the active partial
`0.5 -> 0.25` decay.

Constant aux-scale `0.25` starts at `8.5076` at step 250. This is much cleaner
than constant aux-scale `0.5` seed5 (`8.5156`), though not quite as strong as
the best seed6/7 starts or aux-scale `1.0` (`8.5051`/`8.5046`/`8.5055`). The
lower residual strength does reduce the seed5 early penalty; the question is
whether it preserves the 500/1000 advantage.

Constant aux-scale `0.25` fails by step 500: `7.2465`, far behind aux-scale
`0.5` seed5 (`7.2339`) and aux-scale `1.0` (`7.2345`). This is a clean negative.
The residual auxiliary branch cannot be too weak; the useful regime is around
`0.5`, with `1.0` already showing late weakness and `0.25` losing the early
readout scaffold.

After killing aux-scale `0.25`, GPU0 was assigned to BF200 K2 superpose
aux-scale `0.5`: `bf200_sota_readoutdelta_superposek2_base_aux05_seed5_1500_20260609`.
This tests whether the distributed residual readout is a general
parameter-efficiency improvement across table sizes, not only a BF300 frontier
effect.

BF200 K2 starts at `8.5168` at step 250. This is weak versus BF200 baseline and
current-meta starts (`8.5093`/`8.5088`), but close to the plain BF200
readoutdelta start (`8.5172`). Because BF300 K2 seed5 also started weak and then
recovered by 500/1000, BF200 K2 is being carried to 500 before deciding.

BF200 K2 reaches `7.2409` at step 500. This is roughly tied with BF200
readoutdelta seed5 (`7.2403`) but weaker than the seed6 readoutdelta check
(`7.2386`). It is not an obvious BF200 improvement yet, but the gap is small
enough to carry to 750; the BF300 K2 branch also needed later checkpoints to
show its value.

BF200 K2 fails at step 750: `5.3909`, far behind BF200 baseline/current-meta and
readoutdelta trajectories. This is a clean negative. The K2 superpose win is not
a general low-parameter-table improvement; at BF200 the extra salted row appears
to add collision/interference pressure instead of usable capacity.

After killing BF200 K2, GPU0 was assigned to
`bf300_sota_readoutdelta_superposek2_base_aux05_nonorm_seed5_1500_20260609`.
This adds `ENGRAM_SUPERPOSE_NORMALIZE=0`, disabling the usual
`sqrt(1 + aux_scale^2)` normalization for the base+aux superpose readout. The
motivation is that no-base and aux-scale `0.25` both showed the base lexical row
matters; the current normalized K2 path slightly downscales that base row, so
the no-normalization test asks whether preserving base-row magnitude improves
the branch.

No-normalization K2 starts at `8.5089` at step 250. This is better than
normalized K2 seed5's weak start (`8.5156`), but worse than layer-readout K2
(`8.4965`) and not clearly better than normalized K2 seed6/seed7
(`8.5051`/`8.5046`). It is being carried to 500, but the first signal is only
modestly positive.

No-normalization K2 reaches `7.2344` at step 500. That is slightly worse than
normalized readoutdelta K2 seed5 (`7.2339`) and far behind layer-readout K2
(`7.2317`). Preserving base-row magnitude by removing the RMS normalization does
not improve the branch. The run was killed at 500 and GPU0 was reassigned to a
seed6 replication of layer-readout K2.

Layer-readout K2 seed6 was launched as
`bf300_sota_layerreadout_superposek2_base_aux05_seed6_1500_20260609`. This is a
direct replication of the promising full-readout K2 seed5 branch; if it holds,
the evidence shifts from a lucky trace to a real readout-structure improvement.

After killing BF400 K2, GPU1 was assigned to a partial auxiliary decay:
`bf300_sota_readoutdelta_superposek2_base_aux05to025_s750_500_seed5_1500_20260609`.
This keeps the winning base+aux K2 structure, starts at aux-scale `0.5`, and
decays only to `0.25` after step 750. The prior full `0.5 -> 0` decay was too
destructive late; this test asks whether reducing, but not erasing, the auxiliary
branch can preserve the strong 1000-step scaffold while reducing late
interference.

Partial `0.5 -> 0.25` decay starts at `8.5156` at step 250, exactly matching
the constant aux-scale `0.5` seed5 run before the schedule begins. This is a
sanity check that the launch/config is comparable; the actual test begins after
the schedule starts at step 750.

At step 500, partial `0.5 -> 0.25` decay is `7.2339`, again matching the
constant aux-scale `0.5` seed5 run before the schedule begins. The launch is
therefore a clean paired comparison; any divergence should appear only after
step 750.

At step 750, partial `0.5 -> 0.25` decay is `5.3516`, still matching constant
aux-scale `0.5` seed5. This is expected because the schedule starts at 750. The
first meaningful divergence point is step 1000, after the auxiliary scale has
begun decaying toward `0.25`.

At step 1000, partial `0.5 -> 0.25` decay is `3.5004`. This is better than
BF300 current-meta (`3.5010`) and aux-scale `1.0` (`3.5026`), but slightly worse
than constant aux-scale `0.5` seed5/seed7 (`3.4998`/`3.4998`). Partial decay is
not an upgrade at 1000, but it is close enough to carry to 1250 because its
purpose is to reduce late interference, not maximize the 1000-step scaffold.

At step 1250, partial `0.5 -> 0.25` decay reaches `3.3458`. This is better than
constant aux-scale `0.5` seed5 (`3.3480`) and seed6 (`3.3461`), but still behind
seed7 (`3.3452`). The partial decay does what it was intended to do: it gives up
a little of the 1000-step scaffold and improves the late basin. It remains
worth carrying to final.

Partial `0.5 -> 0.25` decay final is `3.2462` with `69,477 MiB` peak memory.
The late 1250 improvement did not survive to the final checkpoint: this is worse
than constant aux-scale `0.5` seed5/seed6/seed7 (`3.2454`/`3.2460`/`3.2459`) and
roughly ties current-meta BF400 (`3.2462`). The negative lesson is useful:
reducing auxiliary strength can improve the 1250 basin, but the final fit still
wants the constant residual branch.

After the partial-decay final, GPU1 was assigned to a structural readout test:
`bf300_sota_layerreadout_superposek2_base_aux05_seed5_1500_20260609`. This keeps
BF300 K2 base+aux `0.5`, but switches from shared readout plus per-layer readout
delta (`ENGRAM_LAYER_READOUTS=0`, `ENGRAM_LAYER_READOUT_DELTA=1`) back to full
per-layer readouts (`ENGRAM_LAYER_READOUTS=1`, `ENGRAM_LAYER_READOUT_DELTA=0`).
The question is whether superposed rows want fully independent layer readers, or
whether the readoutdelta compression is part of the K2 win.

Layer-readout K2 starts very strong: `8.4965` at step 250 and `7.2317` at step
500. Both are better than readoutdelta K2 seed5 (`8.5156`/`7.2339`), reproduced
BF300 baseline (`8.5045`/`7.2410`), and reproduced BF400 baseline
(`8.5045`/`7.2303`, except it is essentially tied at 500). This is the first
post-K2 structural branch with a clearly better early trajectory; it is being
carried forward.

Layer-readout K2 remains strong at step 750: `5.3487`. This beats all
readoutdelta K2 replications (`5.3516`/`5.3508`/`5.3505`), BF300 current-meta
(`5.3510`), and reproduced BF400 baseline/current-meta (`5.3460`/`5.3497` is
mixed: still behind the old BF400 baseline, but ahead of BF400 current-meta).
The branch is now the strongest structural follow-up to K2 and is being carried
to 1000/final.

At step 1000, layer-readout K2 seed5 falls back to `3.5040`. That is worse than
readoutdelta K2 seed5/seed7 (`3.4998`/`3.4998`) and BF300 current-meta
(`3.5010`). The early advantage therefore does not by itself establish a better
final basin. The run is still being carried to 1250/final because prior K2
replications sometimes recovered after a weak 1000 checkpoint, but the evidence
for full per-layer readouts has weakened.

At step 1250, layer-readout K2 seed5 reaches `3.3475`. This is a recovery from
the weak 1000 checkpoint and beats constant-aux K2 seed5 at 1250 (`3.3480`), but
it is still behind constant-aux K2 seed6/seed7 (`3.3461`/`3.3452`) and partial
decay (`3.3458`). The branch is being allowed to finish because it is close, but
it is no longer the leading candidate.

Layer-readout K2 seed5 final is `3.2481` with `69,441 MiB` peak memory. This is
worse than BF300 readoutdelta K2 seed5/seed6/seed7
(`3.2454`/`3.2460`/`3.2459`) and worse than reproduced BF400 SOTA-meta
(`3.2458`). The strong early path was therefore a mirage: full per-layer
readouts optimize early but do not land in the better final basin.

Layer-readout K2 seed6 starts at `8.5139` at step 250, which does not reproduce
seed5's unusually strong `8.4965` start. This makes the full-readout signal look
seed-fragile rather than structurally robust. It is being carried to 500 as a
cheap confirmation point, but the current default should remain readoutdelta K2
unless later checkpoints reverse this.

Layer-readout K2 seed6 reaches only `7.2406` at step 500, far weaker than seed5
layer-readout (`7.2317`) and worse than the readoutdelta K2 seed5 launch
(`7.2339`). The run was killed at 500 and GPU0 was reassigned to a more distinct
structural test: BF300 readoutdelta K2 plus scheduled hot-row dropout
(`ENGRAM_HIT_DROPOUT_FINAL=0.10`, `ENGRAM_HIT_DROPOUT_MIN_HITS=1024`,
schedule start 250 over 500 steps). This tests whether the K2 branch still
over-relies on ultra-hot rows during training, not just at pruning/eval time.

Hot-row dropout K2 starts identically to the paired constant K2 seed5 branch at
250 (`8.5156`) because dropout only begins ramping at that checkpoint. By step
500, after the schedule has begun, it is `7.2388`, worse than constant K2 seed5
(`7.2339`) and worse than the best seed traces. This is early negative evidence:
forcing the model away from hot rows during training appears to damage the
scaffold before any regularization benefit is visible. The run is being carried
to 750 as a cheap recovery check, not because it is currently promising.

At step 750, hot-row dropout K2 is `5.3527`, still worse than paired constant K2
seed5 (`5.3516`) and worse than all three constant K2 replications
(`5.3516`/`5.3508`/`5.3505`). This confirms the branch is not recovering. It was
killed at 750. The result matches the older project-level lesson: hot rows are
important and compressible, but naive training-time suppression damages the
useful scaffold instead of producing a better final model.

GPU0 was reassigned to BF300 readoutdelta K2 plus learned per-layer head mix:
`bf300_sota_readoutdelta_superposek2_base_aux05_layerheadmix_seed5_1500_20260609`
with `ENGRAM_LAYER_HEAD_MIX=1`. This follows the older SOTA report, where
layer-head mix was one of the few structural variants with a real early signal,
and tests whether the K2 distributed-code branch benefits from learned
bigram/trigram weighting per layer.

Layer-head-mix K2 starts at `8.5160` at step 250. That is effectively identical
to paired constant K2 seed5 (`8.5156`) and does not reproduce the older early
layer-head-mix advantage. The learned weights are already nonuniform
(`l2: 0.552/0.448`, `l8: 0.538/0.462` over the two ngram heads), but this has
not translated into an early loss win. It is being carried to 500 because the
weights may need a little runway, but the transfer signal is weak.

At step 500, layer-head-mix K2 becomes interesting: `7.2264`, clearly better
than paired constant K2 seed5 (`7.2339`), the other K2 seed traces, BF300
current-meta (`7.2410`), and current-meta BF400 (`7.2389`). The learned weights
move further toward the first ngram head (`l2: 0.591/0.409`, `l8: 0.586/0.414`),
suggesting K2 benefits from letting each layer rebalance bigram/trigram readout
after the initial scaffold forms. This branch is now worth carrying through
750/1000.

At step 750, layer-head-mix K2 is still strong at `5.3496`. It beats paired
constant K2 seed5 (`5.3516`) and seed6 (`5.3508`), and is only slightly behind
seed7 (`5.3505` is actually worse; layer-head-mix is ahead of all three K2
replications at this checkpoint). The learned weights continue to favor the
first ngram head (`l2: 0.605/0.395`, `l8: 0.607/0.394`). This is the best
structural continuation of K2 so far and should be carried to 1000/final.

At step 1000, layer-head-mix K2 is `3.5006`. This gives back some of the strong
500/750 advantage and is slightly worse than constant K2 seed5/seed7
(`3.4998`/`3.4998`), but it remains better than BF300 current-meta (`3.5010`)
and much better than the full layer-readout branch (`3.5040`). The learned
head-mix weights are stable around a first-head preference
(`l2: 0.619/0.381`, `l8: 0.602/0.398`). This is no longer a clear mid-run
breakout, but it is still a credible final candidate and is being carried to
1250/1500. Seed6 replication is running to check whether the 500/750 gain is
robust.

At step 1250, layer-head-mix K2 seed5 becomes the strongest June-branch late
checkpoint so far: `3.3445`. This beats constant K2 seed5/seed6/seed7
(`3.3480`/`3.3461`/`3.3452`), partial auxiliary decay (`3.3458`), and
layer-readout K2 (`3.3475`). The final checkpoint is now the critical test. If
the 1250 advantage survives, learned per-layer head weighting may be the first
post-K2 structural idea to improve final loss, not just parameter efficiency.

Layer-head-mix K2 seed5 final is `3.2449` with `69,621 MiB` peak memory. This
beats all June BF300 constant-K2 replications (`3.2454`/`3.2460`/`3.2459`), the
reproduced BF400 SOTA-meta endpoint (`3.2458`), and current-meta BF400
(`3.2462`). It is still behind the older archived May 27 absolute best
(`3.2411`), but it is the best result found on the new June instance and the
first post-K2 structural branch to improve final loss rather than only matching
BF400 with fewer rows. The caveat is seed robustness: seed6 has not yet
replicated the 500/1000 path.

After layer-readout seed5 finished, GPU1 was assigned to a hash-quality test:
BF300 readoutdelta K2 with `ENGRAM_AVALANCHE_HASH=1`
(`bf300_sota_readoutdelta_superposek2_base_aux05_avalanche_seed5_1500_20260609`).
This keeps the winning K2 structure fixed and changes the base ngram hash from
the prime-product path to the avalanche path. If hot rows or collision geometry
are partly caused by the base hash, this should move the early hit/fit pattern.

Avalanche-hash K2 starts at `8.5082` at step 250, which is better than constant
K2 seed5's `8.5156` but not better than the stronger non-avalanche seed traces.
By step 500 it falls to `7.2452`, clearly worse than constant K2 seed5
(`7.2339`) and the current-meta BF300/BF400 traces. The branch was killed at 500.
This repeats the older result: stronger avalanche mixing changes hit statistics
slightly, but does not create a better optimization path in this setup.

GPU1 was reassigned to BF300 readoutdelta K2 plus coarse layer signs:
`bf300_sota_readoutdelta_superposek2_base_aux05_layersigns_seed5_1500_20260609`
with `ENGRAM_LAYER_SIGNS=1`. This is the independent companion to the
layer-head-mix test on GPU0; older work suggested layer signs had a real early
signal, and K2 may or may not make that signal usable late.

Layer-signs K2 starts at `8.5096` at step 250. This is better than paired
constant K2 seed5 (`8.5156`) but weaker than the stronger layer-head-mix 500
trajectory once that branch gets moving. It remains live to 500 because the
coarse signed layer view is a different mechanism from learned head weighting.

At step 500, layer-signs K2 is `7.2394`, worse than paired constant K2 seed5
(`7.2339`) and far behind layer-head-mix K2 (`7.2264`). The older layer-sign
early signal does not transfer cleanly into the K2 branch. This is now a recycle
candidate unless a cheap 750 checkpoint is needed for symmetry.

Layer-signs K2 was killed at 500 and GPU1 was reassigned to a direct
layer-head-mix replication:
`bf300_sota_readoutdelta_superposek2_base_aux05_layerheadmix_seed6_1500_20260609`.
Since seed5 layer-head-mix is the first clearly positive post-K2 structural
branch, the most useful next evidence is seed robustness rather than another
orthogonal weak knob.

Layer-head-mix seed6 starts strong at step 250: `8.5062`. This beats
layer-head-mix seed5 (`8.5160`) and paired constant K2 seed5 (`8.5156`), and
supports the idea that learned layer/head weighting is not just a seed5 quirk.
The run is being carried to 500/750 for the stricter comparison, where seed5 had
its clearest advantage.

At step 500, layer-head-mix seed6 falls to `7.2417`, failing to reproduce
seed5's `7.2264` jump and trailing paired constant K2 seed5 (`7.2339`). The
250-step improvement was therefore not enough evidence. Seed6 is being kept only
long enough for a 750 recovery check; if it remains weak, the layer-head-mix
signal should be treated as promising but seed-fragile.

At step 750, layer-head-mix seed6 recovers to `5.3489`, which beats
layer-head-mix seed5 (`5.3496`) and all constant K2 replications
(`5.3516`/`5.3508`/`5.3505`). The weak 500 checkpoint was not fatal. Across
seed5 and seed6, learned layer-head mix now has a consistent 750-step advantage;
the question is whether seed6 also holds through 1000/1250/final.

At step 1000, layer-head-mix seed6 weakens to `3.5024`, worse than seed5
layer-head-mix (`3.5006`), constant K2 seed5/seed7 (`3.4998`/`3.4998`), and
BF300 current-meta (`3.5010`). The 750 advantage does not reliably predict the
1000 basin. This makes the seed5 final improvement promising but not robustly
established; another replication is needed before treating layer-head mix as the
new default.

At step 1250, layer-head-mix seed6 recovers to `3.3455`. This is worse than
layer-head-mix seed5 (`3.3445`) but better than constant K2 seed5/seed6
(`3.3480`/`3.3461`) and partial decay (`3.3458`), and roughly tied with constant
K2 seed7 (`3.3452`). Seed6 therefore supports a modest late benefit, but not the
full seed5 final-strength story. It is being carried to final.

Layer-head-mix seed6 final is `3.2458` with `69,621 MiB` peak memory. This
matches the reproduced BF400 SOTA-meta endpoint (`3.2458`) and beats
current-meta BF400 (`3.2462`), but it is behind layer-head-mix seed5 (`3.2449`)
and constant K2 seed5 (`3.2454`). The update is therefore useful for parameter
efficiency, but the final-loss improvement is not robust enough yet to make
layer-head-mix the default without more seeds.

After seed6 finished, GPU1 was assigned to another direct replication:
`bf300_sota_readoutdelta_superposek2_base_aux05_layerheadmix_seed8_1500_20260609`.
This keeps the exact BF300/readoutdelta/K2/base+aux0.5/layer-head-mix setup and
only changes `MODEL_SEED=8`. The purpose is to estimate whether seed5's new
June-instance best is a real layer-head-mix effect or a seed-sensitive late
basin.

Layer-head-mix seed8 starts badly: step 250 is `8.5666`, far worse than
layer-head-mix seed5/seed6/seed7 (`8.5160`/`8.5062`/`8.5133`) and also worse
than constant K2 seed5/seed6/seed7 (`8.5156`/`8.5051`/`8.5046`). This is too
weak to spend the full run budget on a replication, so it was killed at 250.
The seed8 result strengthens the read that per-layer head mixing can find a good
late basin, but is not a robust default by itself.

GPU1 was reassigned to a simpler shared head-mix variant:
`bf300_sota_readoutdelta_superposek2_base_aux05_headmix_seed5_1500_20260609`
with `ENGRAM_HEAD_MIX=1` and `ENGRAM_LAYER_HEAD_MIX=0`. This keeps a learned K2
head weighting but shares it across memory layers. The test is meant to separate
the useful part of head reweighting from the seed-fragile per-layer degrees of
freedom.

After the BF400 probe was killed, GPU0 was assigned to the paired seed6
replication:
`bf300_sota_readoutdelta_superposek2_base_aux05_headmix_seed6_1500_20260609`.
This gives the shared-head-mix idea a two-seed early read without spending more
time on the per-layer version that had already failed seed7 and seed8 early.

Shared head-mix seed5 reaches `8.5166` at step 250. This is ordinary: close to
constant K2 seed5 (`8.5156`) and layer-head-mix seed5 (`8.5160`), but not as
strong as constant K2 seed6/seed7 (`8.5051`/`8.5046`) or layer-head-mix seed6
(`8.5062`). It is being carried to 500 because the head-reweighting variants
have shown their clearest separation at 500 rather than 250.

At step 500, shared head-mix seed5 is `7.2359`. This is not the exceptional
layer-head-mix seed5 jump (`7.2264`), but it beats constant K2 seed6/seed7
(`7.2373`/`7.2368`) and is better than layer-head-mix seed6/seed7
(`7.2417`/`7.2483`). The shared version is therefore less explosive than the
best per-layer run, but it may be less seed-fragile; both shared-head seeds are
being carried to the next checkpoint.

Shared head-mix seed6 starts strong at step 250: `8.5061`, essentially matching
the good constant K2 seed6 start (`8.5051`) and layer-head-mix seed6 (`8.5062`).
This supports carrying the paired shared-head run to at least 500/750 rather
than treating seed5's ordinary 250 as a weak branch.

At step 750, shared head-mix seed5 is `5.3504`. This is about tied with the
best constant K2 750 checkpoints (`5.3516`/`5.3508`/`5.3505`) and better than
the failed layer-head-mix seed7 (`5.3549`), but it is not as strong as
layer-head-mix seed6 (`5.3489`) or seed5 (`5.3496`). Shared head-mix therefore
looks like a stabilizing version of the head-reweighting idea rather than a
clear upgrade so far.

Shared head-mix seed6 reaches `7.2402` at step 500. That is weaker than shared
head-mix seed5 (`7.2359`) and constant K2 seed6/seed7
(`7.2373`/`7.2368`), but much better than the failed per-layer seed7 (`7.2483`).
It is being carried to 750 because both layer-head-mix seed6 and several earlier
branches recovered after mediocre 500 checkpoints.

Shared head-mix seed5 reaches `3.4999` at step 1000. This ties the better
constant K2 1000 checkpoints (`3.4998`/`3.4998`) and beats layer-head-mix seed5
(`3.5006`), but it is not a clear new basin. It is being carried to final
because the final endpoint is the only place this branch can prove it is more
than a stabilized variant of constant K2.

Shared head-mix seed6 fails the 750 recovery check: `5.3528`, worse than shared
head-mix seed5 (`5.3504`) and all three constant K2 seeds
(`5.3516`/`5.3508`/`5.3505`). The run was killed at 750. The two-seed read says
shared head-mix is less brittle than per-layer head-mix at 500, but still not
robust enough to declare a default.

Because both shared-head and per-layer-head runs learn a first-head preference
by 500-1000, `train_gpt.py` was patched to accept `ENGRAM_HEAD_MIX_INIT`, a
comma/space-separated list of initial head-mix logits. The patch initializes
either `head_mix_logits` or every row of `layer_head_mix_logits`, validates that
the list length matches `total_hash_heads`, prints the init in the run config,
and passed local plus remote `py_compile`.

GPU0 was then assigned to
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_seed5_1500_20260609`
with shared head mix and `ENGRAM_HEAD_MIX_INIT=0.5,-0.5`. This starts the
two-head softmax near the learned first-head preference observed in the unbiased
head-mix runs. The test asks whether the useful signal is simply a better prior
over the base/auxiliary K2 reads, or whether learning the preference during
training is necessary.

Unbiased shared-head seed5 reaches `3.3451` at step 1250. This is slightly
better than the best constant K2 1250 checkpoint (`3.3452`) and better than
layer-head-mix seed6 (`3.3455`), but still behind layer-head-mix seed5
(`3.3445`). The shared-head branch is now a credible final candidate, though it
has not shown a large new basin.

Unbiased shared-head seed5 final is `3.2456` with `69,621 MiB` peak memory. It
beats current-meta BF400 (`3.2462`), constant K2 seed6/seed7
(`3.2460`/`3.2459`), and layer-head-mix seed6 (`3.2458`), but is behind constant
K2 seed5 (`3.2454`) and layer-head-mix seed5 (`3.2449`). Shared head mixing is
therefore a useful stabilizer/regularizer, not a new SOTA by itself.

Biased shared-head init starts at `8.5179` at step 250, slightly worse than the
unbiased shared-head seed5 (`8.5166`) and constant K2 seed5 (`8.5156`). The
initial softmax preference is visible in the metrics (`h0=0.679`, `h1=0.321`),
but it does not improve the first validation. It is being carried to 500 because
the head-mix variants separate more clearly there.

At step 500, biased shared-head init reaches `7.2347`, slightly better than
unbiased shared-head seed5 (`7.2359`) but still behind constant K2 seed5
(`7.2339`). The result is not a major jump, but it shows that starting near the
learned first-head preference is not harmful and may give a small 500-step
benefit over unbiased shared head mix. It is being carried to 750.

After unbiased shared-head seed5 finished, GPU1 was assigned to a biased-init
seed6 replication:
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_seed6_1500_20260609`.
The paired seed is needed because unbiased shared-head seed6 failed at 750; if
the first-head init helps seed6 avoid that failure, the init is more interesting
than the unbiased learned shared-head knob.

Biased shared-head seed5 reaches `5.3479` at step 750. This is the strongest
750-step checkpoint in the current K2 branch, beating layer-head-mix seed5/seed6
(`5.3496`/`5.3489`), shared-head seed5 (`5.3504`), and all constant K2 seeds
(`5.3516`/`5.3508`/`5.3505`). This is the first clear evidence that seeding the
head-mix preference may improve more than just the 500-step path.

At step 1000, biased shared-head seed5 gives back that advantage: `3.5007`,
worse than unbiased shared-head seed5 (`3.4999`) and the best constant K2
checkpoints (`3.4998`). The biased init may be improving the mid-training read
but over-constraining the later basin. It is being carried to 1250 because some
branches recover late, but the 1000 result weakens the final-SOTA case.

Biased shared-head seed6 starts strong at step 250: `8.5054`, essentially tied
with constant K2 seed6/seed7 (`8.5051`/`8.5046`) and better than unbiased
shared-head seed6 (`8.5061`). It is being carried to 500 to test whether the
biased prior avoids the unbiased seed6 750 failure.

Biased shared-head seed6 reaches `7.2292` at step 500. This is a strong result:
better than constant K2 seed6/seed7 (`7.2373`/`7.2368`), better than constant K2
seed5 (`7.2339`), better than unbiased shared-head seed5/seed6
(`7.2359`/`7.2402`), and only behind the exceptional layer-head-mix seed5
(`7.2264`). This makes the biased first-head prior the most promising
post-layer-head-mix structural idea so far, provided it can avoid seed5's
1000-step fade.

Biased shared-head seed5 recovers at step 1250 to `3.3442`, the strongest
June-branch late checkpoint so far. It beats layer-head-mix seed5 (`3.3445`),
unbiased shared-head seed5 (`3.3451`), layer-head-mix seed6 (`3.3455`), and all
constant K2 1250 checkpoints. The 1000-step fade was therefore not fatal.

Biased shared-head seed6 also reaches `5.3479` at step 750, matching biased
seed5 and beating all earlier K2 750 checkpoints. The first-head init now has a
replicated mid-training signal: both seed5 and seed6 beat the previous best 750
from layer-head-mix seed6 (`5.3489`).

After seed5 finished, GPU0 was assigned to a second direct replication:
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_seed7_1500_20260609`.
The branch has now produced a new June-instance best final on seed5, a weak
seed6 1000 checkpoint, and needs seed7 to distinguish a robust improvement from
a lucky seed5 late basin.

Biased shared-head seed5 final is `3.2448` with `69,621 MiB` peak memory. This
is the best final from the June 9 instance so far, slightly beating
layer-head-mix seed5 (`3.2449`), constant K2 seed5 (`3.2454`), unbiased
shared-head seed5 (`3.2456`), reproduced BF400 SOTA-meta (`3.2458`), and
current-meta BF400 (`3.2462`). It still does not beat the archived May 27
absolute best (`3.2411`), but it is the strongest new structural result from the
current BF300/K2 research branch.

Biased shared-head seed6 fades at step 1000 to `3.5014`, worse than biased seed5
(`3.5007`), unbiased shared-head seed5 (`3.4999`), and the best constant K2
1000 checkpoints (`3.4998`). This does not kill the branch because seed5 also
faded at 1000 and recovered to the best 1250/final seen on this instance, but it
means the biased prior is currently strongest at 500/750 and not yet a clean
late-training improvement across seeds.

Code-path interpretation: head mix applies `softmax(logits) * sqrt(num_heads)`
before the per-head sum, and skips the usual post-sum `1/sqrt(num_heads)`
division. So the biased init is not simply shrinking the auxiliary slot; it is
an RMS-preserving routing prior over the two hash heads. With init `0.5,-0.5`,
the starting weights are approximately `0.679/0.321`, scaled internally by
`sqrt(2)`. That makes the result an addressing/readout allocation effect, not a
parameter-count or memory-savings effect.

Biased shared-head seed6 reaches `3.3450` at step 1250. This is a recovery from
the weak 1000 checkpoint and is slightly better than unbiased shared-head seed5
(`3.3451`), the best constant K2 1250 (`3.3452`), and layer-head-mix seed6
(`3.3455`), but it does not match biased seed5 (`3.3442`) or layer-head-mix
seed5 (`3.3445`). The branch remains worth finishing, but the replicated
late-checkpoint signal is now modest rather than clearly SOTA.

Biased shared-head seed7 starts at `8.5070` at step 250. This is weaker than
biased seed6 (`8.5054`) and constant K2 seed7 (`8.5046`), but still well within
the healthy BF300 K2 range and much better than the weak BF400/layer-head early
paths. It is being carried to 500, where the head-mix variants have separated
more clearly than at 250.

At step 500, biased shared-head seed7 is `7.2344`. This is weaker than biased
seed6 (`7.2292`) and the exceptional layer-head-mix seed5 (`7.2264`), but it
slightly beats biased seed5 (`7.2347`), constant K2 seed6/seed7
(`7.2373`/`7.2368`), and unbiased shared-head seed5 (`7.2359`). It is worth
carrying to 750 because the clearest replicated advantage of the biased init has
been the 750-step checkpoint.

Biased shared-head seed6 final is `3.2453` with `69,621 MiB` peak memory. This
is strong: it beats constant K2 seed5 (`3.2454`), unbiased shared-head seed5
(`3.2456`), layer-head-mix seed6 (`3.2458`), and the reproduced BF400 SOTA-meta
(`3.2458`). It does not beat biased seed5 (`3.2448`) or layer-head-mix seed5
(`3.2449`). The fair read is that `0.5,-0.5` is a useful and likely real
head-routing prior, but not yet a robust absolute-SOTA shift across seeds.

After seed6 freed GPU1, a milder biased-init sweep point was launched:
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit025_seed5_1500_20260609`
with `ENGRAM_HEAD_MIX_INIT=0.25,-0.25`. This tests whether a softer prior can
keep the replicated 500/750 benefit while avoiding the stronger init's
1000-step fade and seed-dependent late behavior.

Biased shared-head seed7 reaches `5.3508` at step 750. This is healthy but no
longer the strong biased-init signature: it is worse than biased seed5/seed6
(`5.3479`/`5.3479`), layer-head-mix seed5/seed6 (`5.3496`/`5.3489`), and
constant K2 seed7 (`5.3505`), while roughly tying constant K2 seed6 (`5.3508`).
This downgrades the `0.5,-0.5` result from robust mid-training improvement to a
seed-dependent but still useful head-routing prior. The run is being carried to
1000 before deciding whether to finish or recycle GPU0.

At step 1000, biased shared-head seed7 recovers to `3.4996`. This is better than
biased seed5/seed6 (`3.5007`/`3.5014`) and slightly better than the best constant
K2 1000 checkpoints (`3.4998`). The 750 weakness therefore did not predict the
1000 behavior. Seed7 is worth carrying further because it now has the best
1000-step result in the K2 branch, even though the 500/750 story is mixed.

At step 1250, biased shared-head seed7 falls back to `3.3453`. This is behind
biased seed5 (`3.3442`), layer-head-mix seed5 (`3.3445`), biased seed6
(`3.3450`), and unbiased shared-head seed5 (`3.3451`). The strong 1000 point did
not translate into a strong late checkpoint. The branch still may finish near
the K2 cluster, but the seed7 trajectory confirms the global biased prior is
not a clean late-SOTA mechanism by itself.

Biased shared-head seed7 final is `3.2454` with `69,621 MiB` peak memory. It
roughly ties constant K2 seed5 (`3.2454`) and is worse than biased seed5/seed6
(`3.2448`/`3.2453`). The global `0.5,-0.5` prior therefore remains useful, but
its advantage is seed-dependent and not a robust final-loss shift.

The milder global prior `ENGRAM_HEAD_MIX_INIT=0.25,-0.25` is weak at step 250:
`8.5211`, with learned head weights already around `0.610/0.390`. This is much
worse than the `0.5,-0.5` seeds and also worse than unbiased shared-head seed5.
It was killed at 250. The result argues that the useful prior is not just
"slightly prefer head 0"; the branch seems to need a stronger initial routing
asymmetry, or a different mechanism entirely.

GPU1 was reassigned to
`bf300_sota_readoutdelta_superposek2_base_aux05_layerheadmixinit05_seed5_1500_20260609`.
This combines the two positive signals: per-layer head-mix flexibility, which
gave the strongest 500-step result but was seed-fragile, and the global
`0.5,-0.5` routing prior, which gave the best June final and the best seed7
1000-step result. The test asks whether initializing all layer-specific head
mixes to the useful global prior stabilizes the per-layer variant.

Layer-head-mix with `0.5,-0.5` init starts at `8.5128` at step 250. This is not
as strong as the best seed6/seed7 starts, but it is better than old
layer-head-mix seed5 (`8.5160`) and global biased seed5 (`8.5179`). The
layer-specific metrics already diverge slightly (`l2 h0=0.692`, `l8 h0=0.674`),
which is the intended behavior. It is worth carrying to 500.

At step 500, layer-head-mix init reaches `7.2316`. This does not match the
exceptional old layer-head-mix seed5 (`7.2264`), but it beats global biased
seed5/seed7 (`7.2347`/`7.2344`), constant K2 seed5/seed6/seed7
(`7.2339`/`7.2373`/`7.2368`), and unbiased shared-head seed5 (`7.2359`). The
layer-specific prior is a positive early result, and the run is being carried to
750.

At step 750, layer-head-mix init seed5 reaches `5.3474`, the best 750-step
checkpoint in the current K2 branch. It beats global biased seed5/seed6
(`5.3479`/`5.3479`), uninitialized layer-head-mix seed5/seed6
(`5.3496`/`5.3489`), and all constant K2 seeds. This is the clearest positive
evidence so far that the per-layer flexibility and the `0.5,-0.5` routing prior
compose usefully.

At step 1000, layer-head-mix init seed5 fades to `3.5027`. This is worse than
global biased seed5/seed6/seed7 (`3.5007`/`3.5014`/`3.4996`), uninitialized
layer-head-mix seed5/seed6 (`3.5006`/`3.5024`), and constant K2 seed5/seed7
(`3.4998`/`3.4998`). The branch now mirrors the earlier pattern: strong
mid-training signal, then a 1000-step fade. It is being carried because biased
seed5 recovered late, but the 1000 result weakens the final-SOTA case.

At step 1250, layer-head-mix init seed5 recovers to `3.3456`, but not enough to
lead the branch. It is worse than global biased seed5 (`3.3442`), uninitialized
layer-head-mix seed5 (`3.3445`), global biased seed6 (`3.3450`), and unbiased
shared-head seed5 (`3.3451`). The initialized per-layer prior therefore has the
best 750-step checkpoint, but the late-window evidence currently favors the
simpler global biased prior or the original uninitialized per-layer run.

Layer-head-mix init seed5 final is `3.2461` with `69,621 MiB` peak memory. This
is worse than global biased seed5/seed6 (`3.2448`/`3.2453`), constant K2 seed5
(`3.2454`), and uninitialized layer-head-mix seed5 (`3.2449`). The final confirms
the 1250 read: initializing per-layer head mix gives the best 750-step scaffold,
but it does not preserve the final-loss benefit.

After global biased seed7 finished, GPU0 was assigned to a direct replication of
the layer-head-mix init branch:
`bf300_sota_readoutdelta_superposek2_base_aux05_layerheadmixinit05_seed6_1500_20260609`.
This is the right replication because the branch currently has the best active
500-step signal, and old uninitialized layer-head-mix was seed-fragile.

Layer-head-mix init seed6 starts healthy at step 250: `8.5082`. This beats
layer-head-mix init seed5 (`8.5128`) and is close to the good global biased
seed6/seed7 starts (`8.5054`/`8.5070`). The replication remains worth carrying
to 500/750.

At step 500, layer-head-mix init seed6 reaches `7.2332`. This is a healthy
replication of the initialized per-layer branch and close to seed5 init
(`7.2316`), but it does not reproduce the stronger global biased seed6 checkpoint
(`7.2292`) or the exceptional uninitialized layer-head-mix seed5 checkpoint
(`7.2264`). The branch remains live to 750 because seed5 init had its clearest
advantage there, but the seed6 500 result already says the initialized per-layer
prior is not a clean upgrade over global biased head mix.

At step 750, layer-head-mix init seed6 fails the recovery check: `5.3527`. This
is worse than layer-head-mix init seed5 (`5.3474`), global biased seed5/seed6
(`5.3479`/`5.3479`), and all constant K2 replications. The run was killed at
750. The combined per-layer freedom plus `0.5,-0.5` prior is therefore a
seed5-positive mid-curve branch, not the next default.

GPU0 was reassigned to
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_freeze_seed5_1500_20260609`.
This keeps the successful global `ENGRAM_HEAD_MIX_INIT=0.5,-0.5` routing prior
but freezes the head-mix logits with `ENGRAM_HEAD_MIX_FREEZE=1`. The test asks
whether the useful effect is the fixed routing allocation itself, or whether the
learned head-mix parameter and its drift are needed for the strong final.

After layer-head-mix init seed5 finished, GPU1 was assigned to the paired frozen
prior seed6:
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_freeze_seed6_1500_20260609`.
This is the most direct way to test whether frozen routing is a real structural
alternative or just another seed5 trace.

Frozen-prior seed5 starts weak at step 250: `8.5261`, worse than learned global
biased seed5 (`8.5179`) and constant K2 seed5 (`8.5156`). The freeze path is
behaving as intended: logged head weights stay at the fixed softmax of
`0.5,-0.5`, about `0.731/0.269`. The run is carried to 500 because 250 has been
noisy for K2 variants, but the early signal is negative.

At step 500, frozen-prior seed5 is still weak: `7.2412`, far behind learned
global biased seed5/seed6/seed7 (`7.2347`/`7.2292`/`7.2344`) and constant K2
seed5 (`7.2339`). Frozen seed5 and the paired seed6 startup were killed. This
is a useful mechanism result: the successful global biased runs do not merely
need a fixed first-head prior. They need the head-mix parameter to learn away
from the raw init; learned seed5 is already around `0.679/0.321` at step 250 and
ends near `0.617/0.383`, while frozen routing stays at `0.731/0.269`.

Next structural code idea: constrained layer deltas for head mix. Full per-layer
head mix can produce very strong checkpoints but is seed-fragile, while shared
head mix is more stable but less expressive. A shared global head-mix plus
per-layer residual logits would test the middle ground: keep the robust global
routing path, but allow layer-specific deviations without giving each layer a
fully independent mix from the start.

`train_gpt.py` was patched with `ENGRAM_LAYER_HEAD_MIX_DELTA=1`. In this mode,
`ENGRAM_HEAD_MIX=1` creates the usual shared `head_mix_logits`, and the new
mode adds zero-initialized per-layer delta logits. The read path uses
`shared_logits + layer_delta_logits[layer]`, so the model starts exactly as the
global biased shared-head run but can learn layer-specific deviations. The logs
print both the global weights (`engram_head_mix_h*`) and the effective per-layer
weights (`engram_head_mix_l*_h*`). Local and remote `py_compile` passed.

Launched:

| Run | Purpose |
| --- | --- |
| `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_layerdelta_seed5_1500_20260609` | Constrained layer-delta version of the best global biased head-mix seed5 run. |
| `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_layerdelta_seed6_1500_20260609` | Paired seed6 replication, because the unconstrained per-layer branch was seed-fragile. |

First checkpoints are mixed but promising enough to continue:

- Seed5 layer-delta reaches `8.5025` at step 250, beating global biased
  seed5/seed6/seed7 starts (`8.5179`/`8.5054`/`8.5070`) and constant K2
  seed6/seed7 (`8.5051`/`8.5046`). The effective layer weights already diverge
  from the global mix: global `h0=0.695`, layer 2 `h0=0.668`, layer 8
  `h0=0.656`.
- Seed6 layer-delta reaches `8.5131` at step 250, weaker than global biased
  seed6/seed7 but still in the healthy K2 range and better than global biased
  seed5. Both runs are carried to 500, where the head-mix variants have been
  more diagnostic.

At step 500 the pair splits:

- Seed5 layer-delta falls to `7.2399`, giving back its strong 250-step start.
  This is worse than global biased seed5/seed7 (`7.2347`/`7.2344`) and constant
  K2 seed5 (`7.2339`).
- Seed6 layer-delta reaches `7.2301`, which is strong: slightly behind global
  biased seed6 (`7.2292`) but better than global biased seed5/seed7 and the
  constant K2 replications.

The constrained delta idea is therefore not uniformly stronger, but seed6 is
good enough to carry both runs to 750. This branch now has the same familiar
shape as several earlier candidates: promising in one seed/checkpoint, mixed in
the paired trace, needing 750/1000 to decide whether it is real.

At step 750, the constrained layer-delta branch becomes more interesting:

- Seed5 recovers to `5.3489`, now competitive with layer-head-mix seed6
  (`5.3489`) and better than constant K2 seed5/seed6/seed7
  (`5.3516`/`5.3508`/`5.3505`), though still behind global biased seed5/seed6
  (`5.3479`/`5.3479`).
- Seed6 reaches `5.3468`, the best 750-step checkpoint observed in the current
  K2/head-mix branch. It beats global biased seed5/seed6 (`5.3479`/`5.3479`),
  layer-head-mix-init seed5 (`5.3474`), and all constant K2 traces.

Both runs are now worth carrying to 1000. The mechanism looks plausible:
global head mix remains relatively first-head-heavy, while layer 8 is allowed to
move more toward the second head than layer 2. This is exactly the constrained
middle ground the patch was meant to test.

At step 1000, the branch fades:

- Seed5 reaches `3.5019`.
- Seed6 reaches `3.5012`.

This gives back seed6's best-in-branch 750 checkpoint and trails the best
global biased/constant K2 1000 checkpoints (`3.4996-3.4998`). It remains close
enough to carry to 1250 because prior K2/head-mix runs have often recovered
after a mediocre 1000. Mechanistically, the layer-specific deltas continue to
separate layer 8 further toward the second head than layer 2, so the code path is
doing the intended constrained routing; the open question is whether that
routing improves final loss or only reshapes the mid-curve.

At step 1250, the branch recovers:

- Seed5 reaches `3.3452`, roughly tied with global biased seed7 and constant K2
  seed7 at the same checkpoint.
- Seed6 reaches `3.3442`, tying global biased seed5 for the best 1250-step
  checkpoint from the June K2/head-mix branch.

This makes the constrained-delta branch credible again after the 1000 fade.
The final checkpoint is the only thing that matters now: it needs to beat or at
least match global biased seed5/seed6 finals (`3.2448`/`3.2453`) to justify
becoming the next default over the simpler shared-head biased prior.

The final layer-delta results are:

- Seed5 final: `3.2451`, peak memory `69,621 MiB`.
- Seed6 final: `3.2445`, peak memory `69,621 MiB`.

Seed6 is the best June-9-instance final so far, narrowly beating global biased
shared-head seed5 (`3.2448`), global biased shared-head seed6 (`3.2453`), and
layer-head-mix seed5 (`3.2449`). It is still behind the archived May 27 absolute
best, `bf400_ngramrows_trigramheavy_0p5_1p5_readhit025_s5_1500_20260527`, whose
final was `3.2411`.

The mechanism is more convincing than the margin. Fixed biased routing failed
early (`7.2412` at 500), and fully per-layer head-mix produced a strong
mid-curve but a weaker final (`3.2461`). The constrained version, where a global
head prior is learned and each layer only learns a small delta on top, gives the
best final in the current branch. Layer 8 consistently moves further toward the
second hash head than layer 2, so the model is using the delta path rather than
leaving it inert.

Because the gain is only `0.0003` over the best global-biased seed and `0.0008`
in the same-seed seed6 comparison, this should not be called robust SOTA yet.
The next check should be direct seed7/seed8 replications of the same constrained
layer-delta configuration. If those hold up, the practical June default becomes:
BF300 + readoutdelta + superpose K2 with base + aux scale `0.5` + shared
head-mix init `0.5,-0.5` + layer head-mix delta.

Seed7 and seed8 replications of this constrained layer-delta recipe were queued
on the two-GPU instance after seed5/seed6 completed. The runs start with the
expected biased head weights (`0.731/0.269`) for the global mix and for the
effective layer mixes before the learned deltas move.

At step 250:

- Seed7 reaches `8.5045`, stronger than seed5 (`8.5025`) only in the sense of
  being the same strong early band and clearly ahead of global biased seed7
  (`8.5126`).
- Seed8 reaches `8.5129`, weaker than seed7 and seed5/seed6, but still in the
  normal range for this branch.

Both are being carried to 500. The early signal does not reject the constrained
layer-delta recipe, and seed7 is especially useful because it directly repairs
the weaker global-biased seed7 trace from the same surrounding branch.

At step 500, the replications turn negative:

- Seed7 reaches `7.2497`.
- Seed8 reaches `7.2479`.

Both are well behind the seed5/seed6 layer-delta traces (`7.2399`/`7.2301`) and
behind the stronger global-biased shared-head traces. This weakens the case that
the seed6 final `3.2445` is a robust structural win. Seed7 was killed at 500 to
free GPU0. Seed8 is being carried to 750 as a recovery check because this branch
has occasionally recovered after a weak 500, but the default interpretation is
now caution: constrained layer-delta is plausible, not robustly established.

GPU0 was reassigned to
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_directresid_seed5_1500_20260609`.
This returns to the more stable global biased shared-head recipe and adds
`ENGRAM_ATTNRES_DIRECT_RESIDUAL=1` with `ENGRAM_ATTNRES_DIRECT_INIT=0.0`. The
hypothesis is different from the previously failed `ENGRAM_ATTNRES_DELTA=1` run:
normal AttnRes still decides the memory weight, but each Engram layer also gets
a learned scalar direct residual from memory to the block output. This tests
whether the useful memory signal wants a small guaranteed path in addition to
the attention-residual router.

Seed8 does recover at step 750: `5.3487`. That is worse than layer-delta seed6
(`5.3468`) and global-biased seed5/seed6 (`5.3479`/`5.3479`), but better than
the failed layer-head-mix seed7 recovery check (`5.3549`) and competitive with
layer-delta seed5 (`5.3489`). The replication remains worth carrying to 1000.
The branch is no longer clearly dead, but the robust-SOTA claim still depends on
whether this recovery survives the late 1000/1250/1500 basin.

The direct-residual run reaches step 0 normally. The metrics show
`engram_direct_resid_l2=0.0` and `engram_direct_resid_l8=0.0`, confirming the
probe starts as a no-op residual path on top of the normal AttnRes router.

At step 1000, layer-delta seed8 reaches `3.5014`. This is worse than the best
global-biased/constant K2 1000 checkpoints (`3.4996-3.4998`) and worse than the
layer-delta seed6 path (`3.5012`), but better than layer-delta seed5 (`3.5019`).
It is being carried to 1250 because the 750 recovery was real enough that the
late basin, not the 1000 checkpoint, should decide the replication.

At step 250, direct-residual seed5 reaches `8.5069`. Against the matched stable
global-biased seed5 comparator (`8.5179`), this is a meaningful early
improvement. The direct residual path is not inert: by step 250 the learned
scalars have moved to `l2=0.2509` and `l8=0.1627`. This is now the most
interesting structural probe in flight; the key question is whether the early
improvement survives 500/750 rather than becoming another mid-curve-only
perturbation.

At step 500, direct-residual seed5 reaches `7.2343`. This is slightly better
than matched global-biased seed5 (`7.2347`), behind the unusually strong
global-biased seed6 checkpoint (`7.2292`), and far better than the layer-delta
replication 500s (`7.2497`/`7.2479`). The path remains worth carrying. The
direct residual scalars also become layer-asymmetric by this point:
`l2=0.3568`, `l8=0.5494`, suggesting the model is using the guaranteed memory
path differently across Engram layers rather than just adding a uniform memory
gain.

Layer-delta seed8 reaches `3.3446` at step 1250. This is weaker than the best
layer-delta seed6 checkpoint (`3.3442`) and matched global-biased seed5
(`3.3442`), but better than global-biased seed6/seed7 (`3.3450`/`3.3453`). The
seed8 replication therefore partially rescues the branch after its bad 500: it
does not make layer-delta clearly superior, but it keeps it competitive enough
to finish.

Direct-residual seed5 reaches `5.3487` at step 750. This gives back some of the
250/500 advantage: it is roughly tied with layer-delta seed8 (`5.3487`), behind
layer-delta seed6 (`5.3468`) and global-biased seed5/seed6 (`5.3479`/`5.3479`),
but still clearly inside the competitive K2/head-mix band. The direct residual
scalars continue to separate strongly by layer (`l2=0.2715`, `l8=0.8267`), so
even if final loss does not improve, this is evidence that layer 8 wants a much
stronger direct memory path than layer 2 under this merge formulation.

At step 1000, direct-residual seed5 fades to `3.5020`, worse than matched
global-biased seed5 (`3.5007`) and the stronger K2/head-mix 1000 checkpoints.
The run was killed at 1000. The useful lesson is mechanistic rather than
performance: a guaranteed memory residual is learned eagerly, especially for
layer 8, but in this unconstrained scalar form it appears to over-contribute by
the late basin.

GPU0 was recycled to
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_attnextra2to8_seed5_1500_20260609`.
This keeps the stable global-biased shared-head recipe and enables
`ENGRAM_ATTNRES_EXTRA_SOURCE_LAYER=2`, `ENGRAM_ATTNRES_EXTRA_TARGET_LAYER=8`.
The hypothesis is that if layer 8 wants a stronger direct/skip memory path, the
cleaner structural test is to let its AttnRes router choose among current state,
Engram memory, and the earlier layer-2 state, rather than forcing an additive
memory scalar.

Layer-delta seed8 final is `3.2449` with `69,621 MiB` peak memory. This is
better than global-biased seed6/seed7 (`3.2453`/`3.2454`) and tied with
layer-head-mix seed5 (`3.2449`), but it does not replicate the seed6 layer-delta
best (`3.2445`) or beat global-biased seed5 by much (`3.2448`). Across
seed5/6/8, constrained layer-delta is competitive and mechanistically clean, but
the final-loss gain is still inside the project noise band. It is a candidate
component, not a robust new default by itself.

After seed8 completed, GPU1 was assigned to the paired extra-source probe:
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_attnextra2to8_seed6_1500_20260609`.
This gives the extra-source structural idea a seed5/seed6 pair immediately if
the early checkpoints are promising; if both starts are weak, both can be
recycled quickly.

The first extra-source attempt fails immediately. Seed5 reaches `8.5382` at
step 250, far worse than the matched global-biased seed5 comparator (`8.5179`).
The metrics explain the failure: because the existing implementation uses a
zero query over three sources, layer 8 starts by assigning about one third of
its router mass to the extra layer-2 state. This did not test a gentle optional
skip; it injected a large extra path from step 0. The seed5/seed6 equal-prob
extra-source runs were killed.

The code was patched to add a learned `engram_attnres_extra_bias` parameter,
initialized by `ENGRAM_ATTNRES_EXTRA_BIAS_INIT` and added only to the extra
source logit before the AttnRes softmax. The parameter is replicated Adam state
with `lr_mul=0.5` and is included in checkpoint config. Corrected runs were then
queued with `ENGRAM_ATTNRES_EXTRA_BIAS_INIT=-4.0`:

- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_attnextra2to8_biasm4_seed5_1500_20260609`
- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_attnextra2to8_biasm4_seed6_1500_20260609`

These corrected runs test the intended structural idea: layer 8 can learn to use
the earlier layer-2 state if useful, but the extra path starts near disabled
instead of consuming one third of the route by construction.

The corrected seed5 run confirms the bias does what it should at step 0:
`engram_attnres_extra_p_mean=0.0091` instead of the failed equal-prob run's
`0.334`. The extra source is now a near-no-op option at initialization, so the
250 checkpoint will be a fairer test of whether the model wants to grow that
route.

At step 250, the `-4.0` corrected extra-source pair is mixed:

- Seed5 reaches `8.5214`, weaker than matched global-biased seed5 (`8.5179`) and
  not worth carrying.
- Seed6 reaches `8.5074`, close to matched global-biased seed6 (`8.5054`) and
  healthy enough to carry to 500.

The extra path remains very small at step 250 (`engram_attnres_extra_p_mean`
about `0.0074`), so `-4.0` may be too conservative. Seed5 `-4.0` was killed, GPU0
was reassigned to a middle-bias probe:
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_attnextra2to8_biasm2_seed5_1500_20260609`
with `ENGRAM_ATTNRES_EXTRA_BIAS_INIT=-2.0`. This tests whether a small but not
nearly-zero extra route is the right structural prior.

At step 500, `-4.0` seed6 reaches `7.2385`. This is not a disaster, but it is
well behind the matched global-biased seed6 checkpoint (`7.2292`). The extra
route remains almost unused (`engram_attnres_extra_p_mean=0.0062`), confirming
that `-4.0` is effectively a no-op prior by 500. The run is weak enough that the
extra-source idea now mostly depends on whether the `-2.0` seed5 probe can use a
larger but still controlled initial route. The `-2.0` run starts with
`engram_attnres_extra_p_mean=0.0635`, a plausible middle point between the bad
equal-prob prior (`0.334`) and the nearly-off `-4.0` prior (`0.0091`).

The extra-source branch is now resolved negative:

- `-2.0` seed5 reaches `8.5180` at step 250, still weaker than matched
  global-biased seed5 (`8.5179`) and much weaker than the useful layer-delta
  seed5 start (`8.5025`).
- `-4.0` seed6 reaches `5.3518` at step 750, worse than matched global-biased
  seed6 (`5.3479`) and layer-delta seed6 (`5.3468`).

Both corrected extra-source runs were killed. The mechanism conclusion is that
adding an earlier layer state as a third AttnRes source is not the right way to
express the layer-8 direct/skip-memory signal. If the extra route starts large,
it damages the model immediately; if it starts small or medium, the loss is not
improved and the route either stays near unused or still hurts.

The next structural patch is `ENGRAM_ATTNRES_LAYER_GAIN=1`: a learned per-layer
log-gain multiplying the existing AttnRes memory weight. It starts as an exact
no-op with `ENGRAM_ATTNRES_LAYER_GAIN_INIT=0.0`, is optimized as replicated Adam
with `lr_mul=0.25`, and logs effective gains for layers 2 and 8. This is a more
targeted version of the direct-residual lesson: layer 8 seemed to want more
memory strength, but the unconstrained direct additive path over-contributed.
Layer-gain keeps the normal router and only lets each Engram layer adjust the
routed memory scale.

Queued pair:

- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_attnlayergain_seed5_1500_20260609`
- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_attnlayergain_seed6_1500_20260609`

At step 250, layer-gain is mixed but mechanistically active:

- seed5: `8.5071`, with `gain_mean=1.023`, `gain_l2=1.154`,
  `gain_l8=1.100`.
- seed6: `8.5152`, with `gain_mean=1.023`, `gain_l2=1.157`,
  `gain_l8=1.099`.

The learned scale is not staying at its no-op initialization. It increases
memory strength for both Engram layers, with layer 2 gaining slightly more than
layer 8. The loss signal is not yet decisive: seed5 is better than the matched
global-biased seed5 start (`8.5179`) and essentially tied with direct-resid
seed5 (`8.5069`), but seed6 is worse than matched global-biased seed6
(`8.5054`). Both runs are being carried to step 500 as the real filter.

At step 500, layer-gain is not strong enough as a two-seed branch:

- seed5: `7.2450`, with `gain_mean=1.060`, `gain_l2=1.292`,
  `gain_l8=1.373`.
- seed6: `7.2317`, with `gain_mean=1.061`, `gain_l2=1.297`,
  `gain_l8=1.376`.

The mechanism is informative: by 500, the model has shifted the layer-8 memory
gain above layer 2, and the AttnRes probability is also higher at layer 8
(`p_l8=0.699` vs `p_l2=0.613` for seed6). But the loss does not beat the
matched global-biased and layer-delta comparators by enough to justify spending
both GPUs. Seed5 was killed at 500; seed6 is being carried to 750 only as a
mechanism check.

At step 750, layer-gain seed6 reaches `5.3492`, still behind matched
global-biased seed6 (`5.3479`) and layer-delta seed6 (`5.3468`). The
mechanistic signal becomes stronger but not useful: `gain_l8` rises to `1.740`
while `gain_l2` is `1.374`, yet the router probability falls back toward
`0.5` (`p_l8=0.507`, `p_l2=0.489`). This looks like the model trying to
increase effective layer-8 memory scale while the router compensates, rather
than a clean loss-improving degree of freedom. The remaining layer-gain run was
killed at 750.

GPU0 was reassigned to a targeted direct-residual probe:
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_directresid_l8only_seed5_1500_20260609`.
This adds `ENGRAM_ATTNRES_DIRECT_LAYERS=8`, leaving the existing
direct-residual feature unchanged by default. The prior direct-residual run
showed layer 8 rapidly learning a larger additive memory path, but the all-layer
version became too blunt late. The layer-8-only probe tests whether the
direct/skip-memory signal is useful when confined to the layer that most clearly
wants it.

After layer-gain was killed, GPU1 was assigned the matching seed6 l8-only run:
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_directresid_l8only_seed6_1500_20260609`.

The first l8-only checkpoint is not strongly positive. Seed5 reaches `8.5156`
at step 250, with `engram_direct_resid_l8=0.166` and `engram_direct_resid_l2=0`
as intended. This is slightly better than matched global-biased seed5
(`8.5179`) but weaker than all-layer direct-resid seed5 (`8.5069`) and
layer-delta seed5 (`8.5025`). The branch is being carried to 500 because the
all-layer direct-resid effect became more interpretable at 500, but this first
point suggests that simply removing the layer-2 direct path does not reproduce
the useful early direct-resid behavior.

At the next filter:

- l8-only seed5 reaches `7.2406` at step 500, with
  `engram_direct_resid_l8=0.562`. This is worse than matched global-biased
  seed5 (`7.2347`) and all-layer direct-resid seed5 (`7.2343`). Seed5 was
  killed at 500.
- l8-only seed6 reaches `8.5053` at step 250, essentially tied with matched
  global-biased seed6 (`8.5054`) and ahead of the layer-gain seed6 start
  (`8.5152`). At step 500, it reaches `7.2291` with
  `engram_direct_resid_l8=0.561`. This is essentially tied with matched
  global-biased seed6 (`7.2292`) and ahead of layer-delta seed6 (`7.2301`).
  Seed6 is being carried to 750.

The current interpretation is that layer-8 direct residual alone does learn the
intended parameter, but is seed-sensitive: seed5 does not recover the all-layer
direct-resid early gain, while seed6 looks competitive through 500. The 750
checkpoint should show whether this is a real branch or another noisy early
tie.

At step 750, l8-only seed6 collapses relative to the useful branches:
`5.3793`, far worse than matched global-biased seed6 (`5.3479`) and layer-delta
seed6 (`5.3468`). The learned direct coefficient on layer 8 rises to `0.888`,
while the normal AttnRes router probability drops back toward `0.5`. This
matches the layer-gain lesson: the model can learn a stronger effective layer-8
memory path, but extra scale is not the missing ingredient and the router tends
to compensate or destabilize. The l8-only branch was killed.

GPU0 was reassigned to `ENGRAM_ATTNRES_DELTA=1`:
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_attnresdelta_seed5_1500_20260609`.
This uses an existing code path where the AttnRes router compares residual
no-op vs memory delta, instead of comparing the current main state against the
memory state as two replacement sources. This is a more literal fit to the way
Engram is used here: memory is added into the residual stream, so the router
should decide whether to add memory, not whether memory is a better replacement
for the main state.

Delta AttnRes starts well. Seed5 reaches `8.5041` at step 250, with router
probability staying near the residual-decision interpretation (`p_mean=0.519`,
`p_l2=0.525`, `p_l8=0.513`) instead of jumping to the `~0.63` memory-heavy
probabilities seen in the standard router. This is better than matched
global-biased seed5 (`8.5179`) and close to layer-delta seed5 (`8.5025`), while
using an arguably cleaner routing semantics. GPU1 was assigned a matching
delta seed6 run:
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_attnresdelta_seed6_1500_20260609`.

Delta seed5 does not hold at 500: `7.2403`, weaker than matched global-biased
seed5 (`7.2347`) and far weaker than the useful layer-delta/head-mix-positive
branches. It was killed at 500. Delta seed6 is still running to its first
filter checkpoint.

Delta seed6 reaches `8.5107` at step 250, weaker than matched global-biased
seed6 (`8.5054`) and the useful l8-only seed6 start (`8.5053`). Combined with
seed5's weak 500, this makes delta AttnRes a negative branch. Seed6 was killed
at 250.

GPU0 was reassigned to an AttnRes gain warmup probe:
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_attnresgainwarm500_seed5_1500_20260609`.
This uses the existing `ENGRAM_ATTNRES_GAIN_WARMUP_STEPS=500` path, ramping the
memory merge gain from `0` to `1.5` over the first 500 steps. The reason to test
it is that layer-gain, direct-resid, and l8-only direct all show the model
learning more memory scale, but then losing quality as the router compensates or
the path overcontributes. Warmup asks whether the useful memory table can learn
before the residual stream is forced to absorb a full-strength memory branch.

The warmup config is confirmed: seed5 step 0 logs `engram_attnres_gain=0.0`.
After delta seed6 was killed, GPU1 was assigned the matching warmup seed6 run:
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_attnresgainwarm500_seed6_1500_20260609`.

Warmup seed5 is weak at step 250: `8.5213`. The mechanism also goes the wrong
way: despite the merge gain being only `0.75`, the router compensates by pushing
memory probability very high (`p_mean=0.745`, `p_l2=0.724`, `p_l8=0.767`).
This is worse than the matched global-biased seed5 (`8.5179`) and much worse
than the useful layer-delta/direct variants. Warmup is therefore likely
negative unless seed6 surprises.

Warmup seed6 is decisively worse: `8.5814` at step 250, with the same router
compensation pattern (`p_mean=0.743`). This resolves gain warmup negative; seed6
was killed at 250.

Warmup seed5 was killed at 250. GPU0 was assigned a softer version of the
layer-sign idea:
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_layersignaux05_seed5_1500_20260609`.
This uses `ENGRAM_LAYER_SIGN_AUX_SCALE=0.5`, which blends a signed auxiliary
readout with the normal readout instead of replacing the memory heads with hard
coarse layer signs. Hard layer signs previously had an early signal but failed
at 500; this test asks whether a softer signed view can preserve useful
anti-collision information without dominating the base readout.

After warmup seed6 was killed, GPU1 was assigned the matching sign-aux seed6:
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_layersignaux05_seed6_1500_20260609`.

Sign-aux is negative on both seeds at 250:

- seed5: `8.5435`
- seed6: `8.5267`

The signed auxiliary view changes the readout geometry too much even when mixed
softly. Both runs were killed at 250. Since hard layer signs and sign-aux both
hurt by 500 or earlier, the next sign-related test shifts from dimension/head
signs to row-level signs.

Both GPUs were reassigned to `ENGRAM_LAYER_ROW_SIGNS=1`:

- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_layerrowsigns_seed5_1500_20260609`
- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_layerrowsigns_seed6_1500_20260609`

This applies a deterministic row-and-dimension sign pattern before memory-head
normalization. It is a collision/decorrelation test rather than an auxiliary
readout branch.

Row-sign seed5 starts strongly: `8.4993` at step 250. This beats matched
global-biased seed5 (`8.5179`), layer-delta seed5 (`8.5025`), all-layer
direct-resid seed5 (`8.5069`), and all previous sign variants. Mechanistically,
row signs lower the effective update/grad ratio (`12.2` vs the usual high teens)
while increasing gradient RMS, suggesting the signed row view changes collision
geometry rather than merely scaling memory. Seed5 is being carried to 500 and
seed6 is still pending its first 250 checkpoint.

Row-sign seed6 reaches `8.5070` at step 250. This is not as strong as seed5,
but it is close to matched global-biased seed6 (`8.5054`) and far healthier
than the failed sign-aux/warmup branches. Both row-sign runs are being carried
to 500.

At step 500:

- seed5 reaches `7.2306`, beating matched global-biased seed5 (`7.2347`) and
  landing close to the best useful 500-step branches.
- seed6 reaches `7.2410`, weak relative to matched global-biased seed6
  (`7.2292`) and layer-delta seed6 (`7.2301`).

Seed6 was killed at 500. Seed5 is being carried to 750, and GPU1 was reassigned
to a third replication:
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_layerrowsigns_seed7_1500_20260609`.
The branch is seed-fragile so far, but seed5 is strong enough that row signs are
the most promising new structural result from this stretch.

At step 750, row-sign seed5 fades to `5.3495`, behind matched global-biased
seed5 (`5.3479`) and layer-delta seed6 (`5.3468`). It was killed at 750. This
keeps the mechanism interesting but not sufficient by itself: row-level signs
produce the strongest early signal, then lose the edge once the router/memory
scale adapts.

GPU0 was reassigned to combine the two most plausible ideas:
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_layerrowsigns_layerdelta_seed5_1500_20260609`.
This sets both `ENGRAM_LAYER_ROW_SIGNS=1` and
`ENGRAM_LAYER_HEAD_MIX_DELTA=1`, testing whether per-layer head-mix adaptation
can preserve the row-sign early collision/decorrelation benefit later in
training.

Row-sign seed7 is weak at 250: `8.5128`. It was killed. GPU1 was reassigned to
the matching row-sign + layer-head-delta seed6:
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_layerrowsigns_layerdelta_seed6_1500_20260609`.

The row-sign + layer-head-delta combo seed5 starts at `8.5061` at step 250. This
is weaker than pure row-sign seed5 (`8.4993`) and close to the earlier
layer-delta seed6 start (`8.5131`/seed5 `8.5025` neighborhood), so it is not an
immediate win. Mechanistically, the layer-delta logits move strongly by 250
(`l2_h0=0.602`, `l8_h0=0.634`), so the combined branch is active and is being
carried to 500.

At step 500, combo seed5 reaches `7.2265`, matching the earlier best 500-step
layer-head-mix signal and beating pure row-sign seed5 (`7.2306`). This is the
best mid-training result from the new row-sign branch. Combo seed6, however,
starts weak at 250 (`8.5157`) and was killed. GPU1 was reassigned to combo
seed7:
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_layerrowsigns_layerdelta_seed7_1500_20260609`.

At step 750, combo seed5 fades badly to `5.3530`, worse than pure row-sign
seed5 (`5.3495`) and the matched/global layer-delta comparators. It was killed
at 750. The combined branch improves the 500-step point but appears to
over-adapt by 750.

GPU0 was reassigned to pure row signs with a lower Engram LR:
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_layerrowsigns_lr3_seed5_1500_20260609`.
This keeps `ENGRAM_LAYER_ROW_SIGNS=1` but lowers `ENGRAM_LR_MUL` from `5.0` to
`3.0`. The rationale is that row signs produce higher gradient RMS and lower
update/grad ratio, then fade later; lowering the table LR tests whether the
early collision/decorrelation benefit can persist without over-updating the
signed rows.

Combo seed7 posts a very strong 250-step result: `8.4969`. This is better than
pure row-sign seed5 (`8.4993`) and combo seed5 (`8.5061`), so the combo branch
is still alive despite seed5 fading at 750 and seed6 failing at 250. GPU1 is
carrying combo seed7 to 500.

At step 500, combo seed7 reaches `7.2324`. This is still healthy, beating pure
row-sign seed5 (`7.2306`) only narrowly in the wrong direction and remaining
near the global-biased BF300 traces, but it does not match combo seed5
(`7.2265`). Since seed7 had the best 250-step point, it is being carried to 750;
that checkpoint is the decisive test because combo seed5 over-adapted by 750.

At step 750, combo seed7 reaches `5.3496`. This does not move the SOTA frontier:
it is worse than global-biased seed5 (`5.3479`) and layer-delta seed6 (`5.3468`).
However, it avoids the severe combo seed5 fade (`5.3530`) and is essentially tied
with pure row-sign seed5 (`5.3495`), so seed7 is being carried to 1000 as a final
recovery check before recycling the GPU.

At step 1000, combo seed7 is `3.5026`, behind the matched global-biased seed5
(`3.5007`) and layer-delta comparators. This resolves the row-sign +
layer-head-delta combo as interesting for early/mid optimization but not a SOTA
trajectory. The run was killed at 1000.

The first LR3-labelled row-sign launch was invalid: the launcher hardcoded
`ENGRAM_LR_MUL=5.0`, so the run named `...layerrowsigns_lr3_seed5...` was
actually a duplicate LR5 configuration. The launcher has been fixed to respect
an external `ENGRAM_LR_MUL`, the duplicate was stopped, and GPU0 was relaunched
as `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_layerrowsigns_lr3fix_seed5_1500_20260609`
with the intended `ENGRAM_LR_MUL=3.0`.

LR3fix seed5 starts at `8.5035` at step 250. That is weaker than pure row-sign
seed5 at LR5 (`8.4993`) and combo seed7 (`8.4969`), but still better than the
matched global-biased seed5 (`8.5179`). The optimizer metrics confirm the lower
LR took effect (`engram_lr=2.4e-2`, update/grad `4.99` instead of the usual
`~12-18`). It is being carried to 500; the useful question is whether lower
update pressure preserves the row-sign benefit after the LR5 branch faded by 750.

At step 500, LR3fix is weak: `7.2383`, worse than pure row-sign seed5 at LR5
(`7.2306`) and worse than the global-biased BF300 seed5 comparator (`7.2347`).
This resolves the lower-LR explanation negatively: the row-sign fade is not
simply due to over-updating the signed rows. The run was killed at 500.

GPU0 was then assigned a direct memory-readout regularization test:
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_outputdrop005_s500_seed5_1500_20260609`.
This keeps the current BF300/superpose/head-mix recipe and ramps
`ENGRAM_OUTPUT_DROPOUT` from `0` to `0.05` over 500 steps. Unlike hot-row
dropout, this does not target hit-count buckets; it asks whether the useful
regularization is about reducing over-reliance on the Engram readout channel
itself.

After combo seed7 was killed, GPU1 was assigned the paired source-dropout test:
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_headdrop005_s500_seed5_1500_20260609`.
This ramps `ENGRAM_HEAD_DROPOUT` from `0` to `0.05` over 500 steps, targeting
robustness across the base/superposed readout heads instead of dropping output
dimensions.

At step 250, output-drop is mostly neutral/slightly weak: `8.5165` with current
dropout `0.025`. It is close to matched global-biased seed5 (`8.5179`) but not
near the strongest row-sign/combo starts. It is being carried to 500 because the
regularization ramps through that point.

Head-drop is more promising at step 250: `8.5040` with current dropout `0.025`.
This is weaker than the best row-sign/combo starts (`8.4993`/`8.4969`) but
stronger than the global-biased seed5 comparator and close to the useful
layer-delta/direct-residual early signals. It is being carried to 500 as the
more interesting readout-source regularization branch.

At step 500, both dropout branches are slightly weak:

- output-drop: `7.2368`
- head-drop: `7.2373`

Both trail the matched global-biased seed5 (`7.2347`) and the strongest
row-sign/combo 500-step points. The negative margin is small, and these branches
are explicitly testing whether light readout regularization helps after 500, so
both are being carried to 750 before a kill decision.

At step 750, both dropout branches fail clearly:

- output-drop: `5.3541`
- head-drop: `5.3552`

Both are worse than global-biased seed5 (`5.3479`), pure row-sign seed5
(`5.3495`), combo seed7 (`5.3496`), and layer-delta seed6 (`5.3468`). This
resolves broad readout-channel regularization negatively. The useful early
signals from this stretch appear to come from collision/decorrelation geometry,
not generic dropout or source masking. Both dropout runs were killed at 750.

The next structural change makes row-sign strength schedulable:
`ENGRAM_LAYER_ROW_SIGN_SCALE`, `ENGRAM_LAYER_ROW_SIGN_SCALE_FINAL`,
`ENGRAM_LAYER_ROW_SIGN_SCALE_SCHEDULE_START`, and
`ENGRAM_LAYER_ROW_SIGN_SCALE_SCHEDULE_STEPS`. Scale `1.0` is the existing hard
row-sign behavior; scale `0.0` becomes the unsigned baseline. The code compiles
locally and on the instance.

Both GPUs were assigned seed5 row-sign fade probes:

- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_layerrowsigns_rowfade0_s500_250_seed5_1500_20260609`
- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_layerrowsigns_rowfade0_s500_500_seed5_1500_20260609`

Both start from the successful pure row-sign setting and keep full strength
through step 500, then fade row signs to zero over either 250 or 500 steps. This
directly tests the current best explanation: row signs improve early collision
geometry but become late interference. If that is right, the 500-step point
should match pure row-sign seed5 and the 750/1000 checkpoints should recover
toward the global-biased or layer-delta trajectories instead of fading.

At step 250, both row-fade runs match the original pure row-sign seed5 exactly:
`8.4993`. This is expected because the row-sign scale is still `1.0`; it also
confirms the new schedule code is preserving existing behavior before the
scheduled fade.

At step 500, both row-fade runs again match pure row-sign seed5: `7.2306`. This
validates the schedule boundary. The important checkpoint is 750: the 250-step
fade is fully unsigned by then, while the 500-step fade is halfway between hard
row signs and the unsigned baseline.

At step 750, row-fade fails decisively:

- fade over 500-750: `5.4038`
- fade over 500-1000: `5.3672`

Both are much worse than hard row-sign seed5 (`5.3495`) and the matched
global/layer-delta trajectories. This rejects the simple "use row signs early,
remove them late" hypothesis. The signed geometry is not just an early
regularizer; once the table and router train under signed rows, fading the signs
creates a representation mismatch. Both row-fade runs were killed at 750.

The next probe keeps the row-sign coordinate system constant from the start but
softens its strength:

- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_layerrowsigns_rowsignscale05_seed5_1500_20260609`
- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_layerrowsigns_rowsignscale075_seed5_1500_20260609`

With the current interpolation, scale `1.0` is hard row signs, scale `0.0` is
unsigned, and intermediate scales turn negative signed dimensions into partial
attenuation rather than a full sign flip. This tests whether softer constant
decorrelation can avoid the late mismatch while keeping some of the early
collision benefit.

At step 250:

- row-sign scale `0.5`: `8.5214`, weak and killed.
- row-sign scale `0.75`: `8.5033`, healthy enough to carry to 500, but still
  weaker than hard row-sign seed5 (`8.4993`).

This suggests the early gain is strength-dependent. Softening the signs removes
some of the collision/decorrelation benefit immediately, rather than preserving
it while fixing the late fade.

At step 500, row-sign scale `0.75` is weak: `7.2408`, worse than hard row-sign
seed5 (`7.2306`) and global-biased seed5 (`7.2347`). It was killed. This
resolves constant soft row-sign interpolation negatively.

After the `0.5` scale run was killed, GPU0 was assigned a targeted interaction:
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_layerrowsigns_hotdrop010_min1024_s250_500_seed5_1500_20260609`.
This combines hard row signs with the previously useful broad hot-row dropout
schedule (`0 -> 0.10`, steps 250-750, min hits 1024). Broad output/head dropout
failed, but hot-row dropout was earlier the regularizer most connected to the
observed hot-row pathology, so this tests whether row signs need targeted
anti-hot-row pressure rather than global readout noise.

After the `0.75` scale run was killed, GPU1 was assigned:
`bf300_sota_readoutdelta_superposek2_base_aux025_headmixinit05_layerrowsigns_seed5_1500_20260609`.
This keeps hard row signs but lowers the superpose auxiliary scale from `0.5`
to `0.25`, testing whether row signs are late-negative because the auxiliary
superposed readout is overconstraining the signed coordinate system.

Row-sign + hotdrop reaches `8.4993` at step 250, matching hard row-sign seed5
as expected because the hotdrop schedule starts at step 250. Aux0.25 has the
intended superpose config and is still at step 0. Both runs are being carried to
500.

At step 500:

- row-sign + hotdrop is `7.2360`, worse than hard row-sign seed5 (`7.2306`).
  It is being carried to 750 because prior hotdrop gains mostly appeared after
  the ramp became active.
- row-sign + aux0.25 is `8.5061` at step 250, weaker than hard row-sign seed5
  (`8.4993`) but not decisively dead.

At step 500, row-sign + aux0.25 becomes interesting: `7.2270`, beating hard
row-sign aux0.5 (`7.2306`) and landing close to combo seed5 (`7.2265`). This
suggests the row-sign branch may be sensitive to auxiliary pressure; lower aux
weakens the 250 start but improves the 500 checkpoint. It is being carried to
750.

At step 750, row-sign + hotdrop is bad: `5.3549`, worse than hard row-sign
seed5 (`5.3495`) and much worse than the best layer-delta path. It was killed.
Hot-row dropout does not fix the row-sign fade in this combination.

Because row-sign + aux0.25 had the strongest new 500-step signal in this cycle,
GPU0 was assigned a seed6 replication:
`bf300_sota_readoutdelta_superposek2_base_aux025_headmixinit05_layerrowsigns_seed6_1500_20260609`.
Seed5 is still being carried to the crucial 750 checkpoint.

At step 750, row-sign + aux0.25 fades to `5.3517`. This is better than the
failed row-fade, hotdrop, broad-dropout, and soft-row-sign variants, but worse
than hard row-sign seed5 (`5.3495`) and the stronger global/layer-delta
comparators. Seed5 was killed at 750, and the seed6 replication was stopped
before its first checkpoint. Lowering superpose auxiliary pressure improves the
500-step signal but still does not solve the 750 fade.

The next two probes keep hard row signs fixed but remove superpose auxiliary
pressure after the useful 500-step window:

- `bf300_sota_readoutdelta_superposek2_base_aux05to0_s500_250_headmixinit05_layerrowsigns_seed5_1500_20260609`
- `bf300_sota_readoutdelta_superposek2_base_aux025to0_s500_250_headmixinit05_layerrowsigns_seed5_1500_20260609`

These test whether the auxiliary readout is useful for early optimization but
becomes late interference. This differs from row-sign fade: the signed memory
coordinate system stays fixed, and only the auxiliary superposed loss/readout
pressure is removed.

At step 250, both aux-decay runs match their corresponding constant-aux
baselines before the schedule starts:

- aux0.5 -> 0: `8.4993`
- aux0.25 -> 0: `8.5061`

Both are being carried to 500; the decisive checkpoint is 750 after the
auxiliary pressure has been removed.

At step 500, both aux-decay runs still match their constant-aux counterparts:

- aux0.5 -> 0: `7.2306`
- aux0.25 -> 0: `7.2270`

This is expected because the schedule begins at step 500. Both are being carried
to 750, the first meaningful test of removing auxiliary pressure while keeping
the signed memory coordinate system fixed.

At step 750, aux-decay does not solve the row-sign fade:

- aux0.5 -> 0: `5.3541`
- aux0.25 -> 0: `5.3510`

The aux0.25 decay is slightly better than constant aux0.25 (`5.3517`), but still
worse than hard row-sign seed5 (`5.3495`) and the stronger global/layer-delta
comparators. Both runs were killed at 750. The row-sign branch now looks
resolved: it reliably creates strong early/mid optimization signals, but every
attempt to preserve or regularize it into the 750 regime has failed.

One final row-sign isolation probe was added after this: aux-only row signs.
`ENGRAM_LAYER_ROW_SIGNS_AUX_ONLY=1` leaves the base memory readout unsigned and
applies per-layer row signs only to auxiliary superposed slots. The hypothesis is
that hard row signs may be useful as a decorrelating distributed-code channel,
but harmful when they also change the base memory coordinate system.

Two BF300 probes were launched:

- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_auxonlylayerrowsigns_seed5_1500_20260609`
- `bf300_sota_readoutdelta_superposek2_base_aux025_headmixinit05_auxonlylayerrowsigns_seed5_1500_20260609`

These should be judged against constant aux0.5/aux0.25 and hard full row-sign
seed5 at 250/500/750 before carrying to final.

At step 250:

- aux-only row signs, aux0.5: `8.5179`
- aux-only row signs, aux0.25: `8.5055`

The aux0.5 run loses the full-row-sign early gain (`8.4993`) and lands near the
unsigned/global head-mix branch. The aux0.25 run matches full row-sign aux0.25
(`8.5061`), so the narrower auxiliary channel is still worth carrying to 500.
Both runs are being carried to 500 for the decisive mid-run signal.

Final aux-only results:

- aux-only row signs, aux0.5: `8.5179 -> 7.2347 -> 5.3479 -> 3.5007 -> 3.3438 -> 3.2440`
- aux-only row signs, aux0.25: `8.5055 -> 7.2426 -> 5.3523 -> 3.5053 -> 3.3455 -> 3.2458`

This is the first row-sign variant that survives to final. The aux0.5 run is
the best June endpoint so far (`3.2440`), beating constant K2 seed5 (`3.2454`),
layer-head-mix-delta seed6 (`3.2445`), and the BF400 SOTA-meta repro (`3.2458`).
It still does not beat the archived May 27 best (`3.2411`). Mechanistically, the
result says the useful part of row signs is not the signed base coordinate
system. The useful part is likely an auxiliary layer-specific distributed code:
when the base readout remains stable and only the superposed channel is signed,
the branch avoids the 750-step degradation seen in full row signs.

Followups launched:

- BF300 aux0.5 aux-only row signs seed6
- BF400 aux0.5 aux-only row signs seed5

At step 250:

- BF300 seed6: `8.5054`
- BF400 seed5: `8.5084`

Both are strong enough to continue. The BF300 seed6 point is much better than
BF300 seed5 (`8.5179`) and matches the stronger seed6 behavior seen in related
branches. The BF400 point is much better than the earlier weak BF400
layer-head-mix probe (`8.5249`), so aux-only row signs may be the first branch
that lets the wider table participate usefully.

At step 500:

- BF300 seed6: `7.2292`
- BF400 seed5: `7.2397`

BF300 seed6 is now stronger than the seed5 aux-only winner at the same point
(`7.2347`), making the aux-only BF300 branch look robust rather than a one-seed
fluke. BF400 is improved versus the earlier BF400 layer-head-mix probe
(`7.2415`) but still behind BF300; it is being carried to 750 as a scaling
recovery check.

At step 750:

- BF300 seed6: `5.3479`
- BF400 seed5: `5.3498`

BF300 seed6 continues to match the successful seed5 aux-only trajectory and does
not show the full-row-sign 750-step fade. BF400 remains behind BF300, but the
gap is small enough to finish it as a useful scaling data point.

Final followup results:

- BF300 aux-only row signs seed6:
  `8.5054 -> 7.2292 -> 5.3479 -> 3.5014 -> 3.3448 -> 3.2453`
- BF400 aux-only row signs seed5:
  `8.5084 -> 7.2397 -> 5.3498 -> 3.5004 -> 3.3453 -> 3.2448`

The best aux-only result remains BF300 seed5 at `3.2440`. Seed6 confirms the
branch is real, but not consistently better than seed5. BF400 catches up late
and beats the old BF400 current-meta repro (`3.2458`) and the constant-aux BF300
seed5 (`3.2454`), but it still does not beat BF300 aux-only seed5. This narrows
the table-scaling story: wider tables can now participate better under the
aux-only distributed-code setup, but row count alone is still not the missing
piece behind the archived `3.2411`.

Next structural combo launched:

- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_auxonlylayerrowsigns_layerheadmixdelta_seed5_1500_20260609`
- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_auxonlylayerrowsigns_layerheadmixdelta_seed6_1500_20260609`

This combines the new aux-only row-sign branch with the earlier
layer-head-mix-delta branch. It tests whether signed auxiliary distributed codes
and layer-specific head reweighting are complementary, or just competing
parameterizations of the same effect.

At step 250:

- combo seed5: `8.5025`
- combo seed6: `8.5131`

The seed5 path is very strong, beating aux-only seed5 (`8.5179`) and matching
the stronger early row-sign family. Seed6 moves the opposite direction versus
aux-only seed6 (`8.5054`). Both are being carried to 500 to distinguish
high-variance complementarity from an early one-seed artifact.

At step 500:

- combo seed5: `7.2399`
- combo seed6: `7.2301`

The seed5 early advantage does not persist to 500; it is worse than aux-only
seed5 (`7.2347`). Seed6 recovers and nearly matches aux-only seed6 (`7.2292`).
The combination currently looks high-variance rather than cleanly additive, but
both are being carried to 750 because this is where row-sign variants either
fade or become useful.

At step 750:

- combo seed5: `5.3489`
- combo seed6: `5.3468`

Seed6 is the best 750-step point in this aux-only family so far, beating the
plain aux-only seed5/seed6 750 values (`5.3479`) and avoiding the full-row-sign
fade. Seed5 remains behind. Seed5 was recycled at 750, while seed6 is being
carried to final and a seed7 combo replication was launched.

At step 1000, combo seed6 is `3.5012`, only marginally ahead of aux-only seed6
(`3.5014`). The combo still may improve the 750 transition, but it has not yet
opened a meaningful late-loss gap.

Final combo seed6:

- `8.5131 -> 7.2301 -> 5.3468 -> 3.5012 -> 3.3448 -> 3.2445`

This matches the earlier layer-head-mix-delta best but does not beat the
plain aux-only seed5 winner (`3.2440`). The 750-step bump did not turn into a
better endpoint, so layer-head-mix-delta is not clearly additive with aux-only
row signs.

Combo seed7 was launched as a robustness check. It reached:

- seed7: `8.5045 -> 7.2497`

The 250 point was promising, but the 500 point was weak. Seed7 was killed at
500. This makes the layer-head-mix-delta combination look high variance rather
than a clean additive improvement.

Next structural probe launched on the recycled GPU:

- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_auxonlylayerrowsigns_attnreslayergain_seed5_1500_20260609`

This keeps the aux-only row-sign winner and enables learned per-layer
attn-residual gain (`ENGRAM_ATTNRES_LAYER_GAIN=1`, init `0.0`). The motivation
is to let the two Engram read layers choose different residual strength without
changing the memory addressing/code geometry.

At step 250, the attn-res layer-gain probe is `8.5071`. This is much better
than aux-only seed5 at 250 (`8.5179`) and good enough to continue. The learned
layer gains have already moved above one (`l2=1.154`, `l8=1.100`), so the branch
is actively changing the residual path rather than sitting at initialization.

At step 500, the attn-res layer-gain probe falls to `7.2450`, much worse than
plain aux-only seed5 (`7.2347`). The learned gains have grown to about
`l2=1.292`, `l8=1.373`, suggesting the branch over-amplifies the residual path
rather than improving it. The run was killed at 500.

Two followup diagnostics were launched:

- Plain aux-only row signs seed7:
  `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_auxonlylayerrowsigns_seed7_1500_20260609`
- Aux-only row signs with frozen head mix seed5:
  `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_auxonlylayerrowsigns_headmixfreeze_seed5_1500_20260609`

The seed7 run measures variance of the actual current best branch. The frozen
head-mix run tests whether the learned drift from the initial `0.5,-0.5`
two-slot mixture is necessary, or whether the fixed distributed-code split is
already enough.

At step 250:

- plain aux-only seed7: `8.5070`
- frozen head mix seed5: `8.5261`

Seed7 is healthy and continues as a variance check. Frozen head mix is clearly
bad at the first checkpoint, which means the learned movement away from the
initial two-slot mixture is doing useful optimization work. The frozen run was
killed at 250.

Plain aux-only seed8 was launched on the recycled GPU:

- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_auxonlylayerrowsigns_seed8_1500_20260609`

Together with seeds 5/6/7, this should give a clearer estimate of the aux-only
branch variance around the current best.

Plain aux-only seed7 reached step 500:

- seed7: `8.5070 -> 7.2344`

This keeps seed7 in the competitive band and close to seed5 (`7.2347`) at the
same checkpoint, so it is being carried forward.

Plain aux-only seed8 reached step 250:

- seed8: `8.5069`

Seeds 7 and 8 both start in-family, reinforcing that the aux-only branch's early
signal is robust across seeds.

At the next checkpoints:

- seed7: `8.5070 -> 7.2344 -> 5.3508`
- seed8: `8.5069 -> 7.2539`

Seed7 weakens by 750 and is no longer a best-run contender, but is being carried
as a final variance point. Seed8 is clearly weak by 500 and was killed.

Seed7 later recovered somewhat:

- seed7: `8.5070 -> 7.2344 -> 5.3508 -> 3.4996`

The 750 point is weak, but the 1000 point is better than seed5/seed6 aux-only
(`3.5007`/`3.5014`), so seed7 is being carried to final as a useful variance
case.

BF400 aux-only seed6 was launched on the freed GPU:

- `bf400_sota_readoutdelta_superposek2_base_aux05_headmixinit05_auxonlylayerrowsigns_seed6_1500_20260609`

BF400 seed5 finished at `3.2448`; this checks whether wider-table scaling is
seed-limited or consistently behind the BF300 seed5 winner.

BF400 seed6 reached step 250 at `8.5117`, weaker than BF400 seed5 (`8.5084`) but
not weak enough to stop before the 500 checkpoint.

BF400 seed6 reached step 500 at `7.2660`, clearly worse than BF400 seed5
(`7.2397`) and the BF300 aux-only family. It was killed at 500. This makes
BF400 aux-only look consistently behind BF300 at the useful settings rather than
merely seed-limited.

Plain aux-only seed7 reached step 1250:

- seed7: `8.5070 -> 7.2344 -> 5.3508 -> 3.4996 -> 3.3457`

Seed7 remains a variance point and is unlikely to beat the seed5 best.

A final head-mix diagnostic was launched on the freed GPU:

- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit00_auxonlylayerrowsigns_seed5_1500_20260610`

This keeps the aux-only row-sign branch but initializes the two head-mix logits
equally (`0,0`) instead of the current base-biased `0.5,-0.5`. Frozen head mix
showed learned drift is necessary; this tests whether starting closer to a
balanced base/aux mixture improves the early trajectory.

Layer-head-mix seed7 starts at `8.5133` at step 250. This is close to constant
K2 seed5 (`8.5156`) but weaker than layer-head-mix seed6 (`8.5062`) and does
not by itself establish robustness. It is being carried to 500/750, where the
layer-head-mix mechanism has shown larger movement than at 250.

At step 500, layer-head-mix seed7 is `7.2483`, which is weak. It is worse than
constant K2 seed5/seed6/seed7 (`7.2339`/`7.2373`/`7.2368`) and worse than
layer-head-mix seed5/seed6 (`7.2264`/`7.2417`). Because seed6 recovered at 750
after a weak 500, seed7 is still worth carrying to 750 as a recovery check, but
its current path argues against treating layer-head-mix as robust yet.

At step 750, layer-head-mix seed7 is `5.3549`, failing the recovery check. This
is worse than layer-head-mix seed5/seed6 (`5.3496`/`5.3489`) and all constant K2
replications (`5.3516`/`5.3508`/`5.3505`). The run was killed at 750 and GPU0
was recycled.

GPU0 was then assigned to a BF400 layer-head-mix seed5 probe:
`bf400_sota_readoutdelta_superposek2_base_aux05_layerheadmix_seed5_1500_20260609`.
This keeps the seed5-positive learned layer/head weighting and increases table
rows from BF300 to BF400. The branch is an absolute-SOTA probe: if memory fits
and the 500-step loss is strong, it tests whether layer-head-mix plus more rows
can close the remaining gap to the archived May 27 best `3.2411`; if early loss
or memory is bad, it should be recycled quickly.

BF400 layer-head-mix fits but is weak early: step 250 is `8.5249` and step 500
is `7.2415`, with about `95.4 GiB` allocated during training. The 500 point is
worse than BF300 layer-head-mix seed5 (`7.2264`) and no better than the earlier
weak BF400 K2 probe. The run was killed at 500, so simply adding rows to the
seed5 layer-head-mix branch does not look like the path to the archived `3.2411`
result.

For reference, the earlier constant-aux seed7 final is `3.2459` with
`69,477 MiB` peak memory. This is worse than seed5
constant-aux (`3.2454`) and just behind old BF400 SOTA-meta (`3.2458`), but it
beats BF300 current-meta (`3.2472`), earlier BF300 readoutdelta (`3.2466`), and
current-meta BF400 (`3.2462`). Across seeds 5/6/7, constant-aux BF300 now
consistently reaches BF400-quality final loss with BF300 table size, and one seed
sets the best observed final.

## Current live summary - 2026-06-10

Aux-only row signs now have a refreshed plot/CSV pass:

- `summary_auxonly_20260610.csv`
- `auxonly_variants_vs_params_20260610.png`
- `auxonly_variants_curves_20260610.png`

The refreshed parse covers 13 aux-only-style logs, with 6 complete finals. The
best complete run remains:

- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_auxonlylayerrowsigns_seed5_1500_20260609`: `3.2440`

The completed plain aux-only BF300 seed spread is now:

- seed5: `3.2440`
- seed6: `3.2453`
- seed7: `3.2451`

This makes aux-only row signs a real improvement over the previous June branch,
but still not a confirmed improvement over the archived May 27 best
`3.2411`. The useful conclusion is structural: signed layer-specific codes help
when restricted to the auxiliary superposed rows, while signing the base memory
coordinate system is harmful or at least not robust.

The equal head-mix initialization run is active on GPU1:

- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit00_auxonlylayerrowsigns_seed5_1500_20260610`

It starts from `head_mix_h0=head_mix_h1=0.5` instead of the usual base-biased
`0.5,-0.5` logits. This tests whether starting closer to the learned final
mixture helps, since frozen head-mix was bad but successful runs drift toward
more auxiliary weight.

GPU0 was reassigned to a structural K=3 distributed-code probe:

- `bf300_sota_readoutdelta_superposek3_base_aux03536_auxonlylayerrowsigns_seed5_1500_20260610`

This keeps one base row and adds two signed auxiliary salted rows. The aux scale
is `0.3535533906`, so the total auxiliary L2 energy roughly matches the K=2,
aux=`0.5` winner after superpose normalization. This isolates whether spreading
the auxiliary code across more independent rows helps, rather than just giving
the memory branch more amplitude.

K=3 fit in memory but failed the 500-step check:

- 250: `8.5042`
- 500: `7.2503`

The good 250 did not hold. By 500 it was clearly worse than K=2 aux-only, so the
run was killed and GPU0 was recycled. The current read is that simply spreading
the same auxiliary energy across two salted aux rows is not enough; it may dilute
per-row learning or make the downstream readout harder even though early loss
briefly improves.

Equal head-mix init has reached:

- 250: `8.5166`
- 500: `7.2359`
- 750: `5.3504`
- 1000: `3.4999`
- 1250: `3.3453`
- 1500: `3.2461`

This is not a direct improvement over normal seed5/seed6 aux-only. The final is
worse than the aux-only seed5/seed6/seed7 spread (`3.2440`/`3.2453`/`3.2451`).
The diagnostic leans toward the original base-biased head-mix initialization
being acceptable; the model learns toward more aux usage without needing to
start balanced.

GPU0 is now running the closer "k rows then read out values" mechanism test:

- `bf300_sota_sketchk2_slotattn_aux05_auxonlylayerrowsigns_nolayerdelta_seed5_1500_20260610`

This uses `ENGRAM_SKETCH_K=2`, `ENGRAM_SKETCH_INCLUDE_BASE=1`,
`ENGRAM_SKETCH_SLOT_READOUT=1`, and `ENGRAM_SKETCH_SLOT_ATTENTION=1`, with
aux-only row signs. This path projects each slot separately and attends over slot
values, so it is closer to the proposed K/V readout over multiple retrieved rows.
It disables `ENGRAM_LAYER_READOUT_DELTA` because the current slot-readout path
rejects that combination, making this a mechanism probe rather than a direct
SOTA configuration.

Slot-attention reached step 250 at `8.5234`, clearly weak and slower than the
superpose path. It was killed. The mechanism is implemented and runs, but this
configuration is not competitive enough to occupy the GPU.

GPU0 was then assigned to a cheaper SOTA-compatible count-sketch probe:

- `bf300_sota_readoutdelta_sketchk2_base_aux05_auxonlylayerrowsigns_seed5_1500_20260610`

This uses `ENGRAM_SKETCH_K=2`, `ENGRAM_SKETCH_INCLUDE_BASE=1`, and
`ENGRAM_SKETCH_AUX_SCALE=0.5`, while keeping `ENGRAM_LAYER_READOUT_DELTA=1`.
Unlike slot-attention, it sums signed sketch rows before projection, so it is
less expressive but directly comparable to the K=2 superpose aux-only branch.

Sketch-sum seed5 is mixed but worth carrying:

- 250: `8.5241`
- 500: `7.2329`
- 750: `5.3533`

The 250 point is weak, the 500 point is stronger than the aux-only seed5 and
seed7 500s and close to the better seed6 trajectory, but the 750 point fails.
This makes signed sketch summation look like a transient early/mid-training
benefit rather than a stable improvement at fixed aux scale. GPU1 was assigned a
seed6 replication:

- `bf300_sota_readoutdelta_sketchk2_base_aux05_auxonlylayerrowsigns_seed6_1500_20260610`

Seed6 reached 250 at `8.5065`, so it is being carried to 500. GPU0 was recycled
from the failed seed5 fixed-scale sketch run into a scheduled sketch probe:

- `bf300_sota_readoutdelta_sketchk2_base_aux05to01_s500_500_auxonlylayerrowsigns_seed5_1500_20260610`

This keeps the same sketch branch through step 500, then decays sketch aux scale
from `0.5` to `0.1` over steps 500-1000. The hypothesis is direct from the
fixed-scale trace: the signed aux sketch may be useful early, but too noisy or
too constraining once the base memory has matured.

Seed6 fixed-scale sketch reached 500 at `7.2441`, failing to replicate seed5's
500-step recovery (`7.2329`). It was killed. That makes fixed-scale sketch a
weak/noisy branch; only the scheduled-decay seed5 run remains worth observing.
Older superpose aux-decay runs already exist and were not better, so they were
not relaunched.

The scheduled-decay sketch run matched fixed-scale through 500 and then failed
harder at 750:

- 250: `8.5241`
- 500: `7.2329`
- 750: `5.3573`

It was killed. This falsifies the "sketch helps early but should be annealed
away" explanation for this schedule. Current sketch verdict: the signed
count-sketch family is not a promising SOTA path in this setup. The only useful
signal was a transient seed5 500-step improvement that did not persist and did
not replicate on seed6.

The final plot refresh for this block parsed 18 runs, 7 complete finals, and
kept the same best complete endpoint:

- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_auxonlylayerrowsigns_seed5_1500_20260609`: `3.2440`

Correction from code review on 2026-06-10: the `ENGRAM_LAYER_ROW_SIGNS_AUX_ONLY`
flag was a no-op for the normal summed `ENGRAM_SUPERPOSE_K > 1` path. It only
affected the slot-readout path. The summed lookup helper did not receive
`layer_id`, so it could not apply per-layer row signs to auxiliary slots before
the slot sum. A paired BF400 check confirmed this: the layer-readout K2 run with
and without aux-only row signs had identical step-250 metrics (`8.5048`) before
the patch, differing only in timing.

The code is now patched so `_lookup_combined_memory_heads(..., layer_id=...)`
applies layer row signs to auxiliary superpose/sketch slots before summing.
Therefore, the earlier `auxonlylayerrowsigns` superpose conclusions should be
read as conclusions about K2 base+aux superposition, not as evidence that
aux-only row signs themselves helped. The `3.2440` result remains the best
complete June endpoint in this branch, but its mechanism is now: BF300
readoutdelta + K2 base/aux superpose + biased head mix. The signed auxiliary
channel needs retesting with the patched path.

Post-patch live retest:

- `bf400_sota_layerreadouts_superposek2_base_aux05_norowsigns_seed5_1500_20260610`
- `bf400_sota_layerreadouts_superposek2_base_aux05_auxonlylayerrowsigns_fixed_seed5_1500_20260610`

The unsigned control reached 250/500/750/1000/1250/final at
`8.5048 -> 7.2401 -> 5.3493 -> 3.5026 -> 3.3460 -> 3.2459`, with peak
allocated memory about `89.8 GiB`. The fixed signed run reached
250/500/750/1000 at `8.4995 -> 7.2323 -> 5.3458 -> 3.5011`, confirming both
that the patch is active and that aux-only row signs help once applied on the
summed superpose path. The gap shrinks from 500 to 1000 but remains positive;
this is the first real evidence for the signed auxiliary channel. The earlier
evidence was only evidence for K2 superposition.

The fixed signed BF400 layer-readout run later reached 1250/final at
`3.3471 -> 3.2463`, slightly worse than the unsigned control's
`3.3460 -> 3.2459`. That says the signed auxiliary channel's early benefit fades
by late training in the layer-readout BF400 setting. The more direct test was
launched on idle GPU0:

- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_auxonlylayerrowsigns_fixed_seed5_1500_20260610`

This reruns the previous `3.2440` BF300 readoutdelta K2/headmix branch with the
patched aux-only row-sign path actually active.

That BF300 fixed seed5 run reached 250/500/750 at
`8.5168 -> 7.2433 -> 5.3529`. This is a negative signal: the old no-op seed5
version was `8.5179 -> 7.2347 -> 5.3479`, so the real signed auxiliary path is
slightly better at 250 but worse by 500 and 750. It was killed at 750. Seed6 was
launched on the second GPU to check whether this is seed-specific; it reached
250/500 at `8.5087 -> 7.2355`, also worse than the old no-op seed6
(`8.5054 -> 7.2292`). It was killed at 500.

To distinguish "signed aux channel is bad" from "hard sign flip is too strong",
GPU0 was reassigned to:

- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_auxonlylayerrowsignscale05_fixed_seed5_1500_20260610`

This keeps the patched aux-only row-sign path active but sets
`ENGRAM_LAYER_ROW_SIGN_SCALE=0.5`.

The scale-0.5 run reached 250/500 at `8.5092 -> 7.2441`. The 250 was much
better than hard-sign seed5 (`8.5168`) and old no-op seed5 (`8.5179`), but the
500 was again worse than no-op (`7.2347`). It was killed at 500. Interpretation:
softened signs can add useful early decorrelation, but leaving them active still
hurts by mid-training.

A second softened probe was launched on GPU1:

- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_auxonlylayerrowsignscale025_fixed_seed5_1500_20260610`

This sets `ENGRAM_LAYER_ROW_SIGN_SCALE=0.25`.

Scale-0.25 reached 250/500/750 at `8.5109 -> 7.2389 -> 5.3485`, also
early-positive and better than hard signs by 500, but still slightly worse than
old no-op seed5 (`8.5179 -> 7.2347 -> 5.3479`). GPU0 was then assigned to a
scheduled fade:

- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_auxrowsignscale05to0_s250_250_fixed_seed5_1500_20260610`

This starts row-sign scale at `0.5`, then linearly fades to `0.0` over steps
250-500. The hypothesis is that the signed auxiliary channel is useful only as
an early collision-breaking scaffold.

The fade run reached 250/500 at `8.5092 -> 7.2816`, matching constant scale-0.5
before the schedule begins and then failing badly by 500. It was killed. This
rejects the simple "use signed aux rows early, then fade them away" schedule;
the transition itself appears disruptive or the model does not adapt cleanly to
removing that coordinate system.

Scale-0.25 later reached 1000 at `3.5047`, also behind the old no-op seed5
(`3.5007`). This closes the patched row-sign scale family as negative for now:
hard signs hurt, softened signs help early but do not beat the no-op K2
superpose branch by mid/late training.

With both GPUs freed from the row-sign branch, two structural probes were
launched:

- `bf300_sketchk2_slotreadout_slotmix_base_aux05_norowsigns_seed5_1500_20260610b`
- `bf300_readoutdelta_sketchk2_combinemix_base_aux05_norowsigns_seed5_1500_20260610`

The first separately reads out the base and auxiliary sketch slots, then learns a
slot mix. The second keeps the SOTA-compatible readoutdelta path but learns a
pre-readout combine mix over sketch slots. These test multi-row retrieval/mixing
without the now-negative row-sign mechanism.

Structural checkpoints:

- slot-readout + slot-mix: `8.5045 -> 7.2269`
- readoutdelta + sketch combine-mix: `8.5144 -> 7.2330 -> 5.3571`

The slot-readout result is the first promising non-row-sign result in this block:
it is much better than the old seed5 K2/no-op at 250/500
(`8.5179 -> 7.2347`) and even beats old seed6 at 500 (`7.2292`). This supports
the multi-row readout hypothesis more strongly than signed row superposition.
At 750/1000, slot-readout reached `5.3495 -> 3.5011`, slightly worse than old
no-op seed5 (`5.3479 -> 3.5007`), so its large 500-step lead mostly faded but
did not collapse. Combine-mix collapsed by 750 and was killed. A seed6
slot-readout replication was launched:

- `bf300_sketchk2_slotreadout_slotmix_base_aux05_norowsigns_seed6_1500_20260610`

That seed6 replication reached 250/500 at `8.5066 -> 7.2345` after the step-0
eval. This did not reproduce the seed5 500-step advantage (`7.2269`) and was
also behind the old no-op seed6 K2 branch (`7.2292`), so it was killed at 500.
The result leaves slot-readout as a plausible structural idea, but not yet a
seed-stable improvement.

The original seed5 slot-readout run continued to 1250 and reached `3.3455`,
again tracking the late-stage SOTA band but not clearly improving it. A followup
was launched to test whether the late fade is because the slot-readout branch
does not have enough per-layer output flexibility:

- `bf300_sketchk2_slotreadout_slotmix_layerreadouts_base_aux05_norowsigns_seed5_1500_20260610`

This keeps the sketch-slot separate readouts and learned slot mix, disables row
signs, and enables full `ENGRAM_LAYER_READOUTS=1` rather than
`ENGRAM_LAYER_READOUT_DELTA=1` because the current sketch slot-readout path
rejects the delta readout mode.

The plain seed5 slot-readout run finished at `3.2458`, with trajectory
`8.5045 -> 7.2269 -> 5.3495 -> 3.5011 -> 3.3455 -> 3.2458`. This is close to
the June SOTA band but does not beat the earlier `3.2440`, and remains behind
the archived all-time `3.2411`. The seed5 result therefore says the separate
slot readout/mix is a real early optimizer improvement but not, by itself, a
new endpoint.

The full per-layer-readout slot variant reached 250 at `8.4967`, the strongest
early checkpoint in this structural family so far. That run is continuing. With
GPU0 freed by the completed plain slot-readout run, a direct slot-attention
variant was launched:

- `bf300_sketchk2_slotattention_base_aux05_norowsigns_seed5_1500_20260610`

This tests the user-proposed "k hashes pull k rows, then attention reads out the
rows" mechanism directly: `ENGRAM_SKETCH_SLOT_READOUT=1`,
`ENGRAM_SKETCH_SLOT_ATTENTION=1`, no row signs, no learned slot mix, and no
layer readout delta.

Followup checkpoints:

- Full per-layer-readout slot mix: `8.4967 -> 7.2331 -> 5.3490 -> 3.5024`
- Slot-attention: `8.5064 -> 7.2318 -> 5.3551`

The per-layer variant has the strongest 250-step result but gives most of it
back by 500/750/1000, much like the plain slot-mix branch. It was killed at
1000 because `3.5024` was behind plain slot-mix (`3.5011`) and the old no-op
seed5 (`3.5007`). Slot-attention is cleanly behind learned slot-mix at 250, but
it partially catches up at 500: weaker than the plain slot-mix seed5 (`7.2269`),
better than the full per-layer slot-mix 500 (`7.2331`), and better than the old
no-op seed5 (`7.2347`). It is being carried to 750 before deciding.
At 750 it fell to `5.3551`, clearly behind the slot-mix family and old no-op
seed5, so it was killed. The direct attention-over-rows mechanism does not look
competitive in this form.

GPU1 was reassigned to combine the two partial ideas:

- `bf300_sketchk2_slotattention_layerreadouts_base_aux05_norowsigns_seed5_1500_20260610`

This tests dynamic slot attention plus full per-layer output readouts. It keeps
row signs off and does not use layer readout delta. It reached `8.5148` at 250,
worse than plain slot-attention and the learned slot-mix variants, so it was
killed immediately.

With GPU0 freed, a count-sketch-style variant was launched:

- `bf300_readoutdelta_sketchk2_dimsigns_balanced_base_aux05_seed5_1500_20260610`

This returns to the SOTA-compatible readoutdelta/headmix path, uses K2 sketch
base+aux summation, leaves row signs off, and adds balanced per-dimension sketch
signs plus balanced scalar sketch signs. This is closer to a true signed
count-sketch than the earlier row-sign tests, because the sign pattern is per
coordinate before the slot sum rather than a whole-row flip shared by all
coordinates.

A matching signed-dimension slot-readout run was also launched:

- `bf300_sketchk2_slotreadout_slotmix_dimsigns_balanced_base_aux05_seed5_1500_20260610`

This keeps the learned slot-mix idea that had the best 500-step behavior, but
adds balanced count-sketch dimension signs to see whether signed coordinate
superposition preserves the early slot-mix win better than unsigned slot-mix.
It reached 250 at `8.4993`, better than unsigned slot-mix (`8.5045`) and summed
signed count-sketch (`8.5012`), though still behind the full per-layer slot-mix
early spike (`8.4967`). This is the best count-sketch-style early result so far
and was carried to 500. It reached `7.2334` at 500, losing most of the early
edge and landing far behind unsigned slot-mix (`7.2269`), so it was killed.
Conclusion for this block: signed coordinate superposition improves some early
checkpoints, but like the row-sign family it has not preserved the improvement
through mid-training.

The summed readoutdelta count-sketch run reached 250 at `8.5012`. That is a
positive early signal versus the unsigned/plain slot-mix 250 (`8.5045`) and the
old no-op K2 seed5 (`8.5179`), though still behind the full per-layer slot-mix
250 (`8.4967`) that later faded. The important distinction is that this branch
keeps the SOTA-compatible readoutdelta path, so it is worth carrying to 500.
At 500 it reached `7.2385`, losing the early advantage and falling behind the
old no-op seed5 (`7.2347`), so the summed signed count-sketch branch was killed.

GPU0 was reassigned to a training-time hot-row regularization test:

- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_hitdrop10_min1024_seed5_1500_20260610`

This returns to the SOTA-compatible unsigned K2 superpose/readoutdelta/headmix
path and applies `ENGRAM_HIT_DROPOUT=0.10` only to rows whose live hit histogram
has reached at least 1024 hits. The intent is to test the earlier hot-row
dominance hypothesis directly: force the model not to rely perfectly on the
ultra-hot rows while preserving normal reads for cold rows.

A second hot-row dropout variant was launched on GPU1:

- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_hitdrop10_min256_seed5_1500_20260610`

This keeps dropout at 10% but lowers the threshold to 256 hits, testing whether
regularizing a broader hot-row band works better than only targeting ultra-hot
rows.

The `min1024` hot-dropout run reached 250 at `8.5069`. That is better than the
old unsigned/no-op K2 seed5 early checkpoint (`8.5179`), but weaker than the
best slot/count-sketch early variants. It is being carried to 500 because the
regularizer may primarily matter after the hit histogram has accumulated enough
hot rows.
At 500 it reached `7.2386`, so the hot-row dropout regularizer was hurting by
mid-training and the run was killed.

The `min256` variant reached 250 at `8.5062`, effectively the same early
behavior as `min1024`. It is still running to 500. A fresh matched no-dropout
control was launched because many comparisons here rely on the older no-op logs:

- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_norowsigns_control_seed5_1500_20260610`

This control uses the same current code path as the dropout runs, but with
`ENGRAM_HIT_DROPOUT=0`.
At 500, `min256` reached `7.2373`, also worse than the no-dropout K2 branch.
Both 10% hot-dropout thresholds therefore appear negative by mid-training.
The fresh BF300 no-dropout control reached 250 at `8.5179`, exactly matching
the older no-op seed5 250 checkpoint. This confirms the current-code control
path is reproducing the previous baseline and that the hot-dropout comparisons
are not confounded by a drift in launch/config. It then reached 500 at
`7.2347`, again matching the previous no-op seed5 trajectory. At 750 it reached
`5.3479`, and at 1000 it reached `3.5007`, continuing to reproduce the older
baseline. It finished at `3.2441` with peak allocated memory `69621 MiB`,
essentially reproducing the previous no-row-sign branch and sitting just behind
the earlier `3.2440` run.

After killing `min256`, GPU1 was assigned to a BF400 matched control:

- `bf400_sota_readoutdelta_superposek2_base_aux05_headmixinit05_norowsigns_control_seed5_1500_20260610`

This checks whether the current no-row-sign K2/readoutdelta/headmix path at
larger BF can approach the archived BF400 record without the row-sign,
count-sketch, slot-attention, or hot-dropout mechanisms.
The BF400 control initially spent longer in startup, then allocated about 95 GiB
and began training normally. It reached 250 at `8.5084`, better than the BF300
control at 250, but by 500 it was `7.2397`, worse than the BF300 control
(`7.2347`). This is another instance of the recurring pattern: larger/more
structured memory improves early training but often loses the advantage by the
mid-training checkpoint. At 750 it reached `5.3498`, still behind the BF300
control (`5.3479`). By 1000 it recovered slightly to `3.5004`, just ahead of
BF300's `3.5007`, so the final checkpoint is still worth collecting for the
parameter-count curve.

With GPU0 freed by the completed BF300 control, a BF400 learned slot-mix run was
launched:

- `bf400_sketchk2_slotreadout_slotmix_base_aux05_norowsigns_seed5_1500_20260610`

This tests whether the best BF300 structural readout idea, separate base/aux
slot readouts with learned slot mix, benefits from the larger BF400 table. The
BF300 version finished at `3.2458`, so the useful question is whether scaling
the table shifts it back into the `3.241-3.244` band.

At 250 steps, BF400 slot-mix reached `8.5064`. That is not an early improvement
over BF300 slot-mix (`8.5045`), but it is close enough that the run should be
carried to 500 before making a kill/continue decision.
At 500 it reached `7.2318`. This is worse than BF300 slot-mix at 500
(`7.2269`), but slightly better than the archived BF400 SOTA reference at 500
(`7.2329`) and better than the BF400 no-row-sign control (`7.2397`). It should
therefore continue at least to 750/1000; the result is not a clean scaling win
for slot-mix, but it is strong enough to remain a live branch.
At 750 it reached `5.3505`, losing the 500-step advantage: it is worse than
BF300 slot-mix (`5.3495`), worse than the archived SOTA curve (`5.3425`), and
effectively tied/slightly worse than BF400 no-row-sign control (`5.3498`).
That matches the recurring pattern where extra memory structure looks useful
early and then fades. The run was stopped after 750.

A follow-up run was launched to test that interpretation directly:

- `bf400_sketchk2_slotreadout_slotmix_aux05to01_s500_500_norowsigns_seed5_1500_20260610`

It keeps the same BF400 slot-mix setup through step 500, then decays the sketch
auxiliary scale from `0.5` to `0.1` over steps 500-1000. The hypothesis is that
the auxiliary sketch component helps early routing/coverage but becomes a
liability once the base memory and dense model have learned enough.
At 250 it reached `8.5064`, exactly matching the unscheduled BF400 slot-mix
run before the decay begins. That is a useful launch sanity check: any later
divergence should mostly reflect the auxiliary-scale schedule rather than
early-run noise or a mismatched config.
At 500 it reached `7.2318`, again exactly matching the unscheduled BF400
slot-mix run at the schedule boundary. This is the desired controlled setup:
the run is indistinguishable before the intervention, so the 750/1000
checkpoints should directly test whether reducing the auxiliary sketch after
500 helps the branch avoid the observed fade.
At 750 it reached `5.3495`, improving over unscheduled BF400 slot-mix
(`5.3505`) by about `0.0010` and matching BF300 slot-mix at 750. This is a
small positive result for the aux-decay hypothesis, but it is still far behind
the archived SOTA trajectory (`5.3425`). The run should continue to 1000,
because the useful question is whether the schedule prevents the later
slot-mix fade, not whether it fully closes the gap by 750.
At 1000 it reached `3.5029`, worse than the BF400 no-row control (`3.5004`)
and worse than the SOTA-family replay at the same checkpoint (`3.4999`). The
small 750-step improvement did not survive; aux-decay helped the slot-mix fade
slightly at 750 but did not fix the late-curve behavior. The run was stopped at
1000.

With GPU1 freed after the SOTA replay, a stronger paired schedule was launched:

- `bf400_sketchk2_slotreadout_slotmix_aux05to0_s500_500_norowsigns_seed5_1500_20260610`

This is identical through step 500, then decays the sketch auxiliary scale from
`0.5` to `0.0` over steps 500-1000. Comparing final `0.1` versus `0.0` should
tell whether the residual auxiliary sketch is still useful after the handoff
window or whether fully removing it is better.
At 250 it reached `8.5064`, matching the other BF400 slot-mix schedule/control
runs before the decay begins.
At 500 it reached `7.2318`, also matching the other slot-mix schedule/control
runs exactly at the schedule boundary. The controlled comparison remains clean
through the intervention point.
At 750 it reached `5.3470`. This is substantially better than aux-to-`0.1`
(`5.3495`) and unscheduled BF400 slot-mix (`5.3505`), though still behind the
archived SOTA curve (`5.3425`). This is the first slot-mix schedule that clearly
improves the post-500 fade, so the run should continue to 1000/final.

Because slot-mix aux-decay did not fix the 1000-step fade, GPU0 was assigned to
the same late-auxiliary-pressure test on the stronger BF400 K2/headmix branch:

- `bf400_sota_readoutdelta_superposek2_base_aux05to025_s500_500_headmixinit05_norowsigns_seed5_1500_20260610`

This matches the BF400 no-row-sign K2/headmix control (`3.2447`) except that
the superpose auxiliary scale decays from `0.5` to `0.25` over steps 500-1000.
The question is whether the current best branch benefits from reducing the
auxiliary superposed component only after it has contributed early training
signal.
At 250 it reached `8.5084`, exactly matching the BF400 K2/headmix control at
the same checkpoint before the schedule starts. This confirms the launch is a
clean controlled comparison up to the intervention point.
At 500 it reached `7.2397`, again exactly matching the BF400 K2/headmix control
at the schedule boundary. Any divergence at 750/1000 should therefore reflect
the superpose auxiliary decay rather than launch/config drift.
At 750 it reached `5.3481`, better than the matched BF400 K2/headmix control
(`5.3498`) by about `0.0017`. This is not enough to approach the archived SOTA
curve, but it is a clean positive result on the stronger K2 branch, so the run
should continue to 1000/final.

## Structural sweep plot refresh, 2026-06-10

A broader structural plotter was added:

- `experiments/modded_nanogpt_work/reports/make_structural_sweep_20260610.py`

It parses the current synced remote logs, including slot-readout, slot-attn,
count-sketch, hot-dropout, and no-row-sign control families, and writes:

- `summary_structural_20260610.csv`
- `structural_variants_vs_params_20260610.png`
- `structural_variants_curves_20260610.png`

The first refresh parsed 11 structural runs, 2 complete. A later refresh after
the BF300/BF400 control checkpoints parsed 12 runs, still 2 complete. Best
complete in that set remains the plain slot-mix run at `3.2458`; active
BF300/BF400 controls are partial and should be folded into the plot again after
completion.
After BF300 control completion, the structural refresh parsed 12 runs, 3
complete. Best complete in that structural subset is now the BF300 no-row-sign
control at `3.2441`; the BF400 no-row-sign control and BF400 slot-mix remain
active.

The BF400 no-row-sign control completed at `3.2447` with peak allocated memory
`89799 MiB`:

- `8.5084 -> 7.2397 -> 5.3498 -> 3.5004 -> 3.3453 -> 3.2447`

This did not improve on BF300 no-row-sign (`3.2441`), the June best (`3.2440`),
or the archived SOTA (`3.2411`). The result reinforces the main scaling
observation from the BF sweep: larger BF can look slightly better early, but
plainly adding rows does not reliably buy final validation loss.

To separate current-code drift from architectural effects, a fresh current-code
replay of the archived best recipe was then launched on GPU1:

- `bf400_ngramrows_trigramheavy_0p5_1p5_readhit025_current_seed5_1500_20260610`

That replay uses the archived best family: BF400, trigram-heavy ngram row
allocation (`0.5,1.5`), layer readouts, and read-hit scaling exponent `0.25`.
The archived reference run from 2026-05-27 reached `3.2411`; this replay should
tell us whether the all-time result still reproduces in the current tree.
The launch config was checked after startup and has the expected Engram flags:
`engram_normalize_readout=1`, `engram_normalize_memory_heads=1`,
`engram_layer_readouts=1`, `engram_max_ngram=3`,
`engram_ngram_row_factors=0.5,1.5`, and
`engram_read_hit_scale_exponent=0.25`.
Its 250-step checkpoint reached `8.5041`, close to but behind the archived
reference's 250-step `8.4993`.
At 500 it reached `7.2390`, now clearly behind the archived reference's
`7.2329` at the same checkpoint and also behind the active BF400 slot-mix run
(`7.2318`). This replay should still continue to final because the shape can
change later, but the current tree is not trivially reproducing the archived
SOTA trajectory through 500 steps.
At 750 it reached `5.3499`, still behind the archived reference (`5.3425`) and
essentially in the same band as BF400 no-row-sign control (`5.3498`) and the
stopped BF400 slot-mix run (`5.3505`). The replay is therefore showing a
persistent current-code/trajectory gap through mid-training, not just a noisy
early checkpoint.
At 1000 it reached `3.4999`. That remains behind the archived reference
(`3.4964`), but it has partially recovered and is now slightly ahead of the
BF400 no-row-sign control (`3.5004`). The final checkpoint remains worth
collecting before deciding whether this is true code drift or just a worse
trajectory that converges late.
At 1250 it reached `3.3460`, still behind the archived reference (`3.3406`) by
about `0.0054`. The run remains useful as a current-code reproducibility anchor,
but it is unlikely to beat the archived SOTA unless the final checkpoint closes
substantially more than the prior intervals.
It finished at `3.2460`, consistent with the June current-code BF400
ngramrows-family seeds (`3.2461-3.2471`) and worse than the BF400 no-row-sign
control (`3.2447`). The current tree therefore appears not to reproduce the
archived May 27 `3.2411` trajectory despite matching the visible config flags.

## Late auxiliary decay follow-up, 2026-06-10

The two late auxiliary decay runs were stopped at 1000 because both failed the
controlled comparison against their matched BF400 baselines.

Slot-mix aux-to-zero:

- `bf400_sketchk2_slotreadout_slotmix_aux05to0_s500_500_norowsigns_seed5_1500_20260610`
- trajectory: `8.5064 -> 7.2318 -> 5.3470 -> 3.5062`

The 750-step result looked useful: fully decaying the slot auxiliary improved
over aux-to-`0.1` (`5.3495`) and unscheduled BF400 slot-mix (`5.3505`). But by
1000 it was worse than aux-to-`0.1` (`3.5029`) and worse than the BF400
no-row-sign K2/headmix control (`3.5004`). The lesson is that removing the
auxiliary path can reduce the mid-training slot-mix fade, but it destabilizes
or under-supports the later transition.

K2/headmix aux-to-`0.25`:

- `bf400_sota_readoutdelta_superposek2_base_aux05to025_s500_500_headmixinit05_norowsigns_seed5_1500_20260610`
- trajectory: `8.5084 -> 7.2397 -> 5.3481 -> 3.5035`

This was also a clean controlled comparison: it exactly matched the BF400
K2/headmix control at 250 and 500 before the schedule changed. The small
750-step gain over control (`5.3481` vs `5.3498`) did not survive to 1000
(`3.5035` vs `3.5004`). Late auxiliary decay therefore does not currently look
like a SOTA path; the auxiliary pressure seems to be part of the useful trained
system rather than a scaffold that can be safely removed after 500 steps.

The structural sweep artifacts were refreshed after these stops. The refreshed
summary parsed 19 runs, with 7 complete. Best complete remains the BF300
no-row-sign K2/headmix control:

- `bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_norowsigns_control_seed5_1500_20260610`: `3.2441`

## Latent adaptive addressing probe, 2026-06-10

After the auxiliary-decay tests failed, two BF300 latent-addressing probes were
launched to test the stable/adaptive hashing idea more directly:

- `bf300_latentfsq_mixngram_nopart_fix_seed5_1500_20260610`
- `bf300_latentbsq24_mixngram_nopart_fix_seed5_1500_20260610`

Both used `ENGRAM_LATENT=1`, `ENGRAM_LATENT_MIX_NGRAM=1`, no layer partitions,
and the same BF300-sized table shape (`30182400 x 384`). The launcher needed a
small fix first: `ENGRAM_LAYER_PARTITIONS` and
`ENGRAM_LAYER_PARTITION_GROUPS` were hard-exported in
`run_bf_superpose_constant_20260609.sh`, so the script now preserves explicit
environment overrides while defaulting to `1`.

Results through the stop point:

- FSQ latent mix: `18.9401 -> 8.5704 -> 7.3311`
- BSQ24 latent mix: `18.9403 -> 8.5687 -> 7.3187`

Both were stopped at 500 because they were far behind the fixed-hash BF300/BF400
families, which sit around `8.50` at 250 and `7.23-7.24` at 500.

The mechanistic result is still useful. Latent mixing strongly flattened the
row-hit distribution: at 250, FSQ had `hit_max=3328` and BSQ24 had
`hit_max=4844`, versus hundreds of thousands for fixed ngram rows at comparable
checkpoints. At 500, the hot-row maxima were still only `12402` and `29943`.
So adaptive addressing does attack hot-row dominance, but in this form it loses
too much of the useful fixed ngram identity. The next version should preserve a
strong fixed ngram path and add latent addressing as an auxiliary/residual
readout, rather than replacing the fixed-address readout.

## Latent auxiliary readout and no-partition controls, 2026-06-10

A guarded implementation was added for a latent auxiliary readout:

- `ENGRAM_LATENT_AUX_READOUT=1`
- `ENGRAM_LATENT_AUX_SCALE=<scale>`

In this mode the model keeps the normal fixed ngram/superpose readout, computes
a latent readout from the same table, and adds `scale * latent_memory` before
the existing memory normalization/readout path. This was added because full
latent addressing flattened row hits but badly hurt loss, suggesting latent
addresses might be useful only as a weak auxiliary path.

Auxiliary results so far:

- FSQ aux scale `0.5`: `18.9397 -> 8.5149 -> 7.2471`, stopped at 500.
- BSQ24 aux scale `0.5`: `18.9403 -> 8.5311`, stopped at 250.
- FSQ aux scale `0.1`: `18.9396 -> 8.5065 -> 7.2456`, stopped at 500.
- FSQ aux scale `0.03`: `18.9396 -> 8.5090 -> 7.2515`, stopped at 500.

The matched no-partition fixed K2/headmix control is:

- `bf300_k2_headmix_nopart_control_seed5_1500_20260610`
- trajectory: `18.9398 -> 8.5179 -> 7.2347 -> 5.3479 -> 3.5007 -> 3.3442 -> 3.2448`

Two conclusions are clear from the matched partials:

1. Small FSQ latent aux can help very early against the no-partition control
   (`8.5065` or `8.5090` vs `8.5179` at 250).
2. The help does not survive to 500 in the tested constant-scale forms
   (`7.2456`/`7.2515` vs `7.2347` control).

This supports the earlier scaffold interpretation: latent/adaptive addressing
can inject useful early signal, but as a persistent readout it interferes with
the stronger fixed ngram memory. If this direction is revisited, it should be
with an explicit schedule that decays latent auxiliary readout after the early
phase, not with a constant auxiliary scale.

The no-partition control finished as a near tie with the BF400 no-row-sign
K2/headmix control (`3.2448` vs `3.2447`) but did not beat the current structural
sweep best (`3.2441`). Removing layer partitions therefore did not explain the
BF300/BF400 gap and did not move the current-code SOTA. A BF400 no-partition
control was also launched:

- `bf400_k2_headmix_nopart_control_seed5_1500_20260610`

Its full trajectory matched the older BF400 K2/headmix control:
`18.9401 -> 8.5084 -> 7.2397 -> 5.3498 -> 3.5004 -> 3.3453 -> 3.2447`.
This is a clean negative control: no-partition at BF400 does not improve on the
partitioned BF400 line, and it remains behind the best complete current-code
BF300 structural sweep run (`3.2441`).

One follow-up run was launched from the latent-aux result instead of doing a
new broad hyperparameter sweep:

- `bf300_k2_headmix_latentaux_fsq_mixngram_s01to0_250_250_nopart_seed5_1500_20260610`

This keeps the fixed no-partition K2/headmix path, adds FSQ latent-mixed
readout at scale `0.1` for the first 250 steps, then linearly decays the latent
auxiliary readout to `0` by step 500. The reason is narrow: constant scale
`0.1` helped at 250 (`8.5065` vs `8.5179`) but hurt at 500 (`7.2456` vs
`7.2347`), so the only plausible remaining version is an early scaffold that
gets out of the way.

The first launch of this scheduled latent-aux idea accidentally omitted
`ENGRAM_HEAD_MIX=1`; it reached 250 at `8.5191` and was killed as a wrong-config
run. The corrected comparison is:

- `bf300_k2_headmix_latentaux_fsq_mixngram_s01to0_250_250_nopart_headmixfix_seed5_1500_20260610`

The corrected run matched the good constant-aux early result at 250 (`8.5065`)
but failed the decisive 500 checkpoint: `7.2473`, worse than both the
no-partition control (`7.2347`) and the constant scale `0.1` latent-aux run
(`7.2456`). It was stopped at 500. This rules out the simple early-scaffold
version; the latent/adaptive path remains useful mechanistically for flattening
row hits, but not as an additive readout in this setup.

## Sparse optimizer probes, 2026-06-10

After the structural/addressing probes, both GPUs were used for optimizer probes
on the current BF300 K2/readoutdelta/headmix stack:

- `bf300_sota_k2_headmix_rowadagrad_seed5_1500_20260610`
- `bf300_sota_k2_headmix_ifal025to4_seed5_1500_20260610`

Both keep the same BF300 table, K2 superpose readout, `ENGRAM_HEAD_MIX=1` with
`0.5,-0.5` init, read-hit scaling, normalization, and readoutdelta. The only
intended differences are the sparse memory optimizer:

- row-wise Adagrad replaces scalar Adam with one cumulative squared-gradient RMS
  accumulator per row.
- IFAL keeps scalar Adam but applies an inverse frequency-adaptive LR scale,
  clamped to `[0.25, 4.0]`, to downweight very hot rows and raise relatively
  cold rows.

These are aimed at the remaining hot-row/update-imbalance hypothesis rather than
changing the addressing path.

Early optimizer results:

- Row-wise Adagrad was clearly worse at 250: `8.5236`. Its update RMS was much
  lower than scalar Adam (`1.514e-02`), and it was stopped.
- IFAL was close at 250 (`8.5023`) and improved by 500: `7.2304`, better than
  the matched no-partition K2/headmix control (`7.2347`). The advantage did not
  clearly survive: 750 tied control (`5.3479`) and 1000 was worse (`3.5032`).
  Final was `3.2458`, worse than the current-tree best complete run (`3.2441`)
  and the BF300 no-partition control (`3.2448`). This supports the interpretation
  that inverse frequency scaling is an early-learning scaffold here, not a final
  SOTA improvement.
- FAL was launched as the complementary test: if IFAL says "cool hot rows down",
  FAL says "push frequent rows harder". It was worse at 250 (`8.5164`) and was
  stopped. Its mean LR scale was only `0.289`, so in this implementation it
  mostly under-updated rows rather than aggressively training hot rows.
- A BF400 IFAL scaling run was launched:
  `bf400_sota_k2_headmix_ifal025to4_seed5_1500_20260610`. Its first checkpoint
  was slightly ahead of the BF400 control (`8.5074` vs `8.5084`), and its
  500-step checkpoint was clearly better (`7.2319` vs `7.2397`). The advantage
  faded by 750 (`5.3536` vs BF400 control `5.3498`), was essentially tied at
  1000 (`3.5003` vs BF400 control `3.5004`), then edged ahead at 1250
  (`3.3446` vs BF400 control `3.3453`). Final was `3.2446`, a tiny improvement
  over the BF400 control (`3.2447`) but still worse than the current-tree best
  (`3.2441`). The broader pattern remains that cumulative hit-count LR scaling
  changes early dynamics strongly but has not yet produced a clean SOTA move.

Follow-up launched:

- `bf300_sota_k2_headmix_batchfreqnorm_seed5_1500_20260610`

This keeps scalar Adam and the same BF300 K2/headmix stack, but enables
`ENGRAM_SPARSE_BATCH_FREQ_NORM=1`. Unlike FAL/IFAL, this divides each coalesced
row gradient by the number of repeated row hits inside the current sparse
gradient batch. It directly tests whether hot-row dominance is caused by
within-step repeated writes rather than cumulative row frequency across the
whole run.

Batch-frequency normalization reaches `8.5100` at step 250 and `7.2377` at
step 500, worse than the matched BF300 controls and the full IFAL run. It also
lowers update RMS (`3.819e-02` at 250 and `3.557e-02` at 500), so the read is
that this dampens memory learning rather than improving row allocation. The run
was stopped after 500 and GPU1 was reassigned.

Code follow-up added: sparse hit-frequency LR modes now support a schedulable
blend:

- `ENGRAM_SPARSE_HIT_LR_BLEND`
- `ENGRAM_SPARSE_HIT_LR_BLEND_FINAL`
- `ENGRAM_SPARSE_HIT_LR_BLEND_SCHEDULE_START`
- `ENGRAM_SPARSE_HIT_LR_BLEND_SCHEDULE_STEPS`

The optimizer applies `1 + blend * (lr_scale - 1)`, so blend `1` is the current
FAL/IFAL behavior and blend `0` is ordinary scalar Adam. This directly tests the
emerging pattern that IFAL helps early checkpoints but loses the advantage late.
A BF400 scheduled-IFAL run is queued behind GPU0:
`bf400_sota_k2_headmix_ifal_sched1to0_500_250_seed5_1500_20260610`, using IFAL
through step 500 and decaying its blend to zero by step 750.

A matching BF300 scheduled-IFAL run was launched on GPU1 after stopping the
batch-frequency probe:
`bf300_sota_k2_headmix_ifal_sched1to0_500_250_seed5_1500_20260610`.
Its 250-step checkpoint is `8.5023` and its 500-step checkpoint is `7.2304`,
matching full IFAL as expected because the blend is still `1.0` through step
500. At 750, after the blend has decayed to `0.0`, it reaches `5.3555`, worse
than full BF300 IFAL at the same checkpoint (`5.3479`). So the first scheduled
read does not support "IFAL only as early scaffold"; the handoff to ordinary
scalar Adam appears to cost more than the early IFAL benefit saves, at least
with a 500-to-750 decay.

The matching BF400 scheduled-IFAL run has also started:
`bf400_sota_k2_headmix_ifal_sched1to0_500_250_seed5_1500_20260610`.
Its 250-step checkpoint is `8.5074`, also matching full BF400 IFAL before the
scheduled decay phase. Its 500-step checkpoint is `7.2319`, again matching full
BF400 IFAL before the schedule changes behavior. The important BF400 readout is
750/1000/final.

Scheduled-IFAL decision: negative. The BF300 scheduled run reached `3.5053` at
step 1000, worse than full BF300 IFAL (`3.5032`) and the matched controls, so it
was stopped. The BF400 scheduled run reached `5.3568` at step 750, worse than
full BF400 IFAL (`5.3536`) and BF400 control (`5.3498`), so it was also stopped.
This closes the current IFAL-schedule branch: cumulative inverse frequency LR
can improve the early curve, but turning it off after step 500 does not preserve
that gain.

Next structural probe launched after freeing both GPUs:

- `bf300_sota_k2_headmix_norowsigns_ngramread115_085_seed5_1500_20260610`
- `bf400_sota_k2_headmix_norowsigns_ngramread115_085_seed5_1500_20260610`

These keep the no-row-sign K2/readoutdelta/headmix recipe and add fixed
RMS-normalized n-gram read scales `1.15,0.85`. The motivation is that completed
K2/headmix runs consistently learn a roughly `0.61/0.39` head mix, so this tests
whether a mild explicit read prior can encode that preference without adding a
new auxiliary read path or optimizer change.

BF300 n-gram-read-scale reached step 250 at `8.5176`, with normalized logged
scales `n2=1.137`, `n3=0.841`. This is essentially neutral versus the matched
no-row-sign/K2 control start (`8.5179`) and weaker than the strongest early
structural/optimizer perturbations, but it is not a clear failure. Because this
probe is intended to test whether a fixed read prior helps the later basin, the
first real decision point is 500.

N-gram-read-scale decision: negative. BF300 reached `7.2382` at step 500, behind
the no-row-sign/K2 control band and behind the useful slot/IFAL early curves.
BF400 reached `8.5122` at step 250, also behind its matched BF400 control
(`8.5084`) and BF400 IFAL (`8.5074`). Both runs were stopped. A fixed mild
bigram-heavy read prior does not reproduce the learned head-mix benefit; the
model appears to need to learn the mixture dynamically rather than have it
baked into read amplitudes.

Next structural probe launched:

- `bf300_sota_k2_headmix_norowsigns_avalanchehash_seed5_1500_20260610`
- `bf400_sota_k2_headmix_norowsigns_avalanchehash_seed5_1500_20260610`

These keep the current no-row-sign K2/readoutdelta/headmix stack but enable
`ENGRAM_AVALANCHE_HASH=1`, directly testing whether the prime/modulo hash path
is leaving exploitable structure or collision bias in the row assignment.

Avalanche-hash 250-step readout:

- BF300: `8.5116`, better than the matched no-row-sign/K2 start (`8.5179`) and
  worth carrying to 500.
- BF400: `8.5088`, essentially neutral/slightly worse than the BF400 no-row-sign
  control (`8.5084`) and full BF400 IFAL (`8.5074`), but close enough to carry
  to 500 while BF300 remains promising.

Mechanistically this is a clean hash-mixing test: the avalanche function mixes
the n-gram hash integer before modulo/offset; the table shape, layer partitions,
K2 superposition, readoutdelta, and head-mix settings are otherwise unchanged.

Avalanche-hash 500-step readout strengthens the branch:

- BF300: `7.2345`, preserving the 250-step advantage and slightly beating the
  matched no-row-sign/K2 control band.
- BF400: `7.2281`, now clearly ahead of BF400 no-row-sign control (`7.2397`)
  and BF400 IFAL (`7.2319`).

This is the first post-optimizer structural probe in this block that improves
the BF400 500-step curve without adding an auxiliary readout path. Both
avalanche runs are being carried to at least 1000, with 750 as the next check
for whether the hash-mixing gain survives the familiar mid-training fade.

Avalanche-hash decision: early gain, late fade. BF300 reached `5.3508` at step
750, no longer ahead of the best BF300 no-row-sign/control band. BF400 reached
`5.3569` at step 750, clearly worse than BF400 no-row-sign control (`5.3498`)
and BF400 slot-mix (`5.3505`) despite its strong 500-step result. Both runs were
stopped at 750. Stronger hash diffusion improves early optimization, especially
at BF400, but the effect behaves like the other scaffold-style wins and does
not survive into the mid-training basin.

Next hash-layout probe launched:

- `bf300_sota_k2_headmix_norowsigns_hashseed1_seed5_1500_20260610`
- `bf400_sota_k2_headmix_norowsigns_hashseed1_seed5_1500_20260610`

These return to the current no-row-sign K2/readoutdelta/headmix stack and change
only `ENGRAM_HASH_SEED=1`. The avalanche result says hash geometry is not
irrelevant: stronger mixing gave real early gains, then faded. This seed probe
asks the narrower question of whether the ordinary prime/modulo layout has
enough variance across valid hash seeds to move the curve without changing the
hash family itself. If it moves early but not late, that reinforces the
"collision/layout scaffold" interpretation. If it moves the final basin, hash
layout should become a first-class sweep axis rather than a fixed constant.

Hash-seed-1 250-step readout:

- BF300: `8.5139`, better than the matched BF300 no-row-sign/K2 control
  (`8.5179`) but weaker than avalanche hash (`8.5116`).
- BF400: `8.5063`, better than the matched BF400 no-row-sign/K2 control
  (`8.5084`), full IFAL (`8.5074`), and avalanche hash (`8.5088`) at the same
  checkpoint.

This is a clean positive early signal, especially at BF400. Both hash-seed-1
runs are being carried to 500 to test whether this is another short-lived hash
layout scaffold or a curve shift that survives past the warmup basin.

Hash-seed-1 500-step readout:

- BF300: `7.2316`, better than the matched BF300 control band and better than
  BF300 avalanche hash at 500 (`7.2345`), though still close to full IFAL
  (`7.2304`).
- BF400: `7.2353`, still better than BF400 no-row-sign control (`7.2397`) but
  behind BF400 avalanche hash (`7.2281`) and full IFAL (`7.2319`) at the same
  checkpoint.

The seed change is not just a 250-step blip, but the width interaction is mixed:
BF300 improved cleanly while BF400 gave back much of its 250-step advantage by
500. Both runs are being carried to 750, which is the decisive checkpoint for
the recurrent "early scaffold, late fade" pattern.

Hash-seed-1 750-step decision:

- BF300: `5.3520`, faded behind the strongest BF300/control band and was
  stopped.
- BF400: `5.3463`, better than BF400 no-row-sign control (`5.3498`), BF400
  slot-mix (`5.3505`), and BF400 avalanche hash (`5.3569`) at the same
  checkpoint. This run is continuing to final.

This is the first hash/layout perturbation in this block whose BF400 advantage
survived to 750. The freed GPU was assigned to a same-stack BF400 replicate with
`ENGRAM_HASH_SEED=2`:

- `bf400_sota_k2_headmix_norowsigns_hashseed2_seed5_1500_20260610`

The point is not to tune a magic seed by hand; it is to estimate whether the
ordinary hash layout is a meaningful stochastic architectural variable. If seed2
also improves BF400, hash-seed/layout sweeps become a principled axis. If seed2
fails, seed1 is likely a favorable collision/layout draw that still tells us the
row assignment can move loss at fixed parameter count.

BF400 hash-seed-1 reached step 1000 at `3.4995`. That preserves the 750-step
advantage and is slightly ahead of the BF400 current-tree control band logged in
this report (`~3.4999-3.5007`). It is continuing to final. The important final
comparison is against:

- BF400 no-row-sign/K2/headmix control: `3.2447` final.
- BF300 no-row-sign/K2/headmix current-tree best: `3.2441` final.
- Archived old BF400 SOTA replay: `3.2411` final, not reproduced by the current
  tree.

BF400 hash-seed-2 was stopped at step 250: `8.5138`, worse than BF400 control
(`8.5084`) and hash-seed-1 (`8.5063`). That already shows the hash-layout axis
has high variance; seed1 is not a generic improvement from changing any hash
seed. A third BF400 layout sample was launched:

- `bf400_sota_k2_headmix_norowsigns_hashseed3_seed5_1500_20260610`

This gives a minimal three-seed read on whether seed1 is a one-off favorable
collision layout or part of a broad distribution where some ordinary hash
layouts materially outperform the default.

BF400 hash-seed-1 reached step 1250 at `3.3446` and is still healthy. At this
point the run has kept its advantage through the mid-training basin; the final
1500-step readout is the decisive test for whether the seed1 layout can beat
the BF400 current-tree final (`3.2447`) and potentially the BF300 current-tree
best (`3.2441`).

BF400 hash-seed-3 reached step 250 at `8.5059`, slightly better than seed1
(`8.5063`) and clearly better than the BF400 control (`8.5084`). This makes the
seed2 failure more informative rather than discouraging: ordinary hash layout
draws appear to have a meaningful spread, and at least two sampled layouts are
better than the default at the first checkpoint. Seed3 is being carried to 500.

Final hash-layout readout:

- BF400 hash-seed-1 final: `3.2442`.
- BF400 current-tree no-row-sign/K2/headmix control final: `3.2447`.
- BF300 current-tree no-row-sign/K2/headmix best final: `3.2441`.
- Archived old BF400 SOTA final: `3.2411`, still not reproduced by the current
  tree.

Hash-seed-1 is a real BF400 improvement and essentially ties the current-tree
best, missing BF300 by only `0.0001`. It does not recover the archived old SOTA,
but it is the cleanest evidence so far that the row-assignment/layout variable
can survive to final loss rather than only improving early training.

BF400 hash-seed-3 was stopped at 500: `7.2424`, worse than BF400 control
(`7.2397`) and much worse than hash-seed-1 (`7.2353`) at the same checkpoint.
Together with seed2's weak 250, this says the hash-layout distribution is broad:
some layouts are materially favorable, but early 250-step quality alone is not
sufficient to predict survival to 500/final.

Follow-up structural test: disable layer-specific hash multipliers while keeping
the current layer partition grouping.

- `bf300_sota_k2_headmix_norowsigns_layerhash0_seed5_1500_20260610`
- `bf400_sota_k2_headmix_norowsigns_layerhash0_seed5_1500_20260610`

In the current no-row-sign path, `ENGRAM_HASH_SEED` mostly changes the
per-layer n-gram multipliers because `ENGRAM_LAYER_HASHES=1`; the K2 superpose
aux salts are fixed and row signs are off. One correction after reading the code:
the launcher sets `ENGRAM_LAYER_PARTITION_GROUPS=1`, so layer 2 and layer 8 are
not in separate row ranges by default. They share the same partition group and
are decorrelated mostly by their layer-specific hash multipliers. Setting
`ENGRAM_LAYER_HASHES=0` therefore makes the two Engram layers hit the same rows
for the same n-gram, not merely the same relative row in separate partitions.
This directly tests whether per-layer remapping is necessary when layers share a
table region.

Layerhash0 250-step readout:

- BF300: `8.5179`, neutral to the matched BF300 control at displayed precision.
- BF400: `8.5084`, neutral to the matched BF400 control at displayed precision.

Shared relative layer addressing does not break the early curve. Both runs are
being carried to 500, where the previous hash-layout runs started to separate.

Layerhash0 500-step readout:

- BF300: `7.2347`, close to the control/avalanche band but behind the stronger
  BF300 hashseed1/IFAL early curves.
- BF400: `7.2397`, neutral to the matched BF400 control at displayed precision.

This makes layerhash0 look mostly neutral rather than a clear improvement. Both
runs are being carried to 750 once, because the question is whether shared
same-row addressing between Engram layers changes the later basin, not only the
early curve.

Layerhash0 750-step readout:

- BF300: `5.3479`, competitive with the stronger BF300 750 band, so it is being
  carried to 1000.
- BF400: `5.3498`, neutral to the BF400 control and stopped.

BF300 layerhash0 reached step 1000 at `3.5007` and was stopped. The early/mid
benefit did not survive; making both layers hit the same rows for the same
n-gram is not a SOTA path in this setup.

The corrected partition-group reading suggests the next structural test:
`ENGRAM_LAYER_PARTITION_GROUPS=2`, which gives layer 2 and layer 8 genuinely
separate table regions. This tests explicit cross-layer row isolation rather
than only independent hash multipliers inside a shared row pool.

Partition-group-2 runs launched:

- `bf300_sota_k2_headmix_norowsigns_partgrp2_seed5_1500_20260610`
- `bf400_sota_k2_headmix_norowsigns_partgrp2_seed5_1500_20260610`

Partition-group-2 early readout:

- BF300: step 250 `8.5094`, a large improvement over the BF300 current-tree
  control (`8.5179`). This is the strongest BF300 250-step structural signal in
  this hash/layout block.
- BF400: step 250 `8.5096`, initially worse than BF400 control (`8.5084`), but
  step 500 recovered to `7.2354`, better than BF400 control (`7.2397`) and
  essentially tied with BF400 hashseed1 at 500 (`7.2353`).

This is now a serious structural branch. Explicitly isolating layer 2 and layer
8 into separate row regions appears to help BF300 immediately and helps BF400 by
500 despite the weak first checkpoint. Both runs are being carried to at least
750.

BF300 partition-group-2 faded by 500: `7.2396`, worse than the matched BF300
control band, so it was stopped. The 250-step gain was another early scaffold.
BF400 partition-group-2 remains alive because its 500-step value (`7.2354`) is
better than BF400 control and essentially tied with BF400 hashseed1 at 500.

Combination probe launched on the freed GPU:

- `bf400_sota_k2_headmix_norowsigns_partgrp2_hashseed1_seed5_1500_20260610`

This stacks the two BF400-favorable layout changes seen so far: explicit
two-group layer row isolation and the favorable `ENGRAM_HASH_SEED=1` per-layer
multiplier draw. If it improves over both components, layout quality is
composable. If it does not, the hashseed1 gain likely depends on sharing the
same row pool across layers.

Plain BF400 partition-group-2 reached `5.3561` at 750 and was stopped. It looked
good at 500 (`7.2354`) but became worse than the BF400 control (`5.3498`) and
BF400 hashseed1 (`5.3463`) by 750. Explicit layer row isolation alone is
therefore another early scaffold. The partgrp2+hashseed1 combo remains alive as
the composability test.

Next layout-combination probe launched:

- `bf400_sota_k2_headmix_norowsigns_avalanchehash_hashseed1_seed5_1500_20260610`

This combines the two hash-mixing signals in a different way. Avalanche hashing
alone improved early optimization but faded by 750, while hashseed1 survived to
final. The question is whether stronger integer diffusion improves the favorable
hashseed1 multiplier draw or destroys the collision/layout structure that made
it good.

Partgrp2+hashseed1 reached step 250 at `8.5056`, the strongest BF400 250-step
point in this block so far. It beats hashseed1 alone (`8.5063`) and partgrp2
alone (`8.5096`). This is a good early composability signal, but the critical
checks remain 500 and 750 because both partgrp2 and avalanche-like changes have
shown scaffold fade.

Partgrp2+hashseed1 failed at 500: `7.2727`, far worse than either component and
the BF400 control. The early 250 win was not composable. This suggests the
favorable seed1 layout relies on the shared row pool; isolating layers into
separate row regions changes the collision structure enough to destroy the
benefit.

Because hashseed1 is still the only BF400 layout change that survived to final,
a robustness check was launched:

- `bf400_sota_k2_headmix_norowsigns_hashseed1_seed6_1500_20260610`

This keeps `ENGRAM_HASH_SEED=1` but changes `MODEL_SEED=6`, testing whether the
BF400 final improvement is robust to normal model-seed variance.

Avalanche+hashseed1 reached step 500 at `7.2379` and was stopped. It was better
than BF400 control at 500 but worse than hashseed1 alone (`7.2353`) and much
weaker than the earlier avalanche seed0 500 point (`7.2281`). Stronger avalanche
mixing does not compose with the favorable hashseed1 layout.

For the hashseed1 robustness check, a matched BF400 control was launched with
the same model seed:

- `bf400_sota_k2_headmix_norowsigns_control_seed6_1500_20260610`

This makes the seed6 comparison local: hashseed1 seed6 versus control seed6,
instead of comparing seed6 against the seed5 control band.

Hashseed1 seed6 reached `8.5154` at 250, worse than the matched BF400 seed6
control (`8.5117`), and was initially stopped after its 500 readout. The 500
readout then changed the interpretation: hashseed1 seed6 was `7.2469`, while the
matched control seed6 was much worse at `7.2660`. I stopped the run too early
based on the 250 signal. A clean rerun was launched under a new id:

- `bf400_sota_k2_headmix_norowsigns_hashseed1_seed6rerun_1500_20260610`

This will decide the seed6 robustness question at the late checkpoints instead
of over-reading the first validation point. A BF300 current-stack control with
`MODEL_SEED=6` was briefly launched, but was stopped before producing a useful
validation point after I caught the GPU0 scheduling collision. The active seed6
comparison is BF400 hashseed1 rerun versus BF400 control.

Resolved seed6 layout readout:

- BF400 matched control seed6 finished at `3.2455`.
- BF400 hashseed1 seed6 rerun recovered at 500 (`7.2469`) but was worse at 750
  (`5.3571`) and 1000 (`3.5047`) than the matched control path, so it was
  stopped.
- BF300 current-stack control seed6 finished at `3.2452`. Its checkpoints were
  `8.5054`, `7.2292`, `5.3479`, `3.5014`, `3.3445`, and `3.2452`.

This changes the interpretation of hashseed1. The seed5 BF400 hashseed1 final
(`3.2442`) is a legitimate favorable layout draw, but the seed6 rerun says it is
not a robust architectural improvement. The more robust current-tree story is
instead that BF300/BF400 are very close, with BF300 now replicated at `3.2441`
and `3.2452` while using much less table memory than BF400.

## Hit-Frequency Optimizer Scaling

After the hash-layout probes, I added schedulable sparse hit-frequency update
scaling to the Engram sparse optimizers. The motivation was to test whether
hot-row dominance is only a readout/collision effect or also an optimizer
allocation problem: rows with huge hit counts may receive systematically too
much or too little effective update relative to rare rows.

BF400 K2/headmix/no-row-sign probes:

| Run | 250 | 500 | 750 | 1000 | 1250 | 1500 | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control seed5 | 8.5084 | 7.2397 | 5.3498 | - | - | 3.2447 | Current-tree BF400 control. |
| hit LR exponent `-0.25`, blend 0->1 | 8.5084 | 7.2404 | 5.3463 | 3.5000 | 3.3483 | - | Good mid-run, undertrained late; stopped at 1250. |
| IFAL reciprocal-log scale, blend 0->1 | 8.5084 | 7.2312 | 5.3483 | 3.5007 | - | - | Strongest 500, then overcorrected; stopped at 1000. |
| hit LR exponent `-0.25`, half blend | 8.5084 | 7.2405 | 5.3542 | - | - | - | Weaker than full blend by 750; stopped. |
| hit LR exponent `-0.125`, blend 0->1 | 8.5084 | 7.2368 | 5.3472 | 3.4984 | 3.3456 | 3.2463 | Best behaved hit-LR variant, but worse than control final. |

The result is useful but not SOTA-positive. Hot-row-aware update scaling can
move the curve: IFAL and soft negative exponent both improve some intermediate
checkpoints, confirming that frequency-dependent optimizer allocation matters.
But naive attenuation is easy to overdo. Full `-0.25` undertrains late, IFAL's
rare-row boost overcorrects, and the softer `-0.125` schedule still finishes
behind the BF400 control. This points toward measuring per-row loss impact per
hit, or changing row assignment/capacity, rather than simply applying a monotone
hit-count learning-rate correction.

After syncing the latest remote logs, the structural sweep artifacts were
regenerated with 54 runs and 14 complete runs:

- `summary_structural_20260610.csv`
- `structural_variants_vs_params_20260610.png`
- `structural_variants_curves_20260610.png`

The best complete/latest current-tree run in that sweep remains
`bf300_sota_readoutdelta_superposek2_base_aux05_headmixinit05_norowsigns_control_seed5_1500_20260610`
at `3.2441`.

A matched seed7 robustness pair is now running on the two-GPU instance:

- `bf300_sota_k2_headmix_norowsigns_control_seed7_1500_20260610`
- `bf400_sota_k2_headmix_norowsigns_control_seed7_1500_20260610`

This is not a new hyperparameter direction; it is the next clean statistical
check on whether BF300's current-tree efficiency advantage over BF400 is robust
across seeds.

Seed7 first checkpoint:

- BF300 seed7: step 250 `8.5070`.
- BF400 seed7: step 250 `8.5093`.

This is an early but directionally consistent signal: BF300 is ahead of BF400
on the matched seed7 start while using fewer table parameters. This should not
be over-read before 500/750/final, but it supports carrying both runs rather
than killing either. After syncing these partials, the structural sweep contains
56 runs and 14 complete runs; the best complete/latest run is still the BF300
seed5 no-row-sign control at `3.2441`.

Seed7 500-step checkpoint:

- BF300 seed7: step 500 `7.2344`.
- BF400 seed7: step 500 `7.2445`.

The BF300 lead widens from `0.0023` at 250 to `0.0101` at 500. This is a
stronger early/mid-run signal than the first checkpoint and makes the BF300
efficiency story more credible. Both runs should still continue to at least 750
because many previous Engram variants showed early scaffold wins that faded
later, but BF400 is not justifying its extra table parameters on seed7 so far.

Seed7 750-step checkpoint:

- BF300 seed7: step 750 `5.3508`.
- BF400 seed7: step 750 `5.3462`.

The ordering flips by 750: BF400 moves ahead by `0.0046` after trailing by
`0.0101` at 500. This weakens the simple "BF300 is always the better frontier"
interpretation and reinforces the repeated pattern from auxiliary/hash/layout
experiments: early scaffold advantages can reverse late. Both seed7 runs should
continue to 1000/final before making a BF300-vs-BF400 decision.

Seed7 1000-step checkpoint:

- BF300 seed7: step 1000 `3.4996`.
- BF400 seed7: step 1000 `3.4995`.

The two widths are effectively tied at 1000, with BF400 ahead by only `0.0001`.
This is well inside the noise floor and does not justify BF400's larger table
on this checkpoint alone. The important update is trajectory, not the tiny
absolute gap: BF300 led at 250 and 500, BF400 led at 750, and they are tied at
1000. Final loss remains necessary before deciding whether BF300 is the better
current-tree efficiency point or whether BF400 catches up late.

Seed7 1250-step checkpoint:

- BF300 seed7: step 1250 `3.3457`.
- BF400 seed7: step 1250 `3.3449`.

BF400 keeps a very small late edge, now `0.0008`. This is still not practically
meaningful by itself, but the ordering is no longer a single-point fluke:
BF400 is ahead at 750, 1000, and 1250 after losing at 250 and 500. The emerging
interpretation is that the larger table may help late consolidation but the
effect size is tiny relative to seed variance and table cost. The final 1500
loss is the only checkpoint that should be used for the BF300-vs-BF400 call.

Seed7 final:

- BF300 seed7: step 1500 `3.2451`, peak allocated `69621 MiB`.
- BF400 seed7: step 1500 `3.2454`, peak allocated `89799 MiB`.

The final ordering returns to BF300 by `0.0003`, while BF300 uses about `20 GiB`
less peak allocation on this run. This resolves the seed7 robustness check in
favor of BF300 as the better current-tree efficiency point. BF400's midrun
late edge is real enough to avoid dismissing capacity entirely, but the final
effect is below noise and the memory cost is large. After syncing the completed
runs, the structural sweep artifacts contain 56 runs and 16 complete runs; the
best complete/latest current-tree run remains the BF300 seed5 no-row-sign
control at `3.2441`.

The completed seed7 pair did not save checkpoints or hit histograms, because
the launcher hard-set `SAVE_CHECKPOINT=0` and `ENGRAM_SAVE_HIT_HIST=0`.
Therefore the next analysis run should explicitly save both `state_step001500.pt`
and `engram_hit_hist_step001500.pt` for BF300/BF400 controls, enabling row
impact per hit, hot/cold masking, and pruning analysis on the current SOTA
stack rather than relying only on aggregate logged hit statistics.

Checkpoint/hit-hist analysis reruns launched:

- `bf300_sota_k2_headmix_norowsigns_control_seed5_ckpthist_1500_20260610`
- `bf400_sota_k2_headmix_norowsigns_control_seed5_ckpthist_1500_20260610`

Both use the same current-stack no-row-sign/K2/head-mix control settings with
`MODEL_SEED=5`, plus `SAVE_CHECKPOINT=1`, `SAVE_CHECKPOINT_EVERY=1500`, and
`ENGRAM_SAVE_HIT_HIST=1`. These are not intended as new SOTA attempts; they are
artifact-producing reruns so the next step can quantify row usefulness instead
of inferring it from aggregate frequency and final loss alone.

Checkpoint/hit-hist rerun 1250-step checkpoint:

- BF300 ckpthist seed5: step 1250 `3.3440`.
- BF400 ckpthist seed5: step 1250 `3.3453`.

This artifact-producing replicate has BF300 ahead by `0.0013` at 1250. That is
not a new SOTA claim, but it is directionally consistent with the seed7 final:
BF400 can look competitive mid-run, yet the larger table has not produced a
robust final-loss advantage commensurate with its extra memory cost. The main
value of this run remains the saved final checkpoint and hit histogram for
row-usefulness analysis.

Checkpoint/hit-hist rerun final:

- BF300 ckpthist seed5: step 1500 `3.2440`, checkpoint `25 GiB`, hit histogram
  `116 MiB`.
- BF400 ckpthist seed5: step 1500 `3.2447`, checkpoint `33 GiB`, hit histogram
  `154 MiB`.

This reproduces the current-tree BF300 best (`3.2441` -> `3.2440`) while BF400
again lands slightly worse. The artifact-producing rerun therefore strengthens
the working conclusion: the present no-row-sign/K2/head-mix stack has a BF300
sweet spot, and BF400 mostly buys extra rows rather than a reliable final-loss
gain. The saved hit histograms make it possible to move from "hot rows are
frequent" to direct row-frequency/row-norm/pruning measurements.

After syncing the completed artifact reruns, the structural sweep artifacts
contain 58 runs and 18 complete runs. The best complete/latest current-tree
entry is now the BF300 checkpoint/hit-hist replicate at `3.2440`.

Hit-count mask generation from the saved histograms:

- BF300: rows with `hit >= 1024` are `0.84%` of rows but `33.65%` of hits;
  rows with `hit >= 256` are `4.65%` of rows and `50.44%` of hits; rows with
  `hit < 32` are `28.75%` of rows but only `6.39%` of hits.
- BF400: rows with `hit >= 1024` are `0.61%` of rows but `33.21%` of hits;
  rows with `hit >= 256` are `3.29%` of rows and `49.02%` of hits; rows with
  `hit < 32` are `52.29%` of rows but only `13.60%` of hits.

This is the clearest structural signal so far on why BF400 is not obviously
better: the larger table does reduce the hot-row fraction by row count, but the
same small set of very hot rows still carries about one third of all accesses,
while much of the extra BF400 capacity becomes a cold tail. If pruning/masking
confirms the cold tail has low loss impact, BF400's extra memory is mostly
unused capacity under the current hash/addressing scheme rather than useful
learned memory.

Row RMS vs hit-count analysis:

- BF300: `corr(log10(hit + 1), row_rms) = 0.7672`; median hit count `43`;
  p99 hit count `892`; median row RMS `0.2066`; p99 row RMS `0.7008`;
  max row RMS `2.0134`.
- BF400: `corr(log10(hit + 1), row_rms) = 0.7601`; median hit count `30`;
  p99 hit count `691`; median row RMS `0.1838`; p99 row RMS `0.6231`;
  max row RMS `1.8403`.

This makes the hot-row story more concrete. Hit frequency strongly predicts
learned row norm in both BF300 and BF400, so the table is not using rows
uniformly. BF400 reduces ordinary-row hit counts and norms, but the extreme hot
row remains essentially the same maximum hit count (`~4.36M`) and the final loss
does not improve. The current addressing/hash scheme therefore appears to create
a high-frequency backbone plus a broad low-frequency tail, and simply widening
the table mostly cools the tail rather than relieving the hottest backbone rows.

Checkpoint pruning/counterfactual evals started from the saved BF300/BF400
artifacts. Loaded-checkpoint base evals are slightly worse than inline training
finals but close enough to use as local baselines:

- BF300 base eval from checkpoint: `3.24435`.
- BF400 base eval from checkpoint: `3.24525`.

First BF300 hot-row mask result:

- BF300 `hit >= 1024`, zero mode: `3.31539`, a `+0.07104` loss increase over
  BF300 checkpoint base. This masks only `0.84%` of rows but `33.65%` of hits,
  confirming that the hottest rows carry real predictive load.

Additional pruning/counterfactual results:

- BF300 `hit >= 1024`, random mode: `3.31642`, `+0.07207` over base.
- BF400 `hit >= 1024`, zero mode: `3.31560`, `+0.07035` over base.
- BF400 `hit >= 1024`, random mode: `3.31622`, `+0.07098` over base.
- BF300 `hit >= 256`, random mode: `3.39325`, `+0.14890` over base.
- BF400 `hit >= 256`, random mode: `3.39055`, `+0.14530` over base.
- BF300 `hit < 32`, zero mode: `3.24494`, only `+0.00058` over base.
- BF300 `hit < 32`, random mode: `3.24510`, only `+0.00075` over base.
- BF400 `hit < 32`, zero mode: `3.24794`, `+0.00270` over base.
- BF400 `hit < 32`, random mode: `3.24854`, `+0.00330` over base.

Random replacement is not meaningfully better than zero for the hottest rows,
which suggests the damage is not just a zero-vector distribution shift. The
model is relying on the learned content of the hot backbone. The `hit >= 256`
BF300 result is also important: masking `4.65%` of rows / `50.44%` of hits
roughly doubles the loss hit versus masking only `hit >= 1024`, so the useful
backbone extends beyond the extreme top rows.

The first cold-tail result points the other way: BF300 can zero-mask rows with
`hit < 32` (`28.75%` of rows but `6.39%` of hits) with almost no loss movement.
That is a concrete memory-reclamation signal. It suggests a pruned/sparse
runtime can probably drop a large cold fraction after training, even though the
hot backbone is not safely compressible by naive replacement.

The completed cold-tail pair refines this: BF400 can mask a much larger fraction
of rows (`52.29%`) but pays a larger loss hit (`+0.0027` to `+0.0033`) because
that cold tail still contains `13.60%` of accesses. BF300 has less removable row
count, but its cold tail is cleaner. This supports a two-level view of the
Engram table: a frequency/norm backbone that is essential, plus a wide tail
that can be pruned post hoc with small loss, especially when the threshold is
chosen to keep hit mass rather than row count alone.

Next model-iteration runs launched from this analysis:

- `bf300_sota_k2_headmix_norowsigns_ngramrows_balanced_1p0_1p0_seed5_1500_20260610`
- `bf300_sota_k2_headmix_norowsigns_ngramrows_bigramheavy_1p5_0p5_seed5_1500_20260610`

Rationale: the hot-row scan showed the hottest rows are overwhelmingly
2-gram/head0, while the current SOTA row split is trigram-heavy (`0.5,1.5`).
These two BF300 current-stack runs keep total table budget fixed but allocate
more capacity to 2-grams. If either improves, the next direction is nonuniform
row allocation based on observed hit/load structure rather than wider tables.

Ngram row-rebalance 250-step checkpoint:

- Balanced `1.0,1.0`: step 250 `8.5130`.
- Bigram-heavy `1.5,0.5`: step 250 `8.5160`.

The balanced split is slightly ahead by `0.0030`, but neither shows an obvious
early breakout. Bigram-heavy does increase 2-gram/head0 allocation, and its hit
coverage is lower (`hit_frac_ever=0.662` vs balanced `0.843`) with higher
mean hits per touched row (`13.1` vs `10.3`), which suggests concentrating rows
into the 2-gram bucket may worsen cold coverage rather than relieving the useful
hot backbone. These should continue to 500 before deciding, because earlier
Engram variants often reversed between 250 and 750.

Ngram row-rebalance 500-step checkpoint:

- Balanced `1.0,1.0`: step 500 `7.2339`.
- Bigram-heavy `1.5,0.5`: step 500 `7.2443`.

The balanced split is now ahead by `0.0104`, while the bigram-heavy split is
clearly worse at this checkpoint. Balanced is also effectively tied with the
same-seed BF300 current-stack/checkpoint-control trajectory at step 500
(`7.2347` for the original `0.5,1.5` split), so the equal row allocation is not
yet a breakthrough, but it is not damaging the run either. The coverage metrics
make the mechanism look cleaner: balanced reaches `hit_frac_ever=0.922` and
`hit_mean_touched=18.85`, while bigram-heavy reaches only
`hit_frac_ever=0.783` with `hit_mean_touched=22.20`. This argues against
blindly allocating more rows to bigrams despite the hottest rows being
bigram/head0; the current bottleneck may be the persistent hot backbone itself,
not a lack of bigram row capacity. Continue balanced to final. Bigram-heavy is
probably not SOTA-positive, but it is still useful as a negative control if GPU
time is available.

Ngram row-rebalance 750-step checkpoint:

- Balanced `1.0,1.0`: step 750 `5.3544`.
- Bigram-heavy `1.5,0.5`: step 750 `5.3576`.
- Original BF300 same-seed control with `0.5,1.5`: step 750 `5.3479`.

Balanced still beats bigram-heavy, but now trails the original trigram-heavy
allocation by `0.0065`. That weakens the case for changing the SOTA row split:
equal allocation improves row coverage, but better coverage is not translating
into lower validation loss. The observed hot rows are real and important, but
they do not seem bottlenecked by simple bigram row capacity. The stronger
working hypothesis is that the useful signal is carried by a small frequency
backbone plus learned norms/content; redistributing table rows changes tail
coverage more than it changes that backbone.

Ngram row-rebalance 1000-step checkpoint:

- Balanced `1.0,1.0`: step 1000 `3.5052`.
- Bigram-heavy `1.5,0.5`: step 1000 `3.5064`.
- Original BF300 same-seed control with `0.5,1.5`: step 1000 `3.5007`.

This confirms the 750-step read. Both row-rebalanced variants now trail the
original row split by about `0.0045` to `0.0057`, while balanced continues to
beat bigram-heavy internally. The row-factor intervention has the desired
mechanical effect on touched-row coverage but does not improve loss, so it is
best treated as a negative result unless the final checkpoint unexpectedly
reverses. The next iteration should probably stop manipulating raw row counts
and instead target the hot-backbone representation directly: e.g. improve how
multiple addresses combine, regularize/route hot rows, or give hot regions a
learned mixture rather than static extra capacity.

Ngram row-rebalance 1250-step checkpoint:

- Balanced `1.0,1.0`: step 1250 `3.3469`.
- Bigram-heavy `1.5,0.5`: step 1250 `3.3492`.
- Original BF300 same-seed control with `0.5,1.5`: step 1250 `3.3440`.

The result remains negative. Balanced is the better of the two rebalanced row
splits, but still behind the original SOTA meta. Bigram-heavy remains the worst
of the three despite matching the initial intuition from hot-row inspection.
That is a useful falsification: the hot-row effect is not simply “not enough
rows assigned to bigrams.” The model seems to prefer the original trigram-heavy
tail allocation even though the hottest rows are bigram/head0, likely because
trigram/tail coverage supplies marginal information while the hottest bigram
rows remain hot under any static split.

Ngram row-rebalance finals:

- Balanced `1.0,1.0`: final `3.2468`.
- Bigram-heavy `1.5,0.5`: final `3.2486`.
- Original BF300 same-seed control with `0.5,1.5`: final `3.2440` inline
  (`3.2441` in the matched no-row-sign replay).

This closes the row-factor sweep as negative. Equal allocation is consistently
better than bigram-heavy allocation, but both lose to the original trigram-heavy
SOTA split. The final result agrees with the pruning and norm/hit analyses:
Engram has a critical hot backbone, but static row reallocations mostly change
tail coverage and do not relieve the learned hot-row dependency. A better next
test is to improve the representation/readout of multiple candidate rows rather
than change how many rows each ngram bucket owns.

Implementation follow-up: the previous sketch slot-readout path rejected
`ENGRAM_LAYER_READOUT_DELTA`, even though the normal readout path simply adds
the layer-delta value/key projections to the shared projections. The code has
now been patched so slot-readout does the same per slot. This enables a direct
test of the strongest earlier multi-row readout idea without dropping the SOTA
layer-delta component.

New paired BF300 runs launched:

- `bf300_slotreadout_slotmix_layerdelta_base_aux05_norowsigns_seed5_1500_20260610`
- `bf300_slotattention_layerdelta_base_aux05_norowsigns_seed5_1500_20260610`

Both use `ENGRAM_SKETCH_K=2`, `ENGRAM_SKETCH_INCLUDE_BASE=1`,
`ENGRAM_SKETCH_SLOT_READOUT=1`, `ENGRAM_LAYER_READOUT_DELTA=1`, no row signs,
and the same BF300/headmix/attnres/norm/read-hit-scale stack as the current SOTA
meta. Step-0 sanity passed for both at `18.9397`, with the expected slot-mix and
layer-delta config lines present in logs.

Slot-readout+delta 250-step checkpoint:

- Slot-mix + layer-delta: `8.5045`.
- Slot-attention + layer-delta: `8.5152`.

Slot-mix+delta exactly matches the old plain slot-mix 250 checkpoint (`8.5045`)
and is much better than the no-row-sign K2/readoutdelta control (`8.5179`), but
it is not an immediate improvement over the earlier slot-mix path. The learned
slot weights have already moved toward the auxiliary slot (`h0: 0.395/0.605`,
`h1: 0.342/0.658`), which is mechanistically consistent with the prior slot-mix
result. Slot-attention+delta is weak at 250; it is being carried to 500 only
because the earlier attention run partially caught up by 500 before failing at
750.

Slot-readout+delta 500-step checkpoint and decision:

- Slot-mix + layer-delta: `7.2384`.
- Slot-attention + layer-delta: `7.2428`.

Both were killed at 500. The old plain slot-mix run was `7.2269` at the same
checkpoint, and the no-row-sign K2/readoutdelta control was `7.2347`, so adding
layer-delta to separate slot readout did not combine the benefits. Instead it
removed the strongest early/mid slot-mix advantage. Mechanistically, the
slot-mix branch still learned to prefer the auxiliary slot, but the extra delta
projection path appears to make that representation less useful rather than more
layer-adapted. This closes the simple "slot-readout plus SOTA delta" hypothesis
as negative.

Next BF300 slot-mix schedule pair launched:

- `bf300_slotreadout_slotmix_aux05to0_s500_500_norowsigns_seed5_1500_20260610`
- `bf300_slotreadout_slotmix_aux05to0_s750_500_norowsigns_seed5_1500_20260610`

These return to the plain no-delta slot-mix branch that had the best structural
500-step result (`7.2269`) and test whether fading the auxiliary slot to zero
can prevent the late fade. The first run removes aux influence over steps
500-1000; the second keeps the full scaffold through 750 and removes it over
750-1250. This is the cleanest remaining test of the "early scaffold, late
handoff" explanation.

Slot-mix aux-decay 250-step checkpoint:

- Aux `0.5 -> 0`, schedule 500-1000: `8.5045`.
- Aux `0.5 -> 0`, schedule 750-1250: `8.5045`.

Both exactly match the old plain slot-mix 250 checkpoint before either schedule
begins. This is the desired controlled setup: any divergence after 500 or 750
should be attributable to the auxiliary-scale handoff rather than launch drift.

Slot-mix aux-decay 500-step checkpoint:

- Aux `0.5 -> 0`, schedule 500-1000: `7.2269`.
- Aux `0.5 -> 0`, schedule 750-1250: `7.2269`.

Both exactly match the old plain slot-mix 500 checkpoint (`7.2269`). The
controlled boundary is clean. The 500-start schedule now begins decaying the
auxiliary slot; the 750-start schedule should continue to match the unscheduled
trace until the next checkpoint.

Slot-mix aux-decay 750-step checkpoint:

- Aux `0.5 -> 0`, schedule 500-1000: `5.3501`.
- Aux `0.5 -> 0`, schedule 750-1250: `5.3495`.
- Old unscheduled plain slot-mix: `5.3495`.

The 500-start handoff is slightly worse by 750, so removing the auxiliary slot
immediately after the strong 500 checkpoint does not improve the mid-training
fade. The 750-start run still matches the old unscheduled trace exactly at the
handoff boundary. Continue to 1000: if the 750-start schedule helps, it should
first show up there; if the 500-start schedule keeps losing, it can be killed.

Slot-mix aux-decay 1000-step checkpoint:

- Aux `0.5 -> 0`, schedule 500-1000: `3.5119`.
- Aux `0.5 -> 0`, schedule 750-1250: `3.5006`.
- Old unscheduled plain slot-mix: `3.5011`.
- No-row-sign K2/readoutdelta control: `3.5007`.

The 500-start handoff is clearly bad and was killed. It drove the learned slot
mix extremely hard toward the auxiliary slot (`~0.92` aux weight) while the aux
scale was being removed, which is a coherent failure mode: the model tries to
use the disappearing branch more, not less. The 750-start handoff is still live:
it is slightly better than old plain slot-mix and essentially tied with the
no-row-sign control at 1000. This is not a strong positive yet, but it keeps the
late-handoff hypothesis alive through the first post-schedule checkpoint.

Because the 500-start schedule was killed, GPU0 was assigned a more conservative
handoff:

- `bf300_slotreadout_slotmix_aux05to0_s1000_250_norowsigns_seed5_1500_20260610`

This keeps the auxiliary slot fully active through the known-good 1000-step
region, then decays it to zero over steps 1000-1250. It tests whether the right
handoff point is later than 750 rather than whether handoff should happen at all.

Slot-mix aux-decay 1250-step checkpoint:

- Aux `0.5 -> 0`, schedule 750-1250: `3.3538`.
- Old unscheduled plain slot-mix: `3.3455`.
- No-row-sign K2/readoutdelta control: `3.3440`.

The 750-start full handoff is clearly negative by 1250 and was killed. Its slot
mix had moved hard toward the auxiliary slot (`~0.90` aux weight) while the aux
scale was being removed, matching the earlier 500-start failure mode. The result
is useful because it weakens the "slot-mix is an early scaffold that should be
deleted" explanation. The model appears to route real capacity into the
auxiliary slot; removing that path causes a mismatch instead of forcing a clean
handoff.

Follow-up launched on the freed GPU:

- `bf300_slotreadout_slotmix_aux05to025_s1000_250_norowsigns_seed5_1500_20260610`

This keeps the same delayed handoff point but only decays aux from `0.5` to
`0.25`. The targeted question is whether the late damage comes specifically
from deleting a learned branch, while a reduced but still-present auxiliary path
can keep the useful diversity without over-dominating the main readout.

Delayed full-handoff 750-step checkpoint:

- Aux `0.5 -> 0`, schedule 1000-1250: `5.3495`.
- Old unscheduled plain slot-mix: `5.3495`.

The delayed handoff is cleanly controlled through 750: it exactly matches the
unscheduled slot-mix trace before the schedule begins. This means the 1000 and
1250 checkpoints will be interpretable as schedule effects rather than launch
drift. It also means the branch is not showing a new pre-handoff advantage; it
is purely a test of whether waiting until after the known-good 1000 region makes
aux removal less disruptive.

Partial-handoff 250-step checkpoint:

- Aux `0.5 -> 0.25`, schedule 1000-1250: `8.5045`.
- Delayed full-handoff `0.5 -> 0`, schedule 1000-1250: `8.5045`.
- Old unscheduled plain slot-mix: `8.5045`.

The partial-handoff run also matches the controlled slot-mix trace before its
schedule begins. It is therefore a clean paired test against the full-deletion
run: if the `0.25` floor matters, the first meaningful divergence should appear
after step 1000.

Delayed full-handoff 1000-step checkpoint:

- Aux `0.5 -> 0`, schedule 1000-1250: `3.5011`.
- Old unscheduled plain slot-mix: `3.5011`.

This continues the exact match through the schedule boundary. The branch has
not improved the pre-handoff trajectory, but it has preserved the controlled
comparison. The 1250 checkpoint is now the decisive test: previous full-deletion
schedules failed there by routing hard into a branch whose scale was being
removed.

Partial-handoff 500-step checkpoint:

- Aux `0.5 -> 0.25`, schedule 1000-1250: `7.2269`.
- Delayed full-handoff `0.5 -> 0`, schedule 1000-1250: `7.2269`.
- Old unscheduled plain slot-mix: `7.2269`.

The partial-floor run also remains exactly controlled through 500.

Prepared next structural candidate after the aux-handoff pair:

- current BF300 no-row-sign/K2/headmix control path, not slot-readout;
- `ENGRAM_HOT_SPLIT=1`, `ENGRAM_HOT_SPLIT_VALUE_ONLY=1`;
- `ENGRAM_HOT_SPLIT_MIN_HITS=16384`;
- `ENGRAM_HOT_SPLIT_AUX_SLOTS=2`;
- `ENGRAM_HOT_SPLIT_AUX_SCALE=0.025`;
- `ENGRAM_MANUAL_SPARSE_COALESCE=1`.

Rationale: the current pruning analysis says ultra-hot rows are extremely
loss-critical, while cold rows are mostly replaceable. Earlier project history
showed this tiny value-only hot split produced a strong early/mid signal on the
old BF400 stack before fading. The current no-row-sign BF300 stack has not yet
tested that exact targeted hot-backbone capacity mechanism. It is a better next
probe than another global aux-scale schedule because it directly targets the
remaining structural bottleneck identified by the hit-mask and row-norm
analysis.

Delayed full-handoff 1250-step checkpoint:

- Aux `0.5 -> 0`, schedule 1000-1250: `3.3541`.
- Old unscheduled plain slot-mix: `3.3455`.
- No-row-sign K2/readoutdelta control: `3.3440`.

This is the third full-deletion schedule to fail after the aux scale starts
falling. Waiting until 1000 did not fix the failure; by 1250 the learned slot
mix had moved strongly toward the auxiliary slot (`~0.845` aux weight) while
that slot was being removed. The full-deletion hypothesis is now closed as
negative. The run was killed at 1250.

Partial-handoff 1000-step checkpoint:

- Aux `0.5 -> 0.25`, schedule 1000-1250: `3.5011`.
- Delayed full-handoff `0.5 -> 0`, schedule 1000-1250: `3.5011`.
- Old unscheduled plain slot-mix: `3.5011`.

The partial-floor run remains controlled through the schedule boundary. Its
1250 checkpoint is the useful test of whether keeping a nonzero aux branch
prevents the full-deletion mismatch.

Launched next structural probe on freed GPU0:

- `bf300_sota_k2_headmix_norowsigns_hotsplit_valueonly16384_auxslots2_aux0025_seed5_1500_20260610`

This uses the current BF300 no-row-sign/K2/headmix control path and adds only the
tiny value-only hot split described above. It is intentionally not a slot-readout
run: it targets ultra-hot value memory directly while preserving the current
best readout/addressing stack.

Partial-handoff 1250-step checkpoint:

- Aux `0.5 -> 0.25`, schedule 1000-1250: `3.3457`.
- Delayed full-handoff `0.5 -> 0`, schedule 1000-1250: `3.3541`.
- Old unscheduled plain slot-mix: `3.3455`.
- No-row-sign K2/headmix control: `3.3440`.

Keeping a nonzero auxiliary floor prevents the full-deletion collapse, but it
does not beat the current control. Mechanistically, this says the auxiliary
slot is not merely a temporary scaffold to delete; by late training, the learned
slot mix uses it as real model capacity. A small floor avoids the mismatch, but
the extra branch still has not produced a durable advantage over the simpler
K2/headmix readout.

Hot-split value-only early checkpoints:

- Step 250: `8.5093`.
- Step 500: `7.2337`.

This is effectively tied with the current BF300 no-row-sign/K2/headmix control
band through 500 steps. It is not an obvious early win like the old slot-mix
branch, but it is also not broken, so it should be allowed to reach at least 750
and likely final unless it falls behind clearly.

Partial-handoff final:

- Aux `0.5 -> 0.25`, schedule 1000-1250: `3.2459`.
- BF300 no-row-sign/K2/headmix current control: `3.2440`.

This closes the partial-floor slot-mix branch as stable but not useful. It
confirms the full-deletion failure was a capacity/path mismatch, not an inherent
slot-readout instability, but preserving the auxiliary branch still does not
convert the early slot-mix advantage into a final loss improvement.

Hot-split correction:

The first hot-split run above accidentally omitted `ENGRAM_LAYER_READOUT_DELTA=1`.
That made it a weaker no-layer-delta probe rather than the intended "SOTA plus
hot split" test. It was stopped after step 1250:

- Misconfigured no-layer-delta hot split: step 750 `5.3528`, step 1000 `3.5019`,
  step 1250 `3.3465`.

This is negative as a no-layer-delta hot-split control, but it should not be used
as the final verdict on hot split with the actual current SOTA stack.

Corrected hot-split run launched:

- `bf300_sota_k2_headmix_layerdelta_norowsigns_hotsplit_valueonly16384_auxslots2_aux0025_seed5_1500_20260610`
- Same current SOTA stack as the BF300 checkpoint-history control, including
  `ENGRAM_LAYER_READOUT_DELTA=1`.
- Adds only `ENGRAM_HOT_SPLIT=1`, `ENGRAM_HOT_SPLIT_VALUE_ONLY=1`,
  `ENGRAM_HOT_SPLIT_MIN_HITS=16384`, `ENGRAM_HOT_SPLIT_AUX_SLOTS=2`,
  `ENGRAM_HOT_SPLIT_AUX_SCALE=0.025`.

Corrected hot-split first checkpoint:

- Step 250: `8.5191`.

This is weaker than the BF300 SOTA/control 250-step band, so the corrected hot
split is not showing the old early hot-backbone signal. It is still running to
500 before being closed.

Bucketed hit-count counterfactual evals started from the BF300 SOTA checkpoint:

- Base checkpoint eval: `3.244354`.
- Mask rows with exactly 1 training hit: `3.244354`.
- Mask rows with 2-3 training hits: `3.244360`.

The coldest rows are effectively loss-neutral. This is stronger than the earlier
coarse `hit < 32` result because it separates the cold tail into buckets. The
remaining diagnostic question is whether impact per unit hit mass rises
superlinearly in the hot buckets, which would mean hot-row dominance is not just
raw n-gram frequency.

Corrected hot-split 500-step checkpoint:

- Step 250: `8.5191`.
- Step 500: `7.2385`.
- BF300 SOTA/control reference around step 500: `7.2347`.

The corrected hot-split run is behind at both 250 and 500, so it was stopped.
Together with the misconfigured no-layer-delta hot-split run, this makes the
tiny value-only hot-backbone branch negative on the current stack. The earlier
old-stack positive was not robust to the final no-row-sign/K2/headmix/layer-delta
recipe.

Bucketed counterfactuals, complete results:

| Masked hit bucket | Val loss | Delta vs base | Hit fraction | Delta per 1% hit mass |
|---|---:|---:|---:|---:|
| `1` | `3.244354` | `+0.000000` | `0.0000003` | `0.000000` |
| `2-3` | `3.244360` | `+0.000006` | `0.0000077` | `0.008348` |
| `4-7` | `3.244370` | `+0.000015` | `0.0001953` | `0.000793` |
| `8-15` | `3.244357` | `+0.000003` | `0.0036896` | `0.000008` |
| `16-31` | `3.244849` | `+0.000495` | `0.0600476` | `0.000082` |
| `32-63` | `3.251080` | `+0.006726` | `0.1823547` | `0.000369` |
| `64-127` | `3.254644` | `+0.010290` | `0.1448507` | `0.000710` |
| `128-255` | `3.257305` | `+0.012951` | `0.1044565` | `0.001240` |
| `256-511` | `3.261107` | `+0.016753` | `0.0891548` | `0.001879` |
| `512-1023` | `3.261541` | `+0.017187` | `0.0787139` | `0.002183` |
| `1024-2047` | `3.255754` | `+0.011400` | `0.0655399` | `0.001739` |
| `>=2048` | `3.294472` | `+0.050118` | `0.2709890` | `0.001849` |

The transition is sharp: below 16 hits is basically free, 16-31 is weak but
measurable, and 32+ becomes meaningfully loss-bearing. Impact per unit hit mass
also rises strongly across the hot buckets, so row impact is not flat after
dividing by frequency. This suggests that the hot-row effect is not simply raw
n-gram frequency; more frequently accessed rows appear to carry higher-value or
more reusable features per access. The `>=2048` bucket is only `0.354%` of rows,
but zeroing it costs `+0.0501` loss.

Because the low-hit rows have much lower norms in the saved SOTA checkpoint
(`1` hit median RMS `0.044`, `8-15` median RMS `0.141`, `16-31` median RMS
`0.178`, `32-63` median RMS `0.207`), a restrained cold-row norm floor test was
launched:

- `bf300_sota_k2_headmix_layerdelta_norowsigns_rowrmsfloor014_hit31_seed5_1500_20260610`
- `ENGRAM_SPARSE_ROW_RMS_FLOOR=0.14`
- `ENGRAM_SPARSE_ROW_RMS_FLOOR_HIT_MAX=31`

This tests whether the cold/low-mid rows are neutral because they are
under-amplified rather than intrinsically useless. It floors only touched rows
while their cumulative hit count is still at or below 31.

Cold-row RMS floor result:

- Step 250: `8.5229`.
- Matched BF300 SOTA/control step 250: `8.5179`.

The floor was stopped at 250. It worsens early loss while increasing touched-row
parameter RMS, so the low-impact cold rows do not look like a simple
under-amplification problem. Forcing them louder injects noise before it creates
useful memory.

Complementary hot-row RMS cap test launched:

- `bf300_sota_k2_headmix_layerdelta_norowsigns_rowrmscap075_seed5_1500_20260610`
- `ENGRAM_SPARSE_ROW_RMS_CAP=0.75`

Step 250 is exactly matched to the control at `8.5179`, which means the cap has
not affected the early regime. Leave it to step 500 before deciding whether
later high-norm/hot rows are necessary or overdominant.

Hot-row RMS cap result:

- Step 250: `8.5179`, exactly matched to control.
- Step 500: `7.2389`.
- Matched BF300 SOTA/control step 500: `7.2347`.

The cap was stopped at 500. It is only `+0.0042` worse, but it is in the same
small negative band as corrected hot split and does not justify paying for a
full train. The result weakly suggests the hottest/highest-norm rows are not an
obvious overdominance bug.

Next diagnostic launched as eval-only probes on the BF300 SOTA checkpoint:

- Scale rows with at least `2048` training hits by `0.5x`.
- Scale rows with at least `2048` training hits by `1.5x`.

This is cheaper than retraining and directly asks whether the hottest rows are
overused, underused, or already roughly calibrated after normalization and the
learned readout.

First hot-row scale eval results:

| Eval modification | Val loss | Delta vs base eval |
|---|---:|---:|
| Base checkpoint eval | `3.244354` | `+0.000000` |
| `>=2048` hits scaled `0.5x` | `3.245634` | `+0.001280` |
| `>=2048` hits scaled `1.5x` | `3.244913` | `+0.000559` |

Both directions are worse, with downscaling worse than upscaling. The hottest
rows are important, but this does not look like a simple final-readout amplitude
mistuning. Softer `0.8x` and `1.2x` probes were launched to check whether there
is a very local optimum near `1.0x`.

Softer hot-row scale eval results:

| Eval modification | Val loss | Delta vs base eval |
|---|---:|---:|
| `>=2048` hits scaled `0.8x` | `3.244476` | `+0.000122` |
| `>=2048` hits scaled `1.2x` | `3.244491` | `+0.000137` |

The local optimum for the hottest rows is close to `1.0x`. This reinforces the
interpretation that the top hot rows are not simply mis-scaled at inference time.
A second pair of eval-only probes was launched for the transition bucket
`32-127` hits, where bucketed zeroing first showed meaningful loss impact.

Transition-bucket scale eval results:

| Eval modification | Val loss | Delta vs base eval |
|---|---:|---:|
| `32-127` hits scaled `0.8x` | `3.244591` | `+0.000237` |
| `32-127` hits scaled `1.2x` | `3.244426` | `+0.000072` |

This is also negative, though `1.2x` is nearly neutral. The current readout
appears close to locally calibrated by hit bucket: zeroing rows reveals where
information lives, but posthoc bucket amplitude scaling does not recover a
better validation loss.

Medium-hot split structural follow-up launched:

- `bf300_sota_k2_headmix_layerdelta_norowsigns_hotsplit_valueonly128_auxslots2_aux0025_seed5_1500_20260610`
- `bf300_sota_k2_headmix_layerdelta_norowsigns_hotsplit_full128_auxslots2_aux0025_seed5_1500_20260610`

Both use the BF300 SOTA stack and set `ENGRAM_HOT_SPLIT_MIN_HITS=128`,
`ENGRAM_HOT_SPLIT_AUX_SLOTS=2`, `ENGRAM_HOT_SPLIT_AUX_SCALE=0.025`. This shifts
the hot-split idea from ultra-hot rows (`>=16384`) to the bucket range where
counterfactual zeroing first showed strong loss impact. The pair compares
value-only auxiliary context against full key/value auxiliary context.

First medium-hot split checkpoint:

- Value-only `min_hits=128`: step 250 `8.5227`, worse than BF300 control
  `8.5179`; stopped.
- Full key/value `min_hits=128`: step 250 `8.5179`, matched BF300 control.

The value-only branch repeats the negative pattern from the ultra-hot value-only
test. Full key/value splitting is at least not harmful at 250, so the active
pair was revised to:

- Full key/value split at `min_hits=128`.
- Full key/value split at `min_hits=32`.

The `min_hits=32` run tests the first clearly loss-bearing bucket from the
counterfactual table, while `min_hits=128` checks the hotter transition region.

Active full-split first checkpoints:

- Full key/value `min_hits=128`: step 250 `8.5179`, matched BF300 control.
- Full key/value `min_hits=32`: step 250 `8.5179`, matched BF300 control.

Unlike value-only split, full key/value split is not hurting at the first
checkpoint. Keep both to step 500, where the earlier corrected value-only split
and row RMS cap began to show their negative separation.

Implementation issue found: full key/value split was a no-op for the current
SOTA configuration because `apply_hot_split` only handled 2-D address tensors,
while `ENGRAM_SUPERPOSE_K=2` passes superposed addresses. This explains why the
full-split `min_hits=128` and `min_hits=32` logs matched the control exactly and
did not emit `engram_hot_split_*` metrics.

The code was patched so superposed address tensors are supported: a read is
treated as hot if any component row is hot, auxiliary rows are generated for all
components, then summed back into the per-head memory. Fixed v2 runs launched:

- `bf300_sota_k2_headmix_layerdelta_norowsigns_hotsplitv2_full128_auxslots2_aux0025_seed5_1500_20260610`
- `bf300_sota_k2_headmix_layerdelta_norowsigns_hotsplitv2_full32_auxslots2_aux0025_seed5_1500_20260610`

Fixed v2 first checkpoints:

- Full key/value `min_hits=128`: step 250 `8.5143`; hot split fraction `0.309`.
- Full key/value `min_hits=32`: step 250 `8.5105`; hot split fraction `0.484`.
- BF300 SOTA/control step 250: `8.5179`.

This is the first positive result from hot split on the current SOTA stack. The
lower threshold is stronger, matching the bucketed counterfactual result that
rows become meaningfully loss-bearing starting around `32` hits. Both fixed v2
runs continue to step 500.

Fixed v2 step-500 checkpoints:

- Full key/value `min_hits=128`: step 500 `7.2330`; hot split fraction `0.387`.
- Full key/value `min_hits=32`: step 500 `7.2325`; hot split fraction `0.594`.
- BF300 SOTA/control step 500: `7.2347`.

Both remain positive at 500, with `min_hits=32` still stronger. However both
runs then hit `cudaErrorIllegalAddress` in sparse gradient coalescing at the
next optimizer step. The fixed hot-split branch increases touched rows from
about `742k` to `2.15M`, so the next stability test uses the existing manual
sparse coalesce path: `ENGRAM_MANUAL_SPARSE_COALESCE=1`.

Manual sparse coalesce v3 results:

- Full key/value `min_hits=32`: step 250 `8.5196`, step 500 `7.2345`.
- Full key/value `min_hits=128`: step 250 `8.5178`, step 500 `7.2361`.

Manual coalescing survives past the post-500 optimizer step, so it fixes the
crash, but it mostly removes the positive signal. This suggests the exact sparse
gradient accumulation path matters here, not just the forward auxiliary read.

A cleaner stability variant was added: `ENGRAM_HOT_SPLIT_DETACH_AUX=1`. This
keeps the auxiliary hot-split memory in the forward pass, but detaches the
auxiliary row lookup so gradients only flow through the original read rows. This
should keep touched-row count near baseline and avoid the CUDA sparse coalesce
failure without changing the forward perturbation. v4 detach-aux runs launched:

- `bf300_sota_k2_headmix_layerdelta_norowsigns_hotsplitv4_full32_auxslots2_aux0025_detachaux_seed5_1500_20260610`
- `bf300_sota_k2_headmix_layerdelta_norowsigns_hotsplitv4_full128_auxslots2_aux0025_detachaux_seed5_1500_20260610`

v4 detach-aux first checkpoints:

- Full key/value `min_hits=32`, aux scale `0.025`: step 250 `8.5100`;
  touched rows back to baseline (`741877`), hot split fraction `0.484`.
- Full key/value `min_hits=128`, aux scale `0.025`: step 250 `8.5203`;
  stopped.
- BF300 SOTA/control step 250: `8.5179`.

Detach-aux keeps the positive early signal for the `32` threshold while avoiding
the expanded sparse-gradient footprint from v2. The `128` threshold is negative
in this form. The `32`/`0.025` run continues to 500, and a stronger
`32`/`0.05` detach-aux run was launched:

- `bf300_sota_k2_headmix_layerdelta_norowsigns_hotsplitv4_full32_auxslots2_aux005_detachaux_seed5_1500_20260610`

v4 detach-aux step-500 update:

- Full key/value `min_hits=32`, aux scale `0.025`: step 500 `7.2366`.
- BF300 SOTA/control step 500: `7.2347`.

Detach-aux is useful early but loses the 500-step gain, implying that training
the auxiliary rows contributed to the stronger v2 result. The `0.025` detach
run was stopped. A non-detached lower-gradient-footprint version was launched:

- `bf300_sota_k2_headmix_layerdelta_norowsigns_hotsplitv5_full32_auxslots1_aux0025_seed5_1500_20260610`

This keeps auxiliary row training but cuts auxiliary slots from 2 to 1, aiming
to preserve the v2 signal while avoiding the sparse coalesce crash.

Plot/report refresh after adding the fixed hot-split labels:

- `summary_structural_20260610.csv`: 74 runs indexed, 19 complete.
- Best complete/latest remains BF300 SOTA/control at `3.2440`.
- Hot-split v2/v3/v4/v5 runs are now labeled separately in the CSV/plots.

v4 stronger detach-aux first checkpoint:

- Full key/value `min_hits=32`, aux scale `0.05`, detach aux: step 250
  `8.5157`; hot split fraction `0.484`, aux slots `2`.
- BF300 SOTA/control step 250: `8.5179`.
- Prior `0.025` detach-aux step 250: `8.5100`.

The stronger detached forward perturbation is only mildly positive and is worse
than the `0.025` detach setting at the same step. It continues to 500 because it
is not clearly bad, but the current best signal is still non-detached full split
with trainable auxiliary rows, if the sparse-gradient path can be stabilized.

v4 stronger detach-aux step-500 update:

- Full key/value `min_hits=32`, aux scale `0.05`, detach aux: step 500
  `7.2311`; hot split fraction `0.594`, aux slots `2`.
- BF300 SOTA/control step 500: `7.2347`.
- Fixed v2 trainable aux `min_hits=32`, aux scale `0.025`: step 500 `7.2325`
  before crashing on the next sparse coalesce.

This reverses the earlier step-250 read: the stronger detached branch is the
best hot-split result at 500 and is stable past the post-eval optimizer step.
It is not proof of final SOTA yet, but it is the cleanest current candidate
because it captures the hot-split gain without increasing the sparse-gradient
row footprint.

Prepared next stability knob: `ENGRAM_HOT_SPLIT_DEDUP_AUX=1`. This is off by
default. When enabled, the hot-split auxiliary addresses are uniqued before the
embedding lookup and gathered back afterward. The forward values should be
unchanged, but sparse autograd should see one aux lookup per unique auxiliary
row instead of every repeated aux occurrence. This is the next candidate if
trainable aux rows remain promising but continue to trigger CUDA sparse coalesce
failures.

v5 auxslot1 trainable-aux result:

- Full key/value `min_hits=32`, aux scale `0.025`, aux slots `1`, trainable aux:
  step 250 `8.5159`, step 500 `7.2695`.
- It survived past the post-500 optimizer step, so reducing aux slots avoids the
  v2 crash, but the loss is much worse than control and hot-split detach.
- The run was stopped after step 500.

v6 launched on GPU0:

- Full key/value `min_hits=32`, aux scale `0.025`, aux slots `2`, trainable aux,
  `ENGRAM_HOT_SPLIT_DEDUP_AUX=1`.
- Purpose: preserve the stronger v2 forward/training setup while reducing sparse
  gradient duplication before Adam coalescing.

v4 detach-aux `0.05` step-750 update:

- Hot split: step 750 `5.3507`.
- BF300 SOTA/control: step 750 `5.3479`.

The detach branch is stable and had the best step-500 value, but the advantage
does not persist cleanly to 750. It may be a transient optimization effect or
within run noise; keep it running to 1500 before deciding whether it is a real
SOTA improvement.

v4 detach-aux `0.05` step-1000 update:

- Hot split: step 1000 `3.5027`.
- BF300 SOTA/control: step 1000 `3.5007`.

The detach branch is now behind control at both 750 and 1000. It is still useful
as a stable diagnostic, but the current evidence no longer supports it as a
likely final-loss improvement.

v6 dedup-aux first checkpoint:

- Full key/value `min_hits=32`, aux scale `0.025`, aux slots `2`, trainable aux,
  dedup aux: step 250 `8.5082`.
- BF300 SOTA/control step 250: `8.5179`.
- Fixed v2 trainable aux, no dedup, same threshold/scale/slots: step 250
  `8.5105`.
- Dedup metrics at step 250: raw aux rows `1,048,576`, unique aux rows
  `734,158`, unique fraction `0.700`.

This is the strongest hot-split step-250 result so far. It still has the large
touched-row footprint (`2.15M` rows), so the key question is whether dedup avoids
the post-500 CUDA sparse coalesce crash that killed v2.

v6 dedup-aux step-500 update:

- Dedup trainable aux: step 500 `7.2321`.
- BF300 SOTA/control step 500: `7.2347`.
- Fixed v2 trainable aux, no dedup: step 500 `7.2325`, then crashed at the next
  sparse coalesce.
- v4 detach-aux `0.05`: step 500 `7.2311`.

v6 survived past the post-500 optimizer step and continued through at least
step 526, so dedup appears to fix the v2 sparse coalesce failure while retaining
most of the trainable-aux gain. It is now the best stable trainable hot-split
variant, though still not proven at final loss.

v4 detach-aux `0.05` step-1250 update:

- Hot split: step 1250 `3.3460`.
- BF300 SOTA/control: step 1250 `3.3440`.

The detach branch is now behind control at 750, 1000, and 1250. Unless it
recovers unexpectedly at 1500, it should be treated as a transient early gain
rather than a SOTA improvement.

v6 dedup-aux step-750 update:

- Dedup trainable aux: step 750 `5.3519`.
- BF300 SOTA/control step 750: `5.3479`.
- v4 detach-aux `0.05` step 750: `5.3507`.

v6 fixed the crash and gave the best early hot-split result, but it also loses
the edge by 750. That suggests the auxiliary hot-split read may be useful as
early optimization scaffolding while becoming a late-training tax.

v7 launched on GPU0:

- Same as v6, but `ENGRAM_HOT_SPLIT_AUX_SCALE=0.025` decays to `0.0` starting
  at step 500 over 500 steps.
- Purpose: test whether retaining early hot-split pressure while removing the
  late auxiliary read avoids the 750+ regression.

v4 detach-aux `0.05` final:

- Final step 1500 `3.2462`.
- BF300 SOTA/control final: `3.2440`.

The fixed detach hot-split branch is stable and helped at step 500, but it loses
by final eval. It is not a SOTA improvement.

v8 launched on GPU1:

- Same as v4 detach-aux `0.05`, but `ENGRAM_HOT_SPLIT_AUX_SCALE=0.05` decays to
  `0.0` starting at step 500 over 500 steps.
- Purpose: paired with v7, test whether early hot-split scaffolding plus late
  removal works for detached aux as well as trainable/dedup aux.

v7 dedup-aux decay first checkpoint:

- Step 250 `8.5088`.
- v6 dedup-aux no decay step 250: `8.5082`.
- BF300 SOTA/control step 250: `8.5179`.

The decay variant preserves the early hot-split benefit before the schedule
starts. The real test is 750+, after the aux scale has begun falling.

v7 dedup-aux decay step-500 update:

- Step 500 `7.2393`.
- v6 dedup-aux no decay step 500: `7.2321`.
- BF300 SOTA/control step 500: `7.2347`.

This is unexpectedly worse before the scheduled decay has visibly reduced the
logged aux scale. The trainable/dedup decay branch is no longer promising, but
it continues to 750 once to test the intended decay region.

v8 detach-aux decay first checkpoint:

- Step 250 `8.5157`.
- v4 detach-aux no decay step 250: `8.5157`.
- BF300 SOTA/control step 250: `8.5179`.

The detach decay variant preserves the early behavior of the non-decay detach
run. Its useful comparison starts at 750, after the aux scale has decayed.

v7 dedup-aux decay step-750 update:

- Step 750 `5.3530`.
- v6 dedup-aux no decay step 750: `5.3519`.
- BF300 SOTA/control step 750: `5.3479`.

Decay did not fix the trainable/dedup hot-split regression. v7 was stopped after
750.

Added `ENGRAM_HOT_SPLIT_TRAIN_ONLY=1`:

- When enabled, hot split is active during training but bypassed during eval.
- Purpose: test whether hot split is only useful as an optimization scaffold and
  harmful as an inference/eval read path.

v9 launched on GPU0:

- Full key/value `min_hits=32`, aux scale `0.05`, aux slots `2`, detach aux,
  train-only hot split.
- This is the cleanest test of "train with scaffold, evaluate without it."

v8 detach-aux decay step-750 update:

- Step 750 `5.3509`.
- v4 detach-aux no decay step 750: `5.3507`.
- BF300 SOTA/control step 750: `5.3479`.

At half-decay (`aux_scale=0.025`), v8 is still behind control and almost
identical to non-decay v4. Continue to 1000, where the aux scale should be zero,
before discarding the decay hypothesis.

v9 train-only startup:

- Eval line omits hot-split metrics, confirming the eval bypass is active.
- Training continues with hot split active; first useful comparison is step 250.

v9 train-only early checkpoints:

- Step 250 `8.5156`.
- Step 500 `7.2311`.
- BF300 SOTA/control: step 250 `8.5179`, step 500 `7.2347`.

Train-only hot split preserves the early optimization gain while removing the
hot-split eval path. This makes the scaffold hypothesis more plausible than the
auxiliary-read-path hypothesis. Continue to 750, because previous hot-split
variants began losing their advantage there.

v9 train-only step-750 update:

- Step 750 `5.3507`.
- BF300 SOTA/control step 750: `5.3479`.
- v4 detach-aux step 750: `5.3507`.

The mid-run regression still appears even when hot split is bypassed in eval.
That weakens the idea that the loss is mainly due to the auxiliary eval read
path. It looks more like the training scaffold changes the learned trajectory
in a way that is not durable. Continue to 1000 once, then stop if it remains
behind.

v9 train-only step-1000 update:

- Step 1000 `3.5027`.
- BF300 SOTA/control step 1000: `3.5007`.
- v4 detach-aux step 1000: `3.5027`.

Train-only hot split converged to the same mid-run penalty as the normal
detach-aux hot-split branch. v9 was stopped after 1000. The likely conclusion is
that hot split changes the learned trajectory, not merely the eval read path.

v8 detach-aux decay step-1000 update:

- Step 1000 `3.5025`.
- BF300 SOTA/control step 1000: `3.5007`.

At step 1000, the aux scale has decayed to zero and v8 still trails control.
This closes the decay-rescue branch for detached hot split. v8 was stopped
after the 1000 checkpoint to free GPU1.

Launched hot-row dropout v1:

- BF300 SOTA/control config plus train-only `ENGRAM_HIT_DROPOUT=0.10`.
- Applies only to rows with hit count >=32.
- Inverted dropout scaling enabled.
- Dropout decays to `0.0` from step 500 to 1000.

Purpose: test hot-row reliance as a training-time regularization problem rather
than adding an auxiliary read path. Eval remains the normal SOTA read path.

Hot-row dropout v1 result:

- Step 250 `8.5195` versus BF300 control `8.5179`.
- Step 500 `7.2358` versus BF300 control `7.2347`.

This did not reproduce the hot-split early gain. It directly perturbs important
hot rows instead of providing alternative capacity, and at p=0.10 it mildly
hurts. The run was stopped after 500.

Patched and launched AttnRes extra-stream tests:

- Found that `_engram_attnres_merge(..., extra=...)` computed `extra_value` and
  extra softmax weights, but only added the memory term to the output.
- Patched it to add `extra_weight * extra_value` when an extra stream exists.
- Launched paired BF300 SOTA-control variants:
  - `bf300_sota_k2_headmix_layerdelta_norowsigns_attnresextra5to8_seed5_1500_20260610`
  - `bf300_sota_k2_headmix_layerdelta_norowsigns_attnresextra2to8_seed5_1500_20260610`

Purpose: test a real cross-layer residual/extra stream into the layer-8 engram
AttnRes merge, rather than another hot-row or table-size tweak.

AttnRes extra-stream startup check:

- Both `5->8` and `2->8` launched and reached step 0.
- Step-0 logs show `engram_attnres_extra_p_l8 ~= 0.334`, so the patched extra
  branch is active and receives roughly one third of the layer-8 AttnRes softmax
  mass at initialization.
- Throughput after warmup is normal, around `664 ms/step`.

The first useful comparison is step 250.

AttnRes extra-stream `5->8` step-250 update:

- Step 250 `8.5198`.
- BF300 SOTA/control step 250: `8.5179`.
- Extra-stream weight at layer 8 dropped from `~0.334` at init to `~0.251`.

The patch is definitely active, but the source-5 extra stream starts slightly
worse than control. Continue to 500 once before discarding, because the gap is
small and this branch is a structural test rather than a pure early-loss tweak.

AttnRes extra-stream `2->8` step-250 update:

- Step 250 `8.5429`.
- BF300 SOTA/control step 250: `8.5179`.
- Extra-stream weight at layer 8 dropped from `~0.334` at init to `~0.230`, but
  the branch still badly hurt optimization.

The `2->8` source is too disruptive with zero extra-bias init and was stopped
after 250. The failure suggests the default extra stream takes too much softmax
mass initially. A conservative `5->8` follow-up was launched with
`ENGRAM_ATTNRES_EXTRA_BIAS_INIT=-4.0`; the bias is a learned Adam parameter, so
this starts the branch near off but lets training opt into it.

AttnRes extra-stream `5->8` step-500 update:

- Step 500 `7.2415`.
- BF300 SOTA/control step 500: `7.2347`.
- Extra-stream weight at layer 8 dropped further to `~0.176`.

The zero-bias `5->8` branch worsened from a small 250-step miss to a clear
500-step miss and was stopped. A second bracket was launched with
`ENGRAM_ATTNRES_EXTRA_BIAS_INIT=-2.0` while the `-4.0` run continues; together
these test whether the repaired extra stream is useful only when initialized as
a weak optional residual.

AttnRes extra-stream `5->8 bias -4` step-250 update:

- Step 250 `8.5115`.
- BF300 SOTA/control step 250: `8.5179`.
- Zero-bias `5->8` step 250: `8.5198`.
- Extra-stream weight at layer 8 started near `0.0091` and was `0.0069` at step
  250.

This is the first repaired extra-stream result with a positive early signal. The
branch appears useful only when initialized as a weak optional residual rather
than as a third equally weighted AttnRes source. Continue to 500.

AttnRes extra-stream `5->8 bias -2` step-250 update:

- Step 250 `8.5071`.
- BF300 SOTA/control step 250: `8.5179`.
- Bias `-4` step 250: `8.5115`.
- Extra-stream weight at layer 8 started near `0.0635` and was `0.0491` at step
  250.

This is the strongest early result in the repaired extra-stream branch so far,
and it is stronger than the `-4` near-off variant. Continue to 500; if it holds,
the useful regime may be a small but nontrivial cross-layer residual rather than
a hard-gated one.

AttnRes extra-stream `5->8 bias -4` step-500 update:

- Step 500 `7.2380`.
- BF300 SOTA/control step 500: `7.2347`.
- Zero-bias `5->8` step 500: `7.2415`.

Bias `-4` is less harmful than zero-bias but still loses the early edge by 500.
It was stopped after 500. Since bias `-2` had the stronger 250-step result, a
stronger `-1` bracket was launched to locate the useful initial extra-stream
mass between the failed zero-bias case and the promising `-2` case.

AttnRes extra-stream `5->8 bias -2` step-500 update:

- Step 500 `7.2379`.
- BF300 SOTA/control step 500: `7.2347`.
- Bias `-4` step 500: `7.2380`.

Bias `-2` had the best 250-step result but did not hold to 500. It was stopped
after 500. Bias `-1` continues warming on GPU0, and a tighter `-1.5` bracket was
launched on GPU1 to test whether a slightly stronger initial extra stream can
keep the early gain longer.

AttnRes extra-stream `5->8 bias -1` step-250 update:

- Step 250 `8.5139`.
- BF300 SOTA/control step 250: `8.5179`.
- Bias `-2` step 250: `8.5071`.
- Bias `-4` step 250: `8.5115`.
- Extra-stream weight at layer 8 started near `0.155` and was `0.120` at step
  250.

Bias `-1` is still better than control at 250, but weaker than both `-2` and
`-4`. Continue to 500 once, while `-1.5` tests the midpoint between `-1` and
`-2`.

AttnRes extra-stream implementation correction:

- While adding a scale schedule for the extra stream, found that the repaired
  `_engram_attnres_merge(..., extra=...)` path was adding the extra stream
  twice: once before direct residual and once after direct residual.
- Fixed it to a single extra contribution.
- Added default-off schedule knobs:
  - `ENGRAM_ATTNRES_EXTRA_SCALE`
  - `ENGRAM_ATTNRES_EXTRA_SCALE_FINAL`
  - `ENGRAM_ATTNRES_EXTRA_SCALE_SCHEDULE_START`
  - `ENGRAM_ATTNRES_EXTRA_SCALE_SCHEDULE_STEPS`

This means the preceding extra-stream bias sweep should be interpreted as a
high-strength/doubled-extra probe. The useful signal is still real as a
directional observation, but the exact bias values are not clean.

Relaunched clean single-extra pair:

- `bf300_sota_k2_headmix_layerdelta_norowsigns_attnresextra5to8_biasm2_singlefix_seed5_1500_20260610`
- `bf300_sota_k2_headmix_layerdelta_norowsigns_attnresextra5to8_biasm2_singlefix_scale1to0_s250_250_seed5_1500_20260610`

The second run tests the scaffold hypothesis directly: use bias `-2` early, then
decay the extra contribution to zero from step 250 to 500.

Clean single-extra step-250 update:

- Bias `-2`, scale `1.0`: step 250 `8.5209`.
- Bias `-2`, scale `1.0 -> 0.0`: step 250 `8.5209`.
- BF300 SOTA/control step 250: `8.5179`.

The clean single-extra branch at scale `1.0` does not reproduce the earlier
positive 250-step signal. This supports the interpretation that the previous
gain came from the accidental doubled extra contribution. Both scale-1 runs were
stopped after 250.

Relaunched clean scale-decay pair:

- `bf300_sota_k2_headmix_layerdelta_norowsigns_attnresextra5to8_biasm2_singlefix_scale2to0_s250_250_seed5_1500_20260610`
- `bf300_sota_k2_headmix_layerdelta_norowsigns_attnresextra5to8_biasm2_singlefix_scale1p5to0_s250_250_seed5_1500_20260610`

These test whether the earlier doubled-strength signal can be used only as an
early scaffold and then removed before the 500-step regression.

Clean scale-decay step-250 update:

- Scale `2.0 -> 0.0`: step 250 `8.5060`.
- Scale `1.5 -> 0.0`: step 250 `8.5042`.
- BF300 SOTA/control step 250: `8.5179`.
- Clean scale `1.0` step 250: `8.5209`.

The early signal returns once the extra contribution is stronger than scale
`1.0`. Scale `1.5` is currently the best repaired extra-stream result at step
250. The decisive check is step 500, after both schedules have decayed the extra
contribution to zero.

Clean scale-decay step-500 update:

- Scale `2.0 -> 0.0`: step 500 `7.2451`.
- Scale `1.5 -> 0.0`: step 500 `7.2319`.
- BF300 SOTA/control step 500: `7.2347`.

Scale `1.5 -> 0.0` is the first repaired extra-stream run to beat the control at
step 500 after the extra contribution has decayed to zero. Scale `2.0` was too
strong and regressed badly. The scale-2 run was stopped. The scale-1.5 process
was also accidentally stopped while killing scale-2, so it was relaunched from
scratch to get 750/final checkpoints. A scale `1.25 -> 0.0` bracket was also
launched to test whether the optimum is below 1.5.

2026-06-11 live seed/diagnostic update:

- `bf300_sketchk2_slotreadout_slotmix_dimsigns_balanced_base_aux05_seed6_1500_20260611`
  reached step 1000 at `3.5021`, behind the seed5 no-row-sign control checkpoint
  (`3.5007`). It was stopped at 1000 to free GPU0.
- `bf300_sketchk2_slotreadout_slotmix_dimsigns_balanced_base_aux05_seed7_1500_20260611`
  reached step 750 at `5.3472`, slightly better than the seed5 no-row-sign
  control 750 checkpoint (`5.3479`). It remains running to test whether the
  earlier step-500 advantage (`7.2318` vs control `7.2347`) survives to final.
  It then reached step 1000 at `3.5000`, still slightly better than the seed5
  no-row-sign control 1000 checkpoint (`3.5007`), so it is continuing to 1500.
  At step 1250 it regressed to `3.3453`, behind seed5 control (`3.3440`) and
  the seed5 sketch rerun (`3.3442`), but it remains close enough and near enough
  to final to keep for a complete seed-distribution point.
- GPU0 was reassigned to
  `bf300_sota_k2_headmix_layerdelta_norowsigns_hotsplitv2_full32_auxslots2_aux0025_gradhook_seed5_1500_20260611`.
  This keeps the reproduced hot-split v2 config and adds only
  `ENGRAM_SPARSE_GRAD_COALESCE_HOOK=1`, aiming to test whether backward-time
  coalescing avoids the native sparse `grad.coalesce()` crash after the 500 eval
  without switching to the worse manual-coalesce path.
- The grad-hook run reached step 250 at `8.5078`, slightly better than the
  reproduced original hot-split v2 rerun (`8.5105`) and the seed5 no-row-sign
  control (`8.5179`). This preserves the early hot-split signal; the decisive
  test is whether it reaches and continues past the post-500 optimizer step.
- It reached step 500 at `7.2441` and continued to steps 501-503 without the
  original CUDA illegal-address crash. This shows all-run gradient-hook coalescing
  can avoid the crash, but it badly worsens the 500-step curve versus original
  hot-split v2 (`7.2325`) and control (`7.2347`). The run was stopped.
- A delayed-hook follow-up was launched:
  `bf300_sota_k2_headmix_layerdelta_norowsigns_hotsplitv2_full32_auxslots2_aux0025_gradhookstart501_seed5_1500_20260611`.
  It adds `ENGRAM_SPARSE_GRAD_COALESCE_HOOK_START=501`, so steps through 500
  should match original hot-split dynamics while the hook first affects the
  crash-prone post-500 step.
- The structural report script now labels this run as `hot-split full grad hook`
  so it is not mixed with the original unstable `hot-split full superpose`
  family.

2026-06-11 later live update:

- `bf300_sketchk2_slotreadout_slotmix_dimsigns_balanced_base_aux05_seed7_1500_20260611`
  finished at step 1500 with val `3.2461`. Together with seed5 `3.2452` and
  seed6 stopped at step 1000 after `3.5021`, this says the sketch/slotmix branch
  is close but not a SOTA mover against the bf300 no-row-sign control seed5
  final `3.2440`.
- The delayed hook-start-501 hot-split run exactly reproduced the original
  hot-split curve through step 500: step 250 `8.5105`, step 500 `7.2325`. It
  still crashed immediately after the step-500 eval with SIGABRT/CUDA failure.
  This suggests `ENGRAM_SPARSE_GRAD_COALESCE_HOOK_START=501` is one step too
  late; the first crash-prone backward/update after the 500 eval likely still
  sees the debug/current step as 500.
- Two `ENGRAM_SPARSE_GRAD_COALESCE_HOOK_START=500` follow-ups were launched,
  seed5 on GPU0 and seed6 on GPU1. This should preserve all pre-500 hot-split
  behavior but activate the coalescing hook for the first post-500 sparse grad
  path.
- The seed5 start-500 run matched original hot-split at step 250 (`8.5105`) and
  step 500 (`7.2325`) and then continued through at least step 520. This
  confirms the crash boundary was an off-by-one activation issue: start 501 was
  too late, while start 500 preserves the useful pre-500 trajectory and catches
  the failing post-500 sparse grad path.
- The seed6 start-500 run also crossed the same boundary: step 250 `8.5229`,
  step 500 `7.2434`, then continued beyond step 505. The fix is therefore not a
  seed5-only fluke; final/750 losses will decide whether hot-split itself moves
  SOTA or only gives an early-step improvement.
- At step 750, start-500 hot-split is stable but not ahead: seed5 `5.3509`,
  seed6 `5.3503`. These are worse than bf300 control seed5 `5.3479` and sketch
  seed7 `5.3472` by only a few thousandths, so the branch remains close enough
  to finish, but the early 500-step advantage has not clearly persisted.
- Both start-500 hot-split seeds then reached step 1000 but crashed immediately
  afterward in `SparseScalarAdam.step()` inside `coalesce_row_sparse_grad()`;
  seed5 val was `3.5014`, seed6 val `3.5031`. The traceback points to native
  `grad.coalesce()` in the optimizer, not the backward hook, so the hook only
  moved the illegal-address failure from the post-500 step to the post-1000
  step once touched rows grew larger.
- Added `ENGRAM_MANUAL_SPARSE_COALESCE_START`, allowing native coalesce until a
  chosen step and manual row-unique/index_add coalescing afterward. A hot-split
  stability run is now launched with `ENGRAM_MANUAL_SPARSE_COALESCE=1` and
  `ENGRAM_MANUAL_SPARSE_COALESCE_START=1000`.
- Because hot-split is no longer ahead at 750/1000, GPU1 is assigned to a
  cleaner follow-up for the more promising repaired extra-stream branch:
  `attnresextra5to8_biasm2_singlefix_scale1p5to0_s250_250_full`.
- The extra-stream full run reproduced the earlier 250-step signal with val
  `8.5042`. The hot-split manual-start run also reproduced its expected step
  250 value `8.5105`. Next decisive checks are extra-stream step 500 and whether
  hot-split survives past the manual-coalesce activation at step 1000.
- The extra-stream full run also reproduced the repaired best 500-step value:
  `7.2319`, after the extra contribution had decayed to zero. Hot-split
  manual-start matched its expected step 500 value `7.2325`. Extra-stream is now
  the more promising SOTA candidate; hot-split is mostly a stability/pathology
  diagnostic until it proves otherwise later.
- By step 750, both active branches fell back to `5.3509`. This mirrors the
  earlier hot-split result and is worse than the bf300 control seed5 `5.3479`.
  The working interpretation is that both interventions act like early
  optimization scaffolds through 500 rather than durable improvements, unless
  later checkpoints recover.
- At step 1000, hot-split with manual coalesce starting at 1000 reached val
  `3.5014` and continued past step 1030. This confirms the staged manual
  coalescer fixes the repeatable optimizer-side CUDA illegal-address crash at
  native `grad.coalesce()`. Extra-stream reached `3.5020`. Both remain behind
  the bf300 control seed5 1000 value `3.5007` and sketch seed7 `3.5000`.
- The first manual-coalesce hot-split run later produced NaN train losses around
  step 1195 and was killed. The likely issue was that the manual coalescer summed
  duplicate sparse rows in bf16; it has been patched to accumulate in fp32 and
  cast back. A distinct `manualfp32start1000` hot-split run was launched on GPU0.
- Extra-stream reached step 1250 at `3.3465`, behind bf300 control seed5
  `3.3440`. Its early 250/500 improvement is therefore not carrying into later
  training on this seed.
- Extra-stream finished at step 1500 with val `3.2477`, behind the bf300
  control/checkpoint-hist seed5 final `3.2440`. This closes the scale-1.5
  extra-stream branch as another early-scaffold result: it improves the
  250/500 trajectory but does not improve the final basin.
- The fp32 manual-coalesce hot-split rerun reached step 500 with the same curve
  as the earlier hot-split runs: step 250 `8.5105`, step 500 `7.2325`. Its
  purpose is now mainly a stability diagnostic: the decisive test is whether
  fp32 duplicate-row accumulation avoids the previous NaN window after the
  manual coalescer activates at step 1000.
- The fp32 manual-coalesce hot-split rerun reached step 750 at `5.3509`,
  matching the earlier hot-split/extra-stream washout and trailing the bf300
  control seed5 value `5.3479`. It should no longer be considered a SOTA
  candidate, but should continue through step 1000 and the old `~1195` NaN
  window to validate the fp32 manual sparse coalescer.
- The same run reached step 1000 at `3.5014`, then continued beyond step 1015.
  This confirms the fp32 manual coalescer fixes the immediate optimizer-side
  CUDA illegal-address crash that previously happened right after the 1000 eval.
  The remaining stability check is whether it avoids the earlier NaN window
  around `1195-1206`.
- It did not: train loss became NaN again by step 1150 and stayed NaN through
  the old `1195` window. The run was killed. The native-coalesce crash is fixed
  by manual fp32 coalescing, but the hot-split configuration has a separate
  numerical instability later in training, likely from the hot-split
  gradient/update dynamics rather than bf16 duplicate accumulation alone.

Bucketed BF300 checkpoint counterfactuals were also reduced into an
impact-per-hit-mass view. Using the saved bf300 control/checkpoint-hist run as
base eval `3.2444`, zeroing each hit-count bucket gives:

| Hit bucket | Rows | Hits | Loss delta | Delta / 1% hits |
|---|---:|---:|---:|---:|
| `1` | `0.0029%` | `0.0000%` | `+0.0000` | `0.00000` |
| `2-3` | `0.0298%` | `0.0008%` | `+0.0000` | `0.00000` |
| `4-7` | `0.3413%` | `0.0195%` | `+0.0000` | `0.00000` |
| `8-15` | `3.0986%` | `0.3690%` | `+0.0000` | `0.00000` |
| `16-31` | `25.2750%` | `6.0048%` | `+0.0004` | `0.00007` |
| `32-63` | `42.8377%` | `18.2355%` | `+0.0067` | `0.00037` |
| `64-127` | `17.5340%` | `14.4851%` | `+0.0102` | `0.00070` |
| `128-255` | `6.2337%` | `10.4456%` | `+0.0129` | `0.00123` |
| `256-511` | `2.6386%` | `8.9155%` | `+0.0167` | `0.00187` |
| `512-1023` | `1.1671%` | `7.8714%` | `+0.0171` | `0.00217` |
| `1024-2047` | `0.4868%` | `6.5540%` | `+0.0114` | `0.00174` |
| `>=2048` | `0.3541%` | `27.0989%` | `+0.0501` | `0.00185` |

This answers the "impact divided by frequency" question: it is not flat. The
cold tail has far less loss impact per hit than the mid/hot backbone. From
`256+` hits onward, the per-hit impact is broadly similar, while the ultra-hot
bucket dominates total loss mostly because it owns `27.1%` of all accesses.

Follow-up launched from this bucketed-impact read:

- `bf300_sota_k2_headmix_layerdelta_norowsigns_hitlr_pos0125_1to2_s250_250_seed5_1500_20260611`

This keeps the BF300 SOTA/control stack and changes only the sparse Engram
optimizer. It enables positive hit-count LR scaling with exponent `0.125`,
clamped to `[1.0, 2.0]`, and ramps the blend from `0` to `1` over steps
`250-500`. This is deliberately the opposite of IFAL: rather than cooling
frequent rows, it mildly boosts established hot/backbone rows after early row
identity has formed. The rationale is that the bucketed counterfactuals show
the cold tail is low-value per hit, while the `256+` hit buckets have much
higher impact per hit. If this helps, it means the previous IFAL/hot-drop
failures were pushing against useful backbone specialization rather than fixing
over-reliance.

The positive hit-LR run reached step 250 at `8.5179`, exactly matching the
bf300 control path while `engram_hit_lr_blend=0`. This confirms it is a clean
intervention: the first meaningful checkpoint is step 500, after the hot-row
boost has ramped in.

At step 500, positive hit-LR is worse: `7.2369` versus bf300 control `7.2347`
and IFAL's earlier `7.2304`. The run has `engram_hit_lr_scale_mean=1.541` and
`engram_hit_lr_scale_max=2.0`, with update/grad ratio `33.1`, so the bounded
hot-row boost is materially increasing memory update size. It is being carried
to 750 before stopping, because several prior variants only became readable at
750, but the 500-step result argues against simple positive frequency scaling.
At 750 it is clearly worse: `5.3568` versus bf300 control `5.3479`. The update
ratio rises to `64.9`, so the branch was killed. Together with IFAL and FAL,
this closes the simple monotone hit-count LR family: cooling hot rows, boosting
hot rows, and normalizing within-batch repeats all move optimizer statistics but
do not improve the final-relevant curve.

After closing the monotone hit-LR branch, both GPUs were moved back to the most
durable non-control axis seen in the report: BF400 hash layout. BF400
hashseed1 finished at `3.2442`, nearly tying the BF300 best `3.2440` at a
larger parameter count, while hashseed2 and hashseed3 showed that the layout
distribution is broad. Two more BF400 current-stack layout samples were
launched:

- `bf400_sota_k2_headmix_layerdelta_norowsigns_hashseed4_seed5_1500_20260611`
- `bf400_sota_k2_headmix_layerdelta_norowsigns_hashseed5_seed5_1500_20260611`

These keep the same BF400 no-row-sign K2/headmix/readoutdelta SOTA stack and
change only `ENGRAM_HASH_SEED`. This is a structural row-assignment probe at
fixed parameter count, not another optimizer scale tweak. The decision rule is
to stop weak layouts at 250/500 and carry only layouts that show a plausible
chance of surviving to the final basin.

BF400 hash-layout continuation first decisions:

- Hashseed4: step 250 `8.5104`, step 500 `7.2425`; worse than BF400 control
  (`8.5084`, `7.2397`) and killed at 500.
- Hashseed5: step 250 `8.5159`, step 500 `7.2394`; bad early but roughly
  neutral/slightly better than BF400 control at 500, so it continues to 750.
- Hashseed6 was launched on the freed GPU0 as another BF400 layout sample.

This reinforces the broad-layout-distribution read: seed4 is weak, seed5 is
not obviously useful but recovered by 500, and only a subset of layouts deserve
full training.

Hashseed5 reached step 750 at `5.3520`, worse than BF400 control `5.3498`, and
was killed. Hashseed7 was launched on the freed GPU1 while hashseed6 continued
toward its first checkpoint.

Hashseed6 reached step 250 at `8.5189`, clearly worse than BF400 control
`8.5084`, and was killed. Hashseed8 was launched on the freed GPU0.

Hashseed7 reached step 250 at `8.5085`, essentially neutral to BF400 control
`8.5084` and weaker than hashseed1's `8.5063`. It is being carried to 500 once,
because seed5 also recovered by 500 after a poor 250.

Hashseed7 reached step 500 at `7.2399`, slightly worse than BF400 control
`7.2397`, and was killed. Hashseed8 reached step 250 at `8.5114`; this is weak,
but not as bad as seed6, so it continues to 500 once. Hashseed9 was launched on
the freed GPU1.

Hashseed8 reached step 500 at `7.2395`, effectively neutral to BF400 control
but following the same weak-250/neutral-500 pattern as hashseed5, which failed
at 750. It was killed at 500. Hashseed10 was launched on the freed GPU0.

Hashseed9 reached step 250 at `8.5050`, better than BF400 control `8.5084` and
also better than hashseed1's `8.5063` at the same checkpoint. This is the first
new sample in the 20260611 continuation with a seed1-like early signal, so it
continues to 500.

Hashseed9 failed at step 500: `7.2492`, far worse than BF400 control `7.2397`.
Hashseed10 was also clearly bad at step 250: `8.5418`. Both were killed. The
continuation sweep therefore strengthens the existing conclusion: hash layout is
a real fixed-parameter axis, but the distribution is broad and favorable
layouts are rare. In this batch, seeds 4/6/10 were immediately weak, seeds
5/7/8 were neutral-ish but not useful, and seed9 had a misleadingly strong
250-step value that collapsed by 500. Hashseed1 remains the only sampled BF400
layout that survived to a near-SOTA final.

After the seed4-10 continuation, an offline hash-layout pre-screen was tested
before spending more GPU time. The first metric replayed the BF400 SOTA hash
geometry over `20M` training tokens and measured global row-hit concentration
for seeds `0-64`: nonzero row fraction, top-1/top-10/top-100/top-1000 hit mass,
and max-hit/mean-hit. These statistics were essentially invariant across
seeds, including good-ish seed1 and bad seeds. This rules out the simple story
that favorable hash seeds merely make the hot-row distribution globally less
concentrated.

A second offline metric looked at the top `200k` canonical bigrams and top
`200k` canonical trigrams from `10M` training tokens and estimated weighted
collisions among frequent ngram modes under each seed. This varied more across
seeds and selected low-collision candidates `hashseed26` and `hashseed47`.
Those were launched on the BF400 SOTA stack:

- `bf400_sota_k2_headmix_layerdelta_norowsigns_hashseed26_seed5_1500_20260611`
- `bf400_sota_k2_headmix_layerdelta_norowsigns_hashseed47_seed5_1500_20260611`

Both failed the first gate and were killed at step 250:

- Hashseed26: `8.5214`
- Hashseed47: `8.5163`

This is a useful negative result. Coarse hit concentration is not predictive,
and even frequent-ngram collision mass is not sufficient to identify favorable
layouts. The hashseed1 advantage appears to depend on more specific
interactions between which ngrams collide, which layer/head reads them, and how
the trained readout/gate dynamics use those collisions. Blind hashseed search
is therefore low expected value unless paired with a better diagnostic than
global hotness or top-mode collision mass.

With blind hashseed selection de-prioritized, the next probe combined the two
remaining structural ideas that had real early/mid signals but weak finals:
balanced count-sketch slot-mix and a temporary `5->8` AttnRes extra stream. The
motivation is that they target different mechanisms: count-sketch/slot-mix
gives a distributed multi-row code, while the extra stream gives layer 8 a
short-lived cross-layer residual scaffold that decays away by step 500.

Launched:

- `bf300_sketchk2_slotreadout_slotmix_dimsigns_balanced_base_aux05_attnresextra5to8_biasm2_scale1p25to0_s250_250_seed5_1500_20260611`
- `bf300_sketchk2_slotreadout_slotmix_dimsigns_balanced_base_aux05_attnresextra5to8_biasm2_scale1p5to0_s250_250_seed5_1500_20260611`

Both use the count-sketch slot-mix branch (`ENGRAM_SKETCH_K=2`,
balanced dimension/scalar signs, slot readout + slot mix, aux scale `0.5`) and
add AttnRes extra source layer `5` into target layer `8` with bias `-2`; the
extra scale decays to zero over steps `250-500`.

First checkpoints:

- Scale `1.25 -> 0`: step 250 `8.5044`, step 500 `7.2275`.
- Scale `1.5 -> 0`: step 250 `8.4981`, step 500 `7.2311`.

Scale `1.5` gives the best 250-step value seen in this family, but scale `1.25`
is clearly better by 500. This is the first new branch after the hashseed block
that genuinely beats the control at 500 while also being tied to a structural
hypothesis rather than a blind parameter draw. Scale `1.25` is continuing to
750/final. Scale `1.5` was stopped at 500 and replaced with a lower `1.0 -> 0`
bracket to test whether the useful range is below `1.25`.

Scale `1.25 -> 0` reached step 750 at `5.3480`, essentially tied with the
BF300 no-row-sign control/checkpoint curve (`5.3479`). The strong 500-step gain
therefore does fade, but it does not collapse in the way the full slot-mix
deletion handoffs did. This remains a live branch through 1000/final because it
is the cleanest composition so far of distributed row codes with a decayed
cross-layer scaffold. The lower scale `1.0 -> 0` bracket is still before its
first validation checkpoint.

The `1.0 -> 0` bracket reached step 250 at `8.4968`, better than both
`1.25 -> 0` (`8.5044`) and `1.5 -> 0` (`8.4981`). The optimum for this
composition is therefore not the same as the normal SOTA path: once the
distributed slot-mix readout is present, the extra stream wants to be weaker,
not stronger. Scale `1.0` is continuing to 500.

Later readout:

- Scale `1.25 -> 0`: step 1000 `3.5010`.
- Scale `1.0 -> 0`: step 500 `7.2322`.

Scale `1.25` remains the best 500-step version, but by 1000 it is only neutral
to slightly behind the BF300 control path. Scale `1.0` had the best 250-step
value but did not hold the advantage to 500, so it was stopped. This suggests
the combination's useful effect is real but still scaffold-like: it improves
the early/mid optimization path without yet changing the later basin.

The next follow-up keeps the better AttnRes extra scale (`1.25 -> 0`) but lowers
the fixed slot-mix auxiliary scale from `0.5` to `0.25`:

- `bf300_sketchk2_slotreadout_slotmix_dimsigns_balanced_base_aux025_attnresextra5to8_biasm2_scale1p25to0_s250_250_seed5_1500_20260611`

Rationale: the slot-mix family repeatedly routes toward the auxiliary slot over
time. If the late fade is partly aux-branch overdependence rather than the
distributed code itself, a lower fixed auxiliary read should preserve some of
the early distributed-code benefit while reducing the late branch mismatch.

The aux0.25 follow-up failed the first checkpoint: step 250 `8.5138`, much
worse than the aux0.5 scale1.25 run (`8.5044`) and the scale1.0/1.5 brackets.
Lowering the slot auxiliary strength from the start removes too much of the
distributed-code benefit. The branch was killed at 250.

The aux0.5 scale1.25 run reached step 1250 at `3.3452`, behind the BF300
control/checkpoint path (`3.3440`). It continues to final for the full curve,
but it is unlikely to beat SOTA. The remaining useful question is whether this
combined distributed-readout/cross-layer-scaffold uses wider table capacity
better than the base BF400 stack, so a BF400 scale-out was launched:

- `bf400_sketchk2_slotreadout_slotmix_dimsigns_balanced_base_aux05_attnresextra5to8_biasm2_scale1p25to0_s250_250_seed5_1500_20260611`

This keeps the best combo setting found so far and changes only BF300 -> BF400,
giving a direct parameter-count comparison for the updated structural plot.

BF300 combo final:

- Scale `1.25 -> 0`, aux0.5: final `3.2460`.

The branch is therefore a clean negative for SOTA loss: it produced a real
500-step improvement (`7.2275`) but converged worse than the BF300 control
(`3.2440`). This supports the scaffold interpretation. The distributed
slot-mix plus decayed extra stream can improve optimization/transient readout,
but in the current form it does not improve the final learned representation.

BF400 scale-out first checkpoint:

- BF400 scale `1.25 -> 0`, aux0.5: step 250 `8.5056`.

This is better than BF400 control (`8.5084`) and slightly better than BF400
hashseed1's 250-step point (`8.5063`), so it continues to 500. Because BF400
hashseed1 is the only layout change that survived to a near-SOTA final, a
parallel composability test was also launched:

- `bf400_hashseed1_sketchk2_slotreadout_slotmix_dimsigns_balanced_base_aux05_attnresextra5to8_biasm2_scale1p25to0_s250_250_seed5_1500_20260611`

BF400 default-layout combo reached step 500 at `7.2398`, essentially neutral to
slightly worse than BF400 control (`7.2397`) and much worse than BF400
hashseed1 at 500 (`7.2353`). It was killed at 500. The BF400 hashseed1 combo
continues as the only live scale-out/composability test.

One more structural follow-up was launched on the freed GPU:

- `bf300_sketchk2_slotreadout_slotmix_dimsigns_balanced_base_aux05_attnresextra5to8_biasm2_scale0to1p25_s500_500_seed5_1500_20260611`

This inverts the extra-stream schedule. Instead of using the cross-layer stream
as an early scaffold and decaying it away, it starts the extra scale at `0` and
ramps to `1.25` over steps `500-1000`. This tests whether the extra stream is
more useful as a late-basin correction after the slot-mix representation has
formed.

BF400 hashseed1 combo failed at the first checkpoint: step 250 `8.5179`. This
is worse than BF400 control (`8.5084`), BF400 hashseed1 alone (`8.5063`), and
the default-layout BF400 combo (`8.5056`). It was killed at 250. The favorable
hashseed1 layout does not compose with the distributed slot-mix + extra-stream
branch; it appears to be a different route through the collision/readout
landscape rather than a reusable additive component.

To separate "late extra helps slot-mix" from "late extra helps the base SOTA
basin", a paired no-slot control was launched:

- `bf300_sota_k2_headmix_layerdelta_norowsigns_attnresextra5to8_biasm2_scale0to1p25_s500_500_seed5_1500_20260611`

This keeps the BF300 SOTA K2/headmix/layerdelta stack and applies the same late
extra-stream ramp (`0 -> 1.25`, steps `500-1000`) without count-sketch slot
readout.

The no-slot late-extra control was negative by step 500:

- Base SOTA + late extra: step 250 `8.5141`, step 500 `7.2455`.

That is far behind the BF300 control path, so it was killed at 500. The
slot-mix late-extra run looked better but still not SOTA-relevant:

- Count-sketch slot-mix + late extra: step 250 `8.5089`, step 500 `7.2340`,
  step 750 `5.3520`, step 1000 `3.5044`.

This keeps the interpretation narrow. Late extra does not help the base SOTA
basin, and inside slot-mix it mostly changes the mid-training trajectory after
the ramp starts. It also underperforms the earlier decayed-extra combo at the
same 1000-step point (`3.5010`), so the run was killed at 1000.

The next structural probe is a bounded combine-mix readout:

- `bf300_sketchk2_combinemix_bounded01_dimsigns_balanced_base_aux05_seed5_1500_20260611`

This uses `ENGRAM_SKETCH_K=2` with balanced dimension/scalar signs and base
inclusion, but does not use the slot-readout/slot-mix branch. Instead it learns
a bounded pre-combine weighting over the two sketch slots before the normal
memory-head readout. The point is to test whether the distributed-code benefit
survives when it stays on the standard readout path instead of routing through
the separate slot-readout branch that has repeatedly faded late.

Default-layout combine-mix first checkpoint:

- Step 250 `8.5065`.

This is not as strong as the earlier count-sketch sum checkpoint (`8.5012`) and
does not reproduce the best slot-mix early bump. It is still close enough to
carry to 500, especially because the bounded combine weights are only a
conservative perturbation at this point (`~0.94-1.06` across the two slots).

At step 500 the same default-layout combine-mix run reached `7.2360`. That is
respectable but not SOTA-shaped, and the paired hashseed1 run was already much
more interesting, so the default-layout branch was killed at 500.

A paired hash-layout composability run was launched on the freed GPU:

- `bf300_hashseed1_sketchk2_combinemix_bounded01_dimsigns_balanced_base_aux05_seed5_1500_20260611`

Hashseed1 improved the base BF400 table but failed when combined with
slot-mix + extra stream. This paired combine-mix run tests whether the
non-composability came from the slot-readout branch specifically or from the
count-sketch distributed code more generally.

BF300 hashseed1 combine-mix first checkpoint:

- Step 250 `8.5000`, step 500 `7.2321`, step 750 `5.3513`, step 1000
  `3.5013`, step 1250 `3.3455`, final `3.2462`.

This is the first combine-mix result with a genuinely interesting early signal:
it is better than default combine-mix (`8.5065`), better than the earlier
count-sketch sum (`8.5012` by a hair), and much better than the hashseed1
slot-mix + extra composition failure (`8.5179`). The 500-step point preserves
that signal: it beats default combine-mix (`7.2360`) and is in the useful
mid-training band. The 750-step point also remains competitive. By step 1000
the branch is not a breakout, and by 1250 it is behind the BF300 control path
(`3.3455` vs `3.3440`), but it remains close enough to carry to final for the
full curve. The final is `3.2462`, confirming the same pattern: bounded
combine-mix + hashseed1 is a real early/mid optimization improvement but not a
new SOTA final.

A BF400 hashseed1 combine-mix scale-out was launched while the BF300 hashseed1
run continues:

- `bf400_hashseed1_sketchk2_combinemix_bounded01_dimsigns_balanced_base_aux05_seed5_1500_20260611`

The BF400 run has initialized and written step 0; its first useful decision
point is step 250.

BF400 hashseed1 combine-mix first checkpoint:

- Step 250 `8.5009`, step 500 `7.2349`.

This is stronger than the earlier BF400 hashseed1 base checkpoint (`8.5063`) and
much stronger than the BF400 hashseed1 slot-mix + extra composition failure
(`8.5179`). This is the clearest positive composability result from the
count-sketch family so far: keeping the sketch slots on the normal readout path
appears to compose with the favorable hash layout, while the separate
slot-readout branch did not. At 500 the advantage over BF400 hashseed1 base is
small (`7.2349` vs `7.2353`), but it persists rather than reversing, so the run
continues.

At step 750, BF400 hashseed1 combine-mix faded to `5.3555`, which is too weak
for the scale-out branch. It was killed at 750. The pattern now looks like:
bounded combine-mix + hashseed1 improves early optimization and can survive to
500, but at larger BF it does not yet preserve the advantage into the
late/mid-loss regime.

The freed GPU was assigned a targeted aux-decay isolation:

- `bf300_hashseed1_sketchk2_combinemix_bounded01_dimsigns_balanced_base_aux05to0_s750_500_seed5_1500_20260611`

This keeps the BF300 hashseed1 combine-mix setup but decays
`ENGRAM_SKETCH_AUX_SCALE` from `0.5` to `0` over steps `750-1250`. In the
combine path, the base slot is unscaled and the auxiliary sketch slot is
multiplied by `SKETCH_AUX_SCALE` before the normalized sum. This run tests
whether the auxiliary sketch slot is useful early but harmful late.

Aux-decay first checkpoint:

- Step 250 `8.5000`, step 500 `7.2321`, step 750 `5.3513`.

This matches the constant-aux branch through 750, as expected because the decay
does not start until step 750. The real comparison starts at 1000/1250.

At step 1000, aux-decay reached `3.5046`, worse than constant-aux (`3.5013`).
The run was killed. Decaying the auxiliary sketch slot after 750 therefore does
not fix the late fade; it removes useful signal faster than it removes whatever
late drag exists.

After the constant-aux branch finished, GPU1 was assigned a stronger slot
weighting variant:

- `bf300_hashseed1_sketchk2_combinemix_softmax_dimsigns_balanced_base_aux05_seed5_1500_20260611`

The bounded combine-mix weights saturate around `0.9/1.1`; softmax removes that
small-deviation cap while keeping the same hashseed1/count-sketch/base-aux
setup. This tests whether the distributed sketch slot needs stronger learned
weighting to survive late training.

The softmax run has initialized and written step 0. Its first useful checkpoint
is step 250.

Softmax combine-mix first checkpoint:

- Step 250 `8.4984`.

This is stronger than bounded hashseed1 combine-mix (`8.5000`) and the earlier
count-sketch sum (`8.5012`). The learned slot weights already route toward the
auxiliary sketch slot at roughly `60/40`, so this is a real stronger-mix effect
rather than just noise in an otherwise identical initialization. It continues to
500.

At step 500, softmax combine-mix fell to `7.2382`, much worse than bounded0.1
hashseed1 combine-mix (`7.2321`). The run was killed. Uncapped softmax routing
buys a better 250-step point but loses the useful 500-step trajectory.

Two follow-ups were launched after killing aux-decay and softmax:

- `bf300_sota_k2_headmix_layerdelta_norowsigns_hashseed1_seed5_1500_20260611`
- `bf300_hashseed1_sketchk2_combinemix_bounded02_dimsigns_balanced_base_aux05_seed5_1500_20260611`

The first is the missing BF300 hashseed1 base-control: it isolates the hash
layout from combine-mix. The second is a moderate bracket between bounded0.1
and softmax, testing whether a slightly larger but still capped slot deviation
can keep the strong early softmax signal without the 500-step collapse.

First checkpoints:

- BF300 hashseed1 base-control: step 250 `8.5139`, killed.
- BF300 hashseed1 bounded0.2 combine-mix: step 250 `8.5015`, continuing to 500.

The base-control result is important despite being negative: at BF300,
hashseed1 alone does not explain the combine-mix improvement. The combine
mechanism is doing real early work. Bounded0.2 is worse than bounded0.1
(`8.5000`) and softmax (`8.4984`) at 250, but not bad enough to kill before 500.

After killing the base-control, GPU0 was assigned:

- `bf300_hashseed1_sketchk2_combinemix_softmax_dimsigns_balanced_base_aux025_seed5_1500_20260611`

This keeps the stronger softmax slot weighting but lowers the auxiliary sketch
scale from `0.5` to `0.25`, testing whether softmax's 500-step collapse was
caused by over-amplifying the auxiliary sketch slot rather than by softmax
routing itself.

Softmax aux0.25 reached step 250 at `8.5134`, much worse than full-aux softmax
(`8.4984`). It was killed. Lowering the auxiliary sketch scale does not rescue
softmax; it removes the early benefit.

Bounded0.2 reached step 500 at `7.2414`, worse than bounded0.1 (`7.2321`) and
softmax full-aux (`7.2382`). It was killed. The interpolation did not help:
more bounded deviation than 0.1 accelerates the same 500-step degradation.

The freed GPU was assigned a more structural missing-composition test:

- `bf300_hashseed1_sketchk2_combinemix_bounded01_dimsigns_balanced_base_aux05_layerdelta_seed5_1500_20260611`

The prior combine-mix runs had `ENGRAM_LAYER_READOUT_DELTA=0`, while the current
SOTA base stack uses layer readout delta. This run tests whether bounded
combine-mix can compose with layerdelta, rather than competing with a missing
piece of the SOTA recipe.

A paired no-hashseed layerdelta control was launched on the other GPU:

- `bf300_sketchk2_combinemix_bounded01_dimsigns_balanced_base_aux05_layerdelta_seed5_1500_20260611`

Together these isolate whether layerdelta helps combine-mix generally, and
whether the effect depends on hashseed1.

Hashseed1 + layerdelta combine-mix reached step 250 at `8.5087`, worse than
bounded0.1 hashseed1 without layerdelta (`8.5000`). It was killed. Layerdelta
does not simply compose with the favorable hashseed1 combine path; it degrades
the early trajectory.

The no-hashseed layerdelta control was also negative: step 250 `8.5029`, step
500 `7.2483`. It was killed at 500. Layerdelta is therefore not just
non-compositional with hashseed1; it is actively bad for the combine-mix path in
both hash layouts tested.

One code fix/generalization was made before the next run:
`ENGRAM_SKETCH_HIT_HIST_BASE_ONLY` previously applied only in the slot-readout
path. The normal combine path now also records hit history on the base sketch
slot only when that flag is enabled.

The new run on the freed GPU is:

- `bf300_hashseed1_sketchk2_combinemix_bounded01_dimsigns_balanced_base_aux05_basehist_seed5_1500_20260611`

This tests whether auxiliary sketch rows contaminating hit history/read-hit
statistics is part of the late fade in bounded combine-mix.

Hashseed1 base-hit-history reached step 250 at `8.5113`, much worse than the
regular hashseed1 bounded combine-mix (`8.5000`). It was killed. Restricting
hit history to the base sketch slot changes read-hit statistics substantially
and hurts early optimization; auxiliary sketch rows are not merely toxic noise
in the hit history.

The no-hashseed base-hit-history control is different: step 250 `8.5036`,
better than default no-hash bounded combine-mix (`8.5065`). At step 500 it
reaches `7.2269`, a strong positive result: better than default no-hash
combine-mix (`7.2360`), hashseed1 bounded combine-mix (`7.2321`), and the
earlier slot-mix + extra scaffold (`7.2275`). At 750 it reaches `5.3513`; the
large 500-step advantage narrows, but the run remains competitive enough to
carry to 1000. At 1000 it reaches `3.4997`, the best 1000-step point in the
combine-mix family so far.

A BF400 no-hash base-hit-history scale-out was launched in parallel:

- `bf400_sketchk2_combinemix_bounded01_dimsigns_balanced_base_aux05_basehist_seed5_1500_20260611`

This asks whether base-only hit history helps the default hash layout and wider
table, even though it hurts hashseed1.

BF400 no-hash base-hit-history checkpoints:

- Step 250 `8.5031`, step 500 `7.2376`, step 750 `5.3512`.

This is also positive: better than BF400 default-layout combo (`8.5056`) and
BF400 hashseed1 base (`8.5063`). At 500 it remains better than the BF400
default-layout combo path, but worse than BF400 hashseed1 combine-mix (`7.2349`).
By 750 it is much better than BF400 hashseed1 combine-mix (`5.3555`) and remains
alive.

BF300 no-hash base-hit-history completed:

- Step 1250 `3.3449`, step 1500 `3.2454`.

This did not beat the current record (`3.2440`), but it is the best completed
combine-mix/basehist branch and it beat the earlier hashseed1 bounded
combine-mix final (`3.2462`). The interpretation is narrower now: base-only hit
history in the default hash layout improves the mid-training path and slightly
improves the completed combine-mix result, but it is still not enough to move
the SOTA final-loss line.

BF400 no-hash base-hit-history reached step 1250 at `3.3464`, behind BF300
base-hit-history at the same point (`3.3449`). The run was left alive to get the
final point, but the wider table is not showing a late advantage on this branch.

BF400 no-hash base-hit-history completed at `3.2455`. This is effectively tied
with BF300 base-hit-history (`3.2454`) and still behind the current record
(`3.2440`). On this branch, scaling BF from 300 to 400 did not produce a late
loss gain.

Because base-hit-history helped the default-layout combine-mix branch but hurt
hashseed1, a direct current-SOTA control was launched:

- `bf300_sota_k2_headmix_layerdelta_norowsigns_hashseed0_basehist_seed5_1500_20260611`

This isolates whether recording hit history only on the base sketch slot helps
the existing no-row-sign/layerdelta/headmix SOTA recipe without the combine-mix
intervention.

The direct current-SOTA base-hit-history control reached step 250 at `8.5179`,
exactly matching the original seed5 no-row-sign control at step 250. This is a
useful negative/identity check: for the normal SOTA path, the base-only
hit-history flag appears to be either behaviorally inactive at this point or
deterministically equivalent through the first 250 steps. It is being carried to
500 to confirm whether it remains an identity path.

At step 500 it again exactly matched the original control (`7.2347`). The run
was stopped because the flag appears to be an identity path for the normal SOTA
configuration. This also explains why base-only hit history mattered for
combine-mix but not for the current SOTA recipe: the ordinary superpose read
path is already behaving as if the auxiliary slot is not changing the
hit-history/read-hit-scale state that affects training.

GPU1 was reassigned to a seed8 current-SOTA control:

- `bf300_sota_k2_headmix_layerdelta_norowsigns_hashseed0_seed8_1500_20260611`

The goal is to refine the seed-variance estimate around the current
`3.2440-3.2452` cluster, so small differences from basehist/combine-mix can be
interpreted against a better noise floor.

After the identity-path check was stopped, GPU0 was assigned a matching seed9
current-SOTA control:

- `bf300_sota_k2_headmix_layerdelta_norowsigns_hashseed0_seed9_1500_20260611`

Seed-sweep early checkpoints:

- Seed8 step 250 `8.5069`.
- Seed9 step 250 `8.5021`.

Both are substantially better than the seed5 control at step 250 (`8.5179`).
This reinforces the caution from earlier: early and mid-training deltas of a few
thousandths are not reliable on one seed, and even differences around `0.01` can
appear before final convergence. The seed8/seed9 runs are being carried to final
to tighten the final-loss noise estimate.

At step 500, the seed paths diverged:

- Seed8 step 500 `7.2539`, worse than seed5 control (`7.2347`).
- Seed9 step 500 `7.2323`, better than seed5 control (`7.2347`).

The seed8 early advantage did not persist, while seed9 remains a potentially
strong seed. Both are still being carried because this sweep is measuring
variance rather than selecting by early checkpoint.

At step 750:

- Seed8 `5.3501`.
- Seed9 `5.3483`.

Seed9 remains the stronger of the two, but the gap has narrowed into the normal
SOTA-seed band. The final result is likely to matter more than the early
ordering.

At step 1000:

- Seed8 `3.5018`.
- Seed9 `3.4990`.

Seed9 still leads seed8, but this does not yet indicate a clear record-setting
trajectory. The main value remains estimating the final-loss seed spread.

At step 1250:

- Seed8 `3.3442`.
- Seed9 `3.3440`.

The two runs have reconverged near the existing SOTA trajectory. Seed9 remains
slightly better, but only the final checkpoint can decide whether this is a
record or just normal seed variation.

Seed8 completed at `3.2444`, inside the current SOTA seed band but not a new
record. Seed9 was still running at the time of this update. With GPU0 free after
seed8 completed, a seed10 current-SOTA control was launched:

- `bf300_sota_k2_headmix_layerdelta_norowsigns_hashseed0_seed10_1500_20260611`

Seed9 completed at `3.2448`, also inside the seed band and not a new record.
The current completed seed sweep now has seed5 `3.2440`, seed6 `3.2452`, seed7
`3.2451`, seed8 `3.2444`, and seed9 `3.2448`. That puts the observed final
spread at about `0.0012` across these five SOTA-control seeds.

The first seed10 launch accidentally overlapped seed9 on the same GPU and OOMed
during model move-to-CUDA. Its failed startup logs were removed and seed10 was
relaunched on the free GPU1. A seed11 current-SOTA control was also launched on
GPU0:

- `bf300_sota_k2_headmix_layerdelta_norowsigns_hashseed0_seed11_1500_20260611`

Seed10/seed11 early checkpoints:

- Seed10 step 250 `8.5116`.
- Seed11 step 250 `8.5107`.

Both are in the normal early band: better than seed5 at 250, worse than seed9
at 250. They are being carried to final for variance measurement.

At step 500:

- Seed10 `7.2466`.
- Seed11 `7.2452`.

Both are middling and worse than the strongest seed9 500-step value (`7.2323`),
but still useful for the final-loss variance estimate.

At step 750:

- Seed10 `5.3541`.
- Seed11 `5.3530`.

Both are trailing the stronger seed trajectories. They are unlikely to set a new
record, but should still complete to extend the variance sample.

At step 1000:

- Seed10 `3.5048`.
- Seed11 `3.5004`.

Seed11 recovered toward the main band, while seed10 remains weak. Neither looks
like a record candidate; both remain useful as variance samples.

Final seed10/seed11 results:

- Seed10 final `3.2478`.
- Seed11 final `3.2461`.

The completed SOTA-control seed sweep is now:

- Seed5 `3.2440`.
- Seed6 `3.2452`.
- Seed7 `3.2451`.
- Seed8 `3.2444`.
- Seed9 `3.2448`.
- Seed10 `3.2478`.
- Seed11 `3.2461`.

Best remains seed5 at `3.2440`. The observed spread across seed5-11 is
`0.0038`, with most seeds in the `3.2440-3.2461` band and seed10 as a weaker
tail. This puts the completed basehist/combine-mix results (`3.2454`/`3.2455`)
comfortably inside normal SOTA-control seed variation, not a reliable regression
or improvement.

Final synced structural summary: 159 parsed runs, 36 complete. Both remote GPUs
were idle after seed10/seed11 completed.

## 2026-06-11: layerdelta-scale combine-mix tests

The seed sweep established that final-loss differences below a few thousandths
are not reliable on one seed: seed5-11 SOTA controls span `3.2440-3.2478`.
Therefore the next iteration should test a structural hypothesis with a plausible
effect larger than that noise floor.

The strongest new non-SOTA branch was bounded count-sketch combine-mix with
base-only hit history: BF300 `3.2454`, BF400 `3.2455`. Full layer readout delta
does not compose with combine-mix: previous no-hash layerdelta combine-mix was
already bad by step 500 (`7.2483`), and hashseed1 layerdelta combine-mix was bad
by step 250 (`8.5087`). This suggests a possible scale/timing conflict rather
than a guaranteed structural incompatibility.

A new `ENGRAM_LAYER_READOUT_DELTA_SCALE` schedule was added. Default behavior is
unchanged at scale `1.0`; the scale multiplies only the residual per-layer
readout delta projection, not the shared readout projection. Two BF300
base-hit-history combine-mix runs were launched:

- `bf300_sketchk2_combinemix_bounded01_dimsigns_balanced_base_aux05_layerdelta_scale025_basehist_seed5_1500_20260611`
- `bf300_sketchk2_combinemix_bounded01_dimsigns_balanced_base_aux05_layerdelta_0to1_s500_500_basehist_seed5_1500_20260611`

The first asks whether a weak per-layer residual can improve the final basin
without disrupting the combine-mix path. The second asks whether the conflict is
temporal: let combine-mix shape the early representation, then ramp layerdelta
from `0` to `1` over steps `500-1000`.

Initial layerdelta-scale results:

- Scale `0.25`: step 250 `8.5100`, step 500 `7.2441`.
- Ramp `0 -> 1`: step 250 `8.5036`, step 500 `7.2269`.
- Matched base-hit-history combine-mix baseline: step 250 `8.5036`, step 500
  `7.2269`, step 750 `5.3513`, final `3.2454`.

Scale `0.25` is clearly worse by step 500, so it was stopped. The ramp run
exactly reproduces the base-hit-history branch through step 500, as expected
because its layerdelta scale is still zero before then. The meaningful test is
step 750 and later, after the residual starts entering the readout.

Ramp step 750 landed at `5.3503`, slightly better than the matched baseline
`5.3513`. The margin is small, but this is the first combine-mix + layerdelta
variant that did not immediately degrade once layerdelta entered, so it remains
worth running to at least step 1000.

Ramp step 1000 is `3.5026`, worse than the matched baseline `3.4997` by about
`0.003`. That weakens the ramp hypothesis, but the gap is still close to the
SOTA-control noise band and the 750 checkpoint was slightly positive, so the run
is being carried to 1250 once before deciding whether to spend the final 250
steps.

Ramp step 1250 is `3.3446`, slightly better than the matched baseline `3.3449`.
The effect is tiny, but the branch recovered after the weaker 1000 checkpoint,
so it is being carried to final.

Ramp final is `3.2453`, a hair better than the matched base-history combine-mix
baseline (`3.2454`) but still behind the current SOTA-control seed5 result
(`3.2440`) and inside seed noise. The result says that scheduled layerdelta can
compose with combine-mix, unlike full-strength layerdelta from the start, but
the amplitude/timing is not yet a decisive SOTA mover.

Two lower-amplitude siblings are running:

- `bf300_sketchk2_combinemix_bounded01_dimsigns_balanced_base_aux05_layerdelta_0to05_s500_500_basehist_seed5_1500_20260611`
- `bf300_sketchk2_combinemix_bounded01_dimsigns_balanced_base_aux05_layerdelta_0to025_s500_500_basehist_seed5_1500_20260611`

They test whether the `0 -> 1` ramp has the right timing but too much late
residual amplitude.

Ramp `0 -> 0.5` reached step 750 at `5.3499`, slightly better than both the
`0 -> 1` ramp (`5.3503`) and the matched base-history combine-mix baseline
(`5.3513`). This is still a tiny margin, but it supports the amplitude hypothesis
enough to continue to step 1000.

Ramp `0 -> 0.5` reached step 1000 at `3.5000`, essentially tied with the matched
baseline (`3.4997`) and better than the `0 -> 1` ramp (`3.5026`). This suggests
the full-amplitude ramp was too strong by the middle of training; continue the
0.5 branch to 1250.

Ramp `0 -> 0.25` reached step 750 at `5.3502`: better than the matched baseline
(`5.3513`), about tied with ramp `0 -> 1` (`5.3503`), and slightly behind ramp
`0 -> 0.5` (`5.3499`). Carry it to 1000, but the best amplitude signal so far
is `0 -> 0.5`.

The freed GPU was moved to a scheduled latent auxiliary readout probe. The
first launch used the generic superpose wrapper and was stopped before producing
data because that wrapper leaves `ENGRAM_HEAD_MIX=0`. It was relaunched with the
matched no-partition/headmix fixed path:

- `bf300_latentfsq_aux010to0_s250_250_nopart_headmix_seed5_1500_20260611`

This revisits the latent/adaptive-addressing result with a decay schedule. Prior
constant FSQ latent aux scale `0.1` helped early against the no-partition
control (`8.5065` vs `8.5179` at step 250), but hurt by step 500 (`7.2456` vs
`7.2347`). The new run keeps the fixed ngram path, uses no layer partitions
because latent addressing does not support partitions yet, and decays latent aux
scale from `0.1` to `0.0` over steps `250-500`.

Latent headmix step 250 is `8.5065`, reproducing the early-help signal against
the no-partition/headmix control. It should not be judged until step 500 because
the auxiliary scale only starts decaying after the 250 checkpoint.

Latent headmix step 500 is `7.2473`, worse than the no-partition/headmix control
(`7.2347`) and slightly worse than the earlier constant-scale latent aux result
(`7.2456`). Decaying from 250 to 500 therefore does not rescue the branch: the
adaptive/latent path has already shaped a worse trajectory by the time it is
removed.

A shorter scaffold follow-up was launched:

- `bf300_latentfsq_aux010to0_s0_250_nopart_headmix_seed5_1500_20260611`

This keeps the same fixed no-partition/headmix path but decays latent aux from
`0.1` to `0.0` during steps `0-250`, so the adaptive path is gone by the first
real validation checkpoint. The question is whether latent addressing can act as
a very early initialization scaffold without corrupting the 250-500 trajectory.

The early-scaffold latent run reached step 250 at `8.5210`, worse than the
no-partition/headmix control (`8.5179`) and much worse than keeping latent aux
active through step 250 (`8.5065`). It was stopped. The latent result is now
fairly clear: adaptive addressing can provide an active early readout benefit,
but removing it either before or after step 250 does not preserve that benefit,
and keeping it long enough to help early hurts the 500-step trajectory.

GPU0 was moved to a lower-amplitude layerdelta ramp:

- `bf300_sketchk2_combinemix_bounded01_dimsigns_balanced_base_aux05_layerdelta_0to05_s500_500_basehist_seed5_1500_20260611`

This keeps the successful combine-mix/basehist branch and ramps layerdelta from
`0` to `0.5` over steps `500-1000`. It directly tests whether the `0 -> 1` ramp
has the right timing but too much late residual amplitude.

The `0 -> 0.5` ramp completed. Checkpoints:

- step 750: `5.3499` versus combine-mix/basehist `5.3513`
- step 1000: `3.5000` versus combine-mix/basehist `3.4997`
- step 1250: `3.3443` versus combine-mix/basehist `3.3449`
- step 1500: `3.2450` versus combine-mix/basehist `3.2454`

This is a tiny improvement over the combine-mix/basehist control and the full
`0 -> 1` ramp (`3.2453`), but not an improvement over the seed-5 no-row-sign
SOTA control (`3.2440`). The useful signal is mostly structural: lower
layerdelta amplitude does not collapse the branch and may be preferable to full
amplitude, but the final delta is well inside the measured seed spread.

GPU1 is running an even smaller `0 -> 0.25` ramp with the same schedule:

- `bf300_sketchk2_combinemix_bounded01_dimsigns_balanced_base_aux05_layerdelta_0to025_s500_500_basehist_seed5_1500_20260611`

The `0 -> 0.25` ramp reached step 1000 at `3.4995`, essentially tied with the
combine-mix/basehist control and slightly ahead of the `0 -> 0.5` ramp at that
checkpoint. It then reached step 1250 at `3.3441`, the best 1250 checkpoint in
this small amplitude sweep (`3.3449` basehist, `3.3446` full ramp, `3.3443`
0.5 ramp). Final was `3.2451`, so the mid-run edge did not translate into a
SOTA improvement. The lower amplitudes compose slightly better than immediate
full layerdelta, but all fixed/ramped layerdelta variants here end in the same
`3.2450-3.2454` cluster.

Based on that shape, GPU0 was moved to a learned layerdelta-scale run:

- `bf300_sketchk2_combinemix_bounded01_dimsigns_balanced_base_aux05_layerdelta_learnscale025_s500_500_basehist_seed5_1500_20260611`

This adds a bounded learned per-layer value/key scale for the layerdelta branch,
initialized at `0.25`, under the same outer `0 -> 1` schedule over steps
`500-1000`. The effective initial amplitude after the ramp is therefore the
same as the current best-looking fixed `0 -> 0.25` run, but the model can move
value and key residual strengths independently for layers 2 and 8. The code
logs `engram_layerdelta_l{layer}_value_scale` and
`engram_layerdelta_l{layer}_key_scale` so we can see whether the branch wants
more value residual, more key residual, or less layer specialization.

Startup was verified after launch: step 0 logged all four learned scales at
`2.500e-01` (`l2` value/key and `l8` value/key), and the run was active on GPU0
with the intended W&B offline run name.

As expected, the learned-scale run exactly matches the basehist control before
the outer layerdelta ramp begins: step 250 `8.5036`, step 500 `7.2269`. The
learned value/key scales also remain at `2.500e-01` through step 500, because
the outer delta scale is still zero and the learned logits should only become
behaviorally relevant after the 500-step ramp starts.

At step 750, learned-scale reaches `5.3524`, worse than the fixed low-amplitude
ramps (`5.3502` for `0 -> 0.25`, `5.3499` for `0 -> 0.5`) and the matched
basehist control (`5.3513`). The learned gates moved downward rather than
upward: `l2` value/key `0.1485/0.1528`, `l8` value/key `0.1930/0.1687`. That
is useful evidence, but not a loss win: when given freedom, the model reduces
layerdelta amplitude, and the resulting trajectory is weaker than the fixed
ramp variants so far.

At step 1000, learned-scale recovered to `3.4994`, slightly ahead of the
matched combine-mix/basehist control (`3.4997`) and the fixed low-amplitude
ramps (`3.4995` for `0 -> 0.25`, `3.5000` for `0 -> 0.5`). The learned gates
continued downward: `l2` value/key `0.1114/0.1039`, `l8` value/key
`0.1800/0.1431`. This is still inside the known seed/noise band, but it is a
cleaner mechanistic result than the loss alone: the model prefers a small
layerdelta branch, with layer 8 retaining more branch strength than layer 2.
Keep it to 1250/final rather than killing at 1000.

At step 1250, learned-scale is `3.3448`, effectively back to the basehist
control (`3.3449`) and behind the fixed low-amplitude ramps (`3.3441` for
`0 -> 0.25`, `3.3443` for `0 -> 0.5`). The scales kept the same asymmetry:
`l2` value/key `0.1024/0.0826`, `l8` value/key `0.1844/0.1458`. The mechanism
still says "small layerdelta, more retained at layer 8 than layer 2", but the
loss no longer argues for it as a SOTA path.

GPU1 was moved to a non-layerdelta structural probe:

- `bf300_sketchk3_combinemix_bounded01_dimsigns_balanced_base_aux03536_basehist_seed5_1500_20260611`

This is the current count-sketch bounded combine-mix/base-hit-history branch
with `ENGRAM_SKETCH_K=3`. The two auxiliary sketch rows use scale
`0.3535533906`, keeping total auxiliary energy close to the `k=2`, aux `0.5`
branch while testing whether a more distributed code improves collision
averaging without simply increasing the auxiliary readout norm.

Startup was verified: the remote config logged `engram_sketch_k=3`,
`engram_sketch_include_base=1`, `engram_sketch_combine_mix=1`, aux scale
`0.3535533906`, and `hit_hist_base_only=1`; GPU1 reached full training
allocation.

Important prior: two older `ENGRAM_SUPERPOSE_K=3` branches had promising step
250 losses but faded by 500/750. This run is therefore not a generic "try k3
again"; it specifically asks whether the newer signed count-sketch +
combine-mix/basehist readout avoids the old k3 dilution/interference failure.

K3 first checkpoint: step 250 `8.5038`. That is essentially tied with the
matched K2 combine-mix/basehist control (`8.5036`) rather than an early win.
At step 500 it reaches `7.2285`, slightly worse than K2 combine-mix/basehist
(`7.2269`) but much better than the prior superpose-K3 aux-only failure
(`7.2503`). This means count-sketch combine-mix avoided the old severe K3
dilution/interference failure, but it has not beaten the K2 baseline yet.

At step 750, K3 reaches `5.3498`, which is now slightly ahead of the matched K2
combine-mix/basehist control (`5.3513`) and tied with the best low-amplitude
layerdelta ramp at the same checkpoint (`5.3499`). This is not yet a SOTA claim,
but it is the first useful evidence that the newer count-sketch combine-mix
implementation may benefit from a more distributed code after the 500-step
transition. Keep it to 1000/final.

At step 1000, K3 fades to `3.5006`, worse than the matched K2 combine-mix
basehist run (`3.4997`) and the learned-scale checkpoint (`3.4994`). The 750
signal was real enough to continue to final, but the current read is that K3
helps collision averaging transiently and then pays a dilution/optimization
cost once the main model has caught up.

While these runs continued, the code was prepared for a follow-up structural
probe: `ENGRAM_SKETCH_AUX_LEARNED_SCALE`. It adds a default-off learned global
multiplier on top of the scheduled sketch auxiliary scale and logs
`engram_sketch_aux_learned_scale`. The reason to prefer this over another manual
aux-scale sweep is that K3's behavior may be an energy-calibration problem: the
bounded combine-mix logits can only redistribute slot weights around a normalized
mean, while this parameter can learn whether the auxiliary rows as a group
should be louder or quieter.

Related hashing/recsys note: this branch is consistent with the broad direction
of hash-embedding work, but our empirical constraint is sharper. Classic hash
embeddings combine multiple hashed rows instead of allocating one row per item
(Svenstrup et al., 2017, https://arxiv.org/abs/1709.03933). Recent
Probabilistic Hash Embeddings treat hashed embeddings stochastically for online
categorical streams and explicitly target fixed memory under dynamic vocabularies
(Li et al., 2025, https://arxiv.org/abs/2511.20893). Q-R/compositional
embeddings frame multiple small tables/partitions as a memory-efficient way to
retain category identity under compression (Shi et al., 2019,
https://arxiv.org/abs/1909.02107), and recent recsys compression surveys place
hashing/weight-sharing among the main table-compression families
(https://arxiv.org/abs/2408.02304). Our full latent-addressing replacement
already showed that flattening row hits alone is not enough; it destroyed too
much useful fixed ngram identity. The actionable version for engram is therefore
not "replace the deterministic hash", but "keep the deterministic ngram path and
add adaptive/hash-mixture signal weakly, temporarily, or through a learned
combiner."

## Host outage state, 2026-06-15

The two live runs did not produce locally synced final evals before the GPU host
went down:

- `bf300_sketchk2_combinemix_bounded01_dimsigns_balanced_base_aux05_layerdelta_learnscale025_s500_500_basehist_seed5_1500_20260611`
- `bf300_sketchk3_combinemix_bounded01_dimsigns_balanced_base_aux03536_basehist_seed5_1500_20260611`

Authoritative local synced evidence:

- Learned layerdelta-scale has a synced eval through step 1250:
  `3.3448`. The synced log continues with train steps through 1260. A later
  terminal snapshot before the outage showed training near step 1470, but that
  is not present in the local synced log/CSV and should not be treated as a
  completed final result unless recovered from the host disk.
- K3 count-sketch combine-mix has a synced eval through step 1000: `3.5006`.
  The synced log continues with train steps through 1010. A later terminal
  snapshot before the outage showed training around step 1226, but that is not
  present in the local synced log/CSV and should not be treated as a completed
  1250/final result unless recovered from the host disk.

If the host comes back, first action is to rsync `/root/modded-nanogpt/logs/`
again and regenerate `summary_structural_20260610.csv` plus the plots. If the
final evals are missing on disk, treat those two runs as interrupted at the last
synced evals above. The next prepared structural probe is the default-off
`ENGRAM_SKETCH_AUX_LEARNED_SCALE` branch on the K2 combine-mix/base-hit-history
setup, because the strongest unresolved sketch question is whether auxiliary
rows need a learned global amplitude rather than another fixed scalar sweep.
