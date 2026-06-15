# Hot Row Literature Research For Engram, 2026-06-07

Question: looking through CS history, old ML, and recommender systems, what are
principled ways to handle Engram hot rows?

Context from current Engram evidence:

- Hot rows are loss-critical.
- Cold rows are nearly disposable after training; randomizing rows hit fewer
  than about 4 times barely changes eval.
- Naive hot-row dropout/splitting has not improved SOTA.
- Row-wise AdaGrad, FAL, and within-batch frequency normalization did not move
  SOTA.
- The current best is BF400 with scalar-row Adam, shared L2/L8 row space,
  memory/readout normalization, trigram-heavy rows, and hit-scaled reads.

## Literature Map

### Heavy-Hitter Algorithms

Old streaming algorithms are directly relevant because Engram row hits are a
stream of addresses with a very skewed tail.

Relevant ideas:

- Misra-Gries / frequent elements: maintain approximate counters for items whose
  frequency can exceed a threshold, with bounded memory.
- SpaceSaving: maintain top-k candidates and replace low-count counters,
  producing a compact heavy-hitter table.
- Count-Min Sketch: cheap approximate frequencies with overestimation error.
- Conservative update: reduce Count-Min overestimation by only incrementing
  minimum counters.

Engram translation:

- Keep the normal hash table for all rows.
- Maintain a small heavy-hitter summary online.
- Once a row crosses a frequency threshold, promote it to a special hot-row path
  instead of hoping the base hash table handles it.

Why this is more principled than our previous hot-row dropout:

- Dropout suppresses hot rows but does not give them better capacity.
- Heavy-hitter handling says first detect the hot rows, then reserve or adapt
  resources for them.

### Cache Replacement And Admission

Cache history separates two problems that Engram currently mixes:

- "What has been frequent recently?"
- "What deserves scarce memory?"

Relevant ideas:

- LFU: keep frequent items.
- LRU: keep recently used items.
- ARC: adaptively balances recency and frequency.
- TinyLFU: use a compact frequency sketch as an admission policy, not just an
  eviction policy.

Engram translation:

- Use hit history not only for analysis/read scaling but also as an admission
  policy.
- Cold rows could be reset/reused/compacted.
- Hot rows could get stable reserved storage.
- Mid-frequency rows could age out unless they keep accumulating evidence.

Important difference from ordinary caching:

- Engram rows are learned parameters, not just stored values.
- Eviction should preserve optimizer semantics. A reset row should probably
  reset scalar Adam state and hit count together.

### Balls-Into-Bins And The Power Of Two Choices

The "power of two choices" result says that allowing each item to choose the
less-loaded of two random bins dramatically reduces maximum load compared with
one random bin.

Engram translation:

- For each ngram, compute two candidate row addresses.
- During training, assign/update the less-loaded candidate, or use a stable
  deterministic rule based on current hit counts.
- During eval, either use the selected row recorded by a deterministic rule or
  read both with a small combine.

Potential issue:

- If assignment changes over training, row semantics drift.
- A safer version is "two-choice at first touch": choose the lower-hit row when
  a key is first seen, then keep that assignment via a small key-to-slot cache
  or deterministic secondary metadata.

Engram experiment shape:

- `ENGRAM_TWO_CHOICE_ON_FIRST_TOUCH=1`
- keep a small CPU/GPU key assignment cache for heavy/repeated ngrams,
- fallback to normal hash for unseen/rare keys.

### Feature Hashing And Signed Hashes

Feature hashing showed that large sparse feature spaces can be compressed by
hashing features into a fixed vector, often with signed hashes to reduce bias
from collisions.

Engram already uses hashed rows and we tested some sign variants. The literature
suggests two nuanced points:

- Signed collisions help when features are linear additive statistics.
- Learned embedding rows are different: a collision is not just additive noise;
  it forces multiple keys to share trainable content and optimizer state.

Engram implication:

- Signed row views may help early interference, which matches our layer-sign
  early results.
- But permanent learned-row collisions need capacity/assignment management, not
  just signs.

### Sparse Online Learning And Per-Coordinate Optimizers

Old ad-click / online ML literature used per-coordinate learning rates,
FTRL-Proximal, AdaGrad, and feature-frequency-aware regularization. The core
principle is that rare and frequent sparse features need different update
statistics.

Engram tried some optimizer analogs:

- row-wise AdaGrad,
- FAL,
- within-batch frequency normalization,
- scalar-row Adam.

Result so far:

- Scalar-row Adam is useful mostly because it saves memory while preserving
  row-wise adaptive scale.
- Simple row-wise AdaGrad/FAL were weak in this setup.

Interpretation:

- Engram's issue is not only optimizer scale.
- Hot rows are also overloaded semantic/collision centers.
- Per-row LR alone cannot create more representational capacity for a hot row.

### Recommender-System Embedding Practice

Industrial recommenders typically treat embedding tables as the main memory
consumer and use:

- row-wise sparse optimizers,
- frequency-aware regularization,
- sharded embedding tables,
- hot/cold partitioning,
- caching hot embeddings,
- collisionless or dynamic embedding tables for IDs,
- pruning/TTL for stale IDs.

Engram overlap:

- Row-wise optimizer state is already productive.
- Hot/cold partitioning is strongly suggested by our masking data.
- Dynamic/collisionless ID tables map to "special treatment for known hot
  ngrams" rather than pure hashing forever.

Key difference:

- Recsys IDs are persistent entities.
- Engram keys are ngrams/latent hashes; the same hash row may represent many
  keys due to collision. We may need key-level heavy-hitter tracking, not just
  row-level hit counts.

## Recommended Engram Experiments

### 1. Heavy-Hitter Sidecar Table

Highest-priority idea.

Design:

- Keep the current BF400 table unchanged.
- Maintain a top-k heavy-hitter key summary for actual ngram keys, not just row
  addresses.
- Promote the top heavy ngrams into a small sidecar embedding table.
- Read `base_hash_row + sidecar_hot_key_row` for promoted keys.
- Initialize sidecar rows from the current base row on promotion.

Why it fits the evidence:

- Hot rows are loss-critical.
- Cold rows are cheap.
- Naive hot splitting failed because it split by hot rows, not necessarily by
  hot keys inside those rows.
- Sidecar gives extra capacity to high-evidence keys without changing the whole
  hash table.

Implementation risk:

- Need key identity, not just row address.
- Need deterministic promotion behavior for train/eval/checkpoints.

First test:

- Offline from a checkpoint: identify top hot ngram keys if logs preserve them.
- If not, start with row-level sidecar as a lower-fidelity proxy.

### 2. Hot Row Sub-Bucketization By Secondary Hash

Simpler than full key sidecar.

Design:

- For rows above a hit threshold, route to `S` subrows using a secondary hash.
- Read base row plus/substitute subrow.
- Subrows share the same logical row family but split colliding traffic.

Difference from prior hot split:

- Make the split conditional and stable based on key hash.
- Track hits per subrow.
- Avoid tiny always-on aux weights; make the subrow a real capacity lane after a
  threshold.

Risk:

- If threshold is based on row hit count, many unrelated keys can be moved
  together.
- If based on key hit count, we need key summaries.

### 3. Two-Choice Hashing For New/Cold Keys

Design:

- Each key gets two candidate rows.
- On first touch, choose the less-loaded row.
- Keep assignment stable for repeated keys.
- For untracked rare keys, use deterministic lower-count rule or normal hash.

Why it might help:

- Reduces future hot rows by preventing early collision concentration.
- Acts before rows become overloaded.

Main challenge:

- Stable assignment memory can become expensive.
- Without stable assignment, training becomes nonstationary.

### 4. Hot/Cold Optimizer Split

Design:

- Keep scalar Adam for normal/cold rows.
- For hot rows, use lower LR, stronger weight decay, or vector moments.
- For cold rows, keep cheap scalar moments or even delayed updates.

Why lower priority:

- We already tried several optimizer-only hot-row ideas.
- But vector moments only for hot rows could be different: it adds directional
  optimizer capacity exactly where the table is loss-critical.

Potential first test:

- Hot rows with full vector `exp_avg` but scalar `exp_avg_sq`, or vice versa.
- Threshold at hit count `>=1024` or `>=4096`.

### 5. TinyLFU-Style Cold Row Reuse / Pruning

Design:

- Maintain approximate frequency sketch.
- Rows below admission threshold are reset/reused.
- Preserve a fixed hot set.

This is more memory/compression oriented than SOTA oriented.

Why useful:

- Our checkpoint masking says hit `<4` rows are nearly free.
- Could reclaim memory for a hot sidecar or vector moments.

Risk:

- During training, cold rows may become important later.
- Needs an aging schedule, not just absolute counts.

## What I Would Not Prioritize

- Pure row-wise AdaGrad reruns: already weak.
- Pure FAL reruns: already weak.
- More global LR scaling for hot rows: unlikely to solve collision/capacity.
- More hash-seed/avalanche probes: current evidence says hash quality is not
  the main bottleneck.
- Smooth sign handoffs: prior sign schedules caused semantic discontinuities.
- More full-table row count via smaller store dim: projection bottleneck and
  update starvation were already clear.

## Proposed Next Implementation Path

1. Add instrumentation to log key-level heavy hitters, not just row-level hits.
   Use a bounded SpaceSaving-style summary or sampled exact counts.
2. Analyze whether the hottest rows are dominated by one key or many colliding
   keys.
3. If one-key dominated: try hot-key sidecar rows.
4. If many-key dominated: try secondary-hash sub-bucketization.
5. Use cold-row pruning to pay for sidecar/vector state only after confirming
   the hot-key structure.

The key diagnostic is the entropy of key distribution inside hot rows. If hot
rows are mostly single ngrams, optimizer/read scaling is plausible. If hot rows
are mixtures of many keys, we need collision relief and sidecar capacity.

## References

- Misra and Gries, "Finding repeated elements", 1982.
  https://www.cs.utexas.edu/~misra/psp.dir/FindRepeatedElements.pdf
- Metwally, Agrawal, and El Abbadi, "Efficient Computation of Frequent and
  Top-k Elements in Data Streams", 2005.
  https://www.cse.ust.hk/~dimitris/6311/11_10_04/space_saving.pdf
- Cormode and Muthukrishnan, "An Improved Data Stream Summary: The Count-Min
  Sketch and its Applications", 2005.
  https://dimacs.rutgers.edu/~graham/pubs/papers/cm-full.pdf
- Mitzenmacher, "The Power of Two Choices in Randomized Load Balancing", 2001.
  https://www.eecs.harvard.edu/~michaelm/postscripts/mythesis.pdf
- Weinberger et al., "Feature Hashing for Large Scale Multitask Learning",
  2009.
  https://alex.smola.org/papers/2009/Weinbergeretal09.pdf
- Duchi, Hazan, and Singer, "Adaptive Subgradient Methods for Online Learning
  and Stochastic Optimization", 2011.
  https://jmlr.org/papers/v12/duchi11a.html
- McMahan et al., "Ad Click Prediction: a View from the Trenches", 2013.
  https://research.google/pubs/ad-click-prediction-a-view-from-the-trenches/
- Naumov et al., "Deep Learning Recommendation Model for Personalization and
  Recommendation Systems", 2019.
  https://arxiv.org/abs/1906.00091
- Einziger et al., "TinyLFU: A Highly Efficient Cache Admission Policy", 2017.
  https://arxiv.org/abs/1512.00727
- Megiddo and Modha, "ARC: A Self-Tuning, Low Overhead Replacement Cache",
  2003.
  https://www.usenix.org/conference/fast-03/arc-self-tuning-low-overhead-replacement-cache
- Liu et al., "Monolith: Real Time Recommendation System With Collisionless
  Embedding Table", 2022.
  https://arxiv.org/abs/2209.07663
