# B10: candidate monosynaptic connections and factor selectivity

Write-up item B10 (`results/AIM1_FINDINGS_WRITEUP.md`). Unilateral 4factors recordings, 7 sessions.
Regenerate the figure: `python plot_B10.py` (reads only the CSVs here and `style.py`).
Rebuild the CSVs: `scripts/aim1_panels/build_B10.py` (repo git commit 6355c0a, branch restore-lost-work; ~40 s,
single-threaded).

## What the resolver and hub views predict (read this first)
- **Discriminating prediction.** A resolver that settles factor competition by local lateral inhibition needs
  factor-specific **inhibitory** connections between units that prefer competing options, i.e. inhibitory pairs with
  **dissimilar** (negatively correlated) selectivity profiles. That is the orange half-plane in panel c.
- **Non-discriminating.** Excitation between like-tuned units (or shared input making connected pairs similar) is
  expected under both the resolver and the hub view. Panels b and d therefore do not distinguish them.
- **Data.** Only 8 troughs were found, 5 of them in adequately powered pairs (all I->I). Their median similarity
  without location is -0.01 vs -0.00 for matched unconnected pairs (p = 0.51). Too few troughs to test the
  discriminating case either way.

## Panels
- **a. Example sharp peak.** CCG of Newton 08_15 unit 41 (non-classic) -> unit 67 (non-classic), 500 um apart,
  STG 0.012 (35,748 presynaptic spikes). Black: observed spike-pair counts. Grey line: interval-jitter (25 ms)
  expectation, computed analytically by `src/ccg.py`. Grey band: pointwise Poisson 0.05-99.95% interval around that
  expectation. No jitter-surrogate band was saved; the detection test sums lags 1-4 ms (shaded), so it is much more
  sensitive than this pointwise band. The pair was chosen by eye among the sharpest (narrow 1-ms peak, flat baseline).
  Many other flagged peaks ride on broad (+-10 ms) co-modulation.
- **b (main). Similarity without location.** Pearson r across 4 factors (identity, salience, reward, probability) of
  the two units' signed visual-epoch ROC selectivity. Black outline: sharp-peak pairs (n = 112). Grey: session- and
  distance-matched unconnected powered pairs (exact mixture of the matching pools, each connected pair's pool
  weighted equally). Solid / dashed lines: medians +0.43 vs +0.21.
- **c. Peaks vs troughs**, similarity without location per pair (points). Black bar = median. Grey box = 95% range of
  the matched-null median for that many pairs. Orange = resolver prediction for inhibitory pairs.
- **d (secondary). With location** (5 factors): +0.84 vs +0.56. This is dominated by shared spatial tuning of nearby
  units.

## Statistics (`stats.csv`)
| quantity | value |
|---|---|
| powered directed pairs (min detectable STG <= 0.01, zero-lag coincidence <= 0.1) | 12,304 |
| sharp-peak pairs | 113 (E->E 9, E->I 21, I->E 21, I->I 62); 80 units; 42 presynaptic units, up to 14 pairs each |
| without location: peaks vs matched | median +0.426 vs +0.214; pair-level p = 0.010 (2000 matched replicates; write-up: p = 0.013 with 1000) |
| without location, unit level | per-presynaptic-unit median +0.34 vs null +0.17, **p = 0.076**; difference of medians +0.21, unit-bootstrap 95% CI **[-0.10, +0.46]** |
| with location: peaks vs matched | +0.838 vs +0.562, p < 0.0005 (0/2000); unit level p < 0.0005; difference CI [+0.14, +0.42] |
| troughs | 8 listed (I->I 7, I->E 1); 5 powered (all I->I) |

Methods:
- Matched null: for each connected pair, an unconnected powered pair from the same session within +-100 um is
  drawn; the median r over the draws is one replicate (2000 replicates); p = fraction >= observed (one-sided).
  This is identical to `scripts/ic_b5_pair_selectivity.py`, applied to the 4-factor vector.
- Unit-level resampling (pairs share units):
  1. Unit-aggregated test: median per presynaptic unit, then the median over units, compared with the same
     aggregation of each null replicate.
  2. Cluster bootstrap over presynaptic units (5000 resamples). A fresh matched draw is taken per resample for the
     difference CI.
  Postsynaptic units also repeat, so this is the lenient choice of cluster.
- The "+0.43 vs +0.22, p = 0.013" in SUMMARY.md and the write-up was not saved anywhere; this is its first saved
  recomputation. The pair-level result reproduces. **It does not survive unit-level resampling** (p = 0.076, CI
  includes 0). The with-location result does survive.

## Column dictionary
- `B10_example_ccg.csv`: lag_ms (post relative to pre spike), observed_count, jitter_expected_count,
  poisson_lo_0.05pct / poisson_hi_99.95pct (pointwise band), plus pair identifiers, n_pre_spikes, stg_1to4ms.
- `B10_pairs_connected.csv`: one row per sharp-peak pair (all powered) and per trough (all 8; `powered` flags the
  5 used). Columns: session, pre_id/post_id (cluster ids), pre_type/post_type (E classic, I non-classic /
  putative interneuron), distance_um, stg (spike transmission gain, 1-4 ms), min_detectable_stg, r_no_location,
  r_5factor.
- `B10_matched_unconnected.csv`: measure (r_no_location / r_5factor), r, weight (1 / pool size), matched_to_pair
  (index of the connected peak pair whose pool it is in). Histogram = weighted counts.
- `B10_null_medians.csv`: pair_set (peak/trough), measure, level (pair / unit-aggregated), null_median_r (one per
  replicate).
- `stats.csv`: statistic, value, n, test, ci_low, ci_high, n_units, window, unit_class, session_filter.

## Provenance
- CCGs: `scripts/ic_b3_ccg.py` -> `results/intercollicular/b3_ccg/<session>.npy` (commit 8588c11). All ordered
  pairs on different channels; spikes over the span of trials stable for both units, ITIs included. 1-ms bins,
  25-ms interval jitter. Peak: Poisson p < 0.001 on 1-4 ms; asymmetric vs -4..-1 ms (binomial p < 0.001); excess
  larger than the 6-10 ms shoulders (`scripts/ic_b3_summary.py`). Trough: the mirror criteria.
- Selectivity: `results/selectivity_by_type/grid_v13_free_free.npz`, recorded columns. Visual epoch 50-250 ms,
  preferred location, verified trials within each neuron's stable period; selectivity = 2 x (AUC - 0.5).
- Pair-to-row mapping: spike-cache unit ids (`data/spike_cache/*.npz`, v13 verified export order).
