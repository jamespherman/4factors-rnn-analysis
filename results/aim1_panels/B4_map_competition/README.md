# B4. The expected location when the target appears elsewhere, and pre-saccadic suppression by target angle

`B4_map_competition.png` / `.pdf`. To regenerate, run `python plot_B4.py`. It reads only the CSVs in this folder and
`style.py`.

## Panels
**a, b. Population PSTHs, classic units (E, n = 50) and putative interneurons (I, n = 90).**
- Units: those whose preferred contralateral location P is the probability location (export location 1).
- Trials: the target appeared at one of the two **90-deg neighbours** of P. Those locations have 12.5% probability
  in both block types.
- Top plot: rate minus the unit's pre-target baseline, mean +- SEM across units, 20-ms bins. Two block types:
  - black: the expected (high-probability) location is the unit's RF;
  - grey: the expected location is the opposite one.
- Bottom plot: the block difference (black minus grey), with a 95% t CI across units.
- Text: the 0-50 ms block difference and the 200-250 ms level re baseline, in the 50-ms windows of
  `cp_p2_conflict.py`.

**c. % of units with significant pre-saccadic suppression, by target angle from P.**
- Suppression: the -50..0 ms window re saccade onset is below the pre-target baseline (-200..0 ms), paired t-test
  p < 0.01, on trials with the target at that location.
- Units: classic (E, n = 72) and interneurons (I, n = 154).
- Target angles: 90 deg contra-neighbour, 90 deg ipsi-neighbour, 180 deg opposite.
- Error bars: Wilson 95% CI.
- Text: exact McNemar test, opposite vs ipsi-neighbour.

## Results
- **Head start at the expected location.**
  - I units: +1.17 sp/s at 0-50 ms (p = 0.005). It is gone by 50-100 ms (+0.23, p = 0.57).
  - E units: no detectable head start in any window (0-50 ms +0.16, p = 0.59).
  - Pooled: +0.81 sp/s (p = 0.005), as in `p2_conflict/summary.md`.
- **Dip below baseline.**
  - I units fall below baseline from about 150-200 ms: -1.2 sp/s at 150-200 ms (p = 0.15); -2.3 sp/s at 200-250 ms
    (p = 0.005); -1.9 at 250-300 ms (p = 0.005).
  - E units return to baseline: -0.03 at 200-250 ms (p = 0.91).
  - Pooled: -1.5 sp/s at 200-250 ms (p = 0.005). This is the write-up's "-1.5 sp/s". At 150-200 ms the pooled value
    is -0.48 (p = 0.38).
  - The dip is the same in both block types, so it does not depend on the prior.
- **Suppression by angle** (matches `g1_geometry/summary.md`, presac window):
  - I: opposite 33% (51/154) vs 19% (30/154) at each 90-deg neighbour; McNemar 31:10, p = 0.0015.
  - E: 12% (9/72) vs 3% (2/72); 8:1, p = 0.039.

## Hub vs resolver
The data are **compatible with both views and do not discriminate between them.**
- A resolver uses SC's map-wide competition to decide.
- A hub uses the same spatial winner-take-all to execute a choice made upstream. Inputs that update at target onset
  would also cancel the head start.

For that reason no model contrast is drawn.

## Files and columns
- `B4_psth.csv` (plotted curves). Columns:
  - `unit_type`;
  - `curve`: `expected_at_RF`, `expected_opposite` (both re baseline), or `difference_RF_minus_opposite` (raw rates);
  - `bin_center_ms` (re target onset);
  - `mean_sp_s`, `sem_sp_s`;
  - `ci95_lo` and `ci95_hi`, for the difference curve only;
  - `n_units`.
- `B4_unit_psth.csv`: per unit x block x bin. `rate_sp_s`, `rate_minus_baseline_sp_s`, `bin_start_ms`,
  `bin_center_ms`.
- `B4_units.csv`: per unit.
  - `baseline_sp_s`: mean rate -300..0 ms re target onset, all unit-stable trials.
  - Trial counts per block type.
  - `w50_<start>_diff_sp_s`: block difference in 50-ms windows.
  - `w50_<start>_level_sp_s`: rate re baseline on neighbour-target trials, both blocks, in 50-ms windows.
- `B4_suppression.csv` (bars): `unit_type`, `target_location`, `angle_from_P_deg`, `n_suppressed`, `n_units`,
  `pct_suppressed`, `ci95_lo_pct`, `ci95_hi_pct`.
- `stats.csv`: every quoted statistic, with window, unit class and session filter.

## Provenance
- PSTH: re-run of the `cp_p2_conflict.py` (b) logic. The saved `p2_conflict/units.npy` holds only the block
  difference.
  - Data: verified successful trials, unit-stable trials.
  - Sessions: 6 diagonal-target sessions (Newton 08_15 excluded).
  - Saccades occur at least 490 ms after target onset, so the -300..+400 ms window is pre-saccadic.
- Bars: `results/intercollicular/g1_geometry/rows.npy` (`scripts/ic_g1_geometry.py`; the same sessions, all units).
- Builder: `scripts/aim1_panels/build_B4.py`, git `6355c0a`.
