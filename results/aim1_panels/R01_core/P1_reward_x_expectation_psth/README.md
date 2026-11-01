# P1. Reward cue x location expectation: visual response of classic SC units (4factors)

`P1_reward_x_expectation_psth.png` / `.pdf`. Regenerate with `python plot_P1.py` (reads only the CSVs here and `style.py`).

## Panel
- Classic (E) units, n = 50, target at the unit's preferred contralateral location (RF).
- Four curves: reward cue (high = black, low = grey) x target location (expected = solid: the RF is the block's
  high-probability location; unexpected = dashed).
- Units: **rate minus the unit's pre-target baseline** (sp/s; baseline = mean rate -300..0 ms re target onset over the
  same trials). 20-ms bins. Mean across units (SEM in `P1_psth.csv`, not drawn, to keep the panel clean).
- Grey bar: the P3 visual window (50-250 ms) where the original test was done: reward x expectation interaction
  +0.27 SD (95% CI 0.16 to 0.38), t-test across units p = 7e-6, n = 50.
- Darker segment (60-130 ms): the contiguous run of P4 40-ms windows (10-ms steps) around the interaction peak with
  pointwise p < 0.01 (uncorrected). Peak +0.19 SD at 80 ms (p = 0.002). Other isolated p < 0.01 windows (centres 180,
  240, 250 ms) are listed in `P1_interaction_timecourse.csv` but not drawn.

## Cell means, 50-250 ms, sp/s re baseline (mean, 95% CI across units)
| | expected | unexpected |
|---|---|---|
| high reward | 7.9 (4.8-11.0) | 6.9 (4.1-9.6) |
| low reward | 6.3 (3.6-9.0) | 8.2 (5.3-11.1) |

The interaction is a **crossover**: high reward, expected > unexpected (+1.0 sp/s); low reward, unexpected > expected
(-1.9 sp/s). Interaction contrast +2.9 sp/s (95% CI 1.4 to 4.4, p = 4e-4). The high-reward expected curve has the
highest visual peak (16.9 sp/s at 70 ms), but **high-reward unexpected is not the lowest; low-reward expected is.**
Raw cell means do not balance stimulus type (image/bullseye, salience); the P3 coefficient (OLS with stimulus terms,
sqrt count, / residual SD) is the controlled statistic.

## Files
- `P1_psth.csv`: `reward_cue`, `target`, `bin_center_ms`, `mean_sp_s`, `sem_sp_s`, `n_units` (plotted curves).
- `P1_unit_psth.csv`: per unit x cell x bin, `rate_minus_baseline_sp_s`.
- `P1_units.csv`: per unit baseline, trial counts per cell, visual-window cell means, and the original P3
  `p3_visual_reward_x_expected_sd` (unit set checked identical to `p3_interactions/units.csv`).
- `P1_cell_means.csv`: the table above (with SEM and total trials per cell).
- `P1_interaction_timecourse.csv`: P4 reward x expected coefficient per 40-ms window: mean, 95% t CI, p, `sig_p01`.
- `stats.csv`: every quoted number with test, window, unit class, session filter, source.

## Provenance
Selection as `scripts/cp_p3_interactions.py` / `cp_p4_timing.py`: 6 diagonal-target sessions (Newton 08_15 excluded),
verified successful trials, unit-stable trials, target at P (g1_geometry), >= 40 trials, unit saw both block types.
High reward = rewardDuration > 200. Builder `scripts/aim1_panels/build_R01_P1.py`, git e217f9f (~1 min).

## P1b (recommended for the R01): expected minus unexpected, by reward cue
`P1b_expected_minus_unexpected.png/.pdf`, regenerate with `python plot_P1b.py` (reads `P1_unit_psth.csv`,
`P1_units.csv`; writes `P1b_expected_minus_unexpected.csv`, `P1b_window_bars.csv`). Per unit: expected-target minus
unexpected-target response (baseline-subtracted sp/s), mean +- SEM across the same 50 classic units; display smoothed
with a 3-bin (60 ms) boxcar. 50-250 ms: high-reward cue +1.0 +- 0.56 sp/s, low-reward cue -1.9 +- 0.54 sp/s
(interaction +2.9 sp/s, p = 4e-4; controlled estimate +0.27 SD, p = 7e-6). The reward cue flips the sign of the
expectation effect on the visual response, mirroring the behavioural crossover (B1).
