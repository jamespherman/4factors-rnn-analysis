# B5. Does pre-target SC activity predict RT location-specifically?

Figure: `B5_rt_state_specificity.png` / `.pdf`. To regenerate it, run `python plot_B5.py`. It reads only the CSVs in this folder and `style.py`.

## Panel
A forest plot of the RT change per SD of pre-target rate. Each point is 100 x the OLS coefficient on log RT, with its 95% CI. Filled points have p < 0.05.

**Predictor.** For each unit, the rate from -300 to 0 ms before target onset, z-scored within block type. The z-scores are averaged per trial over a group of units.

**Model.** log RT ~ session + target location + reward cue + z. This is plain OLS, as in the source script. A p-value from cluster-robust SEs by block run is also given in the CSV.

**Rows, grouped by where the units' RF (P) is relative to the target:**

| Group | Rows |
|---|---|
| RF at the target | Target at the expected location (z_prior, target-at-expected trials). Target at an unexpected location (z_target). |
| RF at a non-target (competitor) site | Expected location with the target elsewhere (z_prior). Opposite location with the target expected (z_away). Non-probability location with the target expected (z_other). Opposite / non-probability location with the target elsewhere, restricted to trials where the target is not at the RF (z_away_nontarget, z_other_nontarget). |
| Joint model (partial coefficients) | z_prior + z_other, for target elsewhere and for target expected. |
| All units | z_all. |

**Shading.** The orange half-plane is shaded for the competitor rows. Local competition between SC sites predicts that when a non-target site is more active, the saccade to the target is *slower* (coefficient > 0).

## Results
Every estimable competitor coefficient is negative (faster RT), and none is positive.

| Unit group | % RT per SD | p | n |
|---|---|---|---|
| RF at expected location, target elsewhere | -5.7 | 0.014 | 880 |
| RF at opposite location, target expected | -4.7 | 0.010 | 1299 |
| RF at non-probability location, target expected | -3.0 | 0.006 | 2760 |
| RF at target (expected location) | -5.2 | 3e-5 | 1461 |
| **RF at target (unexpected location)** | **-1.2** | **0.50** | 809 |
| All units, target expected | -6.4 | 7e-6 | 2760 |
| All units, target elsewhere | -5.2 | 0.010 | 1654 |

- **The RF-at-unexpected-target coefficient is not significant.** cp_p2 (c) reports -0.8 (p = 0.83) for this group.
- Joint model, target elsewhere: z_prior is -6.0 (p = 0.015) with z_other held fixed; z_other is +0.6 (p = 0.76).
- Joint model, target expected: z_prior -4.1 (p = 0.002), z_other -3.3 (p = 0.015).
- **The source script's joint model (z_prior + z_away) cannot be estimated** (n = 0). Every unit's P is a contralateral location (export location 1 or 2). So z_prior exists only in blocks where location 1 is the high-probability location, and z_away only in blocks where location 3 is. The two are never defined on the same trial. Its output was never saved. The estimable joint model is z_prior + z_other.

## Reconciling the write-up's "-5 to -6% RT per SD whether the RF holds the target, the expected competitor, or neither"
The -5 to -6% range applies to these groups:
- RF at the expected location, whether it holds the target (-5.2) or is the competitor (-5.7).
- All units (-5.2 to -6.4).

The other groups do not fit that range:
- RF at neither the target nor the expected location: -3.0 (non-probability location) to -4.7 (opposite).
- RF at the target when the target is unexpected: -1.2, not significant.

The NOTE_P1_P2 figures are reproduced (-5.2, -5.7, -4.7, -6.4).

## Hub vs resolver
A local competition predicts positive coefficients for the competitor rows (shaded). The data are negative throughout, which fits a non-specific readiness or arousal signal.

The predictor is z-scored within block, so slow drift within a block is not removed. This works against neither view specifically.

## Files and columns
- `B5_coefs.csv` (plotted values), one row per estimate:
  - Model: `model` (single / joint), `target_cond` (high = target at the high-probability location; low = elsewhere), `regressor`.
  - Labels: `label`, `competition_role` (target / competitor / mixed).
  - Estimate: `coef_pct_per_sd`, `ci95_lo`, `ci95_hi`.
  - Tests: `p_ols`, `p_cluster_block`, `n_trials`, `sessions`.
- `B5_trials.csv`, one row per trial:
  - Identifiers and outcome: `session`, `trial_export_idx`, `block_run`, `rt_ms`, `log_rt`.
  - Task: `target_loc`, `high_p_loc`, `reward_high`, `cond`.
  - Predictors: `z_prior`, `z_away`, `z_other`, `z_all`, `z_target`, `z_away_nontarget`, `z_other_nontarget` (mean z of the group; NaN if the group is empty).
  - `n_units_*`: number of units in each group.
- `B5_units.csv`: units that contributed (`session`, `unit_id`, `unit_type`, `P_loc`).
- `stats.csv`: every quoted coefficient with its CI, plain and clustered p, n, test, window, unit class (E and I pooled) and session filter.

## Provenance
- Logic: `scripts/cp_p2c_rt_specificity.py`, which printed only; it is re-run here and the results saved. z_target is added from `scripts/cp_p2_conflict.py` (c).
- Data: verified successful trials, unit-stable trials. Sessions: 6 diagonal-target sessions (Newton 08_15 excluded). Unit RFs come from `results/intercollicular/g1_geometry/rows.npy`.
- Builder: `scripts/aim1_panels/build_B5.py`, git `6355c0a`.
