# B3. Do goal and stimulus factor effects line up across SC units? (visual window, E and I units separately)

Figure: `B3_factor_alignment.png` / `.pdf`. To regenerate, run `python plot_B3.py`. It reads only the CSVs in this
folder and `style.py`.

## Panels
- **a** Per-unit reward effect vs salience effect. **b** Per-unit image (vs bullseye) effect vs salience effect.
  - Classic units (E): filled black. Putative interneurons (I): open grey.
  - The text gives the within-class Spearman r, the within-unit trial-shuffle p, and n.
- **c** Matrices of cross-unit Spearman r between factor effects, one for E and one for I.
  - Lower triangle: full-data r. * marks trial-shuffle p < 0.05.
  - Diagonal (grey italic): split-half reliability of each effect.
  - "prior" = expected (the target is at the block's high-probability location).
- **d** For every pair and both classes:
  - dot: full-data r;
  - grey bar: the 95% band of the within-unit trial-shuffle null;
  - black tick: the reliability ceiling sqrt(SB_a x SB_b), with SB = Spearman-Brown full-data reliability. This is
    the r that two perfectly aligned effects would reach given how noisy each one is. It is the **SC priority map
    prediction**: one shared axis predicts r at the ceiling. This prediction is not specific to the resolver; a
    priority hub delivering one graded priority signal predicts the same.
  - No tick: a reliability is <= 0, so the ceiling is undefined.
  - Shaded rows: pairs that involve a goal factor.

## Method (recomputed; identical to the P3 effects)
Per unit, the model of `scripts/cp_p3_interactions.py` (all 226 units match `p3_interactions/units.csv`,
max |diff| 7e-15):
- Trials: target at the unit's preferred contralateral location P, unit-stable, at least 40 trials.
- Response: sqrt of the spike count in the 50-250 ms window after target onset.
- OLS on reward, image, salience (within bullseyes), face (within images) and reward x image. When the unit saw both
  block types, also expected and reward x expected.
- Effect = coefficient / residual SD.

Nulls and reliability:
- **Trial-shuffle null** (500 permutations): permute y across each unit's trials. This keeps the design and its
  coefficient covariances and removes every true effect. The r is recomputed on each permutation.
- **Unit-label permutation** (5,000): permute one effect across units. Gives `p_unit_perm_2sided` and
  `null_unit_perm_q*`; it agrees with the trial-shuffle result.
- **Split-half**: 50 random halvings of each unit's trials.
  - reliability = r(half 1, half 2) for one factor;
  - cross-half r = mean of r(a1, b2) and r(a2, b1), so no trial noise is shared between the two effects;
  - cross-half ceiling = sqrt(rel_a x rel_b).

## Key results (quoted in `stats.csv`)
- Stimulus-stimulus alignment is real in both classes:
  - image-salience: E r = +0.26 (p = 0.027; shuffle p = 0.046, n = 72); I r = +0.42 (p = 6e-8; shuffle
    p = 0.002, n = 154); pooled r = +0.40 (the write-up value).
- **I units, reward-image: r = +0.229 (Spearman p = 0.004; trial-shuffle p = 0.010; unit-perm p = 0.004; n = 154).
  The review's value is confirmed.** E units go the other way: r = -0.17 (p = 0.14). The pooled +0.12 hides the
  difference between the classes.
- **Per-unit reward effects in the visual window are not reliable.** Their split-half reliability is +0.02 (E) and
  -0.00 (I).
  - The ceiling for any reward pair is therefore about 0.1 or undefined.
  - So these data can neither show nor rule out alignment of reward with the stimulus axis.
  - The I-unit reward-image r is significant against both nulls, but it cannot be interpreted as a shared axis.
- The prior (expected) effect is moderately reliable (0.18 E, 0.34 I) and is **not** aligned with image:
  - cross-half r = -0.01 (E) and -0.02 (I), against ceilings of 0.31 and 0.42;
  - full-data r = -0.01 and -0.06, against ceilings of 0.46 and 0.59.
  - Prior-salience: E +0.08, I -0.13, against an I ceiling of 0.46.
  - This is the informative "no common priority axis" result.
- Unexpected: E units show expected-face r = +0.39 (p = 0.006; shuffle p = 0.012). The face effect has about zero
  split-half reliability, so treat this as unconfirmed.
- Goal x stimulus interaction (P3 units.csv): reward x image = -0.077 SD in E (p = 0.03, n = 72) and -0.042 in I
  (p = 0.08).

## Hub vs resolver
An SC priority map that sums goal and stimulus factors onto one axis predicts r near the ceiling (black ticks).
- Prior vs stimulus: the data sit at zero, inside the null and far from the ceiling.
- Reward: untestable, because reward is unreliable at the unit level.

This bears on a shared priority code in SC. It does not separate hub from resolver: either can deliver or build a
shared axis.

## Files and columns
- `B3_units.csv`: one row per unit. `session`, `unit_id`, `unit_type` (E/I), `P_loc` (export location index),
  `n_trials`, `<factor>_sd` (standardised effect; NaN for expected if the unit saw only one block type).
- `B3_pairs.csv`: one row per class (E / I / all) x factor pair.
  - Observed: `spearman_r_full`, `p_spearman`, `n_units`, `pair_kind`.
  - Trial-shuffle null: `null_trial_shuffle_mean` / `_q025` / `_q975`, `p_trial_shuffle_2sided` (deviation from the
    null mean).
  - Unit-label null: `null_unit_perm_q025` / `_q975`, `p_unit_perm_2sided`.
  - Split-half: `cross_half_r` with `cross_half_r_split_q025` / `_q975` (split-to-split spread, not a sampling CI),
    `within_half_r`, `split_half_rel_a` / `_b`, `ceiling_cross_half`, `ceiling_full_data`,
    `cross_half_r_disattenuated`.
  - Counts: `n_shuffles`, `n_unit_perms`, `n_splits`.
- `B3_reliability.csv`: one row per class x factor. `split_half_r` (mean over splits) with the split quantiles,
  `spearman_brown_full`, `n_units`, `n_splits`.
- `stats.csv`: every r (scipy, trial-shuffle and unit-perm p), every cross-half r with its ceiling, every
  reliability, and the reward x image interaction means. Each row records window, unit class and session filter.

## Provenance
- Data: verified successful trials (`load_export(..., 'verified')`). 6 diagonal-target sessions: Feynman 08_12-08_15
  and Newton 08_13-08_14. Newton 08_15 is excluded because its targets lie on the meridians.
- P comes from `results/intercollicular/g1_geometry/rows.npy`.
- Builder: `scripts/aim1_panels/build_B3.py`, git `6355c0a`, seed 20261101. It runs in about 1 min single-threaded.
