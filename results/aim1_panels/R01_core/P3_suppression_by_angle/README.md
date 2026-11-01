# P3. Pre-saccadic suppression by target angle from the RF (4factors)

`P3_suppression_by_angle.png` / `.pdf`. Regenerate with `python plot_P3.py` (reads only the CSVs here and `style.py`).

## Panel
% of units whose firing in the -50..0 ms window before saccade onset is below their pre-target baseline (-200..0 ms
re target onset; paired t-test across trials, p < 0.01, negative) when the target is at the given angle from the
unit's preferred location: 90 deg contralateral neighbour, 90 deg ipsilateral neighbour, 180 deg opposite.
Filled: classic (E, n = 72); open: putative interneurons (I, n = 154). Error bars: Wilson 95% CI. Brackets: exact
McNemar test (binomial on discordant units), 180 deg vs 90 deg ipsi.
- E: 3% (2/72) at each 90 deg neighbour, 12.5% (9/72) opposite; discordant 8:1, p = 0.039.
- I: 19% (30/154) at each 90 deg neighbour, 33% (51/154) opposite; discordant 31:10, p = 0.0015.
(The contra-neighbour comparison gives the same counts and p.)

## Files
- `P3_suppression.csv`: `unit_type`, `target_location`, `angle_from_P_deg`, `n_suppressed`, `n_units`,
  `pct_suppressed`, `ci95_lo_pct`, `ci95_hi_pct`.
- `stats.csv`: McNemar tests per class.

## Provenance
Copied unchanged from `results/aim1_panels/B4_map_competition/` (panel c; `build_B4.py` from
`results/intercollicular/g1_geometry/rows.npy`, `scripts/ic_g1_geometry.py`; 6 diagonal-target sessions, all units).
Builder `scripts/aim1_panels/build_R01_P3.py`.
