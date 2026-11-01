# P2. Do factor effects line up across SC units? (4factors, visual window, all units pooled)

`P2_factor_alignment_scatter.png` / `.pdf` (a, b side by side); alternative b' in `P2alt_reward_vs_salience.png` / `.pdf`.
Regenerate with `python plot_P2.py` (reads only the CSVs here and `style.py`).

## Panels
Each point is one unit's standardised effect (OLS coefficient / residual SD of sqrt spike count, 50-250 ms after
target onset, target at the unit's preferred location). Filled black: classic (E); open grey: putative interneuron
(I). Statistics are pooled over E and I (Spearman r, scipy p), as in the original P3 report.
- **a stimulus x stimulus**: image (vs bullseye) effect vs salience effect: r = +0.40, p = 4e-10, n = 226.
- **b goal x stimulus**: location-prior (expected) effect vs image effect: r = -0.05, p = 0.52, n = 140 (units
  that saw both block types).
- **b' (alternative)**: reward effect vs salience effect: r = +0.09, p = 0.16, n = 226.

Within-class r (not plotted) are in `stats.csv` (e.g. image-salience E +0.26, I +0.42). Note (from B3): per-unit
reward effects have ~0 split-half reliability, so b' cannot test alignment; b (prior, reliability 0.18-0.34) can.

## Files
- `P2_units.csv`: `session`, `unit_id`, `unit_type`, `P_loc`, `reward_sd`, `expected_sd`, `image_sd`, `salience_sd`
  (copied unchanged from `results/competition/p3_interactions/units.csv`, visual window).
- `stats.csv`: `panel`, `x_effect`, `y_effect`, `pair_kind`, `unit_class` (all = plotted), `spearman_r`, `p`, `n`.

## Provenance
`scripts/cp_p3_interactions.py` (6 diagonal-target sessions, Newton 08_15 excluded; verified successful, unit-stable
trials; >= 40 trials at P). Builder `scripts/aim1_panels/build_R01_P2.py`, git e217f9f. All three r and p match the
original P3 summary.
