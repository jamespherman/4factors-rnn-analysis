# B11: data-fitted RNN units vs recorded neurons, visual-epoch ROC selectivity

Write-up item B11 (`results/AIM1_FINDINGS_WRITEUP.md`). Unilateral 4factors recordings, 7 sessions, 293 neurons.

## Files
- `scatter_recorded_vs_rnn_roc_grid_v13_free_free.{png,pdf}`: the **original figure**, moved here with `git mv`
  from `results/selectivity_by_type/` on 2026-11-01. It was made by `scripts/plot_selectivity_by_type.py`, which
  still writes to the old folder if rerun.
- `B11_rnn_selectivity_scatter.{png,pdf}`: the same content redrawn in the shared grant style from the CSVs by
  `plot_B11.py`. The figure reads only `B11_units.csv` and `style.py`. The per-panel r values are identical to the
  original.
- CSVs rebuilt by `scripts/aim1_panels/build_B11.py` (repo git commit 6355c0a, branch restore-lost-work; seconds).

## What is plotted
- Rows: classic SC neurons vs the RNN E unit fitted to each one; putative interneurons vs their RNN I unit. Unit
  types always match.
- Columns: location (preferred vs opposite), identity (face vs non-face), salience, reward, probability (high vs low).
- Markers: circles Feynman, triangles Newton. Filled = recorded selectivity significant (permutation p < 0.05).
- x = recorded ROC area. y = RNN ROC area, the mean over 6 restarts x 5 Poisson draws.
- r = Pearson per monkey (F, N) across neurons.
- **Slice of the rnn_* arrays.** `rnn_<factor>` has shape [293 neurons, 6 restarts (seeds 1-6), 5 Poisson draws,
  (selectivity, p)]. The figure plots `np.nanmean(rnn[..., 0], axis=(1, 2)) / 2 + 0.5`, i.e. the selectivity
  column, averaged over restarts and draws, converted to ROC area. The recorded value is `recorded[:, 0] / 2 + 0.5`.
  The p column (`[..., 1]`) is used only for "% significant" in `stats.csv`.

## What hub vs resolver predict here
- B11 is a methods check, not a hub-vs-resolver contrast. If fitted RNN units do not carry the recorded factor
  selectivities, their I->E weights cannot reveal whether inhibition is factor-specific.
- The fits keep spatial tuning and reward selectivity.
- They do not keep identity or salience selectivity. At chance means % significant RNN units about 5% (the false
  positive rate); per-neuron r is small but not zero in some groups (identity Feynman E +0.37, salience Feynman E
  -0.28).
- They shift the probability effect to the wrong population sign.
- So "unstructured I->E weights" in these fits cannot support the hub view (motivates the task-trained models of A).

## Statistics (`stats.csv`; one row per factor x monkey x class x statistic)
- Location: r 0.83-0.90. RNN mean |selectivity| is 63-71% of recorded.
- Reward: r 0.35-0.63.
- Probability:
  - Recorded mean signed selectivity -0.02 to -0.05 (lower on high-probability trials).
  - RNN mean +0.05 to +0.13.
  - Per-neuron r is nonetheless +0.36 to +0.56: the fits rank neurons by probability modulation, but the
    population mean has the wrong sign.
- % significant: recorded vs RNN, as in `results/selectivity_by_type/NOTE.md`.
- Columns: statistic, factor, monkey, unit_class (E/I), value, n, test (with SD over restarts / SEM where relevant),
  window, session_filter.

## Column dictionary
- `B11_units.csv`: one row per neuron x factor (293 x 5).
  - session, monkey, unit_id (cluster id from the spike cache, v13 export order), export_row (row in the npz),
    unit_type (E classic / I putative interneuron), factor.
  - recorded_selectivity (2 x (AUC - 0.5); positive = face / high salience / high reward / high probability /
    preferred location higher), recorded_roc_auc, recorded_p (500 label permutations, two-sided),
    recorded_significant (p < 0.05).
  - rnn_selectivity_mean and rnn_roc_auc_mean (plotted y), rnn_selectivity_sd_over_restarts (SD of the 6
    restart means), rnn_frac_draws_significant (fraction of 30 restart x draw scores with p < 0.05).
- `B11_rnn_draws.csv`: every RNN score, one row per neuron x factor x restart_seed x poisson_draw. Columns:
  rnn_selectivity, rnn_p.

## Provenance
- `scripts/compare_selectivity_by_type.py` (grid `results/grid_v13`, variant free_free, restarts 1-6, 5 draws,
  500 permutations) -> `results/selectivity_by_type/grid_v13_free_free.npz` (commit d16f68b). Figure commit 6d62818.
- Visual epoch 50-250 ms after target onset, preferred location (location: preferred chosen on even trials, scored
  on odd). All verified trials within each neuron's stable period.
- RNN rates are Poisson-sampled before scoring, so they carry comparable sampling noise. Recorded variability is
  super-Poisson, so RNN % significant is if anything inflated.
