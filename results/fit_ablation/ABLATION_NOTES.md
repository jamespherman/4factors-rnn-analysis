# Fit ablation (2026-10-13)

Why: the context pilot (`results/context_pilot`) fit spatial selectivity poorly and
inconsistently across seeds, worse than the earlier `results/retrained/cond_target` models.
Each config below changes the pilot's `none_free` setup (no context input, context-conditioned
losses on, 3 drift channels, 70/15/15 split, 2000 epochs / patience 300). Sessions 08_14 and
08_15, seeds 1-3. Spatial r = correlation across recorded neurons between model and recorded
visual-epoch (50-250 ms) pref - opposite difference, model rates averaged over 10 noise draws.
Recorded split-half reliability of this effect is 0.98.

| Config | Change | Spatial r per run (08_14 s1-3, 08_15 s1-3) | Mean |
|---|---|---|---|
| none_free (pilot) | - | +0.21 +0.09 -0.13 / +0.24 +0.17 -0.04 | +0.09 |
| noctxloss | context-conditioned losses off | +0.27 +0.26 +0.25 / +0.36 +0.25 +0.33 | +0.29 |
| nodrift | drift channels off | +0.43 +0.07 +0.13 / +0.28 +0.23 -0.20 | +0.16 |
| bare | both off | +0.15 +0.06 +0.11 / +0.24 -0.28 -0.18 | +0.02 |
| bare80 | both off, 80/10/10 split | +0.32 +0.16 +0.12 / +0.23 +0.34 -0.14 | +0.17 |
| long | 5000 epochs, patience 600 | +0.25 +0.18 -0.13 / +0.23 +0.18 -0.09 | +0.10 |
| barelong | bare, 5000 / 600 | +0.15 +0.06 +0.11 / +0.25 -0.15 -0.18 | +0.04 |
| old cond_target (seeds 42-44, `train_session.py`) | reference | +0.34 +0.34 +0.50 / +0.43 +0.67 +0.37 | +0.44 |

- Longer training does not help: most runs pick an early checkpoint, and the extra epochs
  change little.
- Dropping the context-conditioned losses is the only change that gave consistent fits,
  but these are still below the old models.
- `bare80` is nearly the old configuration yet fits worse. The remaining differences are the
  split (stratified by location x context vs by reward; 10% vs 20% selection set) and
  selection on validation losses normalized by their epoch-1 values vs raw sums.
  `results/fit_replicate` tests this directly (below).
- Held-out Poisson LL (x1000 per neuron per bin) varies more between seeds (about 40)
  than between configs, so it does not separate these configs.

## Replication (`results/fit_replicate`, seeds 1-5, sessions 08_14 / 08_15)

| Config | Spatial r per run (08_14 s1-5 / 08_15 s1-5) | Mean [min, max] |
|---|---|---|
| oldscript (`train_session.py --variant cond_target`, unchanged) | +0.17 +0.13 +0.25 +0.47 +0.35 / +0.36 -0.12 +0.35 +0.42 +0.46 | +0.28 [-0.12, +0.47] |
| oldequiv (`train_context.py`, 80/20 split, no drift, no context losses) | +0.28 +0.08 +0.14 +0.32 +0.33 / +0.24 -0.35 -0.19 +0.31 +0.09 | +0.13 [-0.35, +0.33] |
| noctxloss (drift on, context losses off, 70/15/15), seeds 1-5 | +0.27 +0.26 +0.25 +0.45 +0.38 / +0.36 +0.25 +0.33 +0.29 +0.37 | +0.32 [+0.25, +0.45] |

Conclusions:
- The old models' +0.44 on these sessions was partly lucky seeds: the unchanged old script
  gives +0.28 with new seeds, and one run fails.
- `noctxloss` is at least as good as the old script and the only setup without failed runs.
  The drift input is what stabilizes it: every setup without drift has failed seeds.
  A plausible reason is that without drift inputs, slow rate drift has to be absorbed by
  the recurrent dynamics and the input weights.
- Adopted for the context grid (`results/context_grid`, `slurm/jobs_grid.txt`): drift on,
  context-conditioned losses off (`--lctx-scale 0`), 5 seeds x 3 sessions, plus `none_global`
  and `none_spatial` baselines. `none_free_lc25` and `shared_free_lc25` check whether a
  weak context loss (0.25) still lets the context effects be fit without hurting the fit.
