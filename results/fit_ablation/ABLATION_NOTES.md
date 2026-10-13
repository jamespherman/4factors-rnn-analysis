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
  `results/fit_replicate` tests this directly: the old script with new seeds (`oldscript`),
  and `train_context.py` with the old split sizes and no drift or context losses (`oldequiv`).
- Held-out Poisson LL (x1000 per neuron per bin) varies more between seeds (about 40)
  than between configs, so it does not separate these configs.
