# Aim 1 panels: figures + self-contained data for write-up items B1, B3-B11, C1, D1 (2026-11-01)

James asked (2026-11-01) for a visualisation plus self-contained data for: B1 (answer: what the prior effect is), B3 (all
three panels), B4, B5, B6, B7 (i, ii), B8, B9, B10, B11 (data for the existing figure; move the figure), C1 (primary
and alternative), D1. Wherever possible each panel shows how the data differ from the resolver prediction.

## Output layout (repo `D:\Projects\4factors-rnn-analysis`, branch restore-lost-work)
`results/aim1_panels/<ITEM>_<slug>/` containing
- `<ITEM>_<slug>.png` (300 dpi) and `.pdf` (vector, editable text: `pdf.fonttype 42`)
- one tidy CSV per panel (`<ITEM>_<panel>.csv`; long format, one row per observation or plotted point, explicit units in
  column names, e.g. `rt_ms`, `rate_sp_s`, `time_ms_from_target`); JSON only for nested curves if CSV would be
  awkward
- `plot_<ITEM>.py`: standalone, reads ONLY the CSVs in its folder (numpy/pandas/matplotlib), regenerates the figure;
  this is the file James edits to restyle
- `README.md`: what each panel shows, column dictionary, n, statistics quoted in the write-up and how they were computed,
  source script and data provenance, the resolver/hub prediction drawn and where it comes from
Builder scripts that compute the CSVs from the analysis outputs: `scripts/aim1_panels/build_<ITEM>.py` (provenance;
James does not need them). Top-level `results/aim1_panels/README.md` index. B11's existing figure is moved (git mv) to
`results/aim1_panels/B11_*` and references updated.

## Shared conventions (`scripts/aim1_panels/style.py`, copied into each panel folder so it stays self-contained)
- Recorded data: near-black `#222222`, filled markers; CI as error bars or a light band.
- Resolver prediction: orange `#eb6834`, dashed; priority hub: blue `#2a78d6`; outcome hub: aqua `#1baf7a`
  (first three slots of the validated reference palette, all-pairs safe). Null / chance band: light grey `#d9d9d6`.
- Thin marks (lines 1.5-2 pt), Arial/Helvetica 7-8 pt (grant-figure scale), no top/right spines, legends or direct
  labels, one y-axis per panel, panel size ~2.2 x 2.0 in so items can be combined.
- Every statistic printed on a panel must equal the write-up value (or the write-up is corrected and the change noted).

## Per item: content, source, resolver contrast
- B1 (4factors behaviour, `results/competition/p1_behaviour/trials.csv`): RT % change vs low reward by location class
  (expected / neighbour / opposite), high and low reward lines, per monkey; second panel peak velocity. Report the low-
  reward prior effect explicitly (answers James's question). Contrast: "additive" prediction (reward benefit the same at
  all locations) drawn dashed.
- B3 (`p3_interactions/units.csv`, visual window, E and I units): (a) reward vs salience per unit, (b) image vs salience,
  (c) matrix of cross-unit Spearman r between factor effects; null bands from unit-label permutation. Contrast: a common
  priority axis (SC combining goal and stimulus factors) predicts positive alignment; drawn as the band of r a shared
  axis would give (from the observed stimulus-stimulus alignment r = 0.40).
- B4 (`p4_timing`, `cp_p4_timing.py`, `cp_p2_conflict.py`): PSTH of units with RF at the expected location, target
  elsewhere, expected vs non-expected block, with baseline; bars % pre-saccadic suppression by target angle (classic vs
  interneuron). Contrast: compatible with both views; say so in README (no fake contrast).
- B5 (`cp_p2c_rt_specificity.py` outputs): bars of % RT change per SD pre-target rate by RF class with CIs; dashed bar
  with the sign local competition predicts (positive for "RF at expected competitor").
- B6 (bilateral 08-20, Newton_SC_2026 outputs; belief fit `results/hub_models/belief/`): psychometric P(choose right) vs
  belief-value difference, congruent vs conflict; inset single-target RT vs belief value. Read-only use of the peer
  repo; copy any code needed.
- B7 (bilateral notes/outputs): (i) time course of reward-difference and choice coefficients in the L-R signal with
  null bands; (ii) per-unit competitor-reward coefficients histogram, predicted -10% per 100 ms shift drawn. Contrast:
  where available, the trained resolver networks' values of the same measures (sig_v2 / signature outputs), otherwise
  the analytic prediction, labelled as such.
- B8 (`results/intercollicular/g23_timing`): ipsi vs contra onset latency per unit on the identity line, relay band
  10-27 ms shaded (resolver/commissural relay prediction).
- B9 (bilateral intercollicular outputs): onset-aligned PSTHs side X burst vs side Y ipsiversive response, simulated
  relay dashed; inset pooled cross-SC CCG vs sparse-relay CCG.
- B10 (`results/intercollicular/b3_ccg`, `b5_pair_selectivity`): example sharp-peak CCG with jitter band; histogram of
  profile similarity connected vs matched unconnected pairs; mark the inhibitory troughs (n = 8). README states what the
  resolver predicts for connectivity (factor-specific inhibition between dissimilar units) and that excitatory
  like-to-like does not discriminate.
- B11 (`results/selectivity_by_type/grid_v13_free_free.npz`): CSV of recorded vs RNN ROC selectivity per unit and factor;
  move the existing figure.
- C1 (`results/hub_models/sig_v2`, `rt_shift_pilot`, hand-built results, recorded 08-20 S9 and CI): (a) S9 per network,
  priority vs outcome hub at full engagement (hand-built marked), recorded value as a band; resolvers shown for
  completeness if available; (b) binned RT vs value for one representative B and C network beside the monkey's.
- D1 (`results/hub_models/bicuculline/per_network.csv`): dose-response RT change into injected vs other field, and
  fixation-break rate, per architecture with network range shaded; muscimol point marked.

## Rules
- No new heavy computation locally (PC in daily use): reuse saved outputs; if a panel needs recomputation that takes
  more than ~10 min single-threaded, report instead of running it (it can go to CRC).
- Local Python runs single-threaded (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`), low priority (`nice -n 19`).
- Never edit the peer repos (conflict-task-RNN, Newton_SC_2026); read and copy only.
- If a number in the write-up cannot be reproduced from the saved outputs, flag it; do not silently change it.
- Do not commit; the coordinating session reviews and commits.

## v2 after independent review (incorporated)
1. C1: full-engagement S9 comes from `rt_shift_pilot/runs.csv` (offset 0) and `bicuculline/per_network.csv` ctrl rows
   (check `rho_inj` = S9), not sig_v2 (calibrated). Resolvers (A, AE) are also graded (about -0.2 to -0.3), so the panel
   shows all four architectures and the caption says S9 separates graded from categorical delivery, not resolver from
   hub. Binned RT vs value panel in normalised RT (z or % change); block-mean value columns included as a drift control.
2. B7: the -10%/100 ms line is the planted effect (power >= 0.98), labelled as such; the trained resolvers' own S1/S2
   values are drawn as the model prediction. If resolvers do not show the effects, the write-up sentence "a resolver
   comparing values in SC predicts both effects" is corrected. Null bands at the multiple-window-corrected level.
3. B3: within-class (E, I) correlations, split-half (cross-half) correlations and reliability ceilings, within-unit
   trial-shuffle null; the "priority axis" is labelled as the SC-priority-map prediction (not resolver-specific).
   Check the I-unit reward-image r = +0.23 (p = 0.004) and correct the write-up's "|r| <= 0.12" if confirmed.
4. B9: draw the largest relay that is not excluded (detection bound) alongside the sparse relay.
5. B1: absolute mean RT (or % from the low-reward expected cell) per location class, high vs low reward; session-level
   CIs (7 sessions); the crossover (low reward: unexpected faster in Feynman) stated; Newton weak; dashed line labelled
   "no-interaction (additive) null". Write-up wording "amplifies the prior" corrected.
6. B5: use the computed categories (RF at high-probability location / opposite / other / all) and the joint-model
   partial coefficients; shade the half-plane local competition predicts instead of a dashed bar; reconcile the
   write-up's "-5 to -6%" with the outputs (rerun cp_p2c saving results).
7. B8: per-unit amplitude-corrected relay excess (n = 111) with median CI against the 10-27 ms band labelled
   "commissural relay"; recompute the CI with the method documented; state which g1 block gives 27% vs 25%.
8. B10: main contrast without location (+0.43 vs +0.22), unit-level resampling, powered troughs (5, all I->I) stated.
9. D1: correct write-up numbers against per_network.csv (fixation-break range, C bias sign at low doses, B RT range);
   truncate RT curves where drt_inj is undefined; muscimol on its own axis; caption "no architecture differs".
10. B4: suppression bars from `intercollicular/g1_geometry`; block-split PSTH needs a rerun (units.npy stores only
    the difference); split E and I; "90-degree neighbours", not "every non-target location".
11. Self-contained data: every band/CI/quantile in the CSV; a `stats.csv` per item with each quoted statistic (n, test,
    window, unit class, session filter); units keyed by (session, id, type); README records source paths, git commit,
    and all-trials vs verified-trials. B6 x-axis: actual reward difference (ms); belief-value version secondary
    (circularity, alpha at bound noted). B11: document which slice of rnn_* the figure plots.
