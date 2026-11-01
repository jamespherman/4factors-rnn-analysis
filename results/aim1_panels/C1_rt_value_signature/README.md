# C1 - Single-target RT vs value (S9): graded vs categorical value delivery to SC

Figure: `C1_rt_value_signature.png` (300 dpi) / `.pdf`. Regenerate with `python plot_C1.py` (reads only the CSVs here
and `style.py`). Builder (provenance; runs the simulation): `scripts/aim1_panels/build_C1.py`. Repo commit at build:
`6355c0a` (branch restore-lost-work, uncommitted panel work on top).

## Caption (suggested)
S9 = Spearman correlation between single-target saccade RT and the value of the shown target. (a) In task-trained
networks with value engaged on every trial (w = 1), S9 is negative when SC receives a graded value signal: resolvers
(A, AE) and the priority hub (B). It is about 0 when SC receives a categorical outcome: the outcome hub (C). So S9
separates **graded from categorical delivery**. It does **not** separate resolver from priority hub (A/AE vs B rank-sum
p = 0.25); that contrast is the identity test (A3/S8). Hand-built networks (open diamonds) show the same pattern more
strongly. (c) The B-C contrast holds when RTs are slowed 20-36 ms by raising the readout threshold. **Recorded data:**
08-20 RT correlates with the fitted belief (rho -0.21, 95% CI [-0.32, -0.09], n = 301). It does not correlate with the
true block value of the target (rho +0.05 [-0.06, +0.17]). Within blocks, RT for the rich identity is no faster
(+0.2 ms [-10, +10]). The fitted belief rises over the session (rho with trial number +0.47). (b, e) Given the monkey's
own belief sequence, which spans only 143-197 ms, no trained network reaches -0.21 (all |S9| <= 0.15). **So the
recorded data do not currently support a value effect on RT. C1 is a model prediction only.** The prediction for the
proposed experiments: a graded (priority) hub predicts that single-target RT falls with true block value; an outcome
hub predicts no relation.

## Panels
- **a** Per-network S9 on each network's full trial tables: 08-20 sequence plus 2 synthetic 8-block sessions per seed,
  value = belief input, 2 seeds averaged. Trained networks are dots; the bar is the architecture median. Hand-built
  networks are diamonds. The monkey column shows recorded S9 with 95% bootstrap CIs: 08-20 vs belief (filled), 08-20 vs
  true block-mean value (open), 08-19 vs belief (grey). The band is the 08-20 belief CI.
- **b** Same networks, S9 computed only on the trials of the recorded 08-20 sequence. The models received the monkey's
  fitted beliefs as value input.
- **c** S9 vs median-RT slowing as the hazard threshold theta is raised (offsets 0, 0.1, 0.2, 0.35, 0.5). B and C
  networks, from `results/hub_models/rt_shift_pilot/runs.csv`. A and AE were not run.
- **d** % RT change from each source's mean vs the true value of the shown target. Monkey: block-mean reward of the shown
  identity, 4 quantile bins (block x identity cells). Models: synthetic sessions only, where the value input is a
  delta-rule belief with alpha 0.1, close to block means; 5 quantile bins per network. One representative network per
  architecture, chosen as the one closest to its architecture's median S9 in (a): A_s2, B_s0, C_s5.
- **e** % RT change vs fitted belief on the 08-20 sequence. Bins are the 08-20 belief quintiles, shared by monkey and
  models. Black dashed line: monkey RT minus its own block's mean RT, which removes between-block drift.
- Normalised RT is used because model RTs (110-150 ms) differ from the monkey's (mean 221 ms, median 217 ms).

## What each architecture predicts
A and AE (resolvers) and B (priority hub) receive graded value, so S9 < 0. C (outcome hub) receives a decided outcome,
so S9 is about 0. Its upstream signal is anti-graded on single-target trials (dis-value rho -0.14, vs +0.38 in B).

## Files and columns
- `C1_ab_s9_per_network.csv`: one row per network.
  - `net`, `arch`, `handbuilt`, `role`, `arch_x` (plot position).
  - `s9_rho`: panel a. Mean of 2 seeds.
  - `s9_rho_seed0`, `s9_rho_seed1`.
  - `s9_rho_recseq`: panel b, 08-20-sequence trials only.
  - `s9_rho_synthetic`: synthetic sessions only.
  - `n_single_completed_per_seed`, `median_rt_ms`.
  - `bic_ctrl_rho_inj_rightfield`, `bic_ctrl_rho_oth_leftfield`: the same statistic from the bicuculline control
    runs, per field, 3 seeds.
- `C1_ab_recorded_s9.csv`: one row per session x value type (`belief` = delta-rule belief; `block_mean` = true
  block-mean reward).
  - `rho`, `p`, `n`.
  - `fisher_lo` / `fisher_hi`: Fisher z interval, SE = sqrt(1.06/(n-3)) (Fieller-Hartley-Pearson).
  - `boot_lo` / `boot_hi`: percentile bootstrap, 10000 resamples of trials. This is the plotted interval.
  - `se_fisher`, `se_boot`.
- `C1_c_rt_shift_robustness.csv`: `net`, `arch`, `theta_offset`, `s9_rho`, `median_rt_ms`, `delta_rt_ms_from_offset0`,
  `delta_s9_from_offset0`. Seed means.
- `C1_d_rt_vs_true_value.csv` and `C1_e_rt_vs_belief_0820_sequence.csv`: one row per plotted bin.
  - `source`, `arch` (`data` = monkey; `data_within_block` = monkey with block-mean RT removed), `net`, `value_type`.
  - `bin`, `value_bin_lo_ms` / `value_bin_hi_ms` (outer edges open), `value_mean_ms` (x).
  - `n_trials`, `rt_mean_ms`, `rt_pct_change_from_mean` (y).
  - `ci95_lo` / `ci95_hi`: bootstrap over trials in the bin, 2000 resamples.
  - `source_mean_rt_ms`.
- `C1_sim_runs.csv`: one row per network x seed. `n_single_completed`, `median_rt_ms`, `s9_rho`, `s9_rho_recseq`,
  `s9_rho_synthetic`.
- `C1_model_single_target_trials.csv`: every completed single-target model trial.
  - `net`, `arch`, `handbuilt`, `seed`, `trial_row` (row in the simulated table).
  - `sequence` (`recorded_08-20` / `synthetic`), `target_field`, `value_shown_ms`, `rt_ms`.
- `stats.csv`: every quoted number. Columns: `statistic`, `value`, `n`, `test`, `source`, `note`.

## How the statistics were computed (provenance)
- **Recorded data.** Source: `results/hub_models/belief/08-20_trials_with_belief.csv` and `08-19_...`, made by
  `scripts/srs_belief_fit.py` from the Newton_SC_2026 export. Trials: correct single-target, RT >= 80 ms (n = 301 and
  184), as in srs_belief_fit.py. Value = delta-rule belief (alpha 0.01, fit at its bound) of the shown identity before
  the trial. `block_mean` = mean of that identity's reward draws in the block.
- **Drift controls in `stats.csv`.**
  - Block-mean-value S9.
  - Partial Spearman on trial order: -0.18. This control is linear only and does not remove the slow belief trend.
  - log RT ~ belief + block fixed effects: belief p = 0.33.
  - Within-block rich - lean RT: +0.2 ms, 95% CI [-10.2, +10.3], bootstrap within blocks. The coordinator's estimate
    with a different estimator is +3 ms [-7, +14].
  - Belief vs trial order: rho +0.47, p = 1e-17.
- **Models.** `scripts/aim1_panels/build_C1.py` repeats the `scripts/srs_rt_shift_pilot.py` code path at offset 0:
  `srs_bicuculline.tables` / `run`, seeds 0-1, run seed 1000*seed+7, engaged = 1. It runs on the 21 retained trained
  v2 networks (`results/hub_models/trained_v2`; C seeds 1-3 excluded as training failures) and the 3 hand-built networks
  (`srs_bicuculline.HAND` parameters, shape_vis 0.25).
  - Check: for B and C the result equals `rt_shift_pilot/runs.csv` offset 0 exactly (max difference 0).
  - `bicuculline/per_network.csv` ctrl `rho_inj` is the same statistic restricted to targets in the right field,
    with 3 seeds. Across networks it agrees with S9 (Spearman 0.87; median absolute difference 0.04).
  - Runtime: about 4 min, single thread.
- **Rank-sum tests.** Exact Mann-Whitney on per-network S9. B vs C: p = 0.024 (6 vs 3; the minimum attainable).
  Resolvers vs C: p = 0.004. Resolvers vs B: p = 0.25.
- **Hand-built values in the write-up.** The values -0.32 (priority hub) and +0.02 (outcome hub) come from stage 0
  (`results/hub_models/stage0/draws.csv`). Stage 0 used calibrated engagement (w 0.3 for A and B, 0.1 for C) with RT
  around 205 ms, not w = 1. The hand-built resolver A gives the same -0.32 there. At w = 1 the hand-built values are
  A -0.86, B -0.79, C +0.00.
