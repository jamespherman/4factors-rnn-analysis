# B6: behaviour in the bilateral conflict task (Newton, srsClassic; sessions 08-20 and 08-19)

These are **srsClassic** data (conflict task: one target per hemifield on choice trials; isoluminant targets,
salience 180 vs 20 hue contrast; reward = per-trial draws around block means for identities T1/T2). They are
not the 4factors task. All trials with behaviour; choice trials = two targets, chosen side valid, RT >= 80 ms.

Figure: `B6_bilateral_behaviour.png` (300 dpi) / `.pdf`. Regenerate with `python plot_B6.py` (reads only the CSVs
here and `style.py`).

## Panels
- **a** 08-20: P(choose right) vs ACTUAL reward difference (right minus left target, ms), congruent vs conflict
  trials. Points: quintile bins per condition, Wilson 95% CI. Lines: logistic fit per condition with a trial-bootstrap
  95% band (2,000). Text: the joint-model reward beta (+0.28 per 100 ms, 95% CI +0.08 to +0.48, n 227) and salience beta
  (-0.28 per level, p 0.053) from `choseR ~ salDiff + rewDiff + prevChoice`, reproducing the peer repo's
  `run_competition_behaviour.m` exactly.
- **b** 08-19: same (77 trials; reward beta +0.03, CI -0.58 to +0.64). Descriptive only.
- **c** 08-20: same against the fitted delta-rule **belief**-value difference. Secondary and circular: the belief was
  fitted to these choices, and alpha sits at its lower bound (0.01; beta at its upper bound 20, lapse at 0.5).
- **d** 08-20 single-target trials: RT vs the belief value of the shown identity (quintile bins, mean +- SEM; black)
  and vs the TRUE block-mean value of the shown identity (one point per block x identity, 16 points; open grey), the
  drift control. Spearman rho -0.21 (p 3e-4, n 301) for belief; +0.05 (p 0.35) for block-mean value.
- **e** Within-block RT difference, shown rich minus shown lean identity (median RT per block x identity), one
  point per block; bar = mean, line = block-bootstrap 95% CI. Drift cancels within a block.

## Key result and discrepancy with the write-up
- Choice numbers reproduce (beta +0.28/100 ms; salience p 0.053; conflict RT cost +16.4 ms, CI +0.1 to +25.1).
- Spearman -0.21 (n 301) reproduces, but it does **not** survive drift controls: the alpha = 0.01 belief is mostly a
  slow trend across the session (rho with trial index +0.47), RT also drifts (rho -0.11 with trial index), the true
  block-mean value gives rho +0.05 (p 0.35), the within-block slope is -33 ms/100 ms (CI -89 to +22, p 0.24), and the
  within-block rich-minus-lean RT difference is +3.2 ms (CI -6.6 to +14.0; p 0.58; 8 blocks).
- So "RT tracks value more clearly than choice does" is not supported. Suggested wording: "Single-target RT
  correlates with the fitted belief (Spearman -0.21, n 301), but not with the true block value (+0.05) and not within
  blocks (rich minus lean +3 ms, CI -7 to +14), so the correlation most likely reflects slow drift; we find no reliable
  effect of value on RT in this session."
- This also affects C1/S9 (recorded RT-value rho -0.21 used as the target statistic).

## Hub vs resolver
Behaviour alone does not separate them; no model curve is drawn. For context, `B6_model_S9_sig_v2_per_network.csv`
lists the trained networks' S9 (Spearman RT vs belief value, single-target trials; mean of 10 simulated sessions per
network; `results/hub_models/sig_v2/*/draws.csv`, `scripts/srs_stage0.py`): all architectures range about -0.15 to
+0.06; S9 is item C1's subject.

## Files and columns
- `B6_ab_psychometric_reward_points.csv`: panel, session, condition (congruent/conflict), bin, reward_diff_ms_bin_mean
  / _bin_lo / _bin_hi (ms, right minus left), n_trials, n_chose_right, p_chose_right, ci95_lo, ci95_hi (Wilson).
- `B6_ab_psychometric_reward_fit.csv`: panel, session, condition, reward_diff_ms (grid), p_fit, band95_lo, band95_hi.
- `B6_c_psychometric_belief_points.csv`, `B6_c_psychometric_belief_fit.csv`: same with belief_diff_ms.
- `B6_d_rt_vs_belief_bins.csv`: session, version, bin, value_ms_mean/lo/hi (belief, ms reward), n_trials, rt_ms_mean,
  rt_ms_sem. `B6_d_rt_vs_blockmean_points.csv`: session, conflict_block, shown_identity, value_ms (block mean),
  n_trials, rt_ms_mean, rt_ms_sem. `B6_d_single_target_trials_<ses>.csv`: one row per single-target trial
  (trial_row, conflict_block, block_type, shown_identity, belief_value_ms, block_mean_value_ms, shown_rich, rt_ms).
- `B6_e_within_block_rich_minus_lean.csv`: session, conflict_block, block_type, rich/lean block values, n, median RTs,
  rt_diff_rich_minus_lean_ms.
- `stats.csv`: every quoted statistic (value, ci_lo, ci_hi, p, n, test, window, unit_class, session, note).
- RT = saccade onset minus go signal (ms).

## Provenance
- Builder: `scripts/aim1_panels/build_B6.py` (repo D:\Projects\4factors-rnn-analysis, branch restore-lost-work,
  HEAD 6355c0a, 2026-11-01).
- Choice tables (a, b): `results/aim1_panels/_bilateral_source/competition_choice_trials_<session>.csv`, exported
  read-only by `scripts/aim1_panels/export_bilateral_tables.m` from Newton_SC_2026
  `data/bilateral/competition_choice.mat` (2026-10-31 11:43; code `code/run_competition_choice.m`, same trial selection
  as `run_competition_behaviour.m`, whose `competition_behaviour.mat` betas are reproduced).
- Belief and single-target trials (c-e): `results/hub_models/belief/<ses>_trials_with_belief.csv`
  (`scripts/srs_belief_fit.py`, commit 8c76cf1), built from Newton_SC_2026 `data/export/*_srsClassic_trials.csv`.
  The belief CSV's 08-19 choice set has 75 trials (MATLAB table: 77).
- Peer note: `Newton_SC_2026/docs/NOTE_2026-10-31_competition_for_4factors.md` (2026-10-31 14:19).
