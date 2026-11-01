# B1. Reward cue x location prior in saccade RT and peak velocity (4factors, single targets)

`B1_reward_x_prior.png` / `.pdf`. Regenerate with `python plot_B1.py`; it reads only the CSVs here and `style.py`.

## Panels
2 x 2 grid. Columns are monkeys: Feynman (4 sessions), Newton (3 sessions). Rows are RT and peak velocity.
- x axis: where the target appeared relative to the block's high-probability location: expected (62.5% of
  targets), one of the two 90-deg neighbours (12.5% each), or opposite (180 deg, 12.5%).
- Lines: low-reward cue (grey, open squares) and high-reward cue (black, filled circles). Each point is the mean over
  sessions of the change from that session's low-reward, expected-location cell, with a t-based 95% CI over sessions
  (df = 3 for Feynman, 2 for Newton).
- Top row: RT change in %. Bottom row: peak-velocity change in deg/s.
- Dashed line: the **no-interaction (additive) null** for the high-reward cue. It is the low-reward line shifted by
  the reward effect of an additive model (log RT or PV ~ session + reward + location class + stimulus, no
  interaction). If the reward cue gave the same benefit at every location, the black line would follow it.
- Text: the trial-level simple effect "opposite vs expected" within each reward cue (OLS ~ session + location class +
  stimulus, cluster-robust SE by block run), expressed as exp(b) - 1.

## What it shows
The effect is a **crossover**, not an amplification of the prior.
- Feynman, low-reward cue: saccades to the unexpected opposite location are **faster** than to the expected
  location: -7.7% (100 x log ratio -8.0%; p = 0.009 clustered; uncorrected OLS p = 3e-5).
- Feynman, high-reward cue: they are slower: +10.4% (log -9.95%; p = 0.005 clustered; OLS p = 3e-8).
- Feynman interaction reward x opposite: +17.9% (95% CI 13.0 to 22.7, p = 5e-13). Session level, (opposite -
  expected) for high minus low reward: +19.1 percentage points (95% CI 12.2 to 26.0, t over 4 sessions, p = 0.003).
- Newton: weak. Only the high-reward opposite cell differs from expected: +5.5% (log +5.3%; p = 0.037 clustered;
  OLS p = 1.6e-4). Low-reward neighbour is faster: -3.8% (p = 0.02). Reward x opposite +3.5% (p = 0.098). Reward x
  neighbour +3.4% (p = 0.037). The additive reward effect has the opposite sign to Feynman's: high reward is
  +2.2% slower (p = 9e-4).
- Peak velocity (Feynman): reward x neighbour -31 deg/s (p = 0.008), reward x opposite -42 deg/s (p = 0.017). So the
  high-reward cue raises PV at the expected location and not at the others. The simple effects within each cue are
  not significant.
- **Peak velocity differs between locations for reasons other than the task factors.** Newton's 90-deg neighbour
  targets have about 150 deg/s higher PV in both reward conditions (+157 and +139 deg/s, p < 1e-10). This follows the
  saccade direction and amplitude to that location. It is not a prior or reward effect. Compare PV within a
  location class, across cues, and not across locations.

## Hub vs resolver
This item is behaviour only. Both views predict that goal factors interact in behaviour. The dashed line is a
statistical null (no interaction), not a model prediction. Why the low-reward cue speeds saccades to the opposite
location in Feynman is not explained by either view.

## Files and columns
- `B1_cells.csv` (plotted points): `monkey`; `measure` (`rt_pct` = % RT change, `pv_deg_s` = PV change in deg/s);
  `reward_cue` (low/high); `loc_class` (expected / neighbour_90deg / opposite_180deg); `change_mean` (mean over
  sessions); `ci95_lo`, `ci95_hi` (t CI over sessions); `n_sessions`; `n_trials`; `abs_mean_over_sessions` (absolute
  mean RT in ms, or PV in deg/s).
- `B1_sessions.csv` (one row per session x cue x location class): `n_trials`, `rt_ms_mean`, `pv_deg_s_mean`,
  `rt_pct_change_vs_low_expected`, `pv_deg_s_change_vs_low_expected`.
- `B1_null.csv`: `null_change` = additive-null value for each high-reward cell. For RT the shift is combined
  multiplicatively: (1 + low) x (1 + reward) - 1.
- `stats.csv`: every quoted statistic. Columns: `value`, `ci95_lo`, `ci95_hi`, `p`, `n` (trials, or sessions for
  session-level tests), `test`, `window`, `unit_class`, `session_filter`, `note` (units; 100 x log values), `source`.

## Provenance
- Source: `results/competition/p1_behaviour/trials.csv`, written by `scripts/cp_p1_behaviour.py`.
  - Trials: **all successful** gSac_4factors trials (trialEndState 21 in `session_data.mat`) with a valid block and
    RT > 80 ms, from 7 sessions, 5,242 trials. This is not restricted to the export's verified trials.
  - RT = saccade onset minus fixation-point offset. Block run = consecutive trials of one block (cluster unit).
- Builder: `scripts/aim1_panels/build_B1.py`, at git `6355c0a`, branch restore-lost-work.
- The write-up's "+17.9% (p < 1e-6)" is the log-RT coefficient x 100 of the full P1 model. It is reproduced here.
- Caveat on p-values: the p-values quoted for B1 in review (p < 1e-4 for Feynman low-reward opposite, p = 0.003 for
  Newton) match neither test here.
  - With uncorrected OLS they are 3e-5 and 1.6e-4.
  - With block-clustered SEs (the P1 convention, used on the figure) they are 0.009 and 0.037.
  - The effect sizes agree.
