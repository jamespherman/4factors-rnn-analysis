# P5: pre-target bilateral SC state, monkey vs trained Conflict Task E-I RNNs

The monkey analysis in P4 (Newton, simultaneous left and right SC recordings, sessions 08-20 and 08-19) is applied
here to the trained bilateral SC E-I RNNs of the original Conflict Task: 10 bilateral networks and 10 no-commissure
networks.

## Figure (`P5_model_vs_monkey_bilateral.png`, 300 dpi, plus `.pdf`)
**a.** P(chose right) by tercile of the pre-target bias, where bias = left SC minus right SC population activity in
the 300 ms before target onset. Monkey points are mean +/- binomial SE (from P4). Model points are the mean +/- SEM
across networks, using E units.

**b.** Partial correlation across trials between left-SC and right-SC pre-target population activity. Each dot is one
session or one network (E units); the bar marks the group mean. Grey open dots show the same networks after the input
noise on the fixation channel is removed. That channel is the only input both hemispheres receive.

**c.** Bias coefficient (log-odds per SD of bias). Monkey values are from P4 and control for history, reward side and
gaze. Model values are computed within big-side x salient-side x dT cells (see below).

## Results (primary: dT <= 0 trials, E units, mean +/- SD over 10 networks per group)
| | partial r(L, R) | bias coefficient (log-odds/SD) |
|---|---|---|
| Monkey 08-20 | +0.155 (p = 0.011) | +0.63 (z = 3.78) |
| Monkey 08-19 | +0.144 (p = 0.09) | +0.67 (z = 2.32) |
| Bilateral RNN | +0.44 +/- 0.04 (10/10 p < 0.05) | -0.14 +/- 0.40 (Wilcoxon p = 0.70; 1/10 with \|z\| > 2, and it is negative) |
| No-commissure RNN | +0.34 +/- 0.03 (10/10) | +0.01 +/- 0.27 (p = 1.0) |
| Bilateral, fixation-input noise removed | +0.14 +/- 0.04 (10/10) | |
| No-commissure, fixation-input noise removed | +0.01 +/- 0.02 (1/10) | |
| Bilateral, commissure cut post hoc | +0.32 +/- 0.04 | |

* **a/c. The model's pre-target state does not predict choice.** P(chose right) is 0.50 in every tercile in both
  groups. Main-effects logistic fits (z ~ 0) and the all-dT trial set (coefficient -0.03 +/- 0.08 and -0.03 +/- 0.10)
  give the same answer. The reason is that model choices are almost fully set by the stimulus. With dT <= 0, the
  majority choice in each cell is made on 99.5% of trials (94% over all dT). Pre-target fluctuations decay with
  tau = 50 ms, well before the late-arriving target-locked evidence decides the choice. The monkey effect (log-odds
  0.6/SD) is therefore not reproduced.
* **b. Positive left-right co-fluctuation does appear in the model, from two sources:**
  1. **Shared input noise.** The fixation input reaches E units of both hemispheres with the same (mostly negative)
     sign, and its noise accounts for a partial r of about 0.30–0.33.
  2. **The inhibitory commissure.** With shared noise removed, bilateral networks keep r = +0.14, close to the monkey
     value. No-commissure networks drop to about 0. Commissural I units project to E units in both hemispheres, so
     their private noise reaches both sides as shared inhibition.

  The I units show the push-pull signature that E units lack. In bilateral networks the I-unit partial r is
  -0.083 +/- 0.018 (9/10 significantly negative); in no-commissure networks it is +0.001.

  The group difference (bilateral vs no-commissure) is significant: Mann-Whitney p = 0.0003 for E units and p = 0.0002
  for I units.
* **Sources of trial-to-trial variability in the model:**
  * private state noise on every unit (0.1·sqrt(dt/tau)) per 10-ms step;
  * white input noise of SD 0.05 on each input channel. Only the fixation channel projects to both hemispheres, so it
    is the one shared noise source. Visual and goal channels are topographic and project to one hemisphere only.
    Reward-context channels are not connected in these networks;
  * a random urgency slope per trial and softmax action sampling, which affect choice only;
  * no slow or trial-level state such as history or arousal, and every trial starts from x = 0.

## Interpretation for hub vs resolver (with caveats)
* Positive left-right co-fluctuation in the monkey is **not diagnostic**. Even this model, with only fast white noise,
  produces it in two ways: a shared afferent (the fixation input) and a purely inhibitory commissure. So the monkey's
  +0.15 does not by itself demand an external hub. The I-unit anticorrelation is a model-specific signature that could
  be tested against recorded non-classic neurons.
* The model has no slow pre-target state that carries into the choice. Its competition is resolved entirely within
  the trial by target-locked input. The monkey's pre-target-bias-to-choice link (a 0.6 log-odds/SD effect beyond
  history and gaze) therefore needs something this network lacks: slow, trial-scale state that biases one side and
  persists until the decision. Examples are an external/hub input or slow intrinsic dynamics.
* Caveats:
  * The model co-fluctuation is white-noise and fast; the monkey's is trial-scale (spike counts) and was measured with
    history and gaze regressed out. Magnitudes are not directly comparable, and the model r depends on the arbitrary
    input-noise level.
  * Model choices are near-deterministic (no lapses; P(big | conflict) 0.89), which leaves little choice variance for
    any pre-target signal to explain.
  * Fixation was lengthened by 500 ms beyond training (800–1,000 ms instead of 300–500). Behavior was unchanged:
    P(big | conflict) 0.894 vs 0.891 at native timing, and the rates have settled by about 300 ms
    (`P5_model_fixation_settling.csv`).
  * The window and trial selection were fixed before any results were seen; the cell-adjusted fit was added after
    the main-effects fit separated. Both are reported.

## Method
`scripts/aim1_panels/conflict_model_bilateral_state.py`, run single-threaded on CPU (about 20 min):
* **Simulation.** Each network is run with the repo's sampler and rollout on 3,500 two-target trials (conflict and
  congruent) with its own noise.
* **Analysis window.** The window is [target onset - 300, target onset). Primary set: trials with dT <= 0, about
  2,000 per network, for which the window lies entirely inside fixation. Sensitivity set: all dT (`trial_set = all_dT`
  in the CSVs).
* **Population activity.** Each unit's window-mean rate is z-scored across trials, and the mean is taken per
  hemisphere. This is done for E and I units separately.
* **Choice regression.** Main-effects logistic: chose_right ~ z(bias) + big_right + sal_right + dT dummies. The
  cell-adjusted version uses fixed effects for big x sal x dT cells and drops cells where the choice never varies.
* **Partial correlation.** Covariates are big_right, sal_right, dT dummies and t_go. The p value comes from the
  t distribution.
* **Controls.** Each control is paired with the intact run (same random streams): fixation-channel input noise zeroed,
  and commissure cut (`comm_scale = 0`).

## Files
* `P5_i_choice_by_bias_tercile.csv`, `P5_ii_LR_partial_correlation.csv`, `P5_iii_bias_choice_coefficient.csv`: the
  plotted data (monkey and model, all conditions and unit classes).
* `P5_model_per_network.csv`: all per-network statistics, every condition, trial set and unit class.
* `P5_model_tercile_per_network.csv`
* `P5_model_behavior_check.csv`
* `P5_model_fixation_settling.csv`
* `stats.csv`: group summaries, Wilcoxon tests vs 0, Mann-Whitney tests between groups, paired control effects, and
  the monkey values.
* `run_log.txt`
* `plot_P5.py` plus `style.py`: standalone plot that reads only the CSVs (`python plot_P5.py`).

## Provenance
* **Networks.** `D:\OneDrive - University of Pittsburgh\Code\conflict-task-RNN\results\final_bilat_s0-9` and
  `final_nocomm_s0-9` (`model.pt`, read-only).
* **Model code.** Copied unchanged to `external/conflict_task_rnn/` (config, task, network, rollout, analysis).
  Results summary: that repo's `docs/results_summary.md`.
* **Monkey values.** `../P4_bilateral_state/` (CSVs and `stats.csv`), from Newton_SC_2026 `code/run_pretarget_bias.m`.
