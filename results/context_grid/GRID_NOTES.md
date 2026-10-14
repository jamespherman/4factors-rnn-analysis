# Context grid (2026-10-13)

> **Superseded (2026-10-13):** computed on all exported trials, including fixation breaks and trials with unreliable event times (about 30-50% of trials). See `data/trial_audit` and `--trials verified`; verified-trial results are in `results/context_analysis_verified` and `results/verified_gate`.

Setup from `results/fit_ablation/ABLATION_NOTES.md`: drift on, context-conditioned losses off,
3 sessions x 5 seeds per variant. Full table in `evaluation_report.md`.

## Result: the context comparison is not yet informative

- Spatial fit is now consistent. Mean r is +0.26 to +0.41 across variants, and context
  variants are slightly above `none_free` (+0.28).
- **Positive controls fail.** These models can express every recorded context effect by
  construction, and still reproduce none of them per neuron:
  - `free_free` has a free context weight per unit. Its neuron-wise r with the recorded
    reward-context effect is +0.00, against a recorded split-half reliability of 0.71.
  - `trialvalue_free` is handed each trial's reward and probability. It gets the
    probability effect's sign wrong: +0.81 sp/s vs -1.55 recorded.
- **Block effect overshoots.** All block-context models produce population-mean block
  effects 2-3x the recorded +0.79 sp/s. Without a loss term on the context means, the
  context weights are unconstrained.
- **Likelihood:** I->E constraints lower held-out LL (`none_global` -11.5, `none_spatial`
  -14.9, `shared_global` -13.0, `shared_spatial` -16.2 x1000). Seed spread is about +/-15,
  so this is suggestive at best.
- **Weak context loss:** at 0.25, the context-conditioned loss neither helps the context
  effects nor hurts the spatial fit.

Why: the context effects are about 1 sp/s pre-target. In single-trial and PSTH losses they
sit far below per-bin noise, so nothing in the objective rewards matching them. If even the
upper-bound models cannot match them, a failure of the shared-context models says nothing
about H_ext vs H_local.

## Next: window-mean loss gate (`results/lwin_gate`)

`--lwin-scale` adds a loss on the mean rate in the pre-target (-300..0 ms) and visual
(50..250 ms) windows, per location x context cell. It is measured in units of the
recorded standard error, so it targets exactly the statistics the effects are built from.

Gate: `free_free` and `trialvalue_free` must reproduce the per-neuron reward-pre and
prob-vis effects, without losing spatial fit, before the shared-context comparison is
rerun with this loss.
