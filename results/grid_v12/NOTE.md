# Context grid on the corrected exports, random trial split (2026-10-14)

All 7 SC sessions (pipeline 1.2.0, stable-period masking), 8 variants x 6 restarts, fixed
stratified random split (`--split-seed 0`), window-mean loss. Per-session and session-group
tables: `by_session_report.md` (`scripts/summarize_grid_by_session.py`).

Held-out r with the recorded pre-target reward-context effect per neuron, all sessions
(ceiling 0.55):

| Variant | Mean | Range over restarts |
|---|---|---|
| freeE_free | +0.50 | +0.46 to +0.55 |
| free_free | +0.47 | +0.44 to +0.50 |
| freeI_free | +0.41 | +0.26 to +0.52 |
| sharedI_free | +0.39 | +0.28 to +0.53 |
| shared_free | +0.36 | +0.19 to +0.50 |
| free_free_qshift (control) | +0.35 | +0.25 to +0.49 |
| freeI_free_qshift (control) | +0.30 | +0.17 to +0.46 |
| none_free | +0.13 | +0.01 to +0.30 |

## This split cannot separate context from slow per-neuron fluctuations
- The quarter-period shifted-context controls reach 0.30-0.35, and no-context models reach
  0.29 in Newton 08_13/08_14. With misaligned context, the switch-triggered statistic gets no
  help from the context input (the shifted step contributes zero to the after-minus-before
  difference). So most of these r values come from something else.
- That something is slow per-neuron fluctuation on a ~50-trial scale. Test trials are
  interleaved with training trials, so a model that fits each neuron's slow fluctuations on
  training trials also predicts them on neighbouring test trials. Per-unit slow inputs (shifted
  or not) give it that freedom. The recorded "ceiling" (non-test vs test trials) shares the
  same fluctuations and is inflated in the same way.
- Aligned minus shifted: free_free +0.12, freeI_free +0.11. That difference is the part tied
  to the reward context, and it is small.
- Shared-context models (0.36-0.39) are no better than per-unit models with misaligned context.
  This differs from the old Newton-only data (shared +0.04), so the earlier
  "only per-unit context reproduces the pattern" conclusion does not hold on the corrected data.

## Follow-up: `results/grid_v12s`
Same variants plus a shifted shared control, with `--split switch`: whole +/-30-trial windows
around 2 random reward switches are held out for test and 2 for selection (split seed = restart,
so 6 restarts cover different switches). Model effects are also measured on the test trials
alone (`effects_test`).
