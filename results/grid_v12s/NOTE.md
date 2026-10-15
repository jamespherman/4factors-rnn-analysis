# Context grid with switch-window split (grid_v12s, 2026-10-14)

Corrected exports (pipeline 1.2.0), all 7 sessions, verified trials, unstable trials masked.
Same variants as `results/grid_v12` plus `shared_free_qshift`; 6 restarts. Each restart holds
out ±30-trial windows around 2 random non-adjacent reward switches for test and 2 for
selection (split seed = restart), so slow per-neuron fluctuations cannot leak from training
trials into the test windows, as they did with the random interleaved split of grid_v12.

Reports: `by_session_report.md` (model effects from all trials) and
`by_session_report_model_on_test.md` (model effects from the test windows only, the same
switches as the recorded test effect; the cleaner comparison).

## All sessions, reward pre-target effect (held-out r, mean [min, max] over 6 restarts)

| Variant | Model on all trials | Model on test windows |
|---|---|---|
| shared_free | +0.10 [-0.07, +0.24] | +0.21 [+0.04, +0.45] |
| sharedI_free | +0.08 | +0.19 |
| freeE_free | +0.11 | +0.19 |
| freeI_free | +0.10 | +0.19 |
| free_free | +0.10 [-0.01, +0.16] | +0.16 [+0.06, +0.26] |
| free_free_qshift | +0.01 | +0.12 |
| shared_free_qshift | -0.01 | +0.10 |
| none_free | -0.08 | +0.14 |
| *ceiling* | +0.21 [+0.01, +0.38] | |

Paired over restarts (same split for all variants), mean ± SE, restarts with a positive difference:

| Difference | Model on all trials | Model on test windows |
|---|---|---|
| free − free_qshift | +0.09 ± 0.05 (5/6) | +0.04 ± 0.06 (4/6) |
| shared − shared_qshift | +0.11 ± 0.07 (4/6) | +0.11 ± 0.08 (3/6) |
| free − none | +0.18 ± 0.07 (6/6) | +0.02 ± 0.03 (3/6) |
| shared − none | +0.18 ± 0.08 (5/6) | +0.07 ± 0.05 (4/6) |
| free − shared | −0.00 ± 0.04 (3/6) | −0.04 ± 0.05 (2/6) |
| freeI − sharedI | +0.03 ± 0.05 (3/6) | +0.00 ± 0.03 (3/6) |

## Findings
1. **Measured across reward switches, the per-neuron reward-context pattern is weak.** The
   ceiling (recorded effect on held-out switches vs on the rest) is 0.21, down from 0.55 with
   the interleaved split. Most of the earlier reliability was slow per-neuron drift.
2. **Aligned context helps only slightly.** Models with correctly aligned context beat their
   shifted controls by about 0.1 in r, within 1–2 SE. Against no context the gain is
   +0.18 when model effects use all trials, but only +0.02 to +0.07 on the test windows, where
   the trial composition alone gives the no-context model r = 0.14.
3. **Shared and per-unit context are indistinguishable** (free − shared ≈ 0 in every grouping).
   The earlier conclusion that only per-unit context reproduces the pattern is withdrawn.
4. **Held-out likelihood does not favour context.** Over all sessions every context variant
   has lower held-out LL than none_free (−3 to −8 ×1e-3), driven by Newton 08_15. On Newton
   08_13 + 08_14, the sessions with a population reward effect, aligned variants gain +1 to +3.5
   and the shifted controls lose −1.4 to −2.2, but restart ranges overlap.
5. Per session (see reports): single-session r values span roughly ±0.6 across restarts, so no
   per-session conclusion is possible. Newton 08_14 should be treated with caution (slow drift
   in many neurons; see `data/README.md`).
6. The visual probability effect is not reproduced by any variant (r ≈ 0, ceiling 0.05).
