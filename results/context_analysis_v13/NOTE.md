# Stage A on the 2026-10-29/30 exports (curation round 2, new SC classic rule)

All 7 SC sessions, every successful trial, rates masked outside each neuron's stable period.
293 neurons (103 E, 190 I); previously 270 (116 E, 154 I). Full tables: `stage_a_report.md`.
Previous exports are archived in `data/export_2026-10-14_v1.2/` (results in `context_analysis_v12`).

Newton_08_13 was first exported (2026-10-29) with the pre-2026-10-14 event timing (events shifted
0.45-0.65 s; no target-onset visual transient). The pipeline re-exported it on 2026-10-30 with fixed
timing (44 neurons, 653 trials; transient restored); everything here uses the fixed file.

- Pre-target reward context (K = 30): +0.31 sp/s, p = 6e-05 across 105 switches (E +0.18, p 0.035;
  I +0.38, p 0.0005). v12: +0.28, p 0.0007.
- Per session (mean over switches):

  | Session | Neurons | Reward pre | Block pre |
  |---|---|---|---|
  | Feynman_08_12 | 17 | +0.31 (p 0.15) | +0.31 (p 0.12) |
  | Feynman_08_13 | 39 | -0.01 (p 0.89) | +0.18 (p 0.11) |
  | Feynman_08_14 | 19 | +0.24 (p 0.37) | +0.37 (p 0.23) |
  | Feynman_08_15 | 44 | +0.11 (p 0.51) | +0.13 (p 0.45) |
  | Newton_08_13 | 44 | +0.79 (p 0.0034) | +0.38 (p 0.28) |
  | Newton_08_14 | 63 | +0.64 (p 0.002) | +0.28 (p 0.42) |
  | Newton_08_15 | 67 | +0.10 (p 0.51) | +0.58 (p 0.045) |

  As before, the reward effect comes from Newton 08_13 and 08_14.
- Pre-target block: +0.32 sp/s, p = 0.0006.
- Visual probability component: -0.74 sp/s, p = 9e-10 (v12: -0.80).
