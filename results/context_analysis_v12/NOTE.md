# Stage A on the corrected exports (pipeline 1.2.0), 2026-10-14

All 7 SC sessions (Feynman 08_12-15, Newton 08_13-15), every successful trial, rates
masked outside each neuron's stable period. 270 neurons (116 E, 154 I). Full tables:
`stage_a_report.md`. Previous results (`context_analysis_verified`, Newton only, v1.0
exports) are superseded.

- Labels: reward map and block agree 100% with raw trialInfo in every session.
- Pre-target reward context (K = 30): +0.28 sp/s, p = 0.0007 across 105 switches (E +0.28,
  I +0.24). Smaller than the +0.42 on the old Newton subset, but on many more switches.
- It comes almost entirely from two Newton sessions. Per session (mean over switches):

  | Session | Neurons | Reward pre | Block pre |
  |---|---|---|---|
  | Feynman_08_12 | 11 | +0.45 (p 0.11) | +0.46 (p 0.07) |
  | Feynman_08_13 | 41 | +0.02 (p 0.84) | +0.19 (p 0.13) |
  | Feynman_08_14 | 26 | -0.10 (p 0.66) | +0.23 (p 0.63) |
  | Feynman_08_15 | 40 | +0.11 (p 0.56) | +0.17 (p 0.34) |
  | Newton_08_13 | 33 | +0.80 (p 0.004) | +0.47 (p 0.30) |
  | Newton_08_14 | 60 | +0.61 (p 0.003) | +0.32 (p 0.34) |
  | Newton_08_15 | 59 | +0.09 (p 0.61) | +0.60 (p 0.046) |

  Newton_08_14 has the most slow drift (pipeline note, 2026-10-14). The switch-triggered
  contrast is local (+/-30 trials), so a slow trend mostly cancels, but treat it with caution.
- Pre-target block: +0.35 sp/s, p = 0.0015.
- Visual probability component: -0.80 sp/s, p = 2.5e-10 (was -1.30 on the old subset).
- Slow drift is widespread: the trial-number regressor has |t| > 2 in 69% of neurons.
- Population decoding of context (A2) is near its circular-shift null in every session.
  Decoding now compares equal-size rate-matched I and E sets (sessions with more I than E
  match I to E).
