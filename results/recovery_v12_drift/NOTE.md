# Recovery test with realistic noise (recovery_v12_drift, 2026-10-15)

Same teachers, split and students as `results/recovery_v12`, but each teacher's rates are
multiplied by its recorded neuron's own slow rate fluctuation (`make_synthetic_data.py
--drift-amp`, `slow_gain`). The fluctuation is smoothed over about 15 trials, normalized to
mean 1, and circularly shifted by whole 96-trial reward cycles plus 24 trials, so it carries no
reward-context signal. Its strength is set per teacher (`scripts/calibrate_teacher_drift.py`,
`drift_calibration.json`: null 2, ext 1.25, local 1.5, shared 2) so that the switch-split ceiling
matches the recordings (0.21). Students: 5 variants × 6 restarts × 4 teachers × 7 sessions = 840
fits on CRC.

Reports: `recovery_report_model_on_test.md` (student effects from test windows, as for real data
in `results/grid_v12s`) and `recovery_report.md` (student effects from all trials).

## Reward pre-target effect, pooled over sessions (held-out r, test windows, mean over 6 restarts)

| Teacher | Ceiling | none | shared | sharedI | freeI | free |
|---|---|---|---|---|---|---|
| null | −0.01 | +0.11 | +0.16 | +0.09 | +0.15 | +0.24 |
| ext | 0.19 | +0.08 | +0.21 | +0.23 | +0.35 | +0.39 |
| local | 0.21 | +0.12 | +0.26 | +0.24 | +0.32 | +0.38 |
| shared | 0.23 | +0.15 | +0.13 | +0.14 | +0.22 | +0.29 |
| *real data (grid_v12s)* | 0.21 | +0.14 | +0.21 | +0.19 | +0.19 | +0.16 |

## Shared − free, paired over restarts (mean ± SE)

| | held-out r (test windows) | held-out r (all trials) | held-out LL ×1e-3 |
|---|---|---|---|
| null teacher | −0.08 ± 0.05 | −0.03 ± 0.04 | +1.4 ± 1.5 |
| ext teacher | −0.19 ± 0.06 | −0.08 ± 0.04 | −9.2 ± 5.3 |
| local teacher | −0.12 ± 0.07 | −0.10 ± 0.06 | +0.2 ± 5.3 |
| shared teacher | −0.16 ± 0.06 | −0.14 ± 0.07 | +7.4 ± 4.0 |
| **real data** | **+0.04 ± 0.05** | +0.00 ± 0.04 | +2.7 ± 3.2 |

LL uses `nanmean` over neurons: in Newton 08_13–08_15, 2–3 synthetic neurons have NaN LL in
restarts 3 and 6 (same neurons for every student, so the paired difference is unaffected).

## Findings
1. **At data-like reliability the r contrast no longer discriminates.** Shared − free is
   negative for every teacher, including the shared teacher (−0.16) and the no-context teacher
   (−0.08). Per-unit students score higher whatever the mechanism.
2. **Held-out r on test windows is biased upwards.** With no context in the teacher, per-unit
   students reach 0.24 against a ceiling of −0.01 (0.01 when the student effect uses all trials).
   Scores must be compared with matched synthetic controls, not with zero or the ceiling.
3. **Held-out LL separates mechanisms only weakly**: positive for the shared teacher, negative
   for ext, near zero for local and null, with large restart spread. Real data (+2.7 ± 3.2) are
   within about 1 combined SE of shared, local and null, and about 2 from ext.
4. **The idealized-recovery argument is withdrawn.** `results/recovery_v12/NOTE.md` finding 4
   (real data match the ceiling-scaled shared prediction, ~3 SE from per-unit) does not hold:
   the real r contrast lies above every realistic teacher, so it matches none of them well.
   Conclusion: these fits cannot decide whether reward-context input is shared or per-unit.
5. **The teachers still differ from the data.** Per-unit students beat no-context students by
   0.13–0.31 on every teacher but by only 0.02 on the real data. The synthetic noise is a
   per-neuron gain on each neuron's own slow rate; real fluctuations may be shared across
   neurons or otherwise structured. More reward switches per neuron would be the direct remedy.
