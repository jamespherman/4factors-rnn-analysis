# Recovery test on all sessions (recovery_v12, 2026-10-14)

Corrected exports, all 7 sessions, switch-window split (as `results/grid_v12s`). Teachers per
session are the best `results/grid_v12` restarts by selection LL (manifest.json):
null = none_free; ext = free_free (per-unit context into all units); local = freeI_free (per-unit
context into I units); shared = shared_free with gain 1–2, set so that the SD across neurons of its
context-driven reward effect matches ext's. Students: 5 variants × 3 restarts (restart = split seed).

Reports: `recovery_report_model_on_test.md` (student effects from the test windows; used below)
and `recovery_report.md` (student effects from all trials). Each dataset has its own
`evaluation_report.md`.

## Reward pre-target effect, pooled over sessions (held-out r, mean over 3 restarts)

| Teacher | Ceiling | none | shared | sharedI | freeI | free |
|---|---|---|---|---|---|---|
| null | 0.00 | +0.10 | +0.06 | +0.06 | +0.03 | −0.02 |
| ext | 0.47 | +0.03 | +0.22 | +0.08 | +0.41 | +0.49 |
| local | 0.52 | −0.02 | +0.27 | +0.30 | +0.50 | +0.51 |
| shared | 0.64 | −0.05 | +0.60 | +0.61 | +0.52 | +0.50 |

Paired student differences (same split), mean [min, max] over restarts:

| Teacher | shared − free | sharedI − freeI | free − freeI |
|---|---|---|---|
| null | +0.08 [−0.19, +0.36] | +0.03 | −0.05 |
| ext | −0.28 [−0.30, −0.26] | −0.33 [−0.42, −0.21] | +0.09 [+0.04, +0.11] |
| local | −0.23 [−0.29, −0.18] | −0.19 [−0.27, −0.16] | +0.01 [−0.09, +0.08] |
| shared | +0.10 [+0.02, +0.17] | +0.09 [−0.01, +0.27] | −0.02 |
| **real data (grid_v12s, 6 restarts)** | −0.04 ± 0.05 (SE) | +0.00 ± 0.03 | |

## Findings
1. **The procedure works when the teacher's pattern is reliable.** No false positives on null
   teachers. Per-unit teachers are recovered best by per-unit students, and the shared teacher
   best by shared students. The shared − free difference separates them in every restart:
   −0.23 to −0.28 for per-unit teachers, +0.10 for the shared teacher.
2. **The pilot's shared-teacher gap is closed.** With gain-matched effect size, a shared signal
   produces a neuron-specific pattern (ceiling 0.64) that per-unit students fit almost as well
   (0.50) as shared students (0.60). Per-unit students are therefore not automatically better.
3. **E-vs-I target: weakly identifiable.** For the ext teacher, free beats freeI (+0.09; held-out LL
   +0.5 vs −9.2). For the local teacher the two tie (+0.01), with freeI slightly ahead in LL
   (+4.2 vs +2.8).
4. **Real data fall between the two cases and are not decisive.** On real data shared − free is
   −0.04 ± 0.05, but the real ceiling is 0.21, against 0.47–0.64 for the teachers. Scaling the
   teacher differences by ceiling predicts about −0.12 for a per-unit truth and about +0.03 for a
   shared truth. The observed value lies between and within about 1.5 SE of both.
5. **Teachers are easier than the data.** Their context effect is identical at every switch,
   while recorded neurons' effects vary from switch to switch (ceiling 0.21). A calibrated test
   needs teachers whose ceiling matches the data, e.g. by lower gain or by adding the recorded
   neurons' slow fluctuations to the teacher rates.
