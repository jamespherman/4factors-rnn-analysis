# Recovery pilot (2026-10-14)

Session 08_14 (verified trials), one session only. Teachers are the best 08_14 restarts from
`results/verified_grid`:
- ext: `free_free`, per-unit context into all units
- local: `freeI_free`, per-unit context into I units only
- null: `none_free`

Context weights are scaled by g = 1 or 3. Students: 5 variants x 3 restarts, fixed split,
window-mean loss. r = held-out correlation across the 41 neurons between student and synthetic
"recorded" (teacher-generated) pre-target reward-context effect. Per-dataset tables are in
`<dataset>/evaluation_report.md`.

| Teacher | Ceiling | none | shared | sharedI | freeI | free (all) |
|---|---|---|---|---|---|---|
| null | 0.50 | -0.19 -0.15 +0.27 | +0.10 -0.21 +0.47 | +0.21 +0.18 +0.34 | -0.03 +0.60 +0.25 | -0.17 +0.42 +0.22 |
| ext g1 | 0.66 | -0.45 -0.19 -0.42 | +0.57 +0.14 +0.56 | +0.49 +0.24 +0.62 | +0.58 +0.49 +0.53 | +0.71 +0.70 +0.65 |
| ext g3 | 0.99 | -0.48 -0.78 -0.29 | +0.93 +0.94 +0.93 | +0.93 +0.94 +0.93 | +0.94 +0.92 +0.94 | +0.99 +0.98 +0.99 |
| local g1 | 0.80 | -0.37 +0.27 -0.59 | +0.72 -0.40 -0.41 | +0.48 +0.49 -0.50 | +0.73 +0.75 +0.84 | +0.64 +0.68 +0.73 |
| local g3 | 0.94 | +0.04 -0.08 +0.08 | +0.68 +0.76 +0.49 | +0.74 +0.75 +0.76 | +0.90 +0.93 +0.93 | +0.89 +0.94 +0.94 |

## Findings
1. **A shared context signal can produce neuron-specific context effects.** When the teacher has
   them, shared-context students recover much of the pattern: 0.93 at g3, and 0.14-0.62 at g1,
   where the teacher's effects are about as large as the recorded ones. The recurrent circuit
   turns a common reward-map input into location-dependent effects. So "shared context cannot
   make neuron-specific effects" is false.
2. **That sharpens the real-data result.** On real data, shared students reach +0.04 against
   +0.43 for per-unit students (ceiling 0.64). On teacher data of matched effect size, they
   reach about 60% of the per-unit students (mean 0.42 vs 0.69, ext g1). So the recorded
   reward-context pattern contains a component that this circuit does not generate from a
   global reward signal, unlike the teachers' patterns, which are themselves fit to the data.
   This supports neuron-specific (spatially specific) context input.
3. **E-vs-I origin is only weakly identifiable at real effect sizes.** For the local teacher at
   g1, freeI students beat free students (r 0.77 vs 0.68; delta LL +20 vs +15). For the ext
   teacher at g1, free beats freeI (0.69 vs 0.53; delta LL +7.3 vs +7.0). The direction is right
   but the margins are small, so claims about where the specific input lands need more data.
4. **Single-session r is noisy.** On null data, students range from -0.21 to +0.60, and one
   none_free restart reaches +0.27. Conclusions need all sessions pooled (as in the real-data
   grid) and more restarts.

## For the final run (after the re-export)
- Run the recovery test on all sessions, with teachers at g = 1, plus a teacher whose
  effect size matches each session's recorded effect.
- Use 6 restarts.
- Report the real-data shared/per-unit ratio against the same ratio on teacher data
  (finding 2) as the main calibrated comparison.
