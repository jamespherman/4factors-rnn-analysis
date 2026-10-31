# Retrain priority-hub (B) and outcome-hub (C) networks with realistic RT; re-test S9 (2026-10-31)

## Why
S9 (single-target RT vs belief value) separates graded (B: -0.32) from categorical (C: +0.02) delivery in hand-builds,
and the monkey shows -0.21 (08-20). The trained v2 ensemble cannot test it: with time_cost 5e-4/ms its RTs sit near a
floor (95-145 ms vs 209 ms recorded) and, once calibrated to the monkey's weak value use in choice (engaged fraction
0.2-0.65), every architecture gives ~0. v1 networks (no time cost) were too slow (220-340 ms).
Resolvers (A/AE) are already excluded by S8 and are not retrained.

## Design
- Same v2 settings (nuisance randomised in S2/S3, train_engaged_min 0.2, side bias 0.3, shape_vis 0.25) with
  time_cost in {1e-4, 2e-4} (brackets 209 ms between the v1 and v2 extremes).
- B seeds 0-5; C seeds 0-8 (C seeds 1-3 failed before; more seeds keep >= 3 usable). 30 runs, 4 per preempt GPU.
- Exclusion criteria fixed in advance (as v2): one-sided choice (|P(right) - 0.5| > 0.35 at S3 end) or completion < 0.9.
- Evaluation (srs_stage0.py signature run, 10 sessions per network): calibration to 08-20 choices (engaged fraction,
  side bias) AND RT. Primary: S9 per network, B vs C distributions, recorded -0.21. Secondary: S9 under full
  engagement (w = 1), S7 (SC value scaling on single-target trials), S8 null check.
- Decision rule: B and C separate if their per-network S9 ranges do not overlap at the calibrated setting and the
  separation exceeds the per-session SD of S9 on recorded-size data (~0.05). If neither architecture reaches -0.21
  at calibrated choice behaviour, report that RT-value coupling stronger than choice-value use is unexplained by
  both hubs (itself a constraint: e.g. a value-driven vigor signal separate from choice).

## Caveats
RT in the model is go-signal processing after a 450 ms delay; value sensitivity reflects the pool's state at go.
Matching the RT level by time cost is a tuning choice; the S9 contrast, not absolute RT, is the claim.

## v2 after independent review (incorporated)
1. Engagement calibration mechanically shrinks S9 (disengaged trials erase the shown value), so the primary
   architecture test is S9 at w = 1 (and an S9-vs-w curve); the monkey comparison uses the ratio of RT-value to
   choice-value coupling rather than raw S9 at the calibrated w.
2. Trained C may not be categorical on single-target trials (softmax against the no-go option can be graded). For
   every network report the upstream graded-ness (Spearman of disinhibition with value; slope at go) next to S9. If C
   must be categorical by hypothesis, a variant with a fixed high softmax gain is trained and labelled as such.
3. The fixed 450 ms delay lets networks anticipate go (the recorded task is also fixed, so this is kept faithful), but
   a jittered-delay training variant (+-100 ms) is added if the pilot shows anticipation-driven floor RTs.
4. Pilot first (cheap, local, no retraining): raise the hazard threshold of the existing v2 B and C networks to move
   RT off the floor and measure S9 at w = 1 (scripts/srs_rt_shift_pilot.py). Retraining is launched only if needed.
5. Match median and IQR of RT; set rt_jitter_sd to the recorded spread if needed.
6. A flat time cost gives no normative reason for vigour to scale with value: any S9 is mechanistic, stated as such.
7. Decision rule: rank test plus effect size across networks; excluded seeds reported per architecture.
