# Inter-collicular question: summary of results so far (2026-10-31)

Plan: `plans/2026-10-31_intercollicular_plan.md` (v2, reviewed). Phase 1 details: `PHASE1_NOTE.md`.
Bilateral results (bilateral-SC-recordings session, Newton 2026-08-19/20):
`D:\Google Drive\HermanLab\data\Newton_SC_2026\docs\NOTE_2026-10-31_intercollicular_for_4factors.md`.

## Phase 1 (unilateral): ipsilateral responses look feedforward; opposite-direction pre-saccadic suppression remains
- Visual epoch: non-classic responses at the ipsi 90-deg neighbour are as common as at the equidistant contra
  neighbour (activation 27% vs 25%, suppression 19% vs 26%); classic neurons drop across the midline (36% vs 11%).
- Ipsi responses: no reliable extra latency (excess 3.7 ms, 95% CI -1.3 to 6.8), not saccade-locked, salience-
  but not reward-modulated.
- Pre-saccadic suppression is direction-tuned: 33% of non-classic (12% of classic) units at the 180-deg opposite
  target vs 19% (3%) at either 90-deg neighbour.

## Phase 2 (bilateral, C1/C3)
- Ipsiversive activation in side Y rises with side X's contraversive burst (Δ -6 ms [-8, +18]); a simulated relay
  of the recorded X would lag by 10-27 ms. Points away from relay of neurons like the recorded X (the onset test
  is safe but underpowered, so this is descriptive, not a licensed exclusion).
- Ipsiversive suppression is heterogeneous: in 08-20 right SC it starts with X's burst (relay vs common input
  undecidable); in 08-19 left and one 08-20 left unit it starts in the delay, hundreds of ms before the recorded X
  is modulated (cannot be a relay of that burst).
- Flat cross-SC CCG bound (per-pair suppression r <= 0.02-0.03 at f = 1): with our S = 0.63 (median fractional
  pre-saccadic suppression) and X spikes ~0.10-0.2 per 7-ms efficacy window (from n = 1.4 spikes in -100..0 ms),
  r·N_X ≈ 5-10. A relay is excluded only if it converges from fewer than ~170-500 X neurons; a weak convergent
  relay is not excluded. (Plausibility of that convergence is an open anatomy question.)
- Pending: C2 (population source with saccade covariates).

## Phase 3 (unilateral within-SC CCGs, `b3_ccg/summary.md`)
- 13,578 directed pairs on different channels. Most jitter-significant short-lag excess is broad (±8 ms; 7-12% of
  powered pairs): fast common input. Sharp, asymmetric 1-4 ms peaks (candidate monosynaptic): ~1% of powered pairs
  in every class combination (E->E 0.6%, E->I 0.9%, I->E 0.8%, I->I 1.1%), no clear distance dependence.
- Classic -> non-classic sharp peaks reach 14 ipsi-responsive non-classic units (21/2116 pairs) and none of the
  27 non-ipsi-responsive ones (0/333; Fisher p = 0.1, equal power). Non-classic units, ipsi-responsive ones
  included, receive local input from classic units (consistent with H-local; does not exclude H-target).
- Troughs: 8 (mostly I->I); too few to interpret, as expected (Trepka et al. 2022: troughs ~40x rarer than peaks).
- B4: waveform trough-to-peak (classic 0.40 ms, non-classic 0.35-0.43 ms) and depth do not separate groups
  (all p > 0.1). Nothing points to a distinct commissural-source group.

## Decisions (plan decision rules)
- 162 of 190 non-classic units respond at ipsi locations, so removing them from the I pool (the plan's A-check)
  would leave almost no I units. More to the point, the current RNN already gives every unit every target-location
  input, so ipsi responses can be fit as direct input (H-external is built into the model). The commissural
  question therefore does not change how the existing model's I->E weights should be read.
- The two-SC RNN pilot is not run: Phase 1 and C1 show no relay signature for ipsi activations; the only candidate
  (opposite-direction pre-saccadic suppression) cannot be distinguished from extra-collicular opponent input with
  these data, and the earlier recovery test showed comparable distinctions are not identifiable from fits.
- B2 (single-trial coupling) skipped: H-external predicts the same decoupling as H-target (review point).

## What would decide the remaining question
- Simultaneous bilateral recordings with targets in both hemifields interleaved on the same trials, at matched
  eccentricities (mirror locations), with enough classic units on both sides; then C1-style onset tests of
  pre-saccadic suppression against the opposite SC's burst at the mirror site.
- Causal tests: unilateral SC inactivation while recording the opposite SC (Thomas et al. 2023 design), predicting
  loss of opposite-direction pre-saccadic suppression under H-target but not H-external.
