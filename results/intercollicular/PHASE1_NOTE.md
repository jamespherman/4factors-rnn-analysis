# Phase 1: ipsilateral responses and suppression in one SC (unilateral data, v13 exports)

Plan: `plans/2026-10-31_intercollicular_plan.md` (v2). Scripts: `scripts/ic_g1_geometry.py`, `ic_g23_timing.py`,
`ic_g4_factors.py`; shared tables `src/intercollicular.py`, spikes `src/spikes.py` (validated against the export:
median r 0.98-0.99 between spike counts and export window rates). Per-test tables in `g1_geometry/`,
`g23_timing/`, `g4_factors/` (`summary.md`). Diagonal-target sessions only (6 sessions; Newton 08_15 separately).
Classes: classic SC (E, n=72 here) and putative interneurons / non-classic (I, n=154).

## G1, distance-matched geometry (paired test vs pre-target baseline, p < 0.01)
- Visual epoch: non-classic neurons respond as often at the ipsi 90-deg neighbour as at the contra 90-deg neighbour
  (activated 27% vs 25%; suppressed 19% vs 26%, n.s.). Classic neurons fall off across the vertical meridian
  (activated 36% contra-neighbour vs 11% ipsi-neighbour). Non-classic visual fields are broad and not bounded by
  the midline; visual suppression is not hemifield-specific.
- Pre-saccadic epoch: suppression depends on direction, not hemifield. Non-classic: 19% at either 90-deg
  neighbour, 33% at the 180-deg opposite (opposite vs ipsi-neighbour p = 0.0015). Classic neurons show the same
  pattern (12% vs 3%, p = 0.04).
- Post-saccadic: non-classic suppression ipsi-neighbour 23% vs contra-neighbour 14% (p = 0.02); reafference and
  new eye position contribute here, so not interpreted.
- Newton 08_15 (meridian targets): pre-saccadic suppression is again largest for the opposite (horizontal ipsi,
  61% of non-classic) target.
- 31 non-classic units are pause-type (saccade suppression at >= 3 of 4 locations).

## G2, latency vs amplitude
- Calibration from classic contra responses: latency falls with amplitude (-14 ms per ln unit; r = -0.34; residual
  SD 35 ms, so single-condition latencies are noisy).
- Within unit, ipsi-neighbour activations are 5 ms later than contra (p = 0.016); 2.7 ms of that is expected
  from their smaller amplitude; excess 3.7 ms, 95% CI [-1.3, 6.8] (unit bootstrap). A 2-7 ms relay delay is
  therefore neither shown nor excluded.
- Visual suppression latency, ipsi- minus contra-neighbour within unit: +11 ms, n = 15, p = 0.33.

## G3, target- vs saccade-alignment
- Ipsi responses are not predominantly saccade-locked: non-classic ipsi-neighbour activations 52% saccade-locked,
  opposite-location activations 49%, suppressions 36-37%, vs 36-39% for contra responses. No sign that ipsi
  responses relay the opposite SC's motor burst.

## G4, factor inheritance (sign(response) x ROC selectivity)
- High salience deepens visual-epoch suppression at ipsi locations (+0.107, p = 3e-9) as it does at contra
  locations (+0.095). Ipsi activations and suppressions carry no reward modulation (~0; classic contra +0.05);
  probability and identity effects are small everywhere. Pattern fits visual/feedforward input better than a relay
  of the opposite SC's classic population, but the reward contrast is weak (ipsi vs contra activations p = 0.06).

## Interpretation (against the plan's hypotheses)
- Against H-target as the source of non-classic ipsi responses: they are as frequent as equidistant contra
  responses, carry salience but not reward, are not saccade-locked, and show no reliable extra latency.
- The one feature that fits opposite-SC (or any opponent) input is pre-saccadic suppression tuned to the opposite
  saccade vector (180 deg > 90 deg, in both classes). It cannot distinguish commissural inhibition from
  extra-collicular opponent signals (e.g. SNr) or long-range inhibition within the SC map.
- Under the plan's decision rule, Phase 1 does not leave a strong case for the two-SC model. The bilateral C1
  latency test (when the opposite SC's burst starts relative to this suppression) decides the remaining question.
