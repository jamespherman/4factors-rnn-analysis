# Competition analyses P1-P2 (plans/2026-10-31_competition_plan.md v2)

Task structure (from session_data): two alternating block types (7-9 blocks/session). In each, one probability
location (export loc 1 or 3, which are opposite each other) gets 62.5% of targets, the other three 12.5% each.
Reward (350 vs 160 ms juice) varies trial by trial, is not tied to location or stimulus, and is presumably cued
before the target (high/low tone events). Export loc 1-2 are contralateral and 3-4 ipsilateral in every
diagonal-target session, so the probability manipulation pits a contralateral location against its opposite
ipsilateral location.

## P1 behaviour (`p1_behaviour/summary.md`; 5,242 successful trials)
- Images (face, non-face) give faster, faster-velocity saccades than bullseyes (Feynman: face -14%, non-face -9% RT);
  high vs low salience bullseyes do not differ.
- Reward and probability interact (goal x goal): in Feynman, the high-reward cue speeds saccades to the expected
  location by ~9% but slows them to the opposite, unexpected location (reward x opposite +17.9%, p < 1e-6;
  pooled +11.7%); peak velocity mirrors this. Newton's reward effects are weaker (reward x neighbour +3.4%, p = 0.04).
  The reward cue amplifies the location prior rather than adding a location-independent benefit.
- Probability itself has small RT effects (Newton: neighbour -3.5%, p = 0.07; opposite slower by ~3% in cell means).
- Error trials with saccades after go are rare (<15/session except Feynman 08_15: 80); error-direction analysis not
  feasible.

## P2 probability conflict (`p2_conflict/summary.md`; 140 units with RF at a probability location)
- (a) Prior at the RF raises pre-target activity modestly: +0.24 sp/s (p = 0.047); somewhat more with the
  high-reward cue (+0.30 vs +0.18, n.s.).
- (b) Target elsewhere (90-deg neighbour, low probability in both blocks): the prior's advantage peaks at
  0-50 ms after target onset (+0.81 sp/s, p < 0.01; I units +1.17) and is gone by ~100 ms; it does not reverse.
  Meanwhile activity at all non-target RFs falls below baseline from ~150 ms (-1.5 sp/s): the target suppresses
  other locations regardless of the prior.
- (b2) Target in the RF: expected targets give a slightly earlier/larger first response (+1.4 sp/s at 50-100 ms)
  and then a smaller sustained response (-2.4 sp/s at 100-250 ms; expectation suppression).
- (c) Pre-target activity predicts faster RT, but not location-specifically: units with RF at the high-probability
  location -5.2%/SD (target there) and -5.7%/SD (target elsewhere); the same units when the prior points to the
  opposite (ipsi) location -4.7%/SD; all units together -6.4%/SD. A competition account predicts slower RT when the
  competitor's RF is more active; instead any pre-target SC activity goes with faster saccades (general readiness
  or arousal; slow drift not removed beyond block-wise z-scoring).

## Reading against resolver vs hub
- Behaviour shows real factor competition/integration: the reward cue scales the location prior (expected faster,
  unexpected slower). SC carries the prior (pre-target, small) and the reward cue (pre-target +0.31 sp/s,
  stage A), so both goal signals are present in SC before the target.
- After an unexpected target appears, the prior's representation is cancelled within ~100 ms and the target
  suppresses all other locations: the dynamics expected of a map-wide competition, but equally of inputs that
  update when the target appears.
- The RT-predictive SC state is non-specific, so these data do not show SC state deciding between locations.
