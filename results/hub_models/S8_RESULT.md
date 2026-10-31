# S8: identity coding beyond value in SC excludes a resolver that binds value in SC (2026-10-31)

Plan: `plans/2026-10-31_hub_resolver_models_plan.md` (v2). Models: conflict-task-RNN `code/srs` (snapshot in
`external/conflict_srs`). Recordings and detection analysis: bilateral-SC-recordings
(`D:\Google Drive\HermanLab\data\Newton_SC_2026\docs\NOTE_2026-10-31_competition_for_4factors.md`, last section;
`code\run_single_target_identity_null.m`).

## Logic
A resolver must bind target identity to block value inside SC, so its SC needs identity at each location and carries a
stable per-unit identity code beyond value (conjunction units). Hubs bind upstream and send SC only value/priority, so
their SC carries no identity code beyond value. Real SC could carry a purely sensory shape code (horizontal vs vertical
bar); a positive result would therefore be ambiguous, but a null is not.

## Models
Per-unit true identity effect beyond value: block-level log(T1/T2) on single-target trials, averaged over blocks (value
cancels across the alternating rich identity), units contralateral to the target, noise-free rates mapped to recorded-like
sp/s (`scripts/srs_identity_effect_size.py`; `identity_delta_true*.csv`). Trained v2 ensemble (nuisance randomised in
training, shape coding shape_vis 0.25 in all architectures): fraction of units with |delta| > 0.2 (silent units in the
denominator): A (resolver) 0.42-0.61, AE (resolver, input onto E only) 0.28-0.58, B (priority hub) and C (outcome hub)
<= 0.04. Hand-builds: A 0.40, B/C ~0. C seeds 1-3 excluded as training failures (one-sided choice).

## Recordings (08-20; 301 single-target trials, 8 blocks; 27-31 contralateral units valid in both block types)
No identity coding beyond value: quasi-Poisson mean |z| 0.54-0.62 (chance); per-unit |delta| distribution at the
permutation null in every window (delay mean |delta| 0.131 vs null 95th percentile 0.166); cross-block sign consistency at
null. No value scaling either (population log rate ratio within +-8%). 08-19 also at null (2 blocks per type).

## Detection probability (each model's per-unit delta distribution planted in the real 08-20 counts, 500 plants)
| | visual | delay (primary) | pre-saccadic |
|---|---|---|---|
| A (6 seeds pooled) | 0.97 | 1.00 | 1.00 |
| AE (6 seeds pooled) | 0.77 | 1.00 | 1.00 |
| B (6 seeds) | 0.01 | 0.00 | 0.00 |
| C (3 retained seeds) | 0.00 | 0.00 | 0.01 |
All 12 resolver networks (including the sparsest, AE_s3: delay 0.99) would have been detected at P >= 0.99 in the delay and
pre-saccadic windows; the recordings are at null, so these resolvers are excluded. All hub networks are consistent with the
data (false detection <= 0.05).

## Caveats
- Assumes model log-rate effects transfer to recorded units at recorded rates.
- A resolver with a much sparser or weaker identity code than any trained or hand-built network (e.g. a quarter of units at
  |delta| 0.2: ~20% power) would not be excluded.
- One session with enough blocks (08-20); the recorded populations had few classic units on the side of interest.
- This tests where identity x block value is bound (SC vs upstream), not where the choice is resolved (priority vs outcome
  hub); that contrast rests on S9 and the remaining signatures (pending trained-ensemble analysis).
