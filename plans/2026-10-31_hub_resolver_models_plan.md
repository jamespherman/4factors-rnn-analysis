# Plan: a computational argument for hub vs resolver - task-trained bilateral SC RNNs in the recorded conflict task (v2)

Date: 2026-10-31. v1.1 was reviewed by an independent instance, by conflict-task-rnn-f8 (model owner) and by
bilateral-SC-recordings (analysis owner); v2 merges the three (summary at the end).
Collaborators: conflict-task-rnn-f8 (model code in `conflict-task-RNN/code/srs/`, approved by James),
bilateral-SC-recordings (adapter + its validated pipeline on simulated sessions), 4factors-RNN (this plan; GCP training,
sampling/export, Stage 0 power analysis, perturbations, synthesis).
Background: `results/competition/SYNTHESIS.md`, `plans/2026-10-31_research_hub_models.md`, `plans/2026-10-31_research_wta.md`,
conflict-task-RNN `docs/model_predictions_for_Aim1.md`.

## Scope (stated up front)
In the recorded task (srsClassic) salience has no measurable behavioural effect, so there is no goal-vs-stimulus conflict
for SC to resolve. What models plus these recordings can address is narrower:
1. Where the choice is resolved: graded value delivered to SC, with SC doing the winner-take-all (priority hub, B), vs a
   choice made upstream and delivered near-categorically (outcome hub, C).
2. Where identity x block value is bound to location: upstream (B, C) vs in SC (resolver, A).
The models will also show which task redesign would separate the architectures (luminant salience, time pressure,
variable salient/rewarded onset asynchrony), which may be their most useful output for James.

## Task (as recorded; bilateral-SC-recordings, PLDAPS dictionary)
Fixation 500-800 ms; targets on with fixation; fixed 450-ms delay; go = fixation off; 450-ms window; median RT ~220 ms.
Reward tied to target identity (horizontal vs vertical bar); rich identity reverses at unsignalled blocks (50 single-target
trials, then 12-48 choice trials); per-block means on [37,191] ms x k (separation >= 40k) + trial jitter. Salience =
isoluminant hue contrast 180 vs 20 (one each on choice trials). Behaviour (08-20): reward slope +0.28/100 ms, side bias
P(right) 0.36, RT conflict cost +16 ms; 08-19: no detectable reward effect.

## Model (shared; conflict-task-rnn-f8 code/srs/)
- Two SCs; per SC one E pool at its contralateral target location plus non-pool E units; local I and commissural I.
  Dale's law; tau 50 ms; dt 10 ms.
- Fixation inside SC: tonic SNr-like GABAergic input onto E (fixed tonic level to start) plus fixation drive; go removes
  the drive. Readout: stochastic per-pool hazard p_i(t) = sigmoid(g (r_i - theta)), shared g and theta; first event wins;
  an event before go is a fixation break. No readout-level normalisation.
- One upstream recurrent module (same size and trainability in all architectures) with an inhibitory (SNr) output
  channel per location; no SC->upstream feedback or direct/indirect paths in v2.
- Curriculum: single-target trials with a large reward contrast first (hold fixation, then saccade), then real rewards.
- Reuse: the existing conflict-task-RNN networks with a target-locked goal input are a B-type baseline (resolution
  arrives as input; commissure acts as a brake).

## Architectures (v2 core and secondary)
B PRIORITY HUB (core): upstream combines identity-at-location with value belief and outputs graded per-location value as
  SNr disinhibition; SC does the WTA.
C OUTCOME HUB (core): as B, plus upstream WTA; near-categorical disinhibition at the winner; SC weak or no WTA.
A RESOLVER (secondary): same upstream module, output constrained to carry belief per identity or identity per location
  but not their conjunction (by wiring or decoding penalty); SC binds and arbitrates.
Deferred: D FREE + interventional resolver index (a/(a+b): loss of reward slope when the upstream->SC signal is replaced
  by its value-independent version vs when SC's direct belief input is removed), only after B vs C is settled.
Robustness on the winning contrast only: excitatory-reward variant; commissure on/off; upstream latency.

## Behaviour and nuisance factors (matched behaviour is a precondition for neural comparisons)
- Trial-matched simulation: drive models with the recorded trial sequences (bilateral-SC-recordings exports the real
  trials.csv), so sampling noise matches the recordings trial by trial.
- Belief per trial from a model fitted to each session's choices plus single-target RT/vigor (not a free delta rule:
  50 single-target trials per block let a delta rule converge before choice trials).
- Weak reward use can arise from belief noise, weak value weighting, side bias plus decision noise, or lapses; each predicts
  a different neural reward signal at the same behaviour. Cross these routes as a nuisance factor (report per route and
  marginalised). Cross the locus of side bias (SC vs upstream).
- Noise sources in SC, upstream and belief; their amplitudes fitted per architecture to fixation-period variance and the
  pre-target L-R correlation (not to choice prediction).
- Acceptance region: per-session reward slope, side bias and RT cost with bootstrap CIs; networks are compared only within
  it (rejection or an identical behavioural-matching loss for all architectures). Residual behavioural differences
  between architectures are reported, not tuned away.

## Signatures (each with the contrast it can test; simulated sessions through the bilateral pipeline)
Matched sampling: recorded unit counts per side (08-20: 19 L, 27 R; 08-19: 39 L, 38 R), pool-E in recorded proportions per
side (08-20 right had no classic units), Poisson spikes at recorded rates, recorded trial counts. All units valid on all
trials.
S1 L-R reward difference beyond choice and pre-target state (recorded: none) -> graded (A, B) vs categorical (C).
S2 Competitor-reward normalisation, their exact GLM (recorded: none, power >= 0.98) -> any SC-normalising
   architecture vs not (A, B with WTA vs C).
S7 Value scaling on single-target trials (320 trials on 08-20; recorded: to be run after Stage 0) -> B (scales) vs C
   (not, unless pre-specified graded single-target output).
S8 Identity x block coding in SC on single-target trials (identity selectivity whose sign/gain reverses with block, before
   choice; speed of SC value updating after a reversal vs behaviour) -> A (binding in SC) vs B/C.
S4 Pre-target state -> choice and L-R correlation: calibration constraint (noise fitted to other statistics; then the
   predicted state->choice link and its ratio to the reward effect, z 3.8 vs 2.2, are checked).
S3 Choice information after target onset: calibration only unless upstream latency is fixed independently.
Dropped: S5 fast CCG (trivially flat in rate models). Model-only: S6 residual dynamics.

## Stages and go/no-go
0a Hand-built B, C, A at matched behaviour (concrete builds in conflict-task-rnn-f8's critique) through Poisson ->
   matched sampling -> bilateral pipeline, many draws. Keep a signature only if expected |log LR| > log 3 between the
   architectures it targets at recorded n (power analysis). Also simulate the redesigned task to show what would separate.
0b Trainability: can B, C (and A) with matched capacity be trained into the acceptance region? If not, report and stop.
1  Pilot ensembles: 2-3 architectures x 3-5 seeds x nuisance grid; check retained signatures hold across seeds.
2  Full ensembles (10 seeds per setting) only for surviving signatures.

## Perturbations (pre-registered for James)
- Main deliverable, occlusion test on single-target trials: block-value effect on RT and peak velocity (and SC responses
  near the site if recorded) for targets in the injected vs opposite field, before vs after low-dose bicuculline, indexed to
  the observed fixation-break/intrusion rate rather than nominal dose. Tests whether value reaches SC through local GABA-A
  synapses (prerequisite for every architecture claim). Well powered; block reversals separate value from identity.
- Choice level: offset toward the injected field (pre-registered); reward-slope change exploratory only (underpowered at
  ~250 choice trials; and in B and C value arrives via the blocked SNr synapses, so slope change is not resolver-specific).
- Stated plainly: in this task bicuculline cannot separate resolver from hub; the redesigned task (simulated in 0a) is
  what would.
- Calibration only: high-dose irrepressible saccades (Hikosaka & Wurtz 1983; all architectures should pass via the
  shared tonic SNr fixation input), muscimol deficits, spread modelled as a radius with a rostral-pole variant.

## Final argument
Per signature: class-conditional distribution across networks x sampling draws x nuisance settings and the likelihood ratio
at the recorded value; joint LR only from simulated joint distributions. Pre-registered equivalence rule: if every retained
signature has |log LR| < log 3, the recordings cannot distinguish the architectures; the deliverables become the occlusion
prediction and the redesigned task.

## Roles, compute, timeline
- conflict-task-rnn-f8: code/srs/ (task with recorded trial sequences, belief input, architectures, hazard readout,
  simulate() returning trial table + rates + unit metadata, perturbation hooks), hand-builds; CPU smoke tests only.
- 4factors-RNN: belief model fit, Poisson/matched sampling/export (trials.csv + units.mat per bilateral spec), Stage 0
  power/LR analysis, GCP training (copy of the repo; no edits in place), perturbations, synthesis.
- bilateral-SC-recordings: real trials.csv export for trial-matched simulation; adapter; pipeline runs; S7/S8 on recordings
  after Stage 0.
- No local GPU; light local CPU only. GCP pilots < $20; full ensembles only if Stage 1 passes.
- ETA: code ~1 day; Stage 0 ~1-2 days; pilots/ensembles ~1 day; synthesis after.

## Review summary (v1.1 -> v2)
Adopted: one core contrast (graded vs categorical delivery; B vs C) with A secondary under a matched upstream module and D
deferred with an interventional index; scope stated (salience inert, so the question is where value is bound and the
choice resolved); S5 dropped, S6 model-only, S4 as non-circular calibration, S3 calibration; new S7 (single-target value
scaling) and S8 (identity x block coding); power-based keep/drop rule at recorded n; trial-matched simulation with a
fitted belief model; nuisance crossing of the routes to weak behaviour, noise loci and side-bias locus; behavioural
acceptance region; stochastic hazard readout; curriculum; occlusion test as the main bicuculline deliverable and an
honest statement that bicuculline cannot separate the architectures in this task; simulate the redesigned task; reuse the
existing networks as the B baseline; smaller staged compute.
