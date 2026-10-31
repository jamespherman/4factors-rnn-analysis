# Resolver vs priority hub vs outcome hub in the urgent (variable-timing) conflict task (2026-10-31)

James chose (2026-10-31): the urgent design as in the grant's Fig 7 — go signal, targets at variable
dt = target onset - go in {-150, ..., +150} ms (50 ms steps), 450 ms response deadline from go, tachometric
analysis vs processing time (rPT = saccade onset - target onset). Reward tied to target identity with block reversals
(srsClassic), luminant salience. Design input from conflict-task-rnn-f8 (2026-10-31 message): calibrate and sweep the
value-vs-salience latency gap; keep value target-locked; go-locked urgency rather than a linear time cost; analyse
rPT; check that hubs do not carry salience upstream.

## Questions
Q1. Do the architectures predict different tachometric curves (midpoint, slope, capture at short rPT, asymptote)?
Q2. Do they predict different SC population dynamics: when the rich-target and salient-target populations diverge
    relative to target onset, and whether the divergence latency depends on dt?
Q3. Does the S8 identity test (identity beyond value in SC) still separate resolver from hubs in this task (it is the
    test the proposed bilateral recordings would repeat)?
Q4. S9-like signature: single-target RT vs value (graded vs categorical) in the urgent task.

## Model changes (my copy, external/conflict_srs; peer's code untouched)
- Pull peer's rollout fix (target must be on screen at the event step) — done.
- `task.make_urgent_trials(cfg, n, rng, stage)`: as make_training_trials but delay = -dt with dt drawn from
  cfg.urgent_dts; fix_dur 300-500 ms before go-or-target; luminant salience (sal_gain = 1.0); 50% single-target trials
  (needed for Q3/Q4), 50% choice; reward side x salient side independent (50% conflict).
- Go-locked urgency (config `urgency` > 0): hazard logits += s * relu(t - t_go - go_latency) / 100 ms, s per trial
  ~ lognormal(log(urgency), 0.3), identical for both locations (non-spatial), in training and simulation.
  time_cost = 0.
- Identity latency into SC for the resolver (`id_sc_latency`, default = vis_latency): primary runs set it equal to
  up_latency, so the resolver gets identity no earlier than the hubs' upstream (fair timing); a secondary set leaves it
  at vis_latency (resolver advantage) to show the dependence.
- Curriculum: S1 delayed single-target (learn fixation), S2 urgent timing + true values, no urgency, S3 urgent +
  urgency + noisy beliefs (longer: 3000 it).
## Training (CRC preempt GPUs, free)
- A, B, C x value latency gap up_latency - vis_latency in {20, 50, 80} ms x 4 seeds = 36 runs (+ A with
  id_sc_latency = vis_latency at gap 50: 4 runs). Exclusion as before (one-sided choice, completion < 0.9).
## Analysis (`scripts/srs_urgent_analysis.py`)
- Tachometric curves on simulated conflict trials (rich vs salient on opposite sides): P(choose rich) vs rPT, 25 ms bins,
  4-parameter logistic fit (lower/upper asymptote, midpoint, slope) per network; congruent trials as control.
- SC dynamics: pool rates for the rich-target vs salient-target locations aligned on target onset, per dt; divergence
  latency (first time the rich-minus-salient difference exceeds 2 SD of its pre-target value).
- S8 at the recorded unit/trial yield (single-target trials; identity beyond value), and S9 (single-target RT vs value
  at full engagement).
- Report per architecture and latency gap: median and range over networks; rank tests between architectures.

## Risks / caveats
- Absolute timings depend on assumed latencies; claims are about differences between architectures at matched
  latency and how they scale with the gap.
- The original conflict-task model needed fixed sensory weights to keep capture; here visual weights onto SC are
  fixed in all architectures (w_vis fixed), as in srs.
- REINFORCE may find one-sided solutions; curriculum and exclusion criteria as before.

## v2 after independent review (incorporated)
1. Tachometric midpoint/capture depth are set by latencies, not architecture. Primary comparisons are latency-robust:
   each network's *measured* effective value-arrival latency (rich-minus-poor pool signal on single-target trials)
   is reported, and curve shape (slope, asymptotes) is compared at matched measured latency; a difference counts only
   if it holds at all three gaps and both resolver timing variants.
2. Nominal identity-latency matching is not effective-latency matching (hubs pay upstream + SNr filtering; the
   resolver pays for recurrent binding): analyse against measured latency (item 1); keep both A variants.
3. Value difference as a factor: conflict-trial curves split by |V1 - V2| (and early post-reversal trials).
   Graded hub: midpoint/asymptote scale with the gap; outcome hub: near-step; resolver: its own binding dynamics.
4. Divergence latency vs dt: slope 0 = target-locked, 1 = go-locked (snr_fix sweep not trained: caveat).
5. Urgency primary = non-spatial additive drive onto both SC pools (urgency_locus = sc), identical in A/B/C, passing
   through SC dynamics; hazard-logit urgency as robustness (gap 50, 3 seeds per architecture).
6. Salience strength sweep at test (sal_gain 0.3 / 1 / 2; trained at 1) for capture vs salience scaling.
7. Error trials: on capture errors, does the rich-pool signal still arrive at its usual latency (input-driven race,
   expected for hubs) or is it suppressed/delayed by the salient winner (SC competition)?
8. S8 time-resolved in target-aligned windows.
9. 6 seeds per cell (rank tests can reach p < 0.01 across cells).
Training set: A (id_sc_latency = up_latency), B, C x gaps {20, 50, 80} x 6 seeds = 54; A with id_sc_latency =
vis_latency at gap 50 x 6 = 6; hazard-urgency robustness A/B/C x gap 50 x 3 = 9. Total 69 runs.
