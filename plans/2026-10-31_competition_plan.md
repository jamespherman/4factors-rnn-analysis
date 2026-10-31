# Plan: does SC intrinsic circuitry support competition between task factors? (resolver vs hub) - v2

Date: 2026-10-31. v1 was reviewed by an independent instance; v2 adopts the review (summary at the end).
Literature: `plans/2026-10-31_research_wta.md`, `plans/2026-10-31_research_intercollicular.md`.

## Framing (corrected)
- Resolver: the outcome of goal-vs-stimulus conflict is produced by nonlinear competition inside SC.
  Hub: the outcome is already fixed in SC's inputs (BG/thalamic/cortical loops); SC dynamics execute it.
- Factor-agnostic (map/location-only) inhibition does NOT imply hub: a Trappenberg-style field can resolve
  goal-vs-stimulus conflict with location-only inhibition if factors arrive as location-specific drive. So the
  repo's "unstructured I->E supports external loops" inference does not hold on its own; this will be stated.
- What separates the roles is visible only when two representations are active at once. In the 4factors task this
  happens with blocked probability: prior (goal) activity at the high-probability location competes with the
  stimulus-driven response at the target location on low-probability trials. In the bilateral data it happens on
  two-target choice trials.
- Honest scope: these data can show how factors enter SC and whether SC state shapes the outcome; they cannot by
  themselves show where conflict is resolved. The report will say which experiments would.

## Analyses (in order)
P1. Behaviour (4factors, verified trials; local light): RT (saccade onset - go) by reward, probability, salience,
    identity and their interactions (mixed model with session); conflict costs (low-probability location x high
    salience etc.). Which factor dominates RT.
P2. Probability conflict (4factors): for units whose RF (preferred location) is a probability location (1 or 3):
    (a) pre-target activity when their RF location is high- vs low-probability (block); (b) on trials with the
    target at the opposite location, the time course after target onset of the difference (high-prob RF vs
    low-prob RF blocks): is the pre-target advantage suppressed, when, and does its suppression scale with the
    target-site response; (c) whether residual pre-target activity at the high-probability RF predicts RT on
    low-probability trials beyond factor labels (single-trial regression, session random effects). Optional (d):
    saccade errors toward the high-probability location from raw (non-verified) trials, if eye endpoints are
    available in session_data.
P3. Factor interactions and a common currency: counts at the preferred location ~ salience x identity x reward x
    probability on a variance-stabilised scale, tested against a single-gain + output-nonlinearity model;
    population: alignment of factor coding axes (one "priority" axis or orthogonal) over time.
P4. Entry routes (reduced W1): population/jackknife onsets with amplitude-matched criteria; salience as gain vs
    latency shift (r_high(t) ~ g r_low(t - Delta)); reward/probability as post-target gain (pre-target offset
    separated); probability's visual suppression reported as expectation suppression.
P5. Bilateral conflict data (request to bilateral-SC-recordings, runs in parallel): power gate first (minimum
    detectable early choice probability), behaviour (choice ~ salience, reward; conflict RT cost), early choice
    signal before reward's ~100 ms entry with factor values fixed, normalisation of the contralateral-target
    response by the competitor's value and factor class. Not re-asked: winner-loser cross-SC coupling (already
    bounded).
P6. Synthesis for James (report): what each result implies; explicitly what cannot be answered; decisive experiments
    (conflict task with variable SOA between salient and rewarded stimuli; cdlSNr inactivation with SC recording;
    local SC GABA-A antagonism).

Deferred (needs a purpose-built identifiability test first, likely inconclusive): RNN with restricted (RF-local)
inputs so off-RF/opposite-direction suppression must come from recurrence, compared with free inputs, plus a matched
teacher pair (recurrent vs input-carried suppression). Cut: noise-correlation vs selectivity test; in-silico
factor-group stimulation; two-target probe.

## Decision rules
- P5 early choice signal powered and absent before the goal signal -> favours hub; present -> consistent with
  resolver, not diagnostic (could be FEF); underpowered -> "not answerable".
- P2: SC pre-target state predicting RT/errors beyond labels -> SC state shapes the outcome (does not establish who
  resolves it).

## Compute / ETA
All local, single-thread, idle priority. P1-P2 today (~4 h), P3-P4 tomorrow morning, P6 tomorrow.

## Review summary (v1 -> v2)
Adopted: factor-agnostic inhibition is not evidence for hub; timing separates sources not resolvers; cut W2 (common
modulator mimics competition; wrong kind of competition) and W3 factor stimulation / two-target probe (model lacks
identity/salience, off-RF suppression not identifiable, extrapolation); add probability-conflict analysis, behaviour,
interactions/priority axis; onset methods (jackknife, amplitude matching, gain vs shift); W4 led by a power gate.
