# Plan: task-context inputs and a test of local inhibition (v2, after review)

Status: revised October 2026 after an independent critical review of v1 (summary
of changes in §9). Builds on `RETRAINING_RESULTS.md` and `AUDIT_2026-10.md`.

## 1. Framing

Reward and probability in this task are set by block structure that the monkey
learns from trial history, a computation involving structures beyond SC. The model
simulates each trial independently, so it cannot infer them. We therefore treat
block context as an input "computed elsewhere" and ask what the SC circuit does
with it:

- **H_ext:** context input combined with the visual target suffices to reproduce SC
  factor selectivity; factor-specific local inhibition is not needed.
- **H_local:** reproducing it requires factor-structured local inhibition.

Two facts constrain how this can be tested:

1. **Fitted I→E weights are not identified** (cross-seed correlation ~0 in 08_14
   and 08_15). The test cannot rest on reading weights from single fits.
2. **Every model I unit is a recorded interneuron** (16E/4I, 64E/16I, 52E/13I
   with 4, 16, 13 recorded I). There are no hidden I units, so I→E can only add
   factor selectivity to E units to the extent that recorded interneuron rates
   carry it. The I→E *current* (W_IE · Δr_I) is far better constrained than W_IE,
   and much of the answer is in the recorded interneuron activity itself.

## 2. Task structure

Verified against raw `trialInfo` (all export trials matched to raw trials; the
label-derived reward map agrees with `halfBlock` on 100% of trials, the
label-derived block with `blockNumber` on 98.3–99.8%):

- Probability: block 1 → location 1 frequent (~60% of trials), block 2 →
  location 3. Locations 2 and 4 always low probability.
- Reward: deterministic given half-block and location (half-blocks 1, 3:
  locations 1, 2 large; half-blocks 2, 4: locations 3, 4 large); independent of
  stimulus.
- Only 6–14 reward-map switches and 3–8 block switches per session; block is
  close to "early vs late in session". Irregularities: 08_14 opens with reward-map
  runs of 2–7 trials; 08_15 has half-blocks of 217 and 241 trials (consistent with
  raw `halfBlock`, so real, not a labeling error).

## 3. Stage A: data gates (no training)

The rest of the plan depends on these. All context statistics count switches or
blocks, not trials, as the effective N.

**A1. Context effects that survive drift.** Random train/held-out splits share
slow drift, so the earlier reliability values (reward 0.58, probability 0.50) do
not rule it out. Against a circular-shift null that keeps the block run structure,
visual-epoch reward and probability selectivity are at chance (7% of neurons at p
< 0.05), whereas identity (16%) and salience (11%) are above it. Tests:
- Switch-triggered contrasts: K trials before vs after each switch, signed by
  whether the neuron's preferred location gains or loses value; contingency
  alternates across switches, drift does not. Pre-target (−300–0 ms) and visual
  epochs, K = 15, 30. *Preliminary:* pre-target reward context is real (higher
  rates when the preferred location is currently rewarded; 66–71% of neurons,
  Wilcoxon p = 1e-3 to 2e-5). Visual-epoch reward is inconsistent across K.
  Visual responses at the preferred location are *weaker* when it is the frequent
  location (K = 30: 77% of neurons, p = 1e-6), more like adaptation to frequent
  stimulation than a value signal.
- Probability sign reversal: within the same blocks, the effect at location 1 and
  at location 3 must have opposite signs; drift would give the same sign.
- Recent-history control: regress visual response on whether the previous 1–3
  targets were at the same location. If this explains the "probability" effect,
  it is trial-history adaptation, which needs a history input (or a multi-trial
  simulation), not a block-context input.
- Pre-target analyses only on trials with a valid fixation channel (missing on
  30–45% of trials, audit §7), aligned to target onset.

**A2. Do recorded interneurons carry context and factor information?** Population
decoding (cross-validated, blocked by switch for context) of reward context,
block, location, identity and salience from recorded I vs E neurons, with E
subsampled to the I count and matched for firing rate; pre-target and visual
epochs. Single-neuron ROC alone is not enough. If interneurons carry no context
or factor information beyond E-matched chance, then in the current model class
I→E cannot create factor selectivity, and the H_local test must instead concern
unrecorded interneurons (see B5).

**Gate.** Training with context inputs is worth doing only for effects that pass
A1. If only pre-target reward context passes, the training stage targets that;
the visual-epoch "probability" effect is handled as history, not context.

**Gate outcome (Stage A run, `results/context_analysis/stage_a_report.md`):**

- Blocks alternate 3–7 times per session (not just early vs late); context labels
  match raw `trialInfo` (reward map 100%, block 98–100%).
- **Visual-epoch reward selectivity (the pipeline's reward ROC) is not
  distinguishable from drift**: chance against the circular-shift null (7%), no
  consistent switch-triggered effect. The earlier "reliable" reward/probability
  selectivity (r = 0.58/0.50 across random splits) was largely shared drift.
- **Pre-target context is real and spatially specific**: higher pre-target rates
  when the neuron's preferred location is currently rewarded (+0.7 sp/s, 23
  switches, p = 0.001; 71% of neurons) or frequent (+0.8 sp/s, 11 switches,
  p = 0.006). About 8% of the pre-target rate.
- **Visual responses are weaker at the frequent location** (probability component
  −1.6 sp/s, p = 1.5e-6, surviving the sign-reversal drift control; ~9% of the
  visual response, 0.13 trial SDs). Not explained by the previous 1–3 targets, so
  a block-level effect (expectation suppression or slow adaptation), carried by
  block context rather than short trial history.
- Interneurons: pre-target context modulation in I is similar in sign and size to E
  (less power, 33 I vs 76 E). Population context decoding is not above the
  circular-shift null for I or rate-matched E. **I neurons decode target location
  better than rate-matched E neurons** (08_14: 0.63 vs 0.54; 08_15: 0.73 vs 0.55),
  so the recorded interneurons are spatially selective. Identity and salience are
  not decodable from I-sized populations of either type.
- 79 of 109 neurons prefer location 1, so location-3 effects are poorly sampled.

Implication: the targets for context models are small anticipatory
(pre-target) context signals and a small visual suppression at the frequent
location, about 1/4 to 1/8 the size of the spatial effect. The free-vs-constrained
I→E comparison on effects this size may be underpowered; the recovery test
(Stage D) should be run with these effect sizes before the full grid.

## 4. Stage B: model design

**B1. Context inputs, non-spatial (primary).** The two binary context states
(reward map, block) enter through a rank-1, shared-weight pathway: one weight per
state per population (E, I), added to the recurrent input after the embedding
(`W_ctx`, bypassing the multiplicative attention embedding). Any location-specific
context effect must then be produced by the recurrent circuit. A per-unit free
`W_ctx` (v1 design) lets each unit read "my location is valuable" directly, which
makes H_ext win by construction; it is kept only as an upper bound
(`ctx_free`).

**B2. Drift input in every variant.** A slow time-in-session input (and, if A1
supports it, a recent-target-history input) in all models including the
no-context control, so context weights are not fitting drift.

**B3. Routing.** `ctx_all` (context to E and I), `ctx_i` (to I units only; the
sharpest H_local test: can E selectivity be fit when context reaches only
interneurons, which are themselves held to recorded rates?), `ctx_e`.

**B4. Constraint ladder on I→E** (all with B1 inputs):
- (a) free I→E;
- (b) I→E a learned function of the spatial-preference difference between the I
  and E unit only (factor-agnostic, spatially structured);
- (c) rank-1 global inhibition (extreme case);
- (d) I→E frozen at a permuted copy of a free solution (matched-capacity null).

**B5. Hidden interneurons.** Variant with additional unrecorded I units, since
the current model class assumes all local interneurons are recorded, which biases
toward H_ext. Their weights will be weakly constrained; interpreted only through
the comparisons in Stage D.

**B6. Positive and negative controls.** `trialvalue` (current-target reward and
probability as input), `nocontext` (drift input only).

## 5. Stage C: training protocol (pre-registered)

- Trials split 70/15/15 (train / selection / test), stratified by context ×
  location; the test split is used once, for the final comparison.
- Loss: current `cond_target` loss plus a conditioned PSTH term over location ×
  context (16 cells), scaled so it doesn't dominate checkpoint selection.
- Primary metrics on test trials: Poisson log-likelihood per neuron; selectivity
  r normalized by the recorded noise ceiling (drift-robust selectivity from A1);
  per-unit decomposition of selectivity into input, E→E and I→E currents.
- Decision rule: equivalence test (TOST) between free and constrained I→E,
  margin fixed from the recovery test (Stage D) before looking at real-data
  comparisons, with seeds nested within sessions; 5 seeds per configuration.
- 08_13 (4 I units) has little power for free vs constrained; analyses pool
  sessions but report each.

## 6. Stage D: recovery test (before the full grid)

- Teachers in which I units carry context at the upper end of recorded I
  selectivity, with planted factor-structured I→E at a sweep of strengths, and a
  teacher with spatially specific context input and no I structure (false-positive
  check). 2–3 teachers; student seeds independent of teacher initialization.
- Synthetic Poisson spikes with the real sessions' trial structure and neuron
  counts. Fit students under the ladder; read out with the TOST comparison and the
  current decomposition. Output: detection curves and false-positive rates,
  which set the equivalence margin and tell us whether the question is answerable
  with these data. If it is not, the report says so.

## 7. Stage E: perturbations (secondary)

Within fitted `ctx_all` models: degree-preserving I→E shuffles (keep each unit's
incoming and outgoing inhibition), or removal of only the factor-aligned
component of I→E; changes in selectivity normalized by change in overall fit;
reported as distributions across the seed ensemble.

## 8. Order of work and cost

1. Stage A (hours, no training). Decide what Stage B targets.
2. Code: context and drift inputs, `W_ctx`, constraint ladder, 3-way split,
   current decomposition.
3. Pilot: `ctx_all` free vs ladder (b), 3 seeds, to check the design works.
4. Stage D recovery sweep, sized to fix the margin.
5. Full grid (5 seeds), Stage E, write-up in `specs/CONTEXT_RESULTS.md`.

Training cost ~20 min/run with 4 parallel streams.

## 9. Changes from v1 after review

- Per-unit context weights made H_ext win by construction → non-spatial rank-1
  context as primary (B1).
- All I units are recorded → interneuron information analysis (A2), current
  decomposition as primary readout, hidden-I variant (B5).
- Context confounded with time → drift-robust tests as a gate (A1), drift input in
  all models (B2). Preliminary results already overturn the earlier
  reward/probability reliability values.
- Rank-1 global inhibition also removes spatial surround → constraint ladder (B4).
- No pre-registered decision rule, selection on the evaluation split → 3-way split,
  TOST, metrics fixed in advance (C).
- Recovery teachers with non-discriminating I units guaranteed a null → redesigned
  and moved before the grid (D).
- Out-sum-preserving shuffles break E/I balance → degree-preserving shuffles (E).
- Cut: `linear`-embedding variant, E→E shuffle comparison.
