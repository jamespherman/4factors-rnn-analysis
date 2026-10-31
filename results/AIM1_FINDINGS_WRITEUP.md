# Is SC a hub or a resolver? Findings, model predictions and candidate visualisations (Aim 1; v2, 2026-10-31)

For R01 planning. Each finding lists the evidence, its strength and caveats, and a **Visualisation:** a plot that would
demonstrate it, so pieces can be picked individually. v2 reorganises v1 (git history) by how much weight each piece
can bear, and adds a recommendation on bicuculline (section E). Notes and code are cited per item (repo
`4factors-rnn-analysis`, branch `restore-lost-work`; bilateral analyses by the bilateral-SC-recordings session in
`Newton_SC_2026/docs/`).

**Terms.**
- **Resolver**: SC's own recurrent and inhibitory dynamics weigh goal-directed against stimulus-driven factors and
  pick the winner.
- **Hub**: the weighing happens upstream (basal ganglia-thalamo-cortical loops); SC receives and executes the result.
  - **Priority hub**: SC receives graded priority, delivered by SNr disinhibition.
  - **Outcome hub**: SC receives an already-made choice.

**Data.**
- 4factors unilateral recordings (Feynman, Newton): single targets; reward cue x location prior x salience x identity.
- Bilateral Newton recordings in the srsClassic conflict task (08-19, 08-20): reward tied to target identity with
  unsignalled block reversals; salience as isoluminant hue contrast.

**Models** (from conflict-task-RNN `code/srs`): bilateral SC E-I networks (Dale's law) with an upstream BG-like
module, SNr tonic inhibition and a stochastic saccade readout, trained by REINFORCE on srsClassic. All four
architectures share an identical SC:
- **A**, resolver: identity and belief enter SC through fixed random weights, so SC must bind identity x belief x
  location itself.
- **AE**: as A, but with input onto E units only.
- **B**, priority hub: the upstream module gets identity and belief and sends graded per-location disinhibition.
- **C**, outcome hub: the upstream module is a softmax winner-take-all.

Every architecture also carries a weak sensory shape code in SC.

**How to read the tiers.**
- **A. Solid** (can carry a figure panel and a claim).
- **B. Supporting data** (consistent with a hub; suggestive, not decisive alone).
- **C. Predictions for the proposed experiments** (present as hypotheses, not findings; C1 now has model
  backing and pilot support).
- **D. Model validation and method** (optional, for rigor).
- **E. Bicuculline: recommendation.**

=====================================================================================================================

## A. SOLID: the resolver predicts an identity code that recorded SC lacks
Notes: `results/hub_models/S8_RESULT.md`; plan `plans/2026-10-31_hub_resolver_models_plan.md` (v2, reviewed);
trained-ensemble signatures in `results/hub_models/sig_v2/`.

A1. **Logic.**
- In srsClassic, reward is tied to target identity and reverses between blocks.
- A resolver must bind identity to block value inside SC, so its SC neurons carry identity information beyond value.
- A hub binds identity to value upstream, so SC receives value only.
- A positive identity effect in the recordings would be ambiguous, because it could be sensory shape coding. A null
  is not ambiguous.

**Visualisation:** schematic, two circuits side by side. Resolver: identity and value both enter SC and are combined
there. Hub: they combine upstream and a single value or priority arrow enters SC. Below each, the predicted SC
response to the same target in rich vs poor blocks.

A2. **Every trained resolver network carries the code; no hub network does.**
- Measure: fraction of units (contralateral to the target, single-target trials) with |identity effect beyond
  value| > 0.2.
- Trained resolvers (6 seeds each): A 0.42-0.61, AE 0.28-0.58.
- Trained hubs (B and C): <= 0.04.
- Hand-built networks agree: A 0.40, B and C ~0.
- Robust to calibration: with the networks matched to the monkey's choices, the identity-coding excess (in units of
  mean |z| above a permutation null) is A 3.6, AE 2.7, and B and C 0.0.

**Visualisation:** per-network distributions (violin or ECDF) of the per-unit identity effect, coloured by
architecture, with the recorded 08-20 distribution and its permutation null overlaid.

A3. **Recorded SC is at the null, and would have detected the resolver code.**
- 08-20: 301 single-target trials, 8 blocks, 27-31 contralateral units.
  - Mean |delta| is 0.131, against a null 95th percentile of 0.166.
  - Quasi-Poisson |z| is 0.54-0.62 (chance).
- 08-19 is also at the null.
- Detection test: each network's per-unit effects were planted into the real counts.
  - P(detect), delay window: A 1.00, AE 1.00, B 0.00, C 0.00.
  - Each of the 12 resolver networks is detected at P >= 0.99 in both the delay and pre-saccadic windows.

**Visualisation (the key panel):** P(detect) per network, one dot per network grouped by architecture, for the delay
and pre-saccadic windows, with the recorded outcome marked "null". Resolvers sit at the top, hubs at the floor.
Optional companion: the observed statistic against its distribution under planted-resolver data and under the null.

A4. **Caveats (state these in the text).**
- A resolver with a much sparser or weaker identity code than any network we trained is not excluded. Example: a
  quarter of units at |delta| 0.2 would be detected with only ~20% power.
- Only one session is well powered, and the recordings had few classic units on the side of interest.
- The test assumes model log-rate effects transfer to recorded neurons at recorded rates.
- What it shows: identity x value is bound upstream of SC. It does not show whether SC receives graded priority or a
  decided outcome (C1).
- Natural next step, which is also an aim: repeat this test at higher power in the proposed bilateral recordings.

## B. SUPPORTING DATA: SC looks like a recipient of pre-weighted goal signals

### B1-B5: 4factors data (`results/competition/SYNTHESIS.md`, `NOTE_P1_P2.md`, `p3_interactions/`, `p4_timing/`)

B1. **Reward cue x location prior interact in behaviour.**
- The high-reward cue speeds saccades to the expected location and slows them to the unexpected opposite location.
- Feynman: reward x opposite +17.9% RT (p < 1e-6); peak velocity mirrors it.
- So the reward cue amplifies the prior rather than adding a benefit that ignores location.

**Visualisation:** RT (or % change from low reward) by target location class (expected / 90-deg neighbour /
opposite), with lines for the high and low reward cue, one panel per monkey. The crossing lines show the interaction.

B2. **The same interaction appears in classic SC neurons, set before the target appears.**
- The reward cue enlarges the visual response to an expected target and shrinks the response to an unexpected one:
  reward x expectation +0.27 SD (p = 7e-6); putative interneurons +0.08 (p = 0.07).
- The interaction peaks 80-100 ms after target onset.
- Before the target, the reward cue raises activity (+0.31 sp/s), and the prior raises it slightly at the expected
  location (+0.24 sp/s).
- This is a gain applied by a signal present before the target, not something built up after it.

**Visualisation:** population PSTHs (classic units, target in RF) for the four reward x expectation cells, aligned on
target onset, with the pre-target baseline shaded. Inset: the interaction term over time (sliding window, with CI).

B3. **No common "priority" axis.**
- Stimulus-driven factors (image vs bullseye, salience; onset 50-70 ms) are aligned across neurons (r = 0.40).
- Goal and stimulus effects are not aligned (|r| <= 0.12).
- The goal x stimulus interaction is weak (-0.08 SD, p = 0.03).

**Visualisation:** scatter of per-neuron reward effect vs salience effect, with r and its null band, beside the
salience vs image scatter (aligned). Compact alternative: a small matrix of cross-neuron correlations between factor
effects.

B4. **Competitive dynamics across the whole map after target onset.**
- When the target appears away from the expected location, the prior's head start there is cancelled within
  ~100 ms.
- Every non-target location drops below baseline from ~150 ms (-1.5 sp/s).
- Pre-saccadic suppression is strongest for the opposite (180-deg) target: interneurons 33% vs 19% at 90 deg;
  classic 12% vs 3%.
- This is SC's spatial winner-take-all at work. It is compatible with both views: a hub also uses SC's spatial
  winner-take-all to execute.

**Visualisation:** PSTH of units whose RF is at the expected location, on trials where the target appears elsewhere,
expected vs non-expected block (early advantage, its cancellation, the dip below baseline). Side bar chart: % of
units with pre-saccadic suppression by target angle.

B5. **SC state predicts RT, but not location-specifically.**
- Higher pre-target activity predicts faster saccades whether the RF holds the target, the expected competitor, or
  neither (-5 to -6% RT per SD).
- A competition between locations predicts slower RTs when the competitor's site is more active.
- What the data show instead is general readiness.

**Visualisation:** bars of RT change per SD of pre-target rate for "RF at target", "RF at expected competitor" and
"RF elsewhere" (all negative, overlapping CIs), against a dashed bar with the sign a local competition predicts.

### B6-B7: bilateral conflict task (`Newton_SC_2026/docs/NOTE_2026-10-31_competition_for_4factors.md`)

B6. **Behaviour.**
- Choice:
  - Reward biases choice: +0.28 per 100 ms of reward difference (08-20).
  - Salience barely affects it (p = 0.053).
  - Conflict costs +16 ms.
- RT: single-target RT falls with the belief value of the shown identity (Spearman -0.21, 08-20), using a
  delta-rule belief fitted to choices (`results/hub_models/belief/summary.md`).
- So RT tracks value more clearly than choice does. 08-19 shows little use of reward.

**Visualisation:** psychometric curve, P(choose right) vs fitted belief-value difference, split by congruent vs
conflict trials. Inset: single-target RT vs belief value (binned).

B7. **No reward-difference coding and no competitor normalisation.**
- The left-minus-right SC signal carries the upcoming choice: added choice probability 0.60 at +100-150 ms, 0.80
  pre-saccade.
- It carries no reward difference in any window.
- A unit's response is not reduced by the competitor's reward (power >= 0.98 for a 10% drop per 100 ms).
- A resolver comparing values in SC predicts both effects; a recipient of a pre-weighted signal predicts neither.
- Caveat: the early choice window is underpowered (minimum detectable AUC 0.61-0.69).

**Visualisation:**
- (i) Time course of the reward-difference and choice coefficients in the left-minus-right signal, with null bands.
- (ii) Distribution of per-unit competitor-reward coefficients, centred at zero, with the predicted -10% per 100 ms
  shift drawn in.

### B8-B10: the two colliculi (`results/intercollicular/SUMMARY.md`; bilateral `NOTE_2026-10-31_intercollicular_for_4factors.md`)

B8. **Ipsilateral responses look externally driven.**
- Non-classic units respond at the ipsilateral 90-deg neighbour as often as at the equidistant contralateral one
  (27% vs 25%).
- There is no relay latency (excess 3.7 ms, CI -1.3 to 6.8).
- The responses are not saccade-locked, and are salience- but not reward-modulated.

**Visualisation:** latency scatter (ipsilateral vs contralateral onset per unit) on the identity line, with the
predicted relay band (10-27 ms) shaded.

B9. **Ipsiversive activity is not a delayed copy of the other SC.**
- It rises with the other side's contraversive burst (lag -6 ms, CI -8 to +18); a relay of the recorded neurons
  predicts +10-27 ms.
- Cross-SC CCGs are flat in every group.
- Limit: a relay is excluded only if it would converge from fewer than ~170-500 neurons. A weak, broadly convergent
  commissural input is not excluded.

**Visualisation:** onset-aligned PSTHs of side X's burst and side Y's ipsiversive response, with the simulated relay
as a dashed, shifted curve. Inset: the pooled cross-SC CCG, flat, against the CCG a sparse relay would produce.

B10. **Candidate monosynaptic connections link neurons with similar selectivity.**
- Sharp 1-4 ms CCG peaks occur in about 1% of pairs, across all class combinations.
- Connected pairs have more similar factor profiles than matched unconnected pairs: +0.84 vs +0.56; without
  location, +0.43 vs +0.22 (p = 0.013), mostly face/non-face.
- Shared input would also produce this.
- Inhibitory troughs (8) are too rare to test whether inhibition is factor-specific.

**Visualisation:** an example sharp-peak CCG with its jitter band, and a histogram of profile similarity for
connected vs matched unconnected pairs.

### B11: data-fitted RNNs (`results/selectivity_by_type/NOTE.md`)

B11. **Weights fitted to the data cannot locate the computation.**
- RNN units fitted to individual neurons keep their spatial tuning (r 0.83-0.90, at ~60-70% of recorded magnitude)
  and match reward selectivity.
- They are at chance for identity and salience, although both are model inputs.
- They get the sign of the probability effect wrong.
- So "unstructured I->E weights" cannot by itself support the hub view. This motivates the task-trained approach
  in A.

**Visualisation:** per-factor scatter of recorded vs RNN ROC selectivity (already made, `results/selectivity_by_type/`).

## C. PREDICTIONS FOR THE PROPOSED EXPERIMENTS (hypotheses, not findings)

C1. **Priority hub vs outcome hub: does SC receive graded value or a decided outcome?** Status: the model contrast
is now established; its match to the data is suggestive (one session). It can be presented as a model-backed
prediction with preliminary support.
- **Models (solid).** Single-target RT falls with the value of the shown target only when SC receives a graded
  signal.
  - Hand-built networks: priority hub -0.32, outcome hub +0.02.
  - Trained networks, value engaged on every trial: priority hub -0.10 to -0.50 in all 6 networks (median -0.17);
    outcome hub -0.01 to -0.08 in all 3. The ranges do not overlap (rank-sum p = 0.024).
  - Robust to moving RTs off the floor: raising the readout threshold slows RTs by 25-35 ms and changes no
    network's value by more than 0.05 (`results/hub_models/rt_shift_pilot/`).
  - The outcome hub's upstream signal on single-target trials is not graded with value (rho -0.14), so it behaves
    categorically, as intended.
- **Why the calibrated runs gave ~0 for everything.** Matching the monkey's weak use of value in choice removed the
  value input on most trials, on single-target trials too, which mechanically shrinks this signature. Retraining is
  therefore not needed for this contrast.
- **Data (suggestive).**
  - 08-20: -0.21 (n = 301, p = 0.0003). This is inside the priority-hub range and about 2.3 SE beyond the most
    negative outcome-hub network.
  - Partial engagement only shrinks the signature, so no trained outcome hub reaches -0.21 at any level of
    engagement.
  - 08-19: +0.03 (n = 184, p = 0.72), but the monkey barely used reward in that session.
  - The monkey's RT tracks value more strongly than its choices do. The model's engagement mixture cannot produce
    that, which suggests the monkey's weak choice-value use is specific to choice (e.g. a side or salience policy
    on two-target trials) rather than a loss of value information.
- **Caveats.**
  - Value could speed saccades through a route outside SC's input (e.g. a separate vigour signal acting on
    brainstem or cortex), which the models do not include.
  - One informative session.
- Present as: "a graded (priority) hub predicts that RT and SC activity scale with the value of a lone target; a
  categorical (outcome) hub predicts neither. Pilot data favour the graded hub."

**Visualisation:** RT-value correlation per network (dots) for priority-hub and outcome-hub networks (hand-built
marked separately), with the recorded 08-20 value and its CI as a horizontal band. Alternative: binned RT vs value
curves for one representative network of each type beside the monkey's.

C2. **Variable-timing conflict task** (luminant salience, time pressure, onset asynchrony between the salient and
the rich target).
- Resolver: the outcome depends on when each signal reaches SC, within SC's competitive window.
- Hub: the outcome follows the arrival of the goal signal from the loop.
- No model simulation has been run yet; this is the next modelling step after retraining. Treat as a qualitative
  prediction for now.

**Visualisation (once simulated):** P(choose rich target) vs onset asynchrony, one curve per architecture, with the
architecture-specific inflection marked.

C3. **Higher-powered bilateral recordings** (mirrored, interleaved targets; enough classic units per side).
- They repeat A3 and B7 with enough power to exclude weaker or sparser resolvers than the trained networks: the hub
  predicts a null, the resolver a code.
- They also test commissural competition directly (B9).
- Power analysis (`scripts/srs_identity_power.py`; `results/hub_models/identity_power/summary.md`):
  - Assumptions: per-unit effects drawn from the trained resolvers; recorded-like rates and overdispersion;
    19 single-target trials per identity per block.

    | Resolver | 30 units x 8 blocks | 50 x 8 | 80 x 8 |
    |---|---|---|---|
    | As trained | 1.00 | 1.00 | 1.00 |
    | Half strength | 1.00 | 1.00 | 1.00 |
    | Only a quarter of units coding | 0.83 | 0.97 | 0.99 |
    | Half strength, half of units | 0.90 | 0.96 | 1.00 |

  - The 08-20 yield (27-31 units, 8 blocks) already excludes trained-strength resolvers.
  - About 50 contralateral units x 8 blocks per session excludes sparse resolvers at >= 0.95.
  - With 4 blocks, about 80 units are needed.

**Visualisation:** power vs the number of contralateral units (x), one line per resolver variant, solid for 8 blocks
and dashed for 4. Mark the 08-20 yield and the planned yield, and draw a horizontal line at 0.8 or 0.95.

## D. MODEL VALIDATION AND METHOD (optional, for rigor)

D1. **The models reproduce the classic SC bicuculline/muscimol phenotypes** (`results/hub_models/bicuculline/`).
- Bicuculline (all 24 networks, d 0.05-0.35):
  - Saccades into the injected field speed up (d 0.2: 17-42 ms faster; other field unchanged), and choice biases
    toward it.
  - At d >= 0.35, the networks make irrepressible saccades into that field (fixation breaks on 20-70% of trials,
    99-100% toward it), as Hikosaka & Wurtz (1985) reported.
- Muscimol does the opposite: RT into the field +120-210 ms, and choice moves away from it.
- This supports using the models for prediction.

**Visualisation:** dose-response curves (RT change into the injected vs the other field; fixation-break rate), one
line per architecture with the network range shaded.

D2. **Method: matched sampling and detection by planting.**
- Models are sampled at recorded unit counts, firing rates and overdispersion, and tested with the recorded
  pipeline's statistics against permutation nulls.
- Detection probability comes from planting model effect sizes into the real counts.
- Useful as a methods schematic if space allows.

## E. BICUCULLINE: RECOMMENDATION — not necessary for resolver vs hub; safe to cut

**Recommendation.** Leave bicuculline injection and recording out of the resolver vs hub aim. Nothing in the
argument depends on it. The modelling gives a principled reason to omit it, so its absence is not a gap a reviewer
can fairly point to.

What the simulations show:
- **Behaviour doesn't separate the architectures.** At doses an animal can work under, resolver, priority-hub and
  outcome-hub networks all show the same dose-dependent speeding and side bias. RT vs value does not flatten in any
  of them, because disinhibition compresses RTs toward a floor everywhere. The earlier prediction ("resolver:
  conflict-dependent choice changes; hub: execution deficits only") does not hold.
- **The neural signature is modest.**
  - With recording at the injection site, value modulation shrinks only in the priority hub: to 0.84-0.92 of
    control in 6 of 6 trained networks at d 0.2, vs 0.96-2.5 in 10 of 12 resolvers.
  - That is a 13-25% reduction of a modulation that is itself small (11-16% per 100 ms of value).
  - It has to be measured where firing rates are rising sharply.
  - Two resolver networks overlap the hub range.
  - It cannot separate the priority hub from the outcome hub.
- **Stronger tests already exist.** The identity-code test (A3) and its higher-powered repeat (C3), plus the
  variable-timing task (C2), separate resolver from hub more directly and with less risk.

Optional sentence for the proposal, if a reviewer might ask why SC GABA-A blockade isn't proposed: "In simulations
of all candidate architectures, local GABA-A blockade in SC produced the same behavioural phenotype, the classic
disinhibition of saccades into the injected field (Hikosaka & Wurtz 1985), and only a small neural difference between
them. We therefore do not use it to distinguish resolver from hub."

Caveats on the recommendation:
- It rests on four architectures in one task.
- A resolver built on inhibition that is specific to a factor, rather than to a location, could respond differently
  to GABA-A blockade.
  - Neither our data nor the literature suggest such a resolver. B10 found too few inhibitory connections to test
    factor-specific inhibition at all.
- If a future aim asks how SC's local inhibition shapes the spatial winner-take-all, separately from resolver vs hub,
  bicuculline would become relevant again.
