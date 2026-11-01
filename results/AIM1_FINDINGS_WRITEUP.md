# Is SC a hub or a resolver? Findings, model predictions and candidate visualisations (Aim 1; v3, 2026-11-01)

For R01 planning. Each finding lists the evidence, its strength and caveats, and a **Visualisation:** a plot that would
demonstrate it, so pieces can be picked individually. **v3 (2026-11-01):** the visualisations for B1, B3-B11, C1 and D1
now exist, with self-contained data, in `results/aim1_panels/` (index: `results/aim1_panels/README.md`). Building them
re-checked every number against the saved outputs; corrections are marked "(v3)". The largest: the recorded RT-value
effect in C1 and B6 does not survive drift controls, and the trained resolvers do not show the B7 effects. v2 reorganises v1 (git history) by how much weight each piece
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
- **C. Predictions for the proposed experiments** (present as hypotheses, not findings; C1 is a model
  prediction without data support, v3).
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

B1. **Reward cue x location prior interact in behaviour.** (v3: a crossover, not amplification)
- Feynman, high-reward cue: saccades to the opposite (unexpected) location are 10.4% slower than to the expected one
  (p = 0.005); low-reward cue: they are 7.7% *faster* (p = 0.009). Reward x opposite +17.9% RT. (Means over
  sessions; p from trial-level OLS clustered by block.)
- Feynman peak velocity: the high-reward cue raises it at the expected location only (reward x location p <= 0.02).
- Newton: only the high-reward slowing at the opposite location (+5.5%, p = 0.037); no low-reward effect.
- So the expected-location advantage exists only under the high-reward cue: the reward cue gates the prior.
- Panel: `results/aim1_panels/B1_reward_x_prior/`.

**Visualisation:** RT (% change from the low-reward expected cell) by target location class (expected / 90-deg
neighbour / opposite), with lines for the high and low reward cue and the no-interaction null dashed, one panel per
monkey. The crossing lines show the interaction.

B2. **The same interaction appears in classic SC neurons, set before the target appears.**
- The reward cue enlarges the visual response to an expected target and shrinks the response to an unexpected one:
  reward x expectation +0.27 SD (p = 7e-6); putative interneurons +0.08 (p = 0.07).
- The interaction peaks 80-100 ms after target onset.
- Before the target, the reward cue raises activity (+0.31 sp/s), and the prior raises it slightly at the expected
  location (+0.24 sp/s).
- This is a gain applied by a signal present before the target, not something built up after it.

**Visualisation:** population PSTHs (classic units, target in RF) for the four reward x expectation cells, aligned on
target onset, with the pre-target baseline shaded. Inset: the interaction term over time (sliding window, with CI).

B3. **No common "priority" axis.** (v3: restated within cell class and against reliability)
- Stimulus-driven factors (image vs bullseye, salience; onset 50-70 ms) are aligned across neurons: r = +0.26 in
  classic and +0.42 in putative interneurons.
- The location prior is reliable per neuron but not aligned with the stimulus factors (r ~ 0 against reliability
  ceilings of 0.3-0.6).
- Per-neuron reward effects have ~zero split-half reliability, so reward alignment cannot be tested either way (in
  interneurons reward-image r = +0.23, p = 0.004; in classic -0.17).
- The goal x stimulus interaction is weak (-0.08 SD, p = 0.03).
- A shared axis is what "SC as a priority map" predicts; it does not by itself separate resolver from hub.
- Panel: `results/aim1_panels/B3_factor_alignment/`.

**Visualisation:** scatter of per-neuron reward effect vs salience effect, with r and its null band, beside the
salience vs image scatter (aligned). Compact alternative: a small matrix of cross-neuron correlations between factor
effects.

B4. **Competitive dynamics across the whole map after target onset.**
- When the target appears at a 90-deg neighbour, the expected location's head start, seen in putative interneurons
  (+1.2 sp/s at 0-50 ms), is gone by 50-100 ms. (v3)
- At the 90-deg neighbour site, putative interneurons fall below baseline from ~200 ms (-2.3 sp/s at 200-250 ms);
  classic neurons return to baseline. (v3: measured at 90-deg neighbours only)
- Pre-saccadic suppression is strongest for the opposite (180-deg) target: interneurons 33% vs 19% at 90 deg;
  classic 12% vs 3%.
- This is SC's spatial winner-take-all at work. It is compatible with both views: a hub also uses SC's spatial
  winner-take-all to execute.

**Visualisation:** PSTH of units whose RF is at the expected location, on trials where the target appears elsewhere,
expected vs non-expected block (early advantage, its cancellation, the dip below baseline). Side bar chart: % of
units with pre-saccadic suppression by target angle. Panel: `results/aim1_panels/B4_map_competition/`.

B5. **SC state predicts RT, but not location-specifically.**
- Higher pre-target activity predicts faster saccades for units whose RF holds the expected target, the expected
  competitor or another non-target site (-3 to -6% RT per SD); not significant when the target is at an unexpected
  location (-1.2%, p = 0.5). No competitor coefficient is positive. (v3)
- A competition between locations predicts slower RTs when the competitor's site is more active.
- What the data show instead is general readiness.

**Visualisation:** bars of RT change per SD of pre-target rate by RF category (all negative, overlapping CIs), with
the half-plane a local competition predicts shaded. Panel: `results/aim1_panels/B5_rt_state_specificity/`.

### B6-B7: bilateral conflict task (`Newton_SC_2026/docs/NOTE_2026-10-31_competition_for_4factors.md`)

B6. **Behaviour.**
- Choice:
  - Reward biases choice: +0.28 per 100 ms of reward difference (08-20).
  - Salience barely affects it (p = 0.053).
  - Conflict costs +16 ms.
- RT: single-target RT correlates with the fitted belief value of the shown identity (Spearman -0.21, 08-20), but
  (v3) this does not survive drift controls: the belief tracks trial number (rho +0.47); RT vs true block value
  rho +0.05 (p = 0.35); with block fixed effects belief p = 0.33; within blocks the rich identity is not faster
  (+0.2 ms, CI -10 to +10). There is no reliable value effect on RT. 08-19 shows little use of reward.
- Panel: `results/aim1_panels/B6_bilateral_behaviour/` (x-axis: actual reward difference; belief version secondary).

**Visualisation:** psychometric curve, P(choose right) vs fitted belief-value difference, split by congruent vs
conflict trials. Inset: single-target RT vs belief value (binned).

B7. **No reward-difference coding and no competitor normalisation.**
- The left-minus-right SC signal carries the upcoming choice: added choice probability 0.60 at +100-150 ms, 0.80
  pre-saccade.
- It carries no reward difference in any window.
- A unit's response is not reduced by the competitor's reward (power >= 0.98 for a 10% drop per 100 ms).
- (v3) Neither effect is present, but neither separates the architectures: trained resolver networks also show
  none at these trial counts (reward-difference t ~ +0.2, range -0.9 to +1.1; competitor coefficient -0.006, range
  -0.023 to +0.005, inside the recorded null). The test excludes strong competitor normalisation, not resolution in
  SC. Choice coding passes the Bonferroni-corrected null (11 windows) at +100-150 ms (CP 0.60) and +350-400 ms (0.62).
- Caveat: the early choice window is underpowered (minimum detectable AUC 0.61-0.69).
- Panel: `results/aim1_panels/B7_lr_signal_normalisation/` (trained-resolver values and the planted -10%/100 ms effect
  drawn).

**Visualisation:**
- (i) Time course of the reward-difference and choice coefficients in the left-minus-right signal, with null bands.
- (ii) Distribution of per-unit competitor-reward coefficients, centred at zero, with the predicted -10% per 100 ms
  shift drawn in.

### B8-B10: the two colliculi (`results/intercollicular/SUMMARY.md`; bilateral `NOTE_2026-10-31_intercollicular_for_4factors.md`)

B8. **Ipsilateral responses look externally driven.**
- Non-classic units respond at the ipsilateral 90-deg neighbour as often as at the equidistant contralateral one
  (27% vs 25%).
- No reliable extra latency: amplitude-corrected excess 3.7 ms (CI -2.2 to 7.1, unit-level bootstrap; 111
  unit x stimulus conditions from 41 units). A 10-27 ms relay is excluded; a short (2-7 ms) one is not. (v3)
- The responses are not saccade-locked, and are salience- but not reward-modulated.

**Visualisation:** per-condition amplitude-corrected relay excess with median and CI, against the 10-27 ms
commissural-relay band (the raw ipsi-vs-contra scatter misleads, because latency depends on response amplitude).
Panel: `results/aim1_panels/B8_ipsi_latency_relay/`.

B9. **Ipsiversive activity is not a delayed copy of the other SC.**
- On average it rises with the other side's contraversive burst (lag -6 ms, CI -8 to +18), but (v3) the four
  session x side estimates range from -66 to +27 ms (two inside the relay range, each with 2-8 units): timing neither
  supports nor excludes a relay.
- Cross-SC CCGs are flat in every group (max |z| 1.77).
- Limit: a relay is excluded only if it would converge from fewer than ~170-500 neurons; this is an upper estimate (with
  the detection efficiency at full connectivity, ~60-130). A weak, broadly convergent commissural input is not excluded.
- Panel: `results/aim1_panels/B9_intercollicular_relay/`.

**Visualisation:** onset-aligned PSTHs of side X's burst and side Y's ipsiversive response, with the simulated relay
as a dashed, shifted curve. Inset: the pooled cross-SC CCG, flat, against the CCG a sparse relay would produce.

B10. **Candidate monosynaptic connections link neurons with similar selectivity.**
- Sharp 1-4 ms CCG peaks occur in about 1% of pairs, across all class combinations.
- Connected pairs have more similar factor profiles than matched unconnected pairs: +0.84 vs +0.56 (survives
  resampling by unit); without location, +0.43 vs +0.21 (p = 0.01 across pairs, but p = 0.08 when resampling units;
  v3), mostly face/non-face.
- Shared input would also produce this, and like-to-like excitation is expected under both views.
- What a resolver predicts is factor-specific inhibition between dissimilar units. Troughs: 8 listed, 5 powered (all
  I->I), too few to test it.
- Panel: `results/aim1_panels/B10_ccg_pair_selectivity/`.

**Visualisation:** an example sharp-peak CCG with its jitter band, and a histogram of profile similarity for
connected vs matched unconnected pairs.

### B11: data-fitted RNNs (`results/selectivity_by_type/NOTE.md`)

B11. **Weights fitted to the data cannot locate the computation.**
- RNN units fitted to individual neurons keep their spatial tuning (r 0.83-0.90, at ~60-70% of recorded magnitude)
  and match reward selectivity.
- They are at chance for identity and salience (fraction significant), although both are model inputs.
- They rank neurons by probability modulation (r +0.36 to +0.56) but get the population sign wrong. (v3)
- So "unstructured I->E weights" cannot by itself support the hub view. This motivates the task-trained approach
  in A.

**Visualisation:** per-factor scatter of recorded vs RNN ROC selectivity (moved, with its data, to
`results/aim1_panels/B11_rnn_selectivity_scatter/`).

## C. PREDICTIONS FOR THE PROPOSED EXPERIMENTS (hypotheses, not findings)

C1. **Priority hub vs outcome hub: does SC receive graded value or a decided outcome?** Status (v3): a model
prediction only; the recorded data do not currently support it.
- **Models (solid).** Single-target RT falls with the value of the shown target only when SC receives a graded
  signal (S9 = Spearman of single-target RT with the shown target's value, value engaged on every trial, trained
  readout threshold):
  - Outcome hub C -0.04 [-0.04, -0.01] (3 networks); priority hub B -0.17 [-0.45, -0.10] (6); B vs C rank-sum
    p = 0.024.
  - Resolvers are graded too: A -0.24 [-0.38, -0.16], AE -0.22 [-0.35, -0.18] (A/AE vs B p = 0.25). Hand-built:
    A -0.86, B -0.79, C 0.00.
  - So S9 separates graded delivery (resolver or priority hub) from categorical delivery (outcome hub); it does not
    separate resolver from hub (that is the identity test, A).
  - Robust to slowing RTs by 20-36 ms (raised readout threshold): no network's S9 changes by more than 0.06
    (`results/hub_models/rt_shift_pilot/`).
  - The outcome hub's upstream signal on single-target trials is not graded with value (rho -0.14), so it behaves
    categorically, as intended.
- **Data (v3: not supportive).** 08-20 RT correlates with the fitted belief (-0.21, n = 301, CI -0.32 to -0.09), but
  not with the true block value (rho +0.05, CI -0.06 to +0.17), and within blocks the rich identity is not faster
  (+0.2 ms, CI -10 to +10); the belief tracks trial number (rho +0.47). The -0.21 is most likely slow drift. 08-19:
  +0.03 (n = 184). Given the monkey's own belief sequence (range 143-197 ms), no trained network exceeds -0.15, so
  the task as run was also weakly powered for this signature.
- **Caveats.**
  - Value could speed saccades through a route outside SC's input (e.g. a separate vigour signal acting on
    brainstem or cortex), which the models do not include.
  - The signature needs value to vary widely across lone targets: in the Conflict Task, single-target probes on the
    big- and small-reward sides (with the confound that location-tied value is known before the target appears).
- Present as: "a graded (priority) hub, like a resolver, predicts that RT and SC activity scale with the value of a
  lone target; a categorical (outcome) hub predicts neither."

**Visualisation:** S9 per network (dots) for all four architectures (hand-built marked separately), with the recorded
08-20 value and its drift-controlled values; alternative: normalised RT vs value for one representative network of
each type beside the monkey's. Panel: `results/aim1_panels/C1_rt_value_signature/`.

C2. **Variable-timing conflict task** (luminant salience, time pressure, onset asynchrony between the salient and
the rich target).
- Resolver: the outcome depends on when each signal reaches SC, within SC's competitive window.
- Hub: the outcome follows the arrival of the goal signal from the loop.
- v3: simulations of the urgent design (resolver, priority hub and outcome hub, 69 networks) are running; the
  srsClassic version (reward tied to identity) runs first, a Conflict Task version (reward tied to location) follows.
  Treat as a qualitative prediction until then.

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

D1. **The models reproduce the classic SC bicuculline/muscimol phenotypes** (`results/hub_models/bicuculline/`;
numbers corrected in v3).
- Bicuculline (21 trained + 3 hand-built networks, d 0.05-0.35):
  - Saccades into the injected field speed up (d 0.2: 15-48 ms faster across networks, priority hubs 15-21 ms;
    hand-built 17-57 ms); the other field is unchanged except in one network (C_s0, 11 ms faster).
  - Choice shifts toward the injected field in most networks (exceptions: B_s5 at every dose; value-controlled
    bias of the C networks at d 0.05-0.1).
  - At d 0.35, fixation breaks rise to 0.3-71% of trials (below 20% in B_s2, B_s3, B_s4, C_s4 and the hand-built
    networks), 88-100% of them into the injected field: the irrepressible saccades Hikosaka & Wurtz (1985) reported.
- Muscimol (d 0.5) does the opposite: RT into the field +102 to +241 ms (undefined in 10 of 24 networks, too few
  saccades there), and choice moves away from it.
- No architecture differs (the priority hub is weaker); this supports using the models for prediction, not for
  separating resolver from hub (section E).

**Visualisation:** dose-response curves (RT change into the injected vs the other field; fixation-break rate; choice),
one line per architecture with the network range shaded; muscimol on its own axis. Panel:
`results/aim1_panels/D1_bicuculline_dose_response/`.

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
    control in 6 of 6 trained networks at d 0.2, vs 0.96-2.5 in 10 of 12 resolvers. (Behavioural numbers: D1, v3.)
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
