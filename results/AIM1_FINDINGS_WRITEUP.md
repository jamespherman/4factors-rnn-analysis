# Is SC a hub or a resolver? Findings, model predictions and candidate visualisations (Aim 1; 2026-10-31)

For R01 planning. Each finding lists the evidence, its strength and caveats, and **Visualisation:** a plot that would
demonstrate it, so pieces can be picked individually. Detailed notes and code are given per section (repo
`4factors-rnn-analysis`, branch `restore-lost-work`; bilateral analyses by the bilateral-SC-recordings session in
`Newton_SC_2026/docs/`).

Terms. **Resolver**: SC's own recurrent/inhibitory dynamics weigh goal-directed against stimulus-driven factors and
pick the winner. **Hub**: the weighing happens upstream (basal ganglia-thalamo-cortical loops) and SC receives and
executes the result, either as graded priority delivered by SNr disinhibition (**priority hub**) or as an
already-made choice (**outcome hub**). Data sets: 4factors unilateral recordings (Feynman, Newton; single targets;
reward cue x location prior x salience x identity) and bilateral Newton recordings in the srsClassic conflict task
(08-19, 08-20; reward tied to target identity with unsignalled block reversals; salience as isoluminant hue contrast).

---------------------------------------------------------------------------------------------------------------------

## 1. Recorded SC: goal signals arrive as pre-set gains, not as a computed priority (4factors data)
Notes: `results/competition/SYNTHESIS.md`, `NOTE_P1_P2.md`, `p3_interactions/`, `p4_timing/`.

1a. **Reward cue x location prior interact in behaviour.** The high-reward cue speeds saccades to the expected
location and slows them to the unexpected opposite location (Feynman reward x opposite +17.9% RT, p < 1e-6; peak
velocity mirrors it). The reward cue amplifies the prior rather than adding a location-independent benefit.
**Visualisation:** RT (or % change from low-reward) by target location class (expected / 90-deg neighbour /
opposite), lines for high vs low reward cue, one panel per monkey; the crossing lines show the interaction.

1b. **The same interaction is present in classic SC neurons, and it is set before the target appears.** Reward cue
enlarges the visual response to an expected target and shrinks it to an unexpected one (reward x expectation
+0.27 SD, p = 7e-6; putative interneurons +0.08, p = 0.07), peaking 80-100 ms after target onset; the reward cue
itself raises pre-target activity (+0.31 sp/s) and the prior raises it slightly at the expected location
(+0.24 sp/s). A gain applied by a pre-target signal, rather than something built up after target onset.
**Visualisation:** population PSTHs (classic units, target in RF) for the four reward x expectation cells, aligned
on target onset, with a shaded pre-target baseline and an inset of the interaction term over time (sliding window,
with CI) showing it is present from the first visual response.

1c. **No common "priority" axis.** Stimulus-driven factors (image vs bullseye, salience; onset 50-70 ms) are aligned
across neurons (r = 0.40), but goal and stimulus effects are not (|r| <= 0.12); goal x stimulus interaction is weak
(-0.08 SD, p = 0.03).
**Visualisation:** scatter of per-neuron reward-effect vs salience-effect (and vs image effect), with the r value and
its null band; contrast with a scatter of salience vs image effect (aligned). A single-number alternative: a small
matrix of cross-neuron correlations between factor effects.

1d. **Map-wide competitive dynamics after target onset.** When the target appears away from the expected location,
the prior's head start there is cancelled within ~100 ms, and every non-target location drops below baseline from
~150 ms (-1.5 sp/s). Pre-saccadic suppression is strongest for the opposite (180-deg) target (putative interneurons
33% vs 19% at 90 deg; classic 12% vs 3%). Consistent with SC's spatial winner-take-all; inputs that update at target
onset predict the same.
**Visualisation:** PSTH of units whose RF is at the expected location on trials where the target appears elsewhere,
expected vs non-expected block, showing the early advantage and its cancellation, plus the below-baseline dip;
side bar chart of % units with pre-saccadic suppression by target angle.

1e. **SC state predicts RT, but not location-specifically.** Higher pre-target activity predicts faster saccades
whether the neuron's RF holds the target, the expected competitor or neither (-5 to -6% RT per SD). A competition
between locations predicts slower RTs when the competitor's site is more active; the data show general readiness.
**Visualisation:** RT change per SD of pre-target rate, bars for "RF at target", "RF at expected competitor",
"RF elsewhere" (all negative, overlapping CIs), against a dashed bar showing the sign a local competition predicts.

## 2. Bilateral conflict task: SC carries the choice, not the reward comparison (bilateral-SC-recordings)
Notes: `Newton_SC_2026/docs/NOTE_2026-10-31_competition_for_4factors.md`; `results/competition/SYNTHESIS.md` item 7.

2a. **Behaviour.** Reward biases choice (+0.28 per 100 ms reward difference, 08-20), salience barely (p = 0.053);
conflict costs +16 ms. Single-target RT falls with the belief value of the shown identity (Spearman -0.21, 08-20;
delta-rule belief fitted to choices, `results/hub_models/belief/summary.md`). 08-19 shows little reward use.
**Visualisation:** psychometric curve P(choose right) vs fitted belief-value difference, split by salience
configuration (congruent/conflict); inset: single-target RT vs belief value of the shown target (binned means).

2b. **No reward-difference coding and no competitor normalisation.** The left-minus-right SC signal carries the
upcoming choice (added choice probability 0.60 at +100-150 ms, 0.80 pre-saccade) but not the reward difference in
any window; a unit's response to its target is not reduced by the competitor's reward (power >= 0.98 for a 10% drop
per 100 ms). A resolver comparing values in SC predicts both effects; a circuit receiving a pre-weighted signal
predicts neither.
**Visualisation:** two small panels: (i) time course of the reward-difference coefficient and the choice
coefficient in the L-R population signal, with null bands (choice rises, reward-difference stays at zero); (ii)
per-unit competitor-reward coefficient distribution centred at zero with the predicted -10%/100 ms shift drawn.
Caveat: the early choice window is underpowered (minimum detectable AUC 0.61-0.69).

## 3. The two colliculi: no relay of competition through the commissure
Notes: `results/intercollicular/SUMMARY.md`, `PHASE1_NOTE.md`, bilateral note `NOTE_2026-10-31_intercollicular_for_4factors.md`.

3a. **Ipsilateral responses look externally driven.** Non-classic units respond at the ipsilateral 90-deg neighbour
as often as at the equidistant contralateral one (27% vs 25%), with no relay latency (excess 3.7 ms, CI -1.3 to 6.8),
not saccade-locked, salience- but not reward-modulated.
**Visualisation:** latency scatter (ipsi vs contra response onset per unit) on the identity line, with the predicted
relay offset band (10-27 ms) shaded.

3b. **Bilateral: ipsiversive activity is not a delayed copy of the other SC.** Ipsiversive activation rises with
the other side's contraversive burst (lag -6 ms, CI -8 to +18; a relay of the recorded neurons predicts +10-27 ms);
cross-SC CCGs are flat in every group. A relay is excluded only if it converges from fewer than ~170-500 neurons
(r·N ≈ 5-10); a weak, broadly convergent commissural input is not excluded.
**Visualisation:** overlaid onset-aligned PSTHs of side X's contraversive burst and side Y's ipsiversive response
(with the simulated relay prediction as a dashed, shifted curve); inset: pooled cross-SC CCG (flat) vs the CCG a
sparse relay would produce.

3c. **Candidate monosynaptic connections link neurons with similar selectivity.** Sharp 1-4 ms CCG peaks (1% of pairs,
all class combinations) connect units with more similar factor profiles than matched unconnected pairs (5-factor r
+0.84 vs +0.56; without location +0.43 vs +0.22, p = 0.013, mostly face/non-face identity). Shared input would also
produce this. Inhibitory troughs (8) are too rare to test whether inhibition is factor-specific.
**Visualisation:** example sharp-peak CCG with jitter band; histogram of selectivity-profile similarity for connected
vs matched unconnected pairs.

## 4. Data-fitted RNNs: what the SC circuit fit reproduces
Notes: `results/selectivity_by_type/NOTE.md`, scatter figures there; earlier context-model reports.

4a. **Location tuning reproduced; identity and salience selectivity not; probability sign wrong.** RNN units fit to
individual neurons keep spatial tuning (per-neuron r 0.83-0.90, ~60-70% of recorded magnitude), match or exceed
reward selectivity, are at chance for identity and salience (although both are model inputs), and respond more on
high-probability trials where recorded neurons respond less. So fitted weights cannot identify which route factor
signals take, and "unstructured I->E weights" cannot by itself support the hub view.
**Visualisation:** per-factor scatter of recorded vs RNN ROC selectivity (one panel per factor, E and I units in
separate colours, monkeys by marker shape), already made (`results/selectivity_by_type/`).

## 5. Task-trained models: the resolver predicts an identity code that recorded SC lacks (strongest new result)
Notes: `results/hub_models/S8_RESULT.md`; plan `plans/2026-10-31_hub_resolver_models_plan.md` (v2, reviewed);
models from conflict-task-RNN `code/srs` (bilateral SC E-I networks with Dale's law, upstream BG-like module, SNr
tonic inhibition, stochastic saccade readout; REINFORCE training on srsClassic with curriculum).

Architectures, all with identical SC: **A** resolver (identity and belief enter SC through fixed random weights, so
SC must bind identity x belief x location itself), **AE** (same, input onto E units only), **B** priority hub
(upstream gets identity and belief and sends graded per-location disinhibition via SNr), **C** outcome hub
(upstream softmax winner-take-all, near-categorical disinhibition). Every architecture also carries a weak sensory
shape code in SC (so a positive identity effect would be ambiguous; a null is not).

5a. **Logic.** In srsClassic, reward is tied to target identity and reverses between blocks. A resolver must bind
identity to block value inside SC, so its SC neurons carry identity beyond value. A hub binds upstream, so SC gets
value only.

5b. **Every trained resolver network carries the code; no hub does.** Fraction of units (contralateral to the
target, single-target trials) with |identity effect beyond value| > 0.2: A 0.42-0.61, AE 0.28-0.58 (6 seeds each),
B and C <= 0.04. Hand-built networks agree (A 0.40, B/C ~0).
**Visualisation:** per-network distributions of the per-unit identity effect (violin or ECDF), coloured by
architecture, with the recorded 08-20 distribution and its permutation null overlaid.

5c. **Recorded SC is at the null, and would have detected the resolver code.** 08-20 (301 single-target trials,
8 blocks, 27-31 contralateral units): mean |delta| 0.131 vs null 95th percentile 0.166; quasi-Poisson |z| 0.54-0.62
(chance); 08-19 also null. Planting each network's per-unit effects into the real counts: P(detect) A 1.00, AE 1.00
(delay window), B 0.00, C 0.00; every one of the 12 resolver networks >= 0.99 in the delay and pre-saccadic windows.
**Visualisation:** the key panel: P(detect) per network (dots, grouped by architecture; delay and pre-saccadic
windows) with the recorded result marked "null", i.e. resolvers all at the top, hubs at the floor. Optional companion:
observed statistic vs the distribution of the statistic for planted-resolver data and for the null.

5d. **Caveats.** A resolver with a much sparser or weaker identity code than any network we trained (e.g. a quarter
of units at |delta| 0.2, ~20% power) is not excluded; one well-powered session; the recorded populations had few
classic units on the side of interest; assumes model log-rate effects transfer at recorded rates. This shows where
identity x value is bound (upstream), not whether SC receives graded priority or a decided outcome (section 6).

## 6. Priority hub vs outcome hub: single-target RT vs value
Hand-built networks: Spearman(RT, belief value of the shown identity) is -0.32 for A and B (graded) vs +0.02 for C
(categorical); the recorded value is -0.21, which favours a graded signal (priority hub). [Trained-ensemble result:
PENDING, CRC job running, ~5 pm.]
**Visualisation:** distribution of the RT-value correlation per architecture (per network or per simulated session)
with the recorded -0.21 as a vertical line; or binned RT vs value curves for the three architectures beside the
monkey's.

## 7. In-silico bicuculline: predictions (new today)
Notes: `results/hub_models/bicuculline/summary.md`, `per_network.csv`; plan `plans/2026-10-31_bicuculline_plan.md`
(v2, independently reviewed); `scripts/srs_bicuculline.py`. All 24 networks (hand-built A, B, C; trained A, AE, B
x 6 seeds; C x 3). Bicuculline in the left SC scales GABA-A synapses onto its E units (local and commissural I -> E,
and SNr -> E) by 1 - d, d = 0.05-0.35; ~1,900 trials x 3 seeds per network and condition. Controls: local-only and
SNr-only blockade, muscimol (d 0.5), and a value-clamped upstream.

7a. **The models reproduce the classic bicuculline phenotype (validation).** In every architecture, a growing dose
speeds saccades into the injected field (d 0.2: 17-42 ms faster; other field unchanged) and biases choice toward it.
At d >= 0.35 the networks make irrepressible saccades into that field (fixation breaks on 20-70% of trials, 99-100%
toward the injected field; at d 0.9 on every trial), as Hikosaka & Wurtz (1985) found. Muscimol does the opposite
(RT into the field +120-210 ms, choice away). The models therefore behave like real SC under GABA-A blockade.
**Visualisation:** dose-response curves (x = d; y = RT change into the injected vs the other field, and
fixation-break rate), one line per architecture with the network range shaded. They overlap, which is the point of 7b.

7b. **Behaviour alone does not distinguish resolver from hub.** At tolerable doses (fixation breaks <= 10%), all
architectures show the same dose-dependent speeding and side bias. RT vs value on single-target trials does not
flatten in either class, because disinhibition compresses RTs toward a floor in all of them. The resolver-specific
prediction proposed earlier (choice proportions changing in a conflict-dependent way, against execution deficits
only for a hub) is not borne out: no architecture shows a distinctive behavioural signature.
**Visualisation:** a 2 x 2 of choice bias (change in P(choose injected field)) vs RT speeding, one dot per network
and dose, coloured by architecture: one shared cloud.

7c. **The neural signature: value modulation at the injected site shrinks only in the priority hub.** In B, value
reaches SC by modulating SNr inhibition, so blocking GABA-A at the site removes part of the channel that carries
it. The proportional value modulation of the injected-field SC population (delay window; per 100 ms of belief value)
falls with dose: d 0.2: 0.84-0.92 of control in all 6 trained B networks (hand-built 0.81); d 0.35: 0.73-0.91.
In resolvers, value enters SC through excitatory inputs, and the modulation is preserved or grows (d 0.2: 10 of 12
trained A/AE networks 0.96-2.5; hand-built A 1.04). The opposite field is unchanged in all networks. Clamping the
upstream to be value-free abolishes SC value modulation in B (index ~0) but not in A/AE (median 0.92-0.97).
- Strength: directional and consistent (6/6 B vs 10/12 resolvers), but modest (13-25% at tolerable doses).
- Caveats:
  - It requires recording at the injection site, where rates rise sharply.
  - The value modulation it acts on is small in trained networks (11-16% per 100 ms of value).
  - Two trained resolver networks (A_s0, A_s5) fall in the hub range.
  - Some trained resolvers' upstream carries a non-spatial value signal (rho -0.57 to 0.60).
  - Outcome hubs (C) have almost no SC value modulation to begin with, so bicuculline cannot separate B from C.
- An SNr-terminal-selective blockade would give a larger dissociation (B 0.47-0.85 vs A/AE median ~1.0 at SNr-only
  d 0.5), but it produces fixation breaks on more than half the trials and is not a practical primate experiment.
- Bottom line: bicuculline is an excellent validation and perturbation tool, but a weak test of resolver vs hub.
  The variable-timing task and the identity test (5c) at higher power are stronger.
**Visualisation:** value-modulation index (injected field) vs dose, one thin line per trained network coloured by
architecture (B lines below 1, resolver lines at or above 1), with a dashed line at 1 and the opposite field as a grey
reference. A companion schematic: in B, value enters through the SNr synapse that bicuculline blocks; in A, it
enters through excitation.

7d. **Priority vs outcome hub, preliminary from the same simulations** (uncalibrated, all trials value-engaged):
control single-target RT-value rho in trained networks A -0.29, AE -0.22, B -0.15, C +0.03 (recorded -0.21). The
graded vs categorical split of the hand-builds holds in trained networks. Calibrated numbers come from the CRC run
(section 6).

## 8. Experiments that would decide the remaining questions (and model predictions to accompany them)
- Variable-timing conflict task with luminant salience and time pressure (onset asynchrony between salient and rich
  targets): a resolver's outcome depends on when each signal arrives in SC; a hub's tracks the goal signal's arrival
  from the loop. [Model simulation not yet run.]
- Bilateral recordings with mirrored, interleaved targets and enough classic units per side, to repeat the identity
  (5c) and normalisation (2b) tests at higher power and test commissural competition directly.
- Unilateral SC inactivation while recording the opposite SC (opposite-direction pre-saccadic suppression should
  vanish only if relayed).
