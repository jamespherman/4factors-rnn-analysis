# Is SC the place where reward and salience are resolved, or just where the result is carried out? Literature and methods for the task-trained RNN argument

Prepared 2026-10-31. Citation tags: **[V]** means I checked authors, title, venue and DOI this session against PubMed or the NeurIPS/arXiv pages, and the summary follows the retrieved abstract. **[M]** means the citation is from memory and was not checked; confirm it before citing.

---

## 0. The single most important design point (it comes from the Hikosaka & Wurtz 1985 papers)

Bicuculline in SC does **not** cleanly separate local SC inhibition from basal ganglia (BG) inhibition. Hikosaka & Wurtz (1985 II) showed that muscimol in SNr has "the same general effect as bicuculline in the SC". They concluded that a large share of the GABA-A inhibition onto SC saccade cells comes from SNr. So a local GABA-A block removes two things at the injection site: (i) inhibition from local and commissural SC interneurons (the resolver machinery), and (ii) the nigral channel through which a hub would deliver its weighted priority, and through which Hikosaka's caudate-SNr value pathway delivers reward value anyway.

Two consequences follow:
1. The in-silico bicuculline has to scale **all** GABA-A synapses onto SC units at the site. That includes the BG-to-SC inhibitory projection in the hub model, not only the I-to-E weights inside SC.
2. Both architectures predict a main effect: contralateral disinhibition and a bias toward the injected field. They can only be told apart by **how the perturbation interacts with the factors**, i.e. whether it changes the reward-salience "exchange rate" or only adds an offset (Section 4/6).

A related point: in the real brain, reward is probably carried to SC through SNr in *both* architectures (Hikosaka). So the hub-vs-resolver contrast should be defined by **what has already been computed upstream**, not by the route or sign of the input:
- Hub: the SNr-like output already carries the combined, competition-resolved priority.
- Resolver: the upstream channel carries reward per location, unmixed and unresolved, and SC does the combining.

Input sign (excitatory cortical vs disinhibitory nigral) should therefore be a crossed nuisance factor in both model classes.

---

## 1. Circuit models and physiology of SC-BG target selection

**Lo & Wang 2006**, *Nat Neurosci* 9:956-963, doi:10.1038/nn1722 [V].
- Model: a spiking cortex-BG-SC model. Local SC dynamics give an all-or-none burst that *detects* when upstream cortex crosses threshold.
- The threshold depends only weakly on cortico-collicular efficacy. It is tuned by cortico-striatal plasticity.
- Relevance: this is a clear "SC executes" (hub-like) model. SC is a threshold detector and the BG set the policy.
- Predictions: weakening SC input changes RT little and changes choice little. BG manipulations change the speed-accuracy policy.

**Wei, Rubin & Wang 2015**, *J Neurosci* 35:4052-4064, doi:10.1523/JNEUROSCI.3611-14.2015 [V].
- Extends Lo & Wang. Threshold tuning by cortico-striatal plasticity works only if the indirect pathway counterbalances the direct pathway at the BG output.
- Relevance: if the hub module is modelled on BG, it needs opposing direct and indirect channels. An SNr-only output is too simple.

**Bogacz & Gurney 2007**, *Neural Comput* 19:442-477, doi:10.1162/neco.2007.19.2.442 [V].
- Cortex-BG anatomy implements the multihypothesis sequential probability ratio test (MSPRT).
- The BG output carries a quantity in which each channel's evidence is normalised against all the others (softmax/log-sum-exp).
- Relevance: this is the theoretical justification for a hub whose output is *already normalised by competitors*. It makes a sharp prediction. Under a hub, normalisation by the competitor should show up in the SNr-like input to SC, and SC need not add any of its own. That fits the PI's well-powered finding that the competitor's reward does not normalise SC responses.

**Trappenberg, Dorris, Munoz & Klein 2001**, *J Cogn Neurosci* 13:256-271, doi:10.1162/089892901564306 [V].
- Exogenous and endogenous signals are integrated "by dynamic competition across local collicular interactions" (short-range excitation, long-range inhibition).
- Accounts for gap, distractor, anti-saccade and target-probability effects on saccadic RT.
- Relevance: this is the canonical **resolver** model. The SC architecture used in the resolver RNN should be able to reproduce these behavioural signatures.

**Arai & Keller** (2004 IEEE EMBS proceedings, doi:10.1109/IEMBS.2004.1404299 [V]; full model in Arai & Keller 2005 *Biol Cybern* [M]).
- Distributed SC-cerebellum model for saccade curvature and averaging with competing stimuli.
- Long-range lateral inhibition is replaced by local short-range inhibition plus *spatially tuned SNr inhibition*.
- Relevance: a precedent for putting the competitive inhibition in SNr rather than inside SC.

**Kopec, Erlich, Brunton, Deisseroth & Brody 2015**, *Neuron* 88:367-377, doi:10.1016/j.neuron.2015.08.033 [V].
- In rat, FOF and SC inactivations had time courses *opposite* to the neural encoding of the choice.
- A two-area attractor model in which both areas contribute equally reconciled the two, and predicted supralinear impairment from joint inactivation (confirmed).
- Relevance:
  - Encoding and perturbation sensitivity can dissociate, so both must be modelled.
  - Time-resolved perturbations, plus interaction with trial difficulty, are diagnostic.

**Duan, Pagan, Piet, Kopec, Akrami, Riordan, Erlich & Brody 2021**, *Nat Neurosci* 24:1110-1120, doi:10.1038/s41593-021-00865-x [V].
- A subset of rat SC neurons links context to choice and encodes choice *earlier than frontal cortex*.
- SC activity is needed during context encoding but not during choice formation.
- The authors searched over SC circuit models for ones that matched the data.
- Relevance: SC can do flexible computation itself (a counter-example to "SC only executes"). It is also a methodological precedent for searching across a family of architectures.

**Jun, Bautista, Nunez, Allen, Tak, Alvarez & Basso 2021**, *Nat Neurosci* 24:1121-1131, doi:10.1038/s41593-021-00878-6 [V].
- Unilateral SC inactivation in a Glass-pattern discrimination task biased choices and changed RTs.
- Sequential-sampling fits show "a constant offset to a time-varying evidence signal for the ipsilateral choice".
- Relevance: this matters a lot here. An *additive offset* is exactly the hub signature: SC perturbation shifts the decision variable without changing how evidence is weighted. The authors read it as SC participating in evidence computation. The models should state formally which reading each architecture supports.

**Crapse, Lau & Basso 2018**, *Neuron* 97:181-194, doi:10.1016/j.neuron.2017.12.006 [V].
- SC activity tracks the decision *criterion* independently of report location. Microstimulation shifts the criterion.
- Relevance: SC sets a bias or criterion term, again an offset-like role.

**Hikosaka and colleagues: value pathways.**
- Hikosaka, Kim, Yasuda & Yamamoto 2014, *Annu Rev Neurosci* 37:289-306, doi:10.1146/annurev-neuro-071013-013924 [V].
- Yasuda & Hikosaka 2015, *J Neurophysiol* 113:1681-1696, doi:10.1152/jn.00674.2014 [V].
- Amita et al. 2020, *Nat Commun* 11:1876, doi:10.1038/s41467-020-15802-y [V].
- Findings: CDh carries flexible values via rvmSNr; CDt carries stable values via cdlSNr; both project largely separately to SC. Optogenetic activation of the CDt-to-SNr terminals inhibits SNr, excites SC visual-saccadic neurons, and evokes contralateral saccades; those SC neurons prefer good objects.
- Relevance: reward reaches SC as *disinhibition through SNr*. In the hub model, the hub's output should therefore be an inhibitory SNr-like channel. In the resolver, the per-location reward input should also be allowed to come through that channel (see Section 0).

**Krauzlis lab: SC in attention and selection.**
- Carello & Krauzlis 2004, *Neuron* 43:575-583, doi:10.1016/j.neuron.2004.07.026 [V]: subthreshold SC microstimulation biases target *choice* (pursuit and saccades), separately from motor preparation.
- Herman, Katz & Krauzlis 2018, *Nat Neurosci* 21:1651-1655, doi:10.1038/s41593-018-0271-5 [V]: a left-minus-right SC readout explains detection, including after causal manipulations. This motivates using the bilateral L-R signal as the model's readout.
- Arcizet & Krauzlis 2018, *PLoS Biol* 16:e2005930, doi:10.1371/journal.pbio.2005930 [V]: caudate neurons strongly select the relevant location in covert attention.
- Herman, Arcizet & Krauzlis 2020, *eLife* 9:e53998, doi:10.7554/eLife.53998 [V]: caudate attention modulation *depends on ipsilateral SC activity*. This is a loop, SC to BG and back. **Implication:** the hub architecture should include an SC-to-hub feedback projection (via thalamus), so that "hub" does not mean feedforward only.
- Zénon & Krauzlis 2012, *Nature* 489:434-437, doi:10.1038/nature11497 [V]: SC inactivation produced attention deficits without changing cortical attention modulation, so SC acts through non-cortical routes (BG is a candidate).
- Krauzlis, Bogadhi, Herman & Bollimunta 2018, *Cortex* 102:161-175, doi:10.1016/j.cortex.2017.08.026 [V]: an evolutionary review of tectum, striatum and thalamus loops in selection. It is the "SC-BG hub" framing.

**Mysore & Knudsen 2013**, *Nat Neurosci* 16:473-478, doi:10.1038/nn.3352 [V].
- In barn owl, inactivating the GABAergic nucleus isthmi pars magnocellularis (Imc) abolished both exogenous *and* endogenous competitive interactions in optic tectum.
- Relevance: the strongest precedent for one shared inhibitory competition circuit acting on both stimulus-driven and goal-driven factors. It is *outside* tectum but still midbrain.
- Implication: the "competition is local to SC" hypothesis is falsifiable only for the GABA-A inputs that the injection actually reaches. The primate homologue (parabigeminal / tegmental) is uncertain [M].

**Essig, Hunt & Felsen 2021**, *Commun Biol* 4:719, doi:10.1038/s42003-021-02248-1 [V].
- In mouse, GABAergic SC neurons "do not locally suppress premotor output". Long-range inhibition (inter-collicular or extrinsic) dominates spatial choice, and an attractor model built from intrinsic SC circuitry was sufficient.
- Relevance: bilateral models need explicit commissural inhibition, which must be kept separate from local I-to-E inhibition when simulating a unilateral injection.

**Model predictions for value-vs-salience conflict (synthesis).**
- Resolver models (Trappenberg; Wang-type attractor in SC): SC activity should carry both factors before commitment, show competitor normalisation via lateral inhibition, and lose its weighting function when local inhibition is perturbed.
- Hub models (Lo & Wang; Bogacz & Gurney; Hikosaka): SC receives a priority already combined from both factors (and normalised, in MSPRT). SC perturbations should mostly add offsets or change gain and threshold, whereas upstream (striatal) manipulations should change the weighting.

---

## 2. Locating a computation in multi-area task-trained RNNs

**Multi-area RNNs and lesions.**
- Kleinman, Chandrasekaran & Kao 2021, NeurIPS, "A mechanistic multi-area recurrent network model of decision-making", https://proceedings.neurips.cc (poster 27881) [V via the NeurIPS page]. Multi-area plus Dale's law gives more biologically plausible solutions, and output-relevant information propagates preferentially between areas.
- Follow-up: Kleinman et al. 2023/2025, "The information bottleneck as a principle underlying multi-area cortical representations during decision-making", bioRxiv doi:10.1101/2023.07.12.548742 [V]; later eLife "A cortical information bottleneck during decision-making" [V via search]. The upstream area holds inputs plus choice, the downstream area holds a *minimal sufficient* choice representation, and choice is propagated through inter-area connections selectively aligned with the choice axis.
- **Direct relevance:** "minimal sufficient representation of choice, no residual factor information" is what the hub predicts SC should look like. The analyses carry over directly: decode factor information versus choice per area, and measure the alignment of inter-area weights with factor and choice axes (an "effective input subspace").
- Michaels, Schaffelhofer, Agudelo-Toro & Scherberger 2020, *PNAS* 117:32124-32135, doi:10.1073/pnas.2005087117 [V]: a modular RNN with no neural data in training predicted area-specific dynamics, and module lesions reproduced lesion deficits. Precedent for "task-trained, then compared with data".
- Pinto et al. 2019, *Neuron* 104:810-824, doi:10.1016/j.neuron.2019.08.025 [V]: a modular RNN reproduces task-dependent necessity of areas. A caution: whether an area is necessary depends on the task strategy, not only on anatomy.
- Perich & Rajan 2020, *Curr Opin Neurobiol* 65:146-151, doi:10.1016/j.conb.2020.11.003 [V]: review of "network of networks" models. Data-constrained CURBD (Perich et al. 2020 bioRxiv [M]) infers inter-area currents; it could be applied to the existing data-fit SC RNNs as a complement.

**Modularity and population structure.**
- Yang, Joglekar, Song, Newsome & Wang 2019, *Nat Neurosci* 22:297-306, doi:10.1038/s41593-018-0310-2 [V]: functional clusters emerge; task-variance clustering is a usable unit-level analysis.
- Driscoll, Shenoy & Sussillo 2024, *Nat Neurosci* 27:1349-1363, doi:10.1038/s41593-024-01668-6 [V]: reusable dynamical motifs (attractors, decision boundaries) are implemented by unit clusters when activations are non-negative, and lesioning a cluster gives modular deficits. Relevance: with rate (non-negative) units, look for a "decision-boundary motif" and ask *which module hosts it*.
- Dubreuil, Valente, Beiran, Mastrogiuseppe & Ostojic 2022, *Nat Neurosci* 25:783-794, doi:10.1038/s41593-022-01088-4 [V]: flexible input-output mappings need non-random subpopulation structure acting through gain modulation. Relevance:
  - The PI's finding that goal factors act as pre-target gains is the Dubreuil mechanism.
  - A resolver SC would need such subpopulations (factor-specific gain populations).
  - A hub SC could stay low-rank and unstructured.
  - The low-rank fitting framework (Valente, Pillow & Ostojic 2022 NeurIPS [M]) can test the minimal rank and population count needed in the SC module.
- Mastrogiuseppe & Ostojic 2018, *Neuron* 99:609-623 [M]: low-rank theory, the basis for the above.

**Robustness across seeds and architectures.**
- Maheswaranathan, Williams, Golub, Ganguli & Sussillo 2019, NeurIPS, arXiv:1907.08549 [V]: representational geometry (CCA-type measures) is highly architecture-dependent, but fixed-point topology and linearised dynamics are much more universal.
- **Recommendation:** base the hub/resolver discrimination on topological, dynamical and perturbational signatures, not on representational similarity. Report every signature as a distribution across at least 20 seeds times a hyperparameter grid (nonlinearity, noise, regularisation, E:I, connection sparsity, time constants, hub capacity).
- Song, Yang & Wang 2016, *PLoS Comput Biol* 12:e1004792, doi:10.1371/journal.pcbi.1004792 [V]: the standard E-I (Dale) training framework. It is compatible with the repo's `ei_rnn.py` design.

**Comparing trained units with recorded neurons using the same analyses.**
- Mante, Sussillo, Shenoy & Newsome 2013, *Nature* 503:78-84, doi:10.1038/nature12742 [V]: targeted dimensionality reduction (TDR) plus fixed-point analysis applied identically to PFC and the RNN.
- Pagan et al. 2024/2025, *Nature* 639:421-429, doi:10.1038/s41586-024-08433-6 [V]: context-dependent accumulation has three dynamical solutions, and every network uses a *mixture*. Individuals differ, but a specific behaviour-to-neural link holds. Relevance:
  - Expect hub and resolver to be endpoints of a continuum, so define a continuous "resolver index".
  - Look for a behaviour-to-neural link that holds within each class (e.g. per-session exchange rate versus SC factor coding).
- Galgali, Sahani & Mante 2023, *Nat Neurosci* 26:326-338, doi:10.1038/s41593-022-01230-2 [V]: *residual dynamics* (trial-to-trial deviations around the condition mean) separate recurrent from input-driven contributions. Applied to PFC, it suggested rotations were input-driven.
  - This is the most directly usable data-side analysis for "does SC compute or receive". Estimate SC residual dynamics (bilateral, around target onset) in the data and in both model classes.
  - In a resolver, the residual dynamics along the reward-salience combination axis should be recurrent: an unstable or slow saddle mode during competition.
  - In a hub, that axis should be input-driven and stable, and only the final WTA axis should be unstable.

---

## 3. GABAergic manipulations of primate SC: what is known

**Muscimol (inactivation).**
- McPeek & Keller 2004, *Nat Neurosci* 7:757-763, doi:10.1038/nn1269 [V]: during visual search, saccades with the target in the inactivated field were often misdirected to distractors. Single-target control tasks showed no low-level visual or motor deficit. The authors pose exactly the PI's question ("directly involved in target selection, or merely receives selection-related signals from cortex") and conclude involvement. However, a pure loss of activity in an executing WTA also produces this (Section 4).
- Nummela & Krauzlis 2010, *J Neurophysiol* 104:1538-1548, doi:10.1152/jn.00406.2010 [V]: inactivation biases two-target choice away from the affected field for saccades, pursuit and (less so) button presses, with *no effect on single targets*.
- Nummela & Krauzlis 2011, *J Neurosci* 31:8059-8066, doi:10.1523/JNEUROSCI.5480-10.2011 [V]: early orienting still follows a vector average, but the affected-field weight drops and the other weight rises. So SC contributes to *weighted integration*, not only WTA. A useful model signature: the weights of the early, pre-WTA vector average.
- Lovejoy & Krauzlis 2010, *Nat Neurosci* 13:261-266, doi:10.1038/nn.2470 [V]: profound covert-selection deficit, but only when a distractor carrying conflicting information was in the unaffected field. This is the "competition-only" signature.
- Song, Rafal & McPeek 2011, *PNAS* 108:E1433-E1440, doi:10.1073/pnas.1109656108 [V]: reach target selection deficits, so SC contributes to an effector-general priority map.
- Hikosaka & Wurtz 1983, *Brain Res* 272:368-372, doi:10.1016/0006-8993(83)90586-3 [V], and 1985 I, *J Neurophysiol* 53:266-291, doi:10.1152/jn.1985.53.1.266 [V]: muscimol produces longer latency, lower velocity and inaccurate memory saccades.

**Bicuculline (disinhibition).**
- Hikosaka & Wurtz 1985 I (above) [V]: "stereotyped and apparently irrepressible saccades" to the movement field and loss of fixation.
- Hikosaka & Wurtz 1985 II, *J Neurophysiol* 53:292-308, doi:10.1152/jn.1985.53.1.292 [V]: SNr muscimol mimics SC bicuculline (Section 0).
- Munoz & Wurtz 1993 II, *J Neurophysiol* 70:576-589, doi:10.1152/jn.1993.70.2.576 [V]: bicuculline in *rostral* SC (the fixation zone) *increased* saccade latency. This is an important site and spread confound for any injection near the rostral pole.
- Dias, Kiesau & Segraves 1995, *J Neurophysiol* 74:2744-2748, doi:10.1152/jn.1995.74.6.2744 [V]: bicuculline in FEF gives irrepressible saccades too, a comparison point. Schiller & Tehovnik 2005, *Prog Brain Res* 149:157-171, doi:10.1016/S0079-6123(05)49012-3 [V]: FEF bicuculline facilitates target choice.
- **Gap:** I found no primate study of SC bicuculline on *choice between competing targets* (PubMed search, 1995 onward). The PI's experiment would be new. Practically, it needs low doses and sites away from the rostral pole, because irrepressible saccades will otherwise swamp choice measurements.

**Resolver vs hub predictions for these manipulations (summary).**
Both architectures predict the classic muscimol results (bias away, competition-only deficits) and the bicuculline main effect (bias toward the field, intrusions). They differ in the *factor-specific interaction*:
- **Hub.** The perturbation acts on a received scalar priority difference ΔP.
  - It adds an offset to ΔP and possibly changes WTA gain or threshold.
  - Its effect is the same whether ΔP arose from a reward difference or a salience difference.
  - The indifference line in (Δreward, Δsalience) space *translates* but does not rotate.
  - One caveat: if the hub signal enters SC only through the GABA-A (SNr) channel, bicuculline also *occludes* it. Choices for the injected-field target then become less sensitive to *all* priority (both factors in the hub), so the line again translates, with a possible change of slope along ΔP.
- **Resolver.** Local inhibition takes part in combining the factors (divisive or subtractive normalisation, competition between differently weighted inputs).
  - Perturbation changes the *relative weights*: the indifference line rotates and the exchange rate changes.
  - If the reward input arrives disinhibitorily while salience arrives excitatorily, bicuculline selectively removes the reward effect at the site while sparing the salience effect.
  - The data would show this as a factor-specific loss: reward modulation of SC responses and of choice abolished at the site, salience modulation intact.
  - In the hub with a combined SNr channel, both factors' influence is lost together.

---

## 4. Theory: when does a WTA network "decide" vs "execute"?

**Wang 2002**, *Neuron* 36:955-968, doi:10.1016/S0896-6273(02)01092-9 [V], and **Wong & Wang 2006**, *J Neurosci* 26:1314-1328, doi:10.1523/JNEUROSCI.3733-05.2006 [V].
- Slow recurrent excitation with shared inhibition integrates input and makes a categorical choice near an unstable saddle.
- Two regimes exist, WTA with and without persistent attractors.
- Signatures:
  - RT and slowing scale with the closeness of the input difference to zero.
  - Errors are slower than correct choices.
  - The choice depends on *integrated* input over a long window, near the saddle.
  - In a strongly input-driven regime the choice is set almost instantly by the input sign, with little integration.

**Finkelstein, Fontolan, Economo, Li, Romani & Svoboda 2021**, *Nat Neurosci* 24:843-850 [M for pages and DOI, likely 10.1038/s41593-021-00840-6; the erratum doi:10.1038/s41593-021-00869-7 is V].
- Discrete attractors make choice robust to perturbations late in the trial, but early perturbations or distractors can switch it.
- This is a perturbation-timing signature of where the attractor lives.

**Distinguishing "computes the weighted sum" from "receives it": candidate signatures.**
1. *Graded factor information before commitment.* A resolver SC must hold reward and salience as separable (linearly decodable) quantities for the two targets during the competition window. A hub SC needs only ΔP. The PI's data show no reward-difference coding in the SC L-R signal. The models must establish whether task-trained resolvers *necessarily* show it, or whether they can hide it, e.g. via pure pre-target gains (Dubreuil) that are invisible to an L-R difference readout.
2. *Effective dimensionality of SC input.* Hub: SC input is rank-1 per side for the priority (plus visual drive). Resolver: rank of 2 or more, with separate reward and salience axes. In the model this is measurable directly from input weights. In data, use residual dynamics and communication-subspace analyses.
3. *Exchange-rate invariance under SC perturbation* (Section 3). This is the cleanest behavioural signature.
4. *Competitor normalisation.* MSPRT-type hubs normalise upstream, and resolvers normalise in SC through lateral inhibition (Louie, Grattan & Glimcher 2011, *J Neurosci* 31:10627-10639, doi:10.1523/JNEUROSCI.1237-11.2011 [V], showed divisive value normalisation in LIP). The PI's finding of no normalisation by the competitor's reward in SC is hub-consistent, provided the trained resolvers reliably develop it. That is a quantitative question for the ensembles.
5. *Saddle location.* With the fixed-point and linearisation tools of Sussillo & Barak 2013 [M], find the unstable mode and test whether it lies within SC units (resolver) or within the hub (hub WTA happening upstream). A hub with an SC WTA has a saddle in SC too, but its *stable manifold is aligned with ΔP*, not with separate factor axes.
6. *Dependence on initial state and history.* Pre-target state (the PI's "gains set before target") biases choice in both classes. In a resolver, the pre-target state should change the *weighting* (multiplicative interaction of state with factor). In a hub it should change only the offset, unless the gain sits upstream.
7. *Slowing near the indifference line.* Both predict slowing. The resolver additionally predicts that RT depends on the *components* (|Δreward| and |Δsalience| separately, e.g. conflict cost when they oppose) beyond ΔP. A hub predicts RT is a function of ΔP only, assuming the hub has no conflict-dependent latency of its own (test this in the models).
8. *Hysteresis.* With targets that change across the trial, or sequential trials, a resolver SC in the attractor regime shows hysteresis in the SC state. A hub shows it in the hub state, while SC follows its input. Probably not experimentally accessible, but it is a good model diagnostic.

---

## 5. Recommendations

### (a) Architectures to build (all Dale-constrained, bilateral, task-trained only)
- **Common core.**
  - Left and right SC modules (E:I about 4:1).
  - Local I-to-E and explicit inhibitory commissural projections, kept separate so that a unilateral "injection" is well defined (Essig 2021).
  - A saccade or choice readout from SC E units only, as SC is the final common output.
  - Visual (salience) input enters SC retinotopically and excitatorily in all models.
- **Hub (H).** An upstream module (BG-like, with direct and indirect opposing paths, Wei et al. 2015) receives reward cues and visual input. It projects to SC through an *inhibitory, tonically active SNr-like population*. Add SC-to-hub feedback via a thalamic relay (Herman et al. 2020). SC has no direct access to the reward cue.
- **Resolver (R).** The same upstream module, but its output is constrained to carry reward per location only: an information bottleneck enforced by a decoding penalty on salience, or by architecture (no visual input to the hub). SC must combine the factors.
- **Free (F), the key control.** SC receives raw factors *and* the hub's output, so training chooses where the combination happens. Fit a continuous resolver index (the share of the combination variance computed within SC, from lesion and decoding analyses à la Kleinman). The index can then be swept by penalties or capacity.
- **Nuisance crossings.** Cross reward-channel sign (excitatory vs disinhibitory), hub capacity matched to SC capacity, noise level, regularisation, nonlinearity, and seeds. Use at least 20 seeds per cell.
- **Training.** Use the exact trial structure of both tasks. Match *behaviour*, including the monkeys' reward-salience exchange rate and RT distributions, before any neural comparison.

### (b) Diagnostic signatures and analyses (identical pipelines on models and data)
1. L-R choice axis, and decoding of reward difference and salience difference over time from the SC L-R signal and from single units (TDR, Mante 2013).
2. A competitor-normalisation test exactly as already run on the data.
3. Cross-unit alignment of goal vs stimulus factor coefficients (the PI's "not aligned" result). Determine its null distribution under H and R.
4. Pre-target gain structure and population/cluster analysis (Dubreuil; Yang 2019).
5. Residual dynamics of SC (Galgali 2023): the eigenvalue spectrum along the factor and choice axes, which tests recurrent vs input-driven.
6. RT as a function of ΔP vs factor components (conflict cost).
7. For each signature, compute the class-conditional distributions and the discriminability between H and R (AUC), then the likelihood ratio for the observed data value. Report which signatures are universal within a class (Maheswaranathan 2019) and which depend on hyperparameters.

### (c) In-silico perturbations
1. **Bicuculline in SC.** Scale down *all* inhibitory synapses (local, commissural, and SNr-to-SC) onto units in a retinotopic patch of one SC, with graded dose and spread. Run variants that spare the SNr synapses so their contribution is isolated.
   - Primary readout: the shift vs rotation of the indifference line in (Δreward, Δsalience) space, plus factor-specific loss of SC modulation at the site.
   - Secondary readouts: intrusive saccades (calibrate dose to keep these rare, matching what is experimentally feasible) and a rostral-spread control (Munoz & Wurtz 1993).
2. **Muscimol in SC.** Increase inhibitory conductance at the site. Check that both classes reproduce McPeek & Keller 2004, Nummela & Krauzlis 2010/2011 (choice bias, no single-target effect, reduced vector-average weight) and Lovejoy & Krauzlis 2010. These are calibration tests, not discriminators.
3. **Timed perturbations** (pre-target, target-to-competition window, pre-saccade), following Kopec 2015 and Duan 2021. Resolver: weighting changes only when SC is perturbed during the competition window. Hub: offset at all epochs.
4. **Hub manipulations** (striatal or SNr, simulated only, since SNr inactivation is not feasible). These confirm that, in H, upstream perturbation rotates the exchange rate. This is the mirror image of 1 and gives the argument its logical structure: "rotation lives where the weighting is computed".
5. **Lesion and ablation sanity checks.** Removing local SC inhibition should abolish choice in R but not in H (if H's WTA is upstream). Cutting the hub-to-SC projection should abolish reward effects in H.

Bottom line: the most decisive experiment is SC bicuculline in the conflict task, read out as an exchange-rate rotation versus a translation. The model ensembles should first show that this contrast is robust across seeds, hyperparameters and input-sign assumptions, given that SC GABA-A blockade unavoidably also blocks the nigral input.
