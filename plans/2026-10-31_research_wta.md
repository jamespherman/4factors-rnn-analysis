# SC winner-take-all / competition: report on three papers (+ supplement)

All pages of all four PDFs were read: Munoz & Istvan 1998 (17 pp), Trappenberg et al. 2001 (16 pp), Gopal & Hikosaka 2026 (28 pp), and the Gopal & Hikosaka supplement (10 pp: Figs S1-S5, Tables S1-S4). Quotations are verbatim. Where I note something as unclear, the paper itself is ambiguous or inconsistent.

---

## 1. Munoz & Istvan (1998). "Lateral inhibitory interactions in the intermediate layers of the monkey superior colliculus." J Neurophysiol 79:1193-1209

### Main claim, species, methods, task
- **Species:** 6 rhesus monkeys (c, p, g, a from NEI; l, j from Queen's).
- **Methods:** extracellular single-unit recording in the intermediate SC combined with microstimulation at a distant SC site. Stimulation was either brief trains (often 2 pulses at 500 Hz; 0.1-0.3 ms biphasic pulses; <100 µA) or prolonged low-frequency trains. Sites were in the "fixation zone" (rostral to the 2° isoamplitude line) or the "saccade zone" (caudal to the 10° line) of the ipsilateral or contralateral SC. The 2-10° band was deliberately never stimulated (Fig 3). Inhibition was defined as ≥5 consecutive 1-ms bins below the pre-stimulus mean, with ≥1 empty bin. Excitation was classed as orthodromic or antidromic by collision testing.
- **Tasks:** fixation, fixation-blink, visually guided, gap, memory-guided, and target-flash saccades (Fig 2). Stimulation was delivered while the recorded cell was active: around saccades into the movement field for saccade neurons, and during fixation for fixation neurons.
- **Sample:** 206 neurons: 100 saccade neurons (42 burst, 18 buildup, 40 unclassified), 84 fixation neurons, 22 visual (Table 1).
- **Main claim (abstract):** "a local network of inhibitory interneurons may help shape not only the reciprocal discharge pattern of fixation and saccade neurons but also permit lateral interactions between all regions of the ipsilateral and contralateral SC."

### Evidence for competition / WTA
- **Saccade neurons** (Table 2, Fig 5):
  - Inhibited by ipsilateral fixation-zone stimulation in 28/28 (latency 1.8 ± 0.6 ms).
  - Inhibited by contralateral fixation-zone stimulation in 15/17 (3.6 ± 1.4 ms).
  - Inhibited by ipsilateral saccade-zone stimulation in 27/35 (1.6 ± 1.0 ms; 2 excited, 1 antidromic). The recording and stimulation sites were mostly ≥2 mm apart.
  - Inhibited by contralateral saccade-zone stimulation in 18/20 (4.3 ± 1.4 ms). Inhibition from this site "tended to be weaker", but this was not quantified.
  - Burst and buildup latencies did not differ.
- **Fixation neurons** (Table 3):
  - Inhibited by ipsilateral saccade-zone stimulation in 36/47 (2 excited, 4 antidromic).
  - Inhibited by contralateral saccade-zone stimulation in only 8/24 (33%; 4.6 ± 1.3 ms).
  - Contralateral fixation-zone stimulation produced no inhibition in 0/13. Instead, 7/13 were orthodromically excited (1.8 ± 0.5 ms) and 2 were antidromic. The two fixation zones are therefore mutually excitatory.
- **Latencies:** ipsilateral inhibition was 1.7 ± 0.7 ms (n = 83) and contralateral inhibition 4.1 ± 1.4 ms (n = 36), p < 0.0005. The authors read the ipsilateral latency as monosynaptic and the contralateral latency as longer, possibly via commissural inhibitory neurons or via the ipsilateral fixation zone.
- **Superficial visual neurons** (n = 22) were unaffected by any stimulation.
- **Prolonged stimulation** (Figs 14-17):
  - Low-frequency stimulation of the contralateral fixation zone (50-125 Hz, 500 ms) suppressed burst neurons but left buildup neurons discharging at a low rate.
  - The paper concludes: "the efficacy of inhibition from the fixation zone onto burst neurons is much greater than onto buildup neurons." Fixation neurons and buildup neurons "may compete against one another via local inhibitory connections."
  - During prolonged saccade-zone stimulation, fixation cells resumed firing at the end of each evoked saccade even while stimulation continued (all 18 ipsilateral and all 11 contralateral cases tested).

### The WTA statement itself (Discussion, p. 1207)
"Our results also revealed strong inhibitory connections between different loci within the collicular saccade zones (see Fig. 13). These local inhibitory connections may play a very important role in selection of the next saccade target ... We suggest that such a network may serve as a winner-take-all mechanism before the onset of the saccade-related burst of the saccade neurons. The buildup activity of buildup neurons may represent this process."

### Model or architecture
- The paper proposes a conceptual circuit, not a computational model (Fig 1B). Fixation and saccade cells are reciprocally inhibitory through local interneurons, with selective excitatory or inhibitory drive from "higher inputs."
- Fig 18A-E lays out alternative wirings. The authors favour D and E: inhibitory interneurons with horizontal projections, driven either by recurrent collaterals or by shared afferents.
- Inhibition is long-range across the map (≥2 mm) and crosses the commissure. Fixation zones are mutually excitatory across the two colliculi.

### What is shown versus what is assumed
- **Shown:** at short latency, electrical activation of a distant SC site suppresses saccade-related discharge almost everywhere, in both colliculi.
- **Assumed:**
  - **That the inhibition is intrinsic.** The authors concede it could come from an axon reflex in GABAergic afferents that collateralize within the SC. SNr, zona incerta and the nucleus of the optic tract are named (Fig 18A): "it is possible that electrical stimulation of the SC may have led to activation of at least some of these inhibitory afferents." The argument for local inhibition is parsimony: inhibition dominated almost every pairing, and excitation appeared only between the fixation zones. Afferent collaterals are not excluded.
  - **That the inhibition is used for target selection or WTA.** No multi-target or choice task was run. The WTA role is a speculation.
  - **Distance dependence.** It was not measured: "It remains to be determined whether the distribution of intrinsic excitatory and inhibitory connections onto collicular neurons in the monkey varies systematically with distance between the recording and stimulation sites." They note that McIlwain (1982, cat) found excitation at <1 mm.
  - **Cell types.** No interneurons were identified. Recordings were biased toward large output neurons.

### Limitations
- Electrical microstimulation is non-selective: it activates fibres of passage and afferent collaterals. There was no pharmacology.
- Sampling was coarse: few cells per condition, no systematic site-distance mapping, and different monkeys, days and electrodes. The strength of inhibition could not be compared across sites.
- Behaviour was single-target only.

### Predictions testable with our data
- **Cross-hemisphere asymmetry.** In bilateral recordings, contralateral suppression of saccade-related activity should be weaker or later than ipsilateral, by about 2-3 ms more synaptic delay.
- **Fixation-like or tonic rostral cells.** Their activity should be anti-correlated with caudal saccade-related cells and positively coupled across the two colliculi.
- **CCG signature of local inhibition.** Short-latency (1-3 ms), monosynaptic-like troughs should appear in CCGs from putative inhibitory units onto distant saccade-related units in the same colliculus. Cross-colliculus troughs should be later and weaker.
- **E-I RNN fits.** The paper predicts I→E weights that are broad across the map: inhibition reaches all sites, not only nearby ones.

---

## 2. Trappenberg, Dorris, Munoz & Klein (2001). "A model of saccade initiation based on the competitive integration of exogenous and endogenous signals in the superior colliculus." J Cogn Neurosci 13:256-271

### Main claim, species, methods, task
- **Model:** a 1-D neural field model (Amari type) of the intermediate SC with N = 1001 nodes: 501 buildup/fixation nodes and 500 burst nodes, spanning 5 mm per colliculus with both colliculi on one line. Rate units have sigmoid output.
- **Inputs:** each node receives Gaussian exogenous (visual) and endogenous (task/voluntary) inputs. A global inhibition term u0 = 100 gates the burst nodes. It is released when buildup activity reaches 80% of maximum.
- **Fits:** the model was fit qualitatively to human and monkey saccadic RTs and to monkey SC buildup-cell activity in several paradigms:
  - gap effect;
  - distractor paradigm (Olivier et al. 1999 data);
  - pro- vs antisaccades (human data from Forbes & Klein 1996);
  - target-location probability (80/20; human data);
  - express saccades.
- **Claim:** "our model assumes that this information is integrated by dynamic competition across local collicular interactions."

### Competition mechanism and evidence
- **Interaction kernel:** w_ij = a·exp(−d²/2σa²) − b·exp(−d²/2σb²) − c, with a = 144, b = 44, c = 16, σa = 0.6 mm and σb = 3σa. This is a "Mexican hat" with short-range excitation, longer-range inhibition, and a constant global inhibition c that reaches across both colliculi.
- **Empirical basis (Figs 1B-D, 2):**
  - About 40 buildup neurons were recorded (Olivier et al.; note 1). Only 2 cells are shown in Fig 2.
  - A green distractor appeared 100 ms before the target, at varied map positions relative to the cell's response field.
  - The interaction index ρw (distractor-evoked change in buildup activity, normalised) is about +1 within about 1 mm. It falls to roughly −0.1 to −0.2 beyond about 1.5 mm and stays slightly negative out to 7 mm (Fig 2).
  - In other words, distractors near the response field enhance buildup activity, and distant distractors suppress it.
  - The authors say the profile "is roughly independent of the location of the cell within the SC" (translation invariant), on the basis of 2 cells.
- **Behaviour:**
  - Remote distractors lengthen SRT, and near distractors shorten it (Fig 3A).
  - The slowing peaks when the distractor and target are nearly simultaneous (Fig 3B).
  - Antisaccades are slower because the visual stimulus acts like a remote distractor to the endogenous goal location (Fig 4).
  - Target probability is modelled as a weak pretarget endogenous input (a_endo = 3) at the likely location, which speeds RT especially in the gap condition (Fig 5).
  - Express saccades arise when noise lets the visual burst cross threshold on top of elevated buildup (Fig 6). In my rendering of the PDF, Fig 6B-C showed empty axes. The traces did not render, so I cannot describe them.

### The crucial caveat on where the competition lives
The paper does not claim that the effective kernel is intra-collicular. The key sentence:
"It should be noted that this effective interaction pattern need not result from neural mechanisms within the SC alone. This form of w could, for example, result from a combination of short-range excitatory SC neurons and a target-sensitive inhibition from other brain areas. A prime candidate for extra-collicular inhibition would be projections from the substantia nigra pars reticulata of the basal ganglia (see Hikosaka, 1989 for a review)."

They also acknowledge that the ρw profile "does not provide a direct measurement of the synaptic efficacy. Rather, it is the sum over all cell activities in the layer multiplied with the individual synaptic efficacy." It is an effective (functional) connectivity measure, not anatomy.

### What is shown versus what is assumed
- **Shown:** a single distance-dependent functional interaction profile (near facilitation, remote suppression) in buildup neurons. A Mexican-hat field reproduces a set of RT effects and buildup waveforms.
- **Assumed:**
  - **Endogenous (goal-directed) signals arrive pre-computed from outside the SC.** Their strength, timing and location are set per paradigm. Target probability is simply a larger endogenous input. The SC does not decide which factor wins. It sums inputs and lets the field dynamics pick the peak, so goal and stimulus factors "compete" only as differently weighted inputs at different map locations.
  - The kernel is fixed and has no learning.
  - Burst gating is an instantaneous global switch.
  - Parameters are a qualitative fit.
  - The model is 1-D and does not represent the two colliculi as separate populations with distinct commissural weights. A single translation-invariant kernel spans both, with the midline as just another distance.

### Limitations
- The kernel rests on 2 example cells; the full dataset was "in preparation."
- SRTs were not fit quantitatively.
- There is no explicit SNr, cortex or other loop, yet the authors explicitly allow that the inhibitory part may be extra-collicular.
- Only one class of competing input is spatially distinct (target vs distractor). Same-location competition between factor classes is not modelled.
- The noise is ad hoc.

### Predictions testable with our data
- **Distance-dependent sign of interaction.** Activity at the RF should be facilitated by stimuli within about 1 mm (on the order of 10° near the fovea, less caudally) and suppressed by stimuli farther away. Our single-target task cannot test this directly. The bilateral conflict task, if salient and rewarded stimuli fall at different map distances, can test whether suppression of the losing representation scales with map distance.
- **Normalisation-like suppression.** Suppression should be weak (about 10-20% of the near-facilitation magnitude) and roughly flat with distance beyond about 1.5 mm, including across the midline. The model predicts cross-colliculus suppression of similar size to long-range ipsilateral suppression. Munoz & Istvan suggest it is weaker or later. Bilateral recordings can adjudicate.
- **Timing.** The model predicts a competitive effect only in a narrow temporal window around near-simultaneous onsets (Fig 3B), with interactions emerging at the buildup stage, before the burst.
- **Pretarget bias from probability.** Probability should appear as elevated pretarget buildup at the likely location (Fig 5C; also Dorris & Munoz 1998, Basso & Wurtz 1998). This is testable in our single-target task: compare baseline and buildup by probability condition.
- **E-I RNN.** If competition is intrinsic and Mexican-hat-like, fitted E→E weights should be positive at short map distance and I→E weights broad. If W_rec shows no distance-dependent structure while the fits still reproduce suppression, the suppression is being carried by inputs, which fits the paper's own SNr caveat.

---

## 3. Gopal & Hikosaka (2026). "Multiple groups of neurons in the superior colliculus convert value signals into saccadic vigor." iScience 29:114563 (+ supplement)

### Main claim, species, methods, task
- **Species:** 2 male rhesus monkeys (DWN, 22 sessions; BLY, 10 sessions). The total is stated as 33 sessions, but 22 + 10 = 32. This is an inconsistency in the paper.
- **Recording:** a 24-channel Plexon V-probe (50 or 100 µm spacing) spanning up to 2.4 mm of SC depth. Recordings were mostly in middle and caudal SC, in both colliculi (Fig 1E). Kilosort 2.5 with Phy curation gave 854 units, of which 520 were classified.
- **Task:** visually guided saccade to a single fractal object (about 7°) in the RF, or at the 180° opposite location. Each monkey had 4 sets of 8 fractals: 4 high value (250 ms juice) and 4 low value (50 ms). The objects had been learned over >10 days ("stable value", the caudate-tail/cdlSNr circuit).
  - Each 128-trial block also contained equal- and unequal-value choice trials. **Only single-object RF trials are analysed.** Choice trials and opposite-location trials are not reported.
  - In choice trials the monkeys picked the high-value object >98% of the time.
- **Behaviour (Fig 2):**
  - RT: high 186 ± 4 vs low 211 ± 6 ms (p < 0.001; 26/33 sessions significant).
  - Peak velocity: high 582 ± 40 vs low 555 ± 38 °/s (p < 0.001; 16/33 sessions).
  - Amplitude did not differ (12.78 vs 12.89°).
  - RT and PV modulation were uncorrelated across sessions (r = −0.22, p = 0.22).
- **Cell classes (methods p. e3):**
  - visual, n = 233;
  - visuomotor, n = 88;
  - motor, n = 95;
  - tonic, n = 89 (88 in some places).
  - These sum to 505, not 520. The remainder is unexplained.
  - Classification used baseline (−200 to 0), phasic 1 (0-80), phasic 2 (80-160) and pre-saccadic windows (−100 to −50 vs −40 to 0).

### Key findings
- **Value modulation** (Fig 3, Table S1; target-aligned window 80-170 ms):
  - Visual: 27 ± 2 vs 22 ± 2 sp/s (55% positive, 25% negative, 20% none).
  - Visuomotor: 56 ± 5 vs 46 ± 5 (70% positive, 10% negative).
  - Motor: 37 ± 4 vs 24 ± 3 (74% positive, 4% negative).
  - Presaccadic window (−40 to 0) differences were smaller, for example motor 89 vs 83 (p = 0.031).
  - Baseline was not modulated.
  - All effects survived vigor matching (Table S1).
  - In a pooled GLM, β_value was 0.20 (visual), 0.15 (visuomotor) and 0.21 (motor), all p < 0.001. Cross-validated R² was small: 0.013, 0.063 and 0.207 respectively (Fig S5, Table S2).
  - The Table S2 caption says "visual, visuomotor and tonic," but the rows are motor neurons. This is a caption error.
- **Response phases** (Fig 5; bootstrap over pseudo-populations of 25 neurons by RT tercile; Table S3):
  - E_VIS (about 40 ms peak) is time-locked to the target and does not vary with RT.
  - L_VIS (peak about 100-110 ms) is target-locked.
  - Pre_SAC is saccade-locked, about 9-14 ms before onset in visuomotor and motor cells.
  - Value affects L_VIS and Pre_SAC but not E_VIS (Fig 6, Table S4).
- **Value onset latency** (Fig S4):
  - Population: visual 94 ms, visuomotor 91 ms, motor 103 ms.
  - Single cells: 112 ± 4, 100 ± 4 and 108 ± 3 ms.
  - The differences are not significant (F = 1.71, p = 0.18).
  - The authors compare this to SNr value coding at about 100 ms and propose that cdlSNr disinhibition "interface[s] with visual, visuomotor, and motor neurons in SC, thereby inducing value responses quickly."
- **Neuron-behaviour link** (Fig 8):
  - Value index correlated with RT index in visuomotor (r = −0.40), motor (−0.33) and tonic (−0.43) cells, but not in visual cells (−0.01).
  - Value index correlated with PV index in visual (+0.21), motor (+0.24) and tonic (+0.31) cells.
  - About 77% of cells were value-modulated, but only about 30% were RT-modulated and about 8% PV-modulated (Fig 9).
- **"Tonic neurons", a claimed new class** (Fig 4, Fig 7):
  - High firing at baseline (45 ± 5 sp/s), suppressed to about 12 sp/s at 40-60 ms latency by an object in the RF. The suppression lasts through the saccade.
  - RFs are eccentric (4-25°) and the cells are found in caudal tracks, intermingled with other types.
  - Most (77/88) are unaffected by fixation-spot onset.
  - Value modulation is mixed in sign (42% negative, 38% positive, 20% none) and nil at the population level.
  - The time to minimum orders weakly with RT (157/172/191 ms; p = 0.50, not significant).
  - The authors speculate: "the tonic neurons may be part of the local inhibitory network, presumably the GABAergic neurons ... these local inhibitory neurons may modulate the firing of premotor neurons ... or further inhibit the non-target motor neurons, thereby sharpening the activity in the SC motor maps." They also speculate: "when there are no objects in the RF, higher baseline firing in the tonic neurons may inhibit the interaction between the other classes of SC neurons ... When an object ... appears, tonic neurons lower their firing activity, which would disinhibit visual, visuomotor, and motor neurons."

### Circuit studied, and bearing on resolver vs hub
- Only SC was recorded. SNr, caudate and FEF are invoked as interpretation, not measured. The paper's framing is explicitly hub-like: SC is "an important hub of sensorimotor transformation," acting "as a bridge between circuits that process value information and those that control motor actions." Value "may originate" from cdlSNr, FEF and LIP. The authors argue that only later phases are value-modulated, so that "reward value does not indiscriminately alter sensory representations but selectively modulates the later integrative stages."
- The only local-circuit claims concern sequential propagation (value onset visuomotor 95 → visual 98 → motor 113 ms, "statistically not significant") and the speculative disinhibitory role of tonic neurons.
- **Relevance to resolver vs hub:**
  1. With a single target, there is no competition to resolve. The paper establishes value as an input-like gain on SC responses that arrives at about 90-110 ms, about the same time as SNr value signals. That fits a hub reading, but it is not decisive. Simultaneous SNr-SC recording, inactivation and choice-trial analysis are all absent. The unanalysed choice trials (equal/unequal value pairs) would be the relevant dataset.
  2. Value modulation is absent from the earliest visual response. This argues that value is not a stimulus property computed locally from the visual drive. If our salience effect is present in E_VIS while reward appears only in L_VIS, the two factor classes enter at different times and probably from different sources. Any SC-internal competition between them would then be time-structured.
  3. Tonic, suppressed-by-stimulus cells with mixed value sign are candidate putative inhibitory units. They match our "task-modulated but non-discriminating" putative interneurons: task-modulated, non-selective for the factor at the population level. That is consistent with our core hypothesis that the I population does not carry factor-specific structure. Gopal & Hikosaka do not identify these cells as inhibitory: no waveform or optogenetic tagging, and no CCGs.
  4. The authors argue against a salience confound for value: the objects are fractals with physical features randomised across value. They also cannot separate value from attention or motivation.

### Limitations (paper's own and additional)
- Correlational only. The authors note that the cdlSNr mechanism "must be causally tested."
- The four-category classification is coarse; the authors suggest unsupervised clustering.
- Value, attention, motivation and vigor are entangled.
- Additional problems:
  - Choice trials were collected but not analysed.
  - Effect sizes are small relative to variance (GLM R² 0.01-0.2).
  - Window definitions differ across text, figures and tables (80-160, 80-170, 81-130 and 131-170 ms).
  - Counts are inconsistent (88 vs 89 tonic cells; 33 vs 32 sessions; Table S1 tonic SEMs of 17-18 sp/s look like SDs or typos).
  - Text and Table S3 disagree on whether the visual-cell Pre_SAC(T) peak varies with RT (table p < 0.001).
  - No CCGs, noise correlations or bilateral analysis.

### Predictions testable with our data
- In our single-target task, the onset of reward (and probability) modulation should be about 90-110 ms and absent from the first visual transient. Salience may differ: it is a physical property and may already appear in E_VIS. Compare factor-onset latencies by cell class.
- Suppressed-by-stimulus tonic units should exist in our data. Test whether they are our putative I units, whether their factor modulation is mixed in sign and null on average, and whether their CCGs with E units show troughs. A trough would support local inhibition; flat CCGs would not.
- The relation between value index and RT index per neuron (Fig 8) can be replicated per factor. A resolver predicts that factor modulation in I units predicts behaviour, because they would implement the selection. A hub predicts that only E/motor units carry behaviourally predictive factor signals.

---

## 4. What is shown across the three papers versus what the PI's framing assumes

- **Established:**
  - Short-latency, map-wide inhibition between SC sites, ipsilateral and contralateral, follows electrical stimulation (paper 1).
  - Functional near facilitation and far suppression of buildup activity occur with a distractor (paper 2).
  - SC visual, visuomotor and motor activity is gain-modulated by learned value at about 100 ms (paper 3).
- **Not established:**
  - That the inhibition is intrinsic rather than from afferent collaterals (SNr, ZI, NOT). Paper 1 concedes this.
  - That the SC-level "WTA" is computed by SC circuitry. Paper 2 explicitly names SNr as a candidate source of the inhibitory surround.
  - That any of this resolves competition between factor classes. None of the papers puts two factor classes in conflict. Paper 2 treats goal-directed signals as externally supplied inputs, and paper 3 analyses only single targets.
- "Competitive inhibitory interactions within and between the colliculi" is therefore a fair citation of papers 1 and 2 for the existence of lateral inhibition and a WTA-like effective kernel. It is not a citation for a demonstrated intra-SC resolution of factor conflict.

---

## 5. Synthesis (~400 words)

Taken together, the three papers establish that SC sits inside a competitive architecture. They do not establish where the decision between factor classes is made.

Munoz & Istvan show that stimulating almost any SC site suppresses saccade-related activity almost everywhere else, with about 1.7 ms ipsilateral and about 4 ms contralateral latency. That is the physiological basis for "competitive inhibition within and between the colliculi." But they concede that afferent GABAergic collaterals, explicitly including SNr, could produce the same result, and they never measured distance dependence or ran a selection task.

Trappenberg et al. turn the idea into a Mexican-hat field with near excitation and broad, weak (about −0.1 to −0.2) far inhibition, fit to distractor effects in about 2 example buildup cells. Their endogenous (goal) signals are supplied fully formed from outside, and they state that the inhibitory surround "need not result from neural mechanisms within the SC alone," naming SNr.

Gopal & Hikosaka show that learned value enters SC as a gain at about 90-110 ms, synchronous with SNr value signals, and spares the earliest visual response. They describe tonic, suppressed-by-stimulus cells with mixed-sign value coding, which are plausible inhibitory "carry-out" elements, and they frame SC as a hub.

So the literature most strongly supports a **hub-with-local-execution** picture. SC has the machinery to implement WTA: broad, factor-agnostic lateral inhibition. Goal-directed factors (reward, probability) arrive as pre-weighted inputs, probably via BG/SNr and cortex. Stimulus-driven factors (salience) may enter earlier through the visual path. The resolver hypothesis would need evidence that SC's own inhibition is *factor-specific* or that SC dynamics flip outcomes independently of input. No paper provides that.

**Most decisive analyses with our data:**
1. **Factor-specificity of fitted I→E weights (RNN).** Test whether W_rec[I→E] is predicted by target-E factor selectivity, against a shuffled-label null. Also test whether it is predicted by map distance (a Mexican hat), which the existing papers do support. A resolver needs factor-structured inhibition. A hub predicts distance-only or unstructured inhibition, with factor effects carried by inputs. Complement this by refitting the model with factor information removed from inputs: a resolver should still reproduce choice; a hub should not.
2. **CCG troughs from putative I units.** Measure 1-4 ms troughs from putative I units onto E units. Test whether trough depth depends on the E unit's factor preference (resolver) or only on distance and hemisphere (hub). Ipsilateral troughs should be at about 1-3 ms and contralateral troughs later and weaker.
3. **Bilateral conflict task: timing and anticorrelation.**
   - Find the time at which choice becomes decodable from left vs right SC population difference, and compare it to factor-onset latencies (reward about 100 ms per Gopal & Hikosaka; salience possibly earlier).
   - Check whether, on matched-stimulus trials, trial-by-trial noise correlations between the two colliculi (or between the two competing map loci) are negative and become more negative before the choice. Growing anticorrelation that precedes choice-predictive divergence would point to an SC-internal resolver.
   - Signal arriving as a common input already biased toward the winner, with no growing anticorrelation, points to a hub.
4. **Normalisation and divisive signature.** In conflict trials, test whether the response to each stimulus is reduced, relative to its single-target (unilateral task) response, in proportion to the strength of the competitor (salience level × reward level) and to map distance (Trappenberg kernel). Compare factor-general against factor-specific scaling.
5. **Factor-onset latencies by cell class (single-target data).** Compare salience vs reward vs probability vs identity onset in E vs putative I units, and E_VIS vs L_VIS. If I units are factor-agnostic and reward enters late with SNr-like timing, the evidence favours a hub.
